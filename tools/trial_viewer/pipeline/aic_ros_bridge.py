"""Minimal ROS 2 bridge for AIC data collection.

Subscribes to:
  - /left_camera/image, /center_camera/image, /right_camera/image  (sensor_msgs/Image, rgb8)
  - /joint_states                                                  (sensor_msgs/JointState)
  - /fts_broadcaster/wrench                                        (geometry_msgs/WrenchStamped)
  - /aic_controller/controller_state                               (aic_control_interfaces/ControllerState)

Publishes:
  - /aic_controller/pose_commands                                  (MotionUpdate)

Calls service:
  - /aic_controller/change_target_mode                             (Cartesian)

Designed to be agnostic to LeRobotDataset — returns plain dicts so we can
unit-test exploration policies offline too.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from aic_control_interfaces.msg import (
    ControllerState,
    MotionUpdate,
    TargetMode,
    TrajectoryGenerationMode,
)
from aic_control_interfaces.srv import ChangeTargetMode
from geometry_msgs.msg import Twist, Vector3, Wrench, WrenchStamped
from sensor_msgs.msg import Image, JointState

log = logging.getLogger(__name__)

CAMERA_TOPICS = {
    "left":   "/left_camera/image",
    "center": "/center_camera/image",
    "right":  "/right_camera/image",
}


def image_msg_to_numpy(msg: Image) -> np.ndarray:
    """Decode sensor_msgs/Image to HxWx3 uint8 RGB ndarray.

    Handles rgb8 and bgr8 with arbitrary row-stride padding.
    """
    h, w = msg.height, msg.width
    enc = msg.encoding
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if enc in ("rgb8", "bgr8"):
        # Row stride may exceed w*3 (padding)
        if msg.step == w * 3:
            arr = buf.reshape(h, w, 3)
        else:
            arr = buf.reshape(h, msg.step)[:, : w * 3].reshape(h, w, 3)
        if enc == "bgr8":
            arr = arr[:, :, ::-1]
        return np.ascontiguousarray(arr)
    if enc == "mono8":
        return np.stack([buf.reshape(h, w)] * 3, axis=-1)
    raise ValueError(f"Unsupported image encoding: {enc}")


@dataclass
class BridgeConfig:
    image_size: int = 224          # square output, used when image_hw is None
    image_hw: Optional[Tuple[int, int]] = None  # (h, w); when set, overrides image_size
    obs_timeout_s: float = 5.0     # how long to wait for first messages of each topic
    target_stiffness: float = 85.0
    target_damping: float = 75.0
    frame_id: str = "gripper/tcp"


class AICRosBridge:
    """Owns its own rclpy Node + executor thread."""

    def __init__(self, config: Optional[BridgeConfig] = None):
        self.config = config or BridgeConfig()
        self._lock = threading.Lock()
        self._last_imgs: Dict[str, Optional[Image]] = {k: None for k in CAMERA_TOPICS}
        self._last_joint_states: Optional[JointState] = None
        self._last_controller_state: Optional[ControllerState] = None
        self._last_wrench: Optional[WrenchStamped] = None
        self._node: Optional[Node] = None
        self._executor: Optional[SingleThreadedExecutor] = None
        self._executor_thread: Optional[threading.Thread] = None
        self._is_connected = False
        self._motion_update_pub = None
        self._change_target_mode_client = None

    # --- lifecycle -----------------------------------------------------------

    def connect(self) -> None:
        if self._is_connected:
            return
        if not rclpy.ok():
            rclpy.init()
        self._node = Node("lewm_aic_bridge")
        n = self._node

        # Cameras
        for key, topic in CAMERA_TOPICS.items():
            n.create_subscription(
                Image, topic, lambda m, k=key: self._on_image(k, m),
                qos_profile_sensor_data,
            )

        # Joint states
        n.create_subscription(
            JointState, "/joint_states", self._on_joint_states, qos_profile_sensor_data
        )

        # Controller state
        n.create_subscription(
            ControllerState, "/aic_controller/controller_state",
            self._on_controller_state, 10,
        )

        # FT wrench (best-effort — topic may not always be present)
        n.create_subscription(
            WrenchStamped, "/fts_broadcaster/wrench", self._on_wrench,
            qos_profile_sensor_data,
        )

        # Pose-command publisher
        self._motion_update_pub = n.create_publisher(
            MotionUpdate, "/aic_controller/pose_commands", 10
        )

        # Change-target-mode service client
        self._change_target_mode_client = n.create_client(
            ChangeTargetMode, "/aic_controller/change_target_mode"
        )

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(n)
        self._executor_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._executor_thread.start()

        # Wait for controller service then change to cartesian
        self._wait_and_set_cartesian_mode()
        self._is_connected = True

    def _wait_and_set_cartesian_mode(self, timeout_s: float = 30.0) -> None:
        assert self._change_target_mode_client is not None
        log.info("Waiting for /aic_controller/change_target_mode service ...")
        t0 = time.monotonic()
        while not self._change_target_mode_client.wait_for_service(timeout_sec=1.0):
            if time.monotonic() - t0 > timeout_s:
                raise TimeoutError("change_target_mode service never appeared")
        req = ChangeTargetMode.Request()
        req.target_mode.mode = TargetMode.MODE_CARTESIAN
        log.info("Setting controller to CARTESIAN mode")
        fut = self._change_target_mode_client.call_async(req)
        # call_async returns a Future; we spin on executor thread already, so
        # we can just wait on fut.done().
        t0 = time.monotonic()
        while not fut.done():
            if time.monotonic() - t0 > 10.0:
                raise TimeoutError("change_target_mode call timed out")
            time.sleep(0.05)
        resp = fut.result()
        if resp is None or not getattr(resp, "success", False):
            log.warning("change_target_mode response not success: %r", resp)
        time.sleep(0.5)

    def wait_for_data(self, timeout_s: Optional[float] = None) -> None:
        """Block until at least one of each subscribed topic has arrived."""
        timeout_s = timeout_s or self.config.obs_timeout_s
        t0 = time.monotonic()
        while True:
            if (
                all(self._last_imgs[k] is not None for k in CAMERA_TOPICS)
                and self._last_joint_states is not None
                and self._last_controller_state is not None
            ):
                # wrench is best-effort
                return
            if time.monotonic() - t0 > timeout_s:
                missing = [k for k, v in self._last_imgs.items() if v is None]
                if self._last_joint_states is None: missing.append("joint_states")
                if self._last_controller_state is None: missing.append("controller_state")
                raise TimeoutError(f"Timed out waiting for topics: missing={missing}")
            time.sleep(0.05)

    def disconnect(self) -> None:
        if not self._is_connected:
            return
        try:
            self.send_zero_velocity()
        except Exception:
            pass
        if self._executor is not None:
            self._executor.shutdown()
        if self._node is not None:
            self._node.destroy_node()
        if self._executor_thread is not None and self._executor_thread.is_alive():
            self._executor_thread.join(timeout=1.0)
        self._is_connected = False

    # --- callbacks -----------------------------------------------------------

    def _on_image(self, key: str, msg: Image) -> None:
        with self._lock:
            self._last_imgs[key] = msg

    def _on_joint_states(self, msg: JointState) -> None:
        with self._lock:
            self._last_joint_states = msg

    def _on_controller_state(self, msg: ControllerState) -> None:
        with self._lock:
            self._last_controller_state = msg

    def _on_wrench(self, msg: WrenchStamped) -> None:
        with self._lock:
            self._last_wrench = msg

    # --- observation accessors -----------------------------------------------

    def get_observation(self) -> Dict[str, Any]:
        """Return a flat dict with all observation features.

        Image keys: 'observation.images.left|center|right' -> (H, W, 3) uint8
        State key:  'observation.state' -> (32,) float32
        Plus individual scalars for the policy to use (tcp_pose.position.*).
        """
        with self._lock:
            imgs_msg = {k: v for k, v in self._last_imgs.items()}
            js = self._last_joint_states
            cs = self._last_controller_state
            ws = self._last_wrench

        if any(v is None for v in imgs_msg.values()) or js is None or cs is None:
            raise RuntimeError("observation not ready — call wait_for_data first")

        # Decode + resize images
        if self.config.image_hw is not None:
            target_h, target_w = self.config.image_hw
        else:
            target_h = target_w = self.config.image_size
        imgs: Dict[str, np.ndarray] = {}
        for k, m in imgs_msg.items():
            arr = image_msg_to_numpy(m)
            if arr.shape[0] != target_h or arr.shape[1] != target_w:
                arr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_AREA)
            imgs[k] = arr

        # tcp_pose, velocity, error
        tp = cs.tcp_pose
        tv = cs.tcp_velocity
        te = cs.tcp_error  # length 6 array
        # Joint positions (pad/truncate to 7)
        joints = list(js.position)
        if len(joints) < 7:
            joints = joints + [0.0] * (7 - len(joints))
        else:
            joints = joints[:7]

        # Wrench (zero if not yet received)
        if ws is None:
            wf, wt = [0.0]*3, [0.0]*3
        else:
            wf = [ws.wrench.force.x, ws.wrench.force.y, ws.wrench.force.z]
            wt = [ws.wrench.torque.x, ws.wrench.torque.y, ws.wrench.torque.z]

        state_vec = np.array(
            [
                tp.position.x, tp.position.y, tp.position.z,
                tp.orientation.x, tp.orientation.y, tp.orientation.z, tp.orientation.w,
                tv.linear.x, tv.linear.y, tv.linear.z,
                tv.angular.x, tv.angular.y, tv.angular.z,
                float(te[0]), float(te[1]), float(te[2]),
                float(te[3]), float(te[4]), float(te[5]),
                *joints,
                *wf, *wt,
            ],
            dtype=np.float32,
        )
        # 7 + 6 + 6 + 7 + 6 = 32

        return {
            "observation.images.left":   imgs["left"],
            "observation.images.center": imgs["center"],
            "observation.images.right":  imgs["right"],
            "observation.state": state_vec,
            # scalar helpers for policies that need tcp pose:
            "tcp_pose.position.x": float(tp.position.x),
            "tcp_pose.position.y": float(tp.position.y),
            "tcp_pose.position.z": float(tp.position.z),
        }

    # --- action sending ------------------------------------------------------

    def send_action(self, action: Dict[str, float]) -> None:
        if not self._is_connected or self._motion_update_pub is None or self._node is None:
            raise RuntimeError("bridge not connected")
        twist = Twist()
        twist.linear.x = float(action.get("linear.x", 0.0))
        twist.linear.y = float(action.get("linear.y", 0.0))
        twist.linear.z = float(action.get("linear.z", 0.0))
        twist.angular.x = float(action.get("angular.x", 0.0))
        twist.angular.y = float(action.get("angular.y", 0.0))
        twist.angular.z = float(action.get("angular.z", 0.0))

        msg = MotionUpdate()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = self.config.frame_id
        msg.velocity = twist
        K = self.config.target_stiffness
        D = self.config.target_damping
        msg.target_stiffness = np.diag([K]*6).flatten()
        msg.target_damping = np.diag([D]*6).flatten()
        msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        msg.wrench_feedback_gains_at_tip = [0.0]*6
        msg.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY
        self._motion_update_pub.publish(msg)

    def send_zero_velocity(self) -> None:
        self.send_action({k: 0.0 for k in (
            "linear.x", "linear.y", "linear.z",
            "angular.x", "angular.y", "angular.z",
        )})
