#!/usr/bin/env python3
"""Record observations + commanded actions from a live aic_eval trial to a
single rerun .rrd file. Intended to run on host alongside aic_model.

Subscribes to (via AICRosBridge):
  - cameras left/center/right
  - /joint_states
  - /aic_controller/controller_state    (TCP pose/velocity/error)
  - /fts_broadcaster/wrench             (force/torque)
And additionally:
  - /aic_controller/pose_commands       (MotionUpdate — both .pose and .velocity targets)

Logged entities (in addition to camera streams and the original state vector):
  state/wrench/force/x|y|z              per-axis labelled splits
  state/wrench/force_lateral_norm       √(Fx² + Fy²) — wire-tension proxy
  state/tcp/error_pos_norm              ||tcp_error[0:3]||
  action/pose/position                  Pose target translation (3-vec in base_link)
  action/pose/orientation               Pose target quat (x,y,z,w)
  action/trajectory_mode                Enum value of MotionUpdate.trajectory_generation_mode
  command/pos_residual/x|y|z + _norm    action.pose.position − state.tcp.position
  command/lin_vel_residual/x|y|z + _norm action.velocity.linear − state.tcp.lin_vel
  world/tcp                             current TCP point (per frame)
  world/tcp_path                        TCP trajectory (static, logged once at end)
  world/action/pose_target              pose target point (per frame)
  world/action/pose_path                pose target trajectory (static, at end)
  world/tcp/force_arrow                 wrench force as world-frame arrow at TCP

Note on residuals: CheatCode commands `.pose` (the velocity field is zero),
so `lin_vel_residual` ≈ −measured velocity and is uninformative for those
trials. A velocity-mode policy would leave `.pose` default-constructed
(zero, invalid quaternion) and `pos_residual` would be meaningless instead.
Inspect `action/trajectory_mode` to know which residual is the real one.

On SIGINT/SIGTERM (or --max-seconds), flushes the rrd and exits.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/ubuntu/lewm_run/scripts")
from aic_ros_bridge import AICRosBridge, BridgeConfig  # noqa: E402

from aic_control_interfaces.msg import MotionUpdate  # noqa: E402
import rerun as rr  # noqa: E402


def encode_png(rgb_uint8: np.ndarray) -> bytes:
    """Encode an HxWx3 uint8 RGB image as PNG bytes (cv2 expects BGR)."""
    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def rotate_by_quat(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate 3-vector v by quaternion q in (x,y,z,w) order. Standard form."""
    qx, qy, qz, qw = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    ux = qy * vz - qz * vy
    uy = qz * vx - qx * vz
    uz = qx * vy - qy * vx
    tx = qy * uz - qz * uy
    ty = qz * ux - qx * uz
    tz = qx * uy - qy * ux
    return np.array([
        vx + 2.0 * (qw * ux + tx),
        vy + 2.0 * (qw * uy + ty),
        vz + 2.0 * (qw * uz + tz),
    ], dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output .rrd path")
    ap.add_argument("--meta-out", default=None, help="Optional small JSON summary path")
    ap.add_argument("--rate-hz", type=float, default=10.0)
    ap.add_argument("--max-seconds", type=float, default=300.0)
    ap.add_argument("--name", default="trial")
    ap.add_argument("--obs-timeout-s", type=float, default=30.0,
                    help="How long to wait for first messages on each topic.")
    ap.add_argument("--image-h", type=int, default=224)
    ap.add_argument("--image-w", type=int, default=224)
    args = ap.parse_args()

    bridge = AICRosBridge(BridgeConfig(
        image_size=max(args.image_h, args.image_w),
        image_hw=(args.image_h, args.image_w),
        obs_timeout_s=args.obs_timeout_s,
    ))
    bridge.connect()

    last_action = {
        "linear":      np.zeros(3, dtype=np.float32),
        "angular":     np.zeros(3, dtype=np.float32),
        "pose_pos":    np.zeros(3, dtype=np.float32),
        "pose_quat":   np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),  # identity
        "mode":        0,
        "ts":          0.0,
    }
    action_count = {"n": 0}

    def on_action(msg):
        v = msg.velocity
        p = msg.pose
        last_action["linear"]  = np.array([v.linear.x, v.linear.y, v.linear.z], dtype=np.float32)
        last_action["angular"] = np.array([v.angular.x, v.angular.y, v.angular.z], dtype=np.float32)
        last_action["pose_pos"] = np.array(
            [p.position.x, p.position.y, p.position.z], dtype=np.float32,
        )
        last_action["pose_quat"] = np.array(
            [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w],
            dtype=np.float32,
        )
        last_action["mode"] = int(msg.trajectory_generation_mode.mode)
        last_action["ts"] = time.time()
        action_count["n"] += 1

    bridge._node.create_subscription(
        MotionUpdate, "/aic_controller/pose_commands", on_action, 10
    )

    try:
        bridge.wait_for_data(timeout_s=args.obs_timeout_s)
    except Exception as e:
        print(f"[record] obs not ready: {e}", file=sys.stderr, flush=True)
        bridge.disconnect()
        return 2

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rr.init(args.name, recording_id=args.name, spawn=False)
    rr.save(args.out)

    stop_evt = threading.Event()

    def handle_sig(*_a):
        stop_evt.set()

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    dt = 1.0 / args.rate_hz
    t_start = time.time()
    n = 0
    tcp_positions: list[np.ndarray] = []   # accumulated for static path log at end
    pose_targets:  list[np.ndarray] = []   # accumulated for static pose-target path log
    while not stop_evt.is_set() and (time.time() - t_start) < args.max_seconds:
        loop_t = time.time()
        ts = loop_t - t_start
        try:
            obs = bridge.get_observation()
        except Exception:
            time.sleep(dt)
            continue
        rr.set_time("time", duration=ts)

        for cam_key, obs_key in (
            ("left",   "observation.images.left"),
            ("center", "observation.images.center"),
            ("right",  "observation.images.right"),
        ):
            rr.log(
                f"cam/{cam_key}",
                rr.EncodedImage(contents=encode_png(obs[obs_key]), media_type="image/png"),
            )

        s = obs["observation.state"].astype(float)
        rr.log("state/tcp/position",      rr.Scalars(s[0:3].tolist()))
        rr.log("state/tcp/orientation",   rr.Scalars(s[3:7].tolist()))
        rr.log("state/tcp/lin_vel",       rr.Scalars(s[7:10].tolist()))
        rr.log("state/tcp/ang_vel",       rr.Scalars(s[10:13].tolist()))
        rr.log("state/tcp/error",         rr.Scalars(s[13:19].tolist()))
        rr.log("state/joints",            rr.Scalars(s[19:26].tolist()))
        rr.log("state/wrench/force",      rr.Scalars(s[26:29].tolist()))
        rr.log("state/wrench/torque",     rr.Scalars(s[29:32].tolist()))
        rr.log("state/wrench/force_norm", rr.Scalars([float(np.linalg.norm(s[26:29]))]))

        # Per-axis splits + derived diagnostic scalars.
        rr.log("state/wrench/force/x", rr.Scalars([float(s[26])]))
        rr.log("state/wrench/force/y", rr.Scalars([float(s[27])]))
        rr.log("state/wrench/force/z", rr.Scalars([float(s[28])]))
        rr.log("state/wrench/force_lateral_norm",
               rr.Scalars([float(np.hypot(s[26], s[27]))]))
        rr.log("state/tcp/error_pos_norm",
               rr.Scalars([float(np.linalg.norm(s[13:16]))]))

        # Action — both velocity and pose targets, plus the trajectory mode so
        # the consumer can tell which is the meaningful one.
        rr.log("action/linear",  rr.Scalars(last_action["linear"].tolist()))
        rr.log("action/angular", rr.Scalars(last_action["angular"].tolist()))
        rr.log("action/pose/position",    rr.Scalars(last_action["pose_pos"].tolist()))
        rr.log("action/pose/orientation", rr.Scalars(last_action["pose_quat"].tolist()))
        rr.log("action/trajectory_mode",  rr.Scalars([float(last_action["mode"])]))

        # Residuals.
        pos_resid = last_action["pose_pos"] - s[0:3].astype(np.float32)
        rr.log("command/pos_residual/x",    rr.Scalars([float(pos_resid[0])]))
        rr.log("command/pos_residual/y",    rr.Scalars([float(pos_resid[1])]))
        rr.log("command/pos_residual/z",    rr.Scalars([float(pos_resid[2])]))
        rr.log("command/pos_residual_norm", rr.Scalars([float(np.linalg.norm(pos_resid))]))

        vel_resid = last_action["linear"] - s[7:10].astype(np.float32)
        rr.log("command/lin_vel_residual/x",    rr.Scalars([float(vel_resid[0])]))
        rr.log("command/lin_vel_residual/y",    rr.Scalars([float(vel_resid[1])]))
        rr.log("command/lin_vel_residual/z",    rr.Scalars([float(vel_resid[2])]))
        rr.log("command/lin_vel_residual_norm", rr.Scalars([float(np.linalg.norm(vel_resid))]))

        # 3D entities.
        pos = s[0:3].astype(np.float32)
        tcp_positions.append(pos.copy())
        rr.log("world/tcp", rr.Points3D([pos.tolist()], radii=0.005))

        if last_action["mode"] != 0:
            pose_targets.append(last_action["pose_pos"].copy())
            rr.log(
                "world/action/pose_target",
                rr.Points3D([last_action["pose_pos"].tolist()],
                            radii=0.004, colors=[[80, 180, 255]]),
            )

        # Wrench-force arrow at TCP. The wrench is in the FT sensor frame,
        # which is rigidly attached at the wrist — approximated here as
        # TCP-aligned via the TCP orientation. Scale: 1 cm per Newton.
        quat = s[3:7]
        force_world = rotate_by_quat(s[26:29], quat)
        rr.log(
            "world/tcp/force_arrow",
            rr.Arrows3D(
                origins=[pos.tolist()],
                vectors=[(0.01 * force_world).tolist()],
                colors=[[255, 120, 50]],
                radii=0.002,
            ),
        )

        n += 1
        elapsed = time.time() - loop_t
        time.sleep(max(0.0, dt - elapsed))

    # Static 3-D paths: log once at end so they're visible at every timestep
    # when scrubbing in the viewer.
    if tcp_positions:
        rr.log(
            "world/tcp_path",
            rr.LineStrips3D(
                [np.stack(tcp_positions).tolist()],
                colors=[[120, 200, 120]],
                radii=0.0015,
            ),
            static=True,
        )
    if pose_targets:
        rr.log(
            "world/action/pose_path",
            rr.LineStrips3D(
                [np.stack(pose_targets).tolist()],
                colors=[[80, 180, 255]],
                radii=0.0015,
            ),
            static=True,
        )

    bridge.disconnect()
    duration = time.time() - t_start
    print(f"[record] {n} frames over {duration:.1f}s; "
          f"{action_count['n']} action msgs → {args.out}", flush=True)

    if args.meta_out:
        Path(args.meta_out).write_text(json.dumps({
            "frames": n,
            "duration_s": round(duration, 2),
            "action_msgs": action_count["n"],
            "rate_hz": args.rate_hz,
            "image_h": int(args.image_h),
            "image_w": int(args.image_w),
        }, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
