#!/usr/bin/env python3
"""Record observations + commanded actions from a live aic_eval trial to a
single rerun .rrd file. Intended to run on host alongside aic_model.

Subscribes to (via AICRosBridge):
  - cameras left/center/right
  - /joint_states
  - /aic_controller/controller_state    (TCP pose/velocity/error)
  - /fts_broadcaster/wrench             (force/torque)
And additionally:
  - /aic_controller/pose_commands       (MotionUpdate — the action)

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

    last_action = {"linear": np.zeros(3), "angular": np.zeros(3), "ts": 0.0}
    action_count = {"n": 0}

    def on_action(msg):
        v = msg.velocity
        last_action["linear"] = np.array([v.linear.x, v.linear.y, v.linear.z], dtype=np.float32)
        last_action["angular"] = np.array([v.angular.x, v.angular.y, v.angular.z], dtype=np.float32)
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

        rr.log("action/linear",  rr.Scalars(last_action["linear"].tolist()))
        rr.log("action/angular", rr.Scalars(last_action["angular"].tolist()))

        rr.log("world/tcp", rr.Points3D([s[0:3].tolist()], radii=0.005))

        n += 1
        elapsed = time.time() - loop_t
        time.sleep(max(0.0, dt - elapsed))

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
