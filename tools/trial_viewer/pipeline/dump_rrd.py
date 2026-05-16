#!/usr/bin/env python3
"""Dump each LeRobotDataset episode in dataset_gazebo/ to a rerun .rrd file
and build a metadata index.json for the frontend filter UI.

Usage:
  pixi run python dump_rrd.py [--limit N] [--episodes 0,1,2]
                              [--dataset ~/lewm_run/dataset_gazebo]
                              [--out   ~/lewm_run/site/dist]

Output layout:
  <out>/index.json           one row per episode + summary stats
  <out>/rrd/episode_NNNNN.rrd
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rerun as rr

# observation.state slot layout (32-d), per aic_ros_bridge.py:
#   0:3   tcp position (x,y,z)
#   3:7   tcp orientation quaternion (qx,qy,qz,qw)
#   7:10  tcp linear velocity
#   10:13 tcp angular velocity
#   13:19 tcp error (6-d)
#   19:26 joint positions (7)
#   26:29 wrench force (fx,fy,fz)
#   29:32 wrench torque (tx,ty,tz)


def episode_summary(frames: pd.DataFrame) -> dict:
    state = np.stack(frames["observation.state"].values)  # (T, 32)
    action = np.stack(frames["action"].values)            # (T, 6)
    pos = state[:, 0:3]
    force = state[:, 26:29]
    torque = state[:, 29:32]
    return {
        "exploration_mode": str(frames["exploration_mode"].iloc[0]),
        "length": int(len(frames)),
        "duration_s": round(float(frames["timestamp"].max() - frames["timestamp"].min()), 3),
        "max_force_norm": round(float(np.linalg.norm(force, axis=1).max()), 3),
        "max_force_z": round(float(np.abs(force[:, 2]).max()), 3),
        "max_torque_norm": round(float(np.linalg.norm(torque, axis=1).max()), 3),
        "mean_action_lin": round(float(np.linalg.norm(action[:, 0:3], axis=1).mean()), 4),
        "mean_action_ang": round(float(np.linalg.norm(action[:, 3:6], axis=1).mean()), 4),
        "tcp_z_min": round(float(pos[:, 2].min()), 4),
        "tcp_z_max": round(float(pos[:, 2].max()), 4),
        "tcp_travel": round(float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum()), 4),
    }


def log_frame(row: pd.Series) -> None:
    ts = float(row["timestamp"])
    rr.set_time("time", duration=ts)

    # Cameras — PNG bytes already in the parquet; log encoded to keep rrd small.
    for cam in ("left", "center", "right"):
        png = row[f"observation.images.{cam}"]["bytes"]
        rr.log(f"cam/{cam}", rr.EncodedImage(contents=png, media_type="image/png"))

    s = row["observation.state"]
    rr.log("state/tcp/position",      rr.Scalars(s[0:3].tolist()))
    rr.log("state/tcp/orientation",   rr.Scalars(s[3:7].tolist()))
    rr.log("state/tcp/lin_vel",       rr.Scalars(s[7:10].tolist()))
    rr.log("state/tcp/ang_vel",       rr.Scalars(s[10:13].tolist()))
    rr.log("state/tcp/error",         rr.Scalars(s[13:19].tolist()))
    rr.log("state/joints",            rr.Scalars(s[19:26].tolist()))
    rr.log("state/wrench/force",      rr.Scalars(s[26:29].tolist()))
    rr.log("state/wrench/torque",     rr.Scalars(s[29:32].tolist()))

    rr.log("state/wrench/force_norm", rr.Scalars([float(np.linalg.norm(s[26:29]))]))

    a = row["action"]
    rr.log("action/linear",  rr.Scalars(a[0:3].tolist()))
    rr.log("action/angular", rr.Scalars(a[3:6].tolist()))

    rr.log(
        "world/tcp",
        rr.Points3D([s[0:3].tolist()], radii=0.005),
    )


def dump_episode(frames: pd.DataFrame, ep_idx: int, out_rrd: Path) -> None:
    rr.init(f"lewm_episode_{ep_idx:05d}", recording_id=f"ep{ep_idx:05d}", spawn=False)
    rr.save(str(out_rrd))
    for _, row in frames.iterrows():
        log_frame(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/home/ubuntu/lewm_run/dataset_gazebo")
    ap.add_argument("--out", default="/home/ubuntu/lewm_run/site/dist")
    ap.add_argument("--limit", type=int, default=None,
                    help="Convert only the first N episodes (for smoke tests).")
    ap.add_argument("--episodes", default=None,
                    help="Comma-separated explicit episode indices.")
    args = ap.parse_args()

    ds_root = Path(args.dataset)
    out_root = Path(args.out)
    rrd_dir = out_root / "rrd"
    rrd_dir.mkdir(parents=True, exist_ok=True)

    data_files = sorted((ds_root / "data" / "chunk-000").glob("file-*.parquet"))
    if not data_files:
        print(f"No data files under {ds_root}", file=sys.stderr)
        return 1
    print(f"Reading {len(data_files)} parquet shards from {ds_root}…", flush=True)
    df = pd.concat((pd.read_parquet(p) for p in data_files), ignore_index=True)
    print(f"  total frames: {len(df)}; episodes: {df['episode_index'].nunique()}", flush=True)

    all_eps = sorted(df["episode_index"].unique().tolist())
    if args.episodes:
        wanted = [int(x) for x in args.episodes.split(",")]
    else:
        wanted = all_eps if args.limit is None else all_eps[: args.limit]

    index_rows: list[dict] = []
    t_start = time.time()
    for n, ep in enumerate(wanted):
        frames = df[df["episode_index"] == ep].sort_values("frame_index").reset_index(drop=True)
        out_path = rrd_dir / f"episode_{ep:05d}.rrd"
        dump_episode(frames, int(ep), out_path)
        summary = episode_summary(frames)
        index_rows.append({
            "episode_index": int(ep),
            "rrd": f"rrd/episode_{ep:05d}.rrd",
            "rrd_size_kb": round(out_path.stat().st_size / 1024, 1),
            **summary,
        })
        if (n + 1) % 10 == 0 or n + 1 == len(wanted):
            dt = time.time() - t_start
            rate = (n + 1) / dt
            eta = (len(wanted) - (n + 1)) / rate if rate > 0 else 0
            print(f"  [{n + 1}/{len(wanted)}] ep={ep} → {out_path.name} "
                  f"({summary['length']} frames, "
                  f"{out_path.stat().st_size / 1024:.0f} KB) "
                  f"rate={rate:.1f} ep/s eta={eta:.0f}s",
                  flush=True)

    # Re-stat after the loop: rerun's background writer hasn't always flushed
    # by the time we stat'd inside the loop, which made many sizes read as 1 KB.
    for r in index_rows:
        r["rrd_size_kb"] = round((out_root / r["rrd"]).stat().st_size / 1024, 1)

    index_path = out_root / "index.json"
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": str(ds_root),
        "n_episodes": len(index_rows),
        "episodes": index_rows,
    }
    index_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {index_path} ({len(index_rows)} episodes)")
    total_kb = sum(r["rrd_size_kb"] for r in index_rows)
    print(f"Total rrd size: {total_kb / 1024:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
