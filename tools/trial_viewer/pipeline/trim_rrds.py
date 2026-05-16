#!/usr/bin/env python3
"""Trim each AV1-compressed trial rrd to the interesting time window.

We use the measured TCP velocity (state/tcp/lin_vel + tcp/ang_vel) as the
"arm moving" signal — robust whether the policy commands pose or twist.

  start = first time |v_tcp| > 0.005 m/s         (minus 0.5 s of context)
  end   = last time  |v_tcp| > 0.005 m/s + 1.5 s (motion stopped)
          or recording end if never quiet

We deliberately don't try to detect force-failure mid-trial: the recorded
wrench is untared (~7 N baseline from cable/gripper weight), while the
scoring uses wrench − fts_tare_offset, which we don't carry in the rrd.
Whether a trial was actually force-penalized is captured separately at
index-build time from scoring.yaml.

Trims both the scalar/Points3D series and the camera AssetVideo (re-encodes
the [start, end] segment with libsvtav1). Re-zeroes all timestamps so the
trimmed rrd starts at t=0.

Usage:
  pixi run python trim_rrds.py [--rrd-dir DIR] [--only file.rrd]
                               [--keep-bak]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pyarrow as pa
import rerun as rr


CAM_ENTITIES = ("/cam/left", "/cam/center", "/cam/right")

MOTION_EPS_LIN = 0.005      # m/s — smoothed TCP linear velocity threshold for "arm moving"
MOTION_EPS_ANG = 0.02       # rad/s — smoothed TCP angular velocity threshold
MOTION_SMOOTH_S = 1.0       # rolling-mean window for the motion signal
MOTION_MIN_BLOCK_S = 2.0    # ≥this long of contiguous motion to count as a real run
                            # (filters out smoothing-induced 1s blips)
QUIET_MIN_BLOCK_S = 8.0     # ≥this long of quiet after motion → trial ended here.
                            # Mid-trial pauses in CheatCode are typically <5 s;
                            # the held-pose-after-success period is many seconds.
FORCE_LIMIT = 20.0          # N — matches aic_scoring docs threshold
FORCE_SUSTAIN_S = 1.0       # require force above limit for this long (matches scoring penalty criterion)
FORCE_TAIL_S = 1.0          # seconds to keep after the trigger
SUCCESS_TAIL_S = 1.5        # seconds to keep after last commanded action
START_PAD_S = 0.5           # context before the first action

SVT_PRESET = 10
SVT_CRF = 50


def time_series_ns(table: pa.Table) -> np.ndarray:
    """Return the duration index of a read table as float seconds."""
    col = table.column("time")
    if pa.types.is_duration(col.type):
        return col.to_numpy().astype("int64") / 1e9
    return col.to_numpy().astype(float)


def read_scalars(rec, entity: str) -> Tuple[np.ndarray, np.ndarray]:
    view = rec.view(index="time", contents=entity + "/**")
    tbl = view.select().read_all()
    ts = time_series_ns(tbl)
    col = tbl.column(f"{entity}:Scalars:scalars")
    rows = [np.asarray(col[i].as_py(), dtype=float) for i in range(tbl.num_rows)]
    return ts, np.stack(rows) if rows else np.zeros((0, 0))


def read_points3d(rec, entity: str) -> Tuple[np.ndarray, list[np.ndarray]]:
    view = rec.view(index="time", contents=entity + "/**")
    tbl = view.select().read_all()
    ts = time_series_ns(tbl)
    col = tbl.column(f"{entity}:Points3D:positions")
    pts = [np.asarray(col[i].as_py(), dtype=float).reshape(-1, 3)
           for i in range(tbl.num_rows)]
    return ts, pts


def read_video_asset(rec, entity: str) -> bytes:
    """Pull the static AssetVideo blob for a camera."""
    view = rec.view(index="time", contents=entity + "/**")
    static_tbl = view.select_static().read_all()
    blob_col = static_tbl.column(f"{entity}:AssetVideo:blob")
    blob = blob_col[0].as_py()
    if isinstance(blob, list) and blob and isinstance(blob[0], list):
        return bytes(blob[0])
    return bytes(blob)


def read_video_frame_refs(rec, entity: str) -> np.ndarray:
    """Return the (T,) array of VideoFrameReference timestamps (seconds)."""
    view = rec.view(index="time", contents=entity + "/**")
    tbl = view.select().read_all()
    ts = time_series_ns(tbl)
    return ts


def find_trim_window(
    motion_ts: np.ndarray,
    motion_active: np.ndarray,    # bool array, True when arm is moving
    rec_end_t: float,
) -> Tuple[float, float, str]:
    """Decide (start, end, reason). Reason is one of:
       'quiet'    — arm stopped after motion
       'no-motion' — arm never moved
       'no-end'   — fell back to recording end (motion still active at end)
    """
    # Group contiguous runs (motion-True and quiet-False).
    n = len(motion_active)
    if n == 0:
        return 0.0, rec_end_t, "no-motion"
    runs: list[tuple[int, int, bool]] = []  # (start, end, value)
    rs = 0
    for i in range(1, n):
        if motion_active[i] != motion_active[i - 1]:
            runs.append((rs, i - 1, bool(motion_active[i - 1])))
            rs = i
    runs.append((rs, n - 1, bool(motion_active[n - 1])))

    motion_blocks = [
        (s, e) for (s, e, v) in runs
        if v and float(motion_ts[e] - motion_ts[s]) >= MOTION_MIN_BLOCK_S
    ]
    if not motion_blocks:
        return 0.0, rec_end_t, "no-motion"

    first_motion_start = motion_blocks[0][0]
    start_t = max(0.0, float(motion_ts[first_motion_start]) - START_PAD_S)

    # Find first sustained-quiet block that begins after motion started.
    first_quiet = next(
        ((s, e) for (s, e, v) in runs
         if (not v) and s > first_motion_start
         and float(motion_ts[e] - motion_ts[s]) >= QUIET_MIN_BLOCK_S),
        None,
    )
    if first_quiet is not None:
        # Trial ended at the moment motion stopped (start of the long-quiet run).
        last_motion_t = float(motion_ts[first_quiet[0]])
    else:
        # No sustained quiet — fall back to end of last sustained motion block.
        last_motion_t = float(motion_ts[motion_blocks[-1][1]])
    quiet_end_t = last_motion_t + SUCCESS_TAIL_S

    if quiet_end_t < rec_end_t:
        return start_t, quiet_end_t, "quiet"
    return start_t, rec_end_t, "no-end"


def ffmpeg_trim_segment(in_mp4: Path, out_mp4: Path,
                        start_t: float, end_t: float, fps: float) -> None:
    """Re-encode the [start, end] segment of an AV1 MP4 with libsvtav1.
    Re-encoding gives frame-accurate trim (vs keyframe snap of -c copy)."""
    duration = max(0.05, end_t - start_t)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_t:.3f}",
        "-i", str(in_mp4),
        "-t", f"{duration:.3f}",
        "-c:v", "libsvtav1",
        "-preset", str(SVT_PRESET),
        "-crf", str(SVT_CRF),
        "-pix_fmt", "yuv420p",
        "-r", f"{fps:.6f}",
        "-movflags", "+faststart",
        str(out_mp4),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr.decode("utf-8", errors="replace"))
        raise RuntimeError("ffmpeg trim failed")


def process_one_rrd(src: Path, keep_bak: bool) -> dict:
    print(f"\n=== {src.name} ===", flush=True)
    t0 = time.time()
    rec = rr.dataframe.load_recording(str(src))
    schema = rec.schema()
    ents_by_kind: dict[str, list[str]] = {"scalars": [], "points3d": [], "video": []}
    for c in schema.component_columns():
        if c.component.startswith("Scalars"):
            ents_by_kind["scalars"].append(c.entity_path)
        elif c.component.startswith("Points3D"):
            ents_by_kind["points3d"].append(c.entity_path)
        elif c.component.startswith("AssetVideo"):
            ents_by_kind["video"].append(c.entity_path)
    for k in ents_by_kind:
        ents_by_kind[k] = sorted(set(ents_by_kind[k]))

    if not ents_by_kind["video"]:
        print("  no AssetVideo entities — skipping (run compress_rrds first)",
              flush=True)
        return {"src": str(src), "skipped": True}

    # Compute trim window from smoothed TCP velocity. Smoothing kills single-
    # sample noise spikes that linger after CheatCode finishes (the controller
    # holds the inserted pose; observed steady-state |v_tcp| ~ 0.0001 m/s with
    # occasional 0.02 m/s spikes from joint chatter).
    lv_t, lv = read_scalars(rec, "/state/tcp/lin_vel")
    av_t, av = read_scalars(rec, "/state/tcp/ang_vel")
    lin_norm = np.linalg.norm(lv, axis=1)
    ang_norm = np.linalg.norm(av, axis=1)
    # rolling mean
    dt_est = np.median(np.diff(lv_t)) if len(lv_t) > 1 else 0.1
    window = max(1, int(round(MOTION_SMOOTH_S / max(dt_est, 1e-3))))
    if window > 1 and len(lin_norm) >= window:
        kernel = np.ones(window) / window
        lin_smooth = np.convolve(lin_norm, kernel, mode="same")
        ang_smooth = np.convolve(ang_norm, kernel, mode="same")
    else:
        lin_smooth, ang_smooth = lin_norm, ang_norm
    motion_active = (lin_smooth > MOTION_EPS_LIN) | (ang_smooth > MOTION_EPS_ANG)
    rec_end_t = float(lv_t[-1] if len(lv_t) else 0.0)
    start_t, end_t, reason = find_trim_window(lv_t, motion_active, rec_end_t)
    duration = end_t - start_t
    print(f"  full rec: 0..{rec_end_t:6.2f}s  →  trim window {start_t:6.2f}..{end_t:6.2f}s "
          f"({duration:5.2f}s)  reason={reason}", flush=True)

    new_path = src.with_suffix(".trimmed.rrd")
    name = src.stem
    rr.init(name, recording_id=f"{name}-trimmed", spawn=False)
    rr.save(str(new_path))

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for ent in ents_by_kind["video"]:
            mp4_bytes = read_video_asset(rec, ent)
            in_mp4 = td_path / (ent.strip("/").replace("/", "_") + "-in.mp4")
            out_mp4 = td_path / (ent.strip("/").replace("/", "_") + "-out.mp4")
            in_mp4.write_bytes(mp4_bytes)
            ffmpeg_trim_segment(in_mp4, out_mp4, start_t, end_t, fps=10.0)
            trimmed = out_mp4.read_bytes()
            rr.log(ent, rr.AssetVideo(contents=trimmed, media_type="video/mp4"),
                   static=True)
            # Re-emit VideoFrameReferences at re-zeroed times for every original
            # frame that fell inside the window.
            ts = read_video_frame_refs(rec, ent)
            mask = (ts >= start_t) & (ts <= end_t)
            for t in ts[mask]:
                rr.set_time("time", duration=float(t - start_t))
                rr.log(ent, rr.VideoFrameReference(seconds=float(t - start_t)))
            print(f"    {ent}: {len(trimmed)/1024:.0f} KB MP4", flush=True)

    # Scalars
    for ent in ents_by_kind["scalars"]:
        ts, vals = read_scalars(rec, ent)
        mask = (ts >= start_t) & (ts <= end_t)
        for t, v in zip(ts[mask], vals[mask]):
            rr.set_time("time", duration=float(t - start_t))
            rr.log(ent, rr.Scalars(v.tolist()))

    # Points3D
    for ent in ents_by_kind["points3d"]:
        ts, pts = read_points3d(rec, ent)
        mask = (ts >= start_t) & (ts <= end_t)
        for t, ps in zip(ts[mask], [pts[i] for i, m in enumerate(mask) if m]):
            rr.set_time("time", duration=float(t - start_t))
            rr.log(ent, rr.Points3D(ps.tolist(), radii=0.005))

    rr.disconnect()
    time.sleep(0.5)

    old_size = src.stat().st_size
    new_size = new_path.stat().st_size

    bak = src.with_suffix(".rrd.bak")
    src.rename(bak)
    new_path.rename(src)
    if not keep_bak:
        bak.unlink()

    dt = time.time() - t0
    print(f"  size: {old_size/1024:.0f} → {new_size/1024:.0f} KB "
          f"({100*new_size/old_size:.1f}%)  in {dt:.1f}s", flush=True)

    return {
        "src": str(src),
        "old_bytes": old_size,
        "new_bytes": new_size,
        "duration_s": round(duration, 2),
        "trim_reason": reason,
        "trim_start_orig": round(start_t, 3),
        "trim_end_orig": round(end_t, 3),
        "elapsed_s": round(dt, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rrd-dir", default="/home/ubuntu/lewm_run/site/dist/rrd")
    ap.add_argument("--only", default=None,
                    help="Comma-separated filenames (e.g. trial_01.rrd)")
    ap.add_argument("--keep-bak", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rrds = sorted(Path(args.rrd_dir).glob("trial*.rrd"))
    if args.only:
        wanted = set(args.only.split(","))
        rrds = [p for p in rrds if p.name in wanted]
    if args.limit:
        rrds = rrds[: args.limit]
    if not rrds:
        print("no rrds found", file=sys.stderr)
        return 1

    print(f"processing {len(rrds)} rrds (force_limit={FORCE_LIMIT}N "
          f"sustain={FORCE_SUSTAIN_S}s success_tail={SUCCESS_TAIL_S}s)", flush=True)
    results = []
    for p in rrds:
        try:
            r = process_one_rrd(p, args.keep_bak)
            if not r.get("skipped"):
                results.append(r)
        except Exception as e:
            print(f"  FAILED on {p.name}: {e}", file=sys.stderr, flush=True)

    if results:
        print("\nper-trial summary:")
        for r in results:
            print(f"  {Path(r['src']).name}  {r['duration_s']:5.1f}s  "
                  f"reason={r['trim_reason']:7s}  "
                  f"{r['old_bytes']/1024:.0f} → {r['new_bytes']/1024:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
