#!/usr/bin/env python3
"""Enrich each trimmed trial .rrd with derived diagnostic signals.

NOTE (legacy): This script was used to retroactively add derived signals to
the v1 HF dataset (aic-cheatcode-rollouts-v1). New recordings made with the
updated record_trial.py already include per-axis force, pose targets,
residuals, and 3D entities at capture time, so this script is no longer
needed for fresh data. Kept for one-off migration of older RRDs only.

Adds the following entities alongside the originals (which pass through
unchanged):

  Split-per-axis wrench force (so the Rerun panel labels each line):
    /state/wrench/force/x|y|z

  Derived scalars geared at the wire-catch diagnostic:
    /state/wrench/force_lateral_norm    = sqrt(Fx^2 + Fy^2)
    /state/tcp/error_pos_norm           = ||tcp_error[0:3]||
    /command/lin_vel_residual/x|y|z     = action.linear - measured tcp.lin_vel
    /command/lin_vel_residual_norm      = ||residual||

Rationale: there is only one F/T sensor (at the wrist), so wire-tension
and contact forces can't be separated by signal — but during the CheatCode
approach phase (no plug-target contact possible) any deviation of
force_lateral_norm above the gripped-plug baseline is wire interaction.
The residual catches "robot moved slower than commanded" cases that often
co-occur with a catch.

Usage:
  pixi run python pipeline/enrich_rrds.py [--rrd-dir DIR]
                                          [--only FILE.rrd[,FILE.rrd]]
                                          [--keep-bak] [--limit N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pyarrow as pa
import rerun as rr


def time_series_s(table: pa.Table) -> np.ndarray:
    col = table.column("time")
    if pa.types.is_duration(col.type):
        return col.to_numpy().astype("int64") / 1e9
    return col.to_numpy().astype(float)


def read_scalars(rec, entity: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (timestamps_s, values_shape_(T,K)). K=1 collapses to (T,)."""
    view = rec.view(index="time", contents=entity + "/**")
    tbl = view.select().read_all()
    ts = time_series_s(tbl)
    col = tbl.column(f"{entity}:Scalars:scalars")
    rows = [np.asarray(col[i].as_py(), dtype=float) for i in range(tbl.num_rows)]
    if not rows:
        return ts, np.zeros((0, 0))
    arr = np.stack(rows)
    return ts, arr


def read_points3d(rec, entity: str) -> Tuple[np.ndarray, list[np.ndarray]]:
    view = rec.view(index="time", contents=entity + "/**")
    tbl = view.select().read_all()
    ts = time_series_s(tbl)
    col = tbl.column(f"{entity}:Points3D:positions")
    pts = [np.asarray(col[i].as_py(), dtype=float).reshape(-1, 3)
           for i in range(tbl.num_rows)]
    return ts, pts


def read_video_asset(rec, entity: str) -> bytes:
    view = rec.view(index="time", contents=entity + "/**")
    static_tbl = view.select_static().read_all()
    blob_col = static_tbl.column(f"{entity}:AssetVideo:blob")
    blob = blob_col[0].as_py()
    if isinstance(blob, list) and blob and isinstance(blob[0], list):
        return bytes(blob[0])
    return bytes(blob)


def read_video_frame_ref_times(rec, entity: str) -> np.ndarray:
    view = rec.view(index="time", contents=entity + "/**")
    tbl = view.select().read_all()
    return time_series_s(tbl)


def compute_derived(rec) -> list[Tuple[str, np.ndarray, np.ndarray]]:
    """Return list of (entity_path, timestamps_s, values_1d)."""
    out: list[Tuple[str, np.ndarray, np.ndarray]] = []

    # Wrench: per-axis split + lateral norm.
    try:
        ts_f, F = read_scalars(rec, "/state/wrench/force")
        if F.ndim == 2 and F.shape[1] == 3 and F.shape[0] > 0:
            out.append(("state/wrench/force/x", ts_f, F[:, 0]))
            out.append(("state/wrench/force/y", ts_f, F[:, 1]))
            out.append(("state/wrench/force/z", ts_f, F[:, 2]))
            out.append((
                "state/wrench/force_lateral_norm",
                ts_f, np.sqrt(F[:, 0] ** 2 + F[:, 1] ** 2),
            ))
    except Exception as e:
        print(f"  skip force derives: {e}", flush=True)

    # TCP translational error norm.
    try:
        ts_e, E = read_scalars(rec, "/state/tcp/error")
        if E.ndim == 2 and E.shape[1] >= 3 and E.shape[0] > 0:
            out.append((
                "state/tcp/error_pos_norm",
                ts_e, np.linalg.norm(E[:, :3], axis=1),
            ))
    except Exception as e:
        print(f"  skip tcp_error derives: {e}", flush=True)

    # Velocity-tracking residual: commanded action.linear - measured tcp.lin_vel.
    try:
        ts_v, V = read_scalars(rec, "/state/tcp/lin_vel")
        ts_a, A = read_scalars(rec, "/action/linear")
        if (V.ndim == 2 and V.shape[1] == 3 and V.shape[0] > 0
                and A.ndim == 2 and A.shape[1] == 3 and A.shape[0] > 0):
            # record_trial.py logs both inside the same loop iteration so
            # timestamps should match exactly; interpolate to be robust to
            # any future divergence.
            A_on_v = np.stack(
                [np.interp(ts_v, ts_a, A[:, i]) for i in range(3)],
                axis=1,
            )
            R = A_on_v - V
            out.append(("command/lin_vel_residual/x", ts_v, R[:, 0]))
            out.append(("command/lin_vel_residual/y", ts_v, R[:, 1]))
            out.append(("command/lin_vel_residual/z", ts_v, R[:, 2]))
            out.append((
                "command/lin_vel_residual_norm",
                ts_v, np.linalg.norm(R, axis=1),
            ))
    except Exception as e:
        print(f"  skip residual derives: {e}", flush=True)

    return out


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

    derived = compute_derived(rec)
    if not derived:
        print("  no derived signals computed — skipping", flush=True)
        return {"src": str(src), "skipped": True}

    new_path = src.with_suffix(".enriched.rrd")
    name = src.stem
    rr.init(name, recording_id=f"{name}-enriched", spawn=False)
    rr.save(str(new_path))

    # Video: pass through (static blob + per-frame refs at original times).
    for ent in ents_by_kind["video"]:
        blob = read_video_asset(rec, ent)
        rr.log(ent, rr.AssetVideo(contents=blob, media_type="video/mp4"),
               static=True)
        for t in read_video_frame_ref_times(rec, ent):
            rr.set_time("time", duration=float(t))
            rr.log(ent, rr.VideoFrameReference(seconds=float(t)))

    # Scalars: pass through.
    for ent in ents_by_kind["scalars"]:
        ts, vals = read_scalars(rec, ent)
        for t, v in zip(ts, vals):
            rr.set_time("time", duration=float(t))
            rr.log(ent, rr.Scalars(v.tolist()))

    # Points3D: pass through.
    for ent in ents_by_kind["points3d"]:
        ts, pts = read_points3d(rec, ent)
        for t, ps in zip(ts, pts):
            rr.set_time("time", duration=float(t))
            rr.log(ent, rr.Points3D(ps.tolist(), radii=0.005))

    # Derived: each as its own labelled entity, one scalar per timestamp.
    for ent, ts, vals in derived:
        for t, v in zip(ts, vals):
            rr.set_time("time", duration=float(t))
            rr.log(ent, rr.Scalars([float(v)]))

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
    print(f"  added {len(derived)} derived series  "
          f"size: {old_size/1024:.0f} → {new_size/1024:.0f} KB "
          f"({100*new_size/old_size:.1f}%)  in {dt:.1f}s", flush=True)

    return {
        "src": str(src),
        "old_bytes": old_size,
        "new_bytes": new_size,
        "derived_count": len(derived),
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

    print(f"enriching {len(rrds)} rrds in {args.rrd_dir}", flush=True)
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
            print(f"  {Path(r['src']).name}  "
                  f"+{r['derived_count']} series  "
                  f"{r['old_bytes']/1024:.0f} → {r['new_bytes']/1024:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
