#!/usr/bin/env python3
"""Build site/dist/index.json from a record_smoke_runner run.

Sources of truth per trial:
  - eval/runs/<run_label>/iter_NN/summary.yaml   (cable_type, plug_type, board pose, …)
  - eval/runs/<run_label>/iter_NN/scoring.yaml   (per-tier scores, t3 message, success)
  - eval/<run_label>.csv                         (cached per-iter row)
  - eval/runs/<run_label>/iter_NN/recorder.json  (frame count, duration of the rrd)
  - site/dist/rrd/trial_NN.rrd                   (the recording itself)

Usage:
  pixi run python build_index_from_eval.py --run-label cheatcode_record_<TS>
  # or
  pixi run python build_index_from_eval.py --csv <path>.csv --run-dir <eval/runs/...>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import rerun as rr
import yaml


def load_yaml(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        return yaml.safe_load(p.read_text()) or {}
    except Exception as e:
        print(f"  warn: failed to parse {p}: {e}", file=sys.stderr)
        return {}


def rrd_duration_s(rrd_path: Path) -> float:
    """Inspect an rrd's first/last timestamp on the 'time' index. Returns 0
    if the file is missing or has no time-indexed data."""
    if not rrd_path.exists():
        return 0.0
    try:
        rec = rr.dataframe.load_recording(str(rrd_path))
        for cand in ("/state/tcp/lin_vel/**", "/cam/center/**", "/state/**", "/**"):
            view = rec.view(index="time", contents=cand)
            tbl = view.select().read_all()
            if tbl.num_rows == 0:
                continue
            col = tbl.column("time")
            if pa.types.is_duration(col.type):
                ts = col.to_numpy().astype("int64") / 1e9
            else:
                ts = col.to_numpy().astype(float)
            if len(ts):
                return float(ts.max() - ts.min())
        return 0.0
    except Exception as e:
        print(f"  warn: could not inspect {rrd_path}: {e}", file=sys.stderr)
        return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV path produced by record_smoke_runner.sh")
    ap.add_argument("--run-dir", required=True,
                    help="Per-trial directory, e.g. eval/runs/cheatcode_record_<TS>")
    ap.add_argument("--rrd-dir", default="/home/ubuntu/lewm_run/site/dist/rrd")
    ap.add_argument("--rrd-prefix", default="trial_",
                    help="Prefix used by the recorder when this CSV was produced "
                         "(matches the runner's TRIAL_PREFIX env var).")
    ap.add_argument("--out", default="/home/ubuntu/lewm_run/site/dist/index.json")
    ap.add_argument("--dataset-label", default=None,
                    help="Label shown in the page header (default: derived from CSV name).")
    ap.add_argument("--append", action="store_true",
                    help="Append to existing --out index.json (renumber so trial numbers don't collide).")
    ap.add_argument("--trial-offset", type=int, default=None,
                    help="Add this offset to the CSV iteration numbers before writing. "
                         "With --append, defaults to (max existing trial).")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    run_dir = Path(args.run_dir)
    rrd_dir = Path(args.rrd_dir)
    out_path = Path(args.out)
    label = args.dataset_label or csv_path.stem

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1
    df = pd.read_csv(csv_path)

    existing_episodes: list[dict] = []
    if args.append and out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            existing_episodes = existing.get("episodes", [])
        except Exception:
            pass
    offset = args.trial_offset
    if offset is None:
        offset = max([e.get("trial", 0) for e in existing_episodes], default=0) if args.append else 0

    episodes: list[dict] = list(existing_episodes)
    for _, row in df.iterrows():
        i_csv = int(row["iteration"])
        i = i_csv + offset
        iter_dir = run_dir / f"iter_{i_csv:02d}"

        summary = load_yaml(iter_dir / "summary.yaml").get(f"trial_1", {})
        scoring = load_yaml(iter_dir / "scoring.yaml")
        # Force penalty: tier_2.categories["insertion force"].score < 0 means -12 applied.
        force_cat = (
            scoring.get("trial_1", {})
                   .get("tier_2", {})
                   .get("categories", {})
                   .get("insertion force", {})
        )
        force_penalized = int(float(force_cat.get("score", 0.0)) < 0)
        force_message = str(force_cat.get("message", ""))
        rec_meta = {}
        rmp = iter_dir / "recorder.json"
        if rmp.exists():
            try:
                rec_meta = json.loads(rmp.read_text())
            except Exception:
                pass

        # Find the rrd: support both the default "trial_NN.rrd" and any other
        # TRIAL_PREFIX written by record_smoke_runner.sh.
        rrd_path = rrd_dir / f"{args.rrd_prefix}{i_csv:02d}.rrd"
        trimmed_duration_s = round(rrd_duration_s(rrd_path), 2)
        rrd_rel = str(rrd_path.relative_to(out_path.parent)) if rrd_path.exists() else None
        rrd_kb = round(rrd_path.stat().st_size / 1024, 1) if rrd_path.exists() else 0.0

        image_hw = "224x224"
        if rec_meta.get("image_h") and rec_meta.get("image_w"):
            image_hw = f'{rec_meta["image_w"]}x{rec_meta["image_h"]}'

        episodes.append({
            "trial": i,
            "rrd": rrd_rel,
            "rrd_size_kb": rrd_kb,
            "image_hw":    image_hw,
            "frames":      rec_meta.get("frames", 0),
            "duration_s":  trimmed_duration_s if trimmed_duration_s > 0 else
                           rec_meta.get("duration_s", float(row.get("elapsed_s", 0.0))),
            "rec_duration_s": rec_meta.get("duration_s", 0.0),
            "plug_type":   str(row.get("plug_type", "?")),
            "port_type":   str(row.get("port_type", "?")),
            "cable_type":  str(row.get("cable_type", "?")),
            "target_module": str(row.get("target_module", "?")),
            "success":     int(row.get("success", 0)),
            "force_penalized": force_penalized,
            "force_message": force_message,
            "t3_message":  str(row.get("t3_message", "")),
            "tier1":       round(float(row.get("tier1", 0.0)), 3),
            "tier2":       round(float(row.get("tier2", 0.0)), 3),
            "tier3":       round(float(row.get("tier3", 0.0)), 3),
            "total_score": round(float(row.get("total_score", 0.0)), 3),
            "elapsed_s":   round(float(row.get("elapsed_s", 0.0)), 1),
            "seed":        int(row.get("seed", 0)),
        })

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": label,
        "kind": "eval_rollouts",
        "n_episodes": len(episodes),
        "episodes": episodes,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path} ({len(episodes)} trials)")
    n_with_rrd = sum(1 for e in episodes if e["rrd"])
    print(f"  trials with rrd: {n_with_rrd}/{len(episodes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
