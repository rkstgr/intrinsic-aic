#!/usr/bin/env python3
"""Re-encode the per-trial .rrd files: replace per-frame PNG image streams
with AV1-encoded MP4 video assets + VideoFrameReference per timestamp.

For each input .rrd in <rrd-dir>:
  - Extract every /cam/<name> PNG frame stream with its timestamps.
  - Pipe decoded frames through ffmpeg (libsvtav1 by default) to an MP4.
  - Write a new .rrd that:
      * logs AssetVideo (static) for each camera
      * logs VideoFrameReference at each captured timestamp
      * mirrors every non-camera entity (state/*, action/*, world/*) verbatim

Existing rrds are renamed *.rrd.bak; the new ones take their place.

Usage:
  pixi run python compress_rrds.py [--rrd-dir DIR] [--codec libsvtav1|libaom-av1]
                                   [--crf 50] [--keep-bak]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow as pa
from PIL import Image
import rerun as rr


CAM_ENTITIES = ("/cam/left", "/cam/center", "/cam/right")
SVT_PRESET = 8  # overridden by --preset


def extract_frames(rec, entity_path: str):
    """Return (timestamps_seconds, list_of_png_bytes) for one camera entity."""
    view = rec.view(index="time", contents=entity_path + "/**")
    table = view.select().read_all()
    ts_col = table.column("time")
    # time index returns arrow Duration (ns). Convert to seconds.
    if pa.types.is_duration(ts_col.type):
        ts_ns = ts_col.to_numpy().astype("int64")
        ts_s = ts_ns / 1e9
    else:
        ts_s = ts_col.to_numpy()
    blob_col = table.column(entity_path + ":EncodedImage:blob")
    # blob_col rows are list<list<uint8>>; the outer list wraps a single
    # inner uint8 array (the PNG bytes).
    pngs: list[bytes] = []
    for i in range(table.num_rows):
        v = blob_col[i].as_py()
        if isinstance(v, list) and v and isinstance(v[0], list):
            pngs.append(bytes(v[0]))
        else:
            pngs.append(bytes(v))
    return ts_s, pngs


def encode_to_av1_mp4(pngs: list[bytes], out_mp4: Path, fps: float,
                      codec: str, crf: int) -> None:
    """Decode PNGs and stream raw RGB frames into ffmpeg → AV1 MP4."""
    if not pngs:
        raise ValueError("no frames")
    first = np.array(Image.open(BytesIO(pngs[0])))
    h, w = first.shape[:2]

    if codec == "libsvtav1":
        codec_args = [
            "-c:v", "libsvtav1",
            "-preset", str(SVT_PRESET),
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-svtav1-params", "tune=0",
        ]
    elif codec == "libaom-av1":
        codec_args = [
            "-c:v", "libaom-av1",
            "-cpu-used", "6",
            "-crf", str(crf),
            "-b:v", "0",
            "-pix_fmt", "yuv420p",
        ]
    else:
        raise ValueError(f"unknown codec {codec}")

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}",
        "-r", f"{fps:.6f}",
        "-i", "-",
        *codec_args,
        "-movflags", "+faststart",
        str(out_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for png in pngs:
            arr = np.array(Image.open(BytesIO(png)).convert("RGB"))
            if arr.shape != (h, w, 3):
                arr = np.array(Image.open(BytesIO(png)).convert("RGB").resize((w, h)))
            proc.stdin.write(arr.tobytes())
    finally:
        proc.stdin.close()
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited {rc}")


def copy_scalars_entity(src_rec, entity: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (timestamps_s, values shape (T, K)) for a Scalars entity."""
    view = src_rec.view(index="time", contents=entity + "/**")
    table = view.select().read_all()
    ts_col = table.column("time")
    if pa.types.is_duration(ts_col.type):
        ts_s = ts_col.to_numpy().astype("int64") / 1e9
    else:
        ts_s = ts_col.to_numpy()
    val_col = table.column(entity + ":Scalars:scalars")
    rows = []
    for i in range(table.num_rows):
        rows.append(np.asarray(val_col[i].as_py(), dtype=float))
    return ts_s, np.stack(rows)


def copy_points_entity(src_rec, entity: str) -> tuple[np.ndarray, list[np.ndarray]]:
    view = src_rec.view(index="time", contents=entity + "/**")
    table = view.select().read_all()
    ts_col = table.column("time")
    if pa.types.is_duration(ts_col.type):
        ts_s = ts_col.to_numpy().astype("int64") / 1e9
    else:
        ts_s = ts_col.to_numpy()
    pos_col = table.column(entity + ":Points3D:positions")
    pts: list[np.ndarray] = []
    for i in range(table.num_rows):
        arr = np.asarray(pos_col[i].as_py(), dtype=float).reshape(-1, 3)
        pts.append(arr)
    return ts_s, pts


def process_one_rrd(src: Path, codec: str, crf: int, fps: float, keep_bak: bool) -> dict:
    print(f"\n=== {src.name} ===", flush=True)
    t0 = time.time()
    src_rec = rr.dataframe.load_recording(str(src))
    schema = src_rec.schema()
    all_entities = sorted({c.entity_path for c in schema.component_columns()})

    # Per-entity component map → decide kind
    by_ent = {}
    for c in schema.component_columns():
        by_ent.setdefault(c.entity_path, set()).add(c.component)

    # Treat a camera entity as compressible only if it still has PNG blobs;
    # otherwise it's already an AssetVideo and we skip the rrd entirely.
    cams = [e for e in CAM_ENTITIES
            if e in by_ent and any(c.startswith("EncodedImage") for c in by_ent[e])]
    if not cams:
        print("  no PNG-encoded cameras left — skipping (already compressed)",
              flush=True)
        return {"src": str(src), "old_bytes": src.stat().st_size,
                "new_bytes": src.stat().st_size, "elapsed_s": 0.0}
    scalars_ents = [e for e, comps in by_ent.items()
                    if any(c.startswith("Scalars") for c in comps)]
    points_ents = [e for e, comps in by_ent.items()
                   if any(c.startswith("Points3D") for c in comps)]

    print(f"  entities: {len(all_entities)} total — {len(cams)} cams, "
          f"{len(scalars_ents)} scalars, {len(points_ents)} points3d", flush=True)

    new_path = src.with_suffix(".new.rrd")
    name = src.stem  # e.g. trial_01
    rr.init(name, recording_id=f"{name}-av1", spawn=False)
    rr.save(str(new_path))

    # 1. Camera streams → encode + log AssetVideo + VideoFrameReference per timestamp.
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        cam_sizes: dict[str, int] = {}
        for ent in cams:
            ts_s, pngs = extract_frames(src_rec, ent)
            mp4 = td_path / (ent.strip("/").replace("/", "_") + ".mp4")
            encode_to_av1_mp4(pngs, mp4, fps=fps, codec=codec, crf=crf)
            mp4_bytes = mp4.read_bytes()
            cam_sizes[ent] = len(mp4_bytes)
            print(f"  {ent}: {len(pngs)} frames → {len(mp4_bytes)/1024:.0f} KB MP4",
                  flush=True)

            # Static asset (no time).
            rr.log(ent, rr.AssetVideo(contents=mp4_bytes, media_type="video/mp4"),
                   static=True)
            # Per-frame reference at the captured timestamps.
            for t, _ in zip(ts_s, pngs):
                rr.set_time("time", duration=float(t))
                rr.log(ent, rr.VideoFrameReference(seconds=float(t)))

    # 2. Mirror scalar series.
    for ent in scalars_ents:
        ts_s, vals = copy_scalars_entity(src_rec, ent)
        for t, v in zip(ts_s, vals):
            rr.set_time("time", duration=float(t))
            rr.log(ent, rr.Scalars(v.tolist()))

    # 3. Mirror Points3D series.
    for ent in points_ents:
        ts_s, pts = copy_points_entity(src_rec, ent)
        for t, ps in zip(ts_s, pts):
            rr.set_time("time", duration=float(t))
            rr.log(ent, rr.Points3D(ps.tolist(), radii=0.005))

    # Force flush by reinit. (rr.save with the same recording id stays open.)
    rr.disconnect()
    # Some rerun versions buffer asynchronously; small sleep to let it flush.
    time.sleep(0.5)

    old_size = src.stat().st_size
    new_size = new_path.stat().st_size

    bak = src.with_suffix(".rrd.bak")
    src.rename(bak)
    new_path.rename(src)
    if not keep_bak:
        bak.unlink()

    dt = time.time() - t0
    print(f"  size: {old_size/1024/1024:.1f} MB → {new_size/1024/1024:.1f} MB "
          f"({100*new_size/old_size:.1f}%)  in {dt:.1f}s", flush=True)

    return {"src": str(src), "old_bytes": old_size, "new_bytes": new_size,
            "elapsed_s": round(dt, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rrd-dir", default="/home/ubuntu/lewm_run/site/dist/rrd")
    ap.add_argument("--codec", default="libsvtav1",
                    choices=["libsvtav1", "libaom-av1"])
    ap.add_argument("--crf", type=int, default=50,
                    help="Quality; lower = better. svtav1 sweet spot 35-55.")
    ap.add_argument("--preset", type=int, default=8,
                    help="libsvtav1 preset; higher = faster, lower quality. 0-13.")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--keep-bak", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None,
                    help="Comma-separated filenames (e.g. trial_06.rrd) to compress.")
    args = ap.parse_args()
    global SVT_PRESET
    SVT_PRESET = args.preset

    rrds = sorted(Path(args.rrd_dir).glob("trial*.rrd"))
    if args.only:
        wanted = set(args.only.split(","))
        rrds = [p for p in rrds if p.name in wanted]
    if args.limit:
        rrds = rrds[: args.limit]
    if not rrds:
        print("no rrds found", file=sys.stderr)
        return 1

    print(f"processing {len(rrds)} rrds with codec={args.codec} crf={args.crf}",
          flush=True)
    results = []
    for p in rrds:
        try:
            results.append(process_one_rrd(p, args.codec, args.crf, args.fps,
                                           args.keep_bak))
        except Exception as e:
            print(f"  FAILED on {p.name}: {e}", file=sys.stderr, flush=True)

    if results:
        old = sum(r["old_bytes"] for r in results)
        new = sum(r["new_bytes"] for r in results)
        print(f"\nTOTAL: {old/1024/1024:.1f} MB → {new/1024/1024:.1f} MB "
              f"({100*new/old:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
