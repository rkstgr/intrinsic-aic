# trial_viewer

A small static web app for browsing CheatCode rollouts on the AIC
cable-insertion task — plus the pipeline that produced the data.

Trajectories live on Hugging Face:
[`rkstgr/aic-cheatcode-rollouts-v1`](https://huggingface.co/datasets/rkstgr/aic-cheatcode-rollouts-v1).

## What it is

`viewer/` is a single-page app:

- Left pane: filterable / sortable table of 20 trials (plug type, success,
  Tier-1/2/3, total score, trimmed duration, force-penalty flag).
- Right pane: embedded [Rerun web viewer](https://rerun.io/docs/getting-started/data-out/web)
  loading the selected trial's `.rrd` — 3 synchronized camera streams,
  TCP pose / velocity / error, joints, force/torque wrench, action twist,
  3-D TCP point.

The viewer is fully static (HTML + `<script type="module">` from unpkg).
No backend. The metadata column comes from `index.json`; the rerun JS is
pinned to `0.26.2`.

## Quick start

By default the viewer streams `index.json` and each `.rrd` directly from the
public HF dataset — no local copy needed:

```bash
cd tools/trial_viewer/viewer
python -m http.server 8000
# → http://localhost:8000/
```

If you're on a remote host, SSH-forward the port:

```bash
ssh -L 8000:localhost:8000 <user>@<host>
```

### Pointing at a different data source

The data base URL is configurable via the `?base=` query param. Any URL ending
without `/` will have one appended:

```
# local copy in the same dir as index.html (legacy behaviour)
http://localhost:8000/?base=./

# a different HF dataset
http://localhost:8000/?base=https://huggingface.co/datasets/<user>/<name>/resolve/main/
```

To work fully offline, pre-fetch the dataset and pass `?base=./`:

```bash
pip install huggingface_hub
huggingface-cli download rkstgr/aic-cheatcode-rollouts-v1 \
  --repo-type dataset --local-dir viewer/
```

## Deploying to Vercel

The viewer is fully static, so a free-tier Vercel project works:

1. Connect the repo on vercel.com.
2. **Root Directory** → `tools/trial_viewer/viewer`
3. Framework Preset: **Other** (no build step). `vercel.json` is already in
   place; output is the directory itself.
4. Deploy.

All trajectory bytes are served from HF — Vercel only ships ~12 KB of HTML/JS
per visitor.

## Pipeline

The `pipeline/` directory carries the recording + processing scripts. They
need the AIC eval sim and its pixi environment to actually run.

| script | purpose |
|---|---|
| `record_trial.py` | Subscribes to `/observations`, `/aic_controller/{controller_state,pose_commands}`, `/fts_broadcaster/wrench` from a live AIC trial. Writes a raw `.rrd` with PNG-encoded camera frames. |
| `record_smoke_runner.sh` | Orchestrates N trials end-to-end (gen config → sim → policy → recorder → score). Honors `IMAGE_HW=512x576`, `TRIAL_PREFIX=trial_`. |
| `compress_rrds.py` | Replaces per-frame PNG image streams with AV1-encoded MP4 (`AssetVideo` + `VideoFrameReference`). ~100× smaller. |
| `trim_rrds.py` | Trims each `.rrd` to the motion-relevant window using smoothed TCP velocity. Re-encodes the camera segment with `libsvtav1`. |
| `build_index_from_eval.py` | Builds `index.json` from `eval/runs/<run>/iter_NN/{summary,scoring}.yaml` + `recorder.json` and the trimmed `.rrd` durations. |
| `aic_ros_bridge.py` | Shared ROS subscriber used by `record_trial.py`. |
| `dump_rrd.py` | Legacy converter for the LeRobotDataset exploration data — not used by the current pipeline. |

### Re-record + republish

```bash
# 20 trials, 576×512, seeds 43-62
IMAGE_HW=512x576 TRIAL_PREFIX=trial_ \
  bash pipeline/record_smoke_runner.sh 20 42

# compress, trim, re-index
pixi run python pipeline/compress_rrds.py --preset 10
pixi run python pipeline/trim_rrds.py
pixi run python pipeline/build_index_from_eval.py \
  --csv  ~/lewm_run/eval/cheatcode_record_<TS>.csv \
  --run-dir ~/lewm_run/eval/runs/cheatcode_record_<TS> \
  --rrd-prefix trial_
```

## Known limitations

- **No AV1 hardware encoding on Ampere GPUs.** `av1_nvenc` requires Ada
  Lovelace (RTX 4000 Ada / RTX 50 series). On older GPUs the pipeline uses
  CPU `libsvtav1` preset 10 — ≈ 2-3 min per trial at 576×512.
- **Untared wrench in the rrd.** The recorder captures `/fts_broadcaster/wrench`
  directly. The AIC scoring uses `wrench − fts_tare_offset`. Use the
  `force_penalized` column in `index.json` (parsed from `scoring.yaml`) for
  the canonical "was this trial force-penalized?" answer.
- **Velocity-action only.** `record_trial.py` subscribes to
  `MotionUpdate.velocity`; `CheatCode` commands `MotionUpdate.pose`, so
  `action/linear` and `action/angular` in current rrds are all zero. A
  future recorder update should also log `.pose`.
