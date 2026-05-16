#!/usr/bin/env bash
# CheatCode smoke runner WITH per-trial trajectory recording.
# Forks of smoke_runner.sh that additionally launches record_trial.py per
# iteration and writes one .rrd per trial to the chosen output dir.
#
# Usage:
#   record_smoke_runner.sh <N> [seed_base]
#
# Env:
#   PER_ITER_SECS   Max wall-clock per iteration (default 240).
#   RRD_OUT_DIR     Where rrd files land (default /home/ubuntu/lewm_run/site/dist/rrd).
#   RUN_LABEL       Prefix for the per-run dir under eval/runs (default cheatcode).
set -u
LEWM_RUN=${LEWM_RUN:-/home/ubuntu/lewm_run}
SCRIPTS="$LEWM_RUN/scripts"
DEMO_SCRIPTS="$SCRIPTS/demo_collection"
LOGDIR="$LEWM_RUN/logs"
EVAL_DIR="$LEWM_RUN/eval"
mkdir -p "$LOGDIR" "$EVAL_DIR"

N="${1:?usage: record_smoke_runner.sh <N> [seed_base]}"
SEED_BASE="${2:-42}"

PER_ITER_SECS="${PER_ITER_SECS:-240}"
RRD_OUT_DIR="${RRD_OUT_DIR:-/home/ubuntu/lewm_run/site/dist/rrd}"
RUN_LABEL="${RUN_LABEL:-cheatcode}"
TRIAL_PREFIX="${TRIAL_PREFIX:-trial_}"
# IMAGE_HW=576x512 → records cameras at non-square sizes; default 224x224.
IMAGE_HW="${IMAGE_HW:-224x224}"
IMG_H="${IMAGE_HW%x*}"
IMG_W="${IMAGE_HW#*x}"
mkdir -p "$RRD_OUT_DIR"

RUN_TS=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR="$EVAL_DIR/runs/${RUN_LABEL}_record_${RUN_TS}"
mkdir -p "$RUN_DIR"
CSV_PATH="$EVAL_DIR/${RUN_LABEL}_record_${RUN_TS}.csv"

echo "[rec] starting record-smoke: N=$N seed_base=$SEED_BASE"
echo "[rec] run_dir=$RUN_DIR  csv=$CSV_PATH  rrd_dir=$RRD_OUT_DIR"

export DBX_CONTAINER_MANAGER=docker
export DISPLAY=:1

run_one_iter() {
  local i=$1
  local seed=$2
  local iter_dir="$RUN_DIR/iter_$(printf '%02d' "$i")"
  mkdir -p "$iter_dir"
  local cfg="$iter_dir/engine_config.yaml"
  local sumcfg="$iter_dir/summary.yaml"
  local sim_log="$iter_dir/sim.log"
  local model_log="$iter_dir/model.log"
  local rec_log="$iter_dir/recorder.log"
  local rrd_path="$RRD_OUT_DIR/${TRIAL_PREFIX}$(printf '%02d' "$i").rrd"
  local rec_meta="$iter_dir/recorder.json"

  echo "[rec][iter=$i seed=$seed] generating config"
  cd /home/ubuntu/ws_aic/src/aic
  pixi run --manifest-path /home/ubuntu/ws_aic/src/aic/pixi.toml \
    python "$DEMO_SCRIPTS/gen_random_config.py" \
    --seed "$seed" --out "$cfg" --summary-out "$sumcfg" --num-trials 1 \
    > "$iter_dir/gen.log" 2>&1 \
    || { echo "[rec][iter=$i] gen_random_config FAILED"; return 2; }

  echo "[rec][iter=$i] pre-flight cleanup"
  bash "$SCRIPTS/cleanup.sh" >> "$LOGDIR/cleanup.log" 2>&1 || true
  rm -f /home/ubuntu/aic_results/scoring.yaml
  sleep 2

  local t_start=$(date +%s)

  echo "[rec][iter=$i] launching sim+engine"
  nohup distrobox enter -r aic_eval -- /entrypoint.sh \
    start_aic_engine:=true \
    shutdown_on_aic_engine_exit:=true \
    ground_truth:=true \
    launch_rviz:=false \
    gazebo_gui:=false \
    aic_engine_config_file:="$cfg" \
    > "$sim_log" 2>&1 &
  local SIM_PID=$!
  echo "$SIM_PID" > "$iter_dir/sim.pid"
  disown $SIM_PID || true

  sleep 25

  echo "[rec][iter=$i] launching aic_model (CheatCode)"
  cd /home/ubuntu/ws_aic/src/aic
  nohup env \
    PYTHONPATH="$SCRIPTS${PYTHONPATH:+:$PYTHONPATH}" \
    pixi run --manifest-path /home/ubuntu/ws_aic/src/aic/pixi.toml \
      ros2 run aic_model aic_model \
      --ros-args -p use_sim_time:=true -p policy:=aic_example_policies.ros.CheatCode \
    > "$model_log" 2>&1 &
  local MODEL_PID=$!
  echo "$MODEL_PID" > "$iter_dir/model.pid"
  disown $MODEL_PID || true

  # Give aic_model a moment to come up before subscribing to its action topic.
  sleep 5

  echo "[rec][iter=$i] launching recorder → $rrd_path"
  nohup env \
    PYTHONPATH="$SCRIPTS${PYTHONPATH:+:$PYTHONPATH}" \
    pixi run --manifest-path /home/ubuntu/ws_aic/src/aic/pixi.toml \
      python "$DEMO_SCRIPTS/record_trial.py" \
      --out "$rrd_path" --meta-out "$rec_meta" \
      --rate-hz 10 --max-seconds "$PER_ITER_SECS" \
      --image-h "$IMG_H" --image-w "$IMG_W" \
      --name "${TRIAL_PREFIX}$(printf '%02d' "$i")" \
      > "$rec_log" 2>&1 &
  local REC_PID=$!
  echo "$REC_PID" > "$iter_dir/recorder.pid"
  disown $REC_PID || true

  local SCORE_PATH="/home/ubuntu/aic_results/scoring.yaml"
  local result_yaml=""
  while true; do
    local now=$(date +%s)
    local elapsed=$((now - t_start))
    if (( elapsed >= PER_ITER_SECS )); then
      echo "[rec][iter=$i] TIMEOUT after ${elapsed}s"
      break
    fi
    if [[ -f "$SCORE_PATH" ]]; then
      local mt=$(stat -c '%Y' "$SCORE_PATH" 2>/dev/null || echo 0)
      if (( mt >= t_start )); then
        result_yaml="$SCORE_PATH"
        echo "[rec][iter=$i] scoring.yaml ready (elapsed=${elapsed}s)"
        break
      fi
    fi
    if ! kill -0 "$SIM_PID" 2>/dev/null; then
      echo "[rec][iter=$i] sim PID gone, giving 5s grace"
      sleep 5
      [[ -f "$SCORE_PATH" ]] && result_yaml="$SCORE_PATH"
      break
    fi
    sleep 3
  done

  # Stop the recorder cleanly so it flushes the rrd.
  if kill -0 "$REC_PID" 2>/dev/null; then
    echo "[rec][iter=$i] stopping recorder (PID=$REC_PID)"
    kill -TERM "$REC_PID" 2>/dev/null || true
    # Wait up to 10 s for it to flush.
    for k in $(seq 1 20); do
      kill -0 "$REC_PID" 2>/dev/null || break
      sleep 0.5
    done
    kill -KILL "$REC_PID" 2>/dev/null || true
  fi

  local t_end=$(date +%s)
  local total_elapsed=$((t_end - t_start))
  echo "$total_elapsed" > "$iter_dir/elapsed_s.txt"

  if [[ -n "$result_yaml" ]]; then
    cp "$result_yaml" "$iter_dir/scoring.yaml" || true
    pixi run --manifest-path /home/ubuntu/ws_aic/src/aic/pixi.toml \
      python "$DEMO_SCRIPTS/parse_smoke_result.py" \
      --iteration "$i" --seed "$seed" \
      --scoring "$iter_dir/scoring.yaml" \
      --summary "$sumcfg" \
      --csv "$CSV_PATH" \
      --elapsed-s "$total_elapsed" \
      2>&1 | tail -1
  else
    pixi run --manifest-path /home/ubuntu/ws_aic/src/aic/pixi.toml \
      python "$DEMO_SCRIPTS/parse_smoke_result.py" \
      --iteration "$i" --seed "$seed" \
      --scoring "$iter_dir/scoring.yaml" \
      --summary "$sumcfg" \
      --csv "$CSV_PATH" \
      --elapsed-s "$total_elapsed" \
      --failed \
      2>&1 | tail -1
  fi

  bash "$SCRIPTS/cleanup.sh" >> "$LOGDIR/cleanup.log" 2>&1 || true
  sleep 2
}

for i in $(seq 1 "$N"); do
  seed=$((SEED_BASE + i))
  echo "[rec] ==== iter $i / $N (seed=$seed) ===="
  run_one_iter "$i" "$seed" || echo "[rec][iter=$i] non-zero"
done

echo "[rec] done. CSV: $CSV_PATH"
echo "[rec] artifacts: $RUN_DIR"
echo "[rec] rrds: $RRD_OUT_DIR"
