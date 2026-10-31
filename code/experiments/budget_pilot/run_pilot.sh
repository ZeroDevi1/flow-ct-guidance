#!/usr/bin/env bash
# Run on the existing, user-authorized AutoDL instance only.
# Foreground this wrapper inside nohup/tmux as needed. It shuts THIS instance down.
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
if [[ ${AUTODL_BUDGET_SHUTDOWN_APPROVED:-} != YES ]]; then
  echo 'Set AUTODL_BUDGET_SHUTDOWN_APPROVED=YES only after instance shutdown is authorized.' >&2; exit 2
fi
command -v timeout >/dev/null
SHUTDOWN=$(command -v shutdown) || { echo 'No shutdown command: refuse GPU work' >&2; exit 2; }
OUT=${PILOT_OUT:-/root/autodl-tmp/flowct_runs/pilot_$(date -u +%Y%m%dT%H%M%SZ)}
[[ ! -e "$OUT" ]] || { echo 'Output exists, refusing overwrite'; exit 2; }
mkdir -p "$OUT"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
# Independent local safety timer; control-plane must still verify shutdown/billing.
nohup bash -c 'sleep 1800; sync; for i in 1 2 3; do "$1" -h now; sleep 10; done' _ "$SHUTDOWN" >"$OUT/watchdog.log" 2>&1 </dev/null &
WATCHDOG=$!
printf '%s\n' "$WATCHDOG" > "$OUT/watchdog.pid"
finish() {
  rc=$?
  printf '[EXIT] rc=%s UTC=%s\n' "$rc" "$(date -u +%FT%TZ)" >> "$OUT/launch.log"
  sync
  # Keep watchdog alive if this attempt fails. On success the machine terminates it.
  "$SHUTDOWN" -h now >> "$OUT/launch.log" 2>&1 || echo '[ERROR] shutdown request failed; watchdog remains active' >> "$OUT/launch.log"
  exit "$rc"
}
trap finish EXIT
exec > >(tee -a "$OUT/launch.log") 2>&1
printf '[START] %s\n' "$(date -u +%FT%TZ)"
printf '[BUDGET] pilot timeout=1500s, whole-instance watchdog=1800s; no training, validation only\n'
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'
timeout --signal=TERM --kill-after=15s 180s python -m experiments.budget_pilot.check_controls --device cuda --size 64 --out "$OUT/p0_controls.json"
timeout --signal=TERM --kill-after=30s 1500s python -m experiments.budget_pilot.run_validation \
 --processed "${PROCESSED:-/root/autodl-tmp/data/totalseg/processed}" \
 --ckpt "${PRIOR:-/root/autodl-tmp/checkpoints/prior_256/ema_final.pt}" \
 --out "$OUT/validation" --patients 4 --slices-per-patient 2 --nfe 20 --max-seconds 1200
printf '[DONE] %s\n' "$(date -u +%FT%TZ)"
