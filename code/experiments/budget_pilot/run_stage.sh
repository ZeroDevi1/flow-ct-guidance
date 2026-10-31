#!/usr/bin/env bash
# Phase runner only: NEVER powers off the instance. A verified platform deadline
# must protect the whole paid session, including idle diagnosis and setup time.
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
STAGE=${1:?Usage: run_stage.sh diagnostic|pilot}
[[ "$STAGE" == diagnostic || "$STAGE" == pilot ]] || exit 2
[[ ${PLATFORM_DEADLINE_VERIFIED:-} == YES ]] || { echo 'Verify platform hard shutdown deadline first'; exit 2; }
DEADLINE=${PLATFORM_DEADLINE_EPOCH:?Set verified platform shutdown time as UTC epoch}
[[ "$DEADLINE" =~ ^[0-9]+$ ]] || exit 2
NOW=$(date +%s)
REMAINING=$((DEADLINE-NOW))
(( REMAINING > 180 && REMAINING <= 28800 )) || { echo 'Platform deadline must be 3 minutes to 8 hours away'; exit 2; }
LIMIT=${STAGE_TIMEOUT_SECONDS:-120}
if [[ "$STAGE" == pilot ]]; then LIMIT=${STAGE_TIMEOUT_SECONDS:-1200}; fi
[[ "$LIMIT" =~ ^[0-9]+$ ]] && (( LIMIT > 0 && LIMIT <= 1500 && LIMIT + 60 < REMAINING )) || { echo 'Invalid stage time budget'; exit 2; }
OUT=${STAGE_OUT:-/root/autodl-tmp/flowct_runs/${STAGE}_$(date -u +%Y%m%dT%H%M%SZ)}
[[ ! -e "$OUT" ]] || { echo 'Output exists; refuse overwrite'; exit 2; }
mkdir -p "$OUT"
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
finish() {
 rc=$?
 printf '{"exit_code":%d,"stage":"%s","ended_epoch":%d,"platform_deadline_epoch":%d,"instance_shutdown_requested":false}\n' "$rc" "$STAGE" "$(date +%s)" "$DEADLINE" > "$OUT/launcher_status.json"
 sync
 echo "[EXIT] rc=$rc; instance remains on under platform budget deadline"
 exit "$rc"
}
trap finish EXIT
exec > >(tee -a "$OUT/launch.log") 2>&1
printf '[START] stage=%s UTC=%s task_timeout=%ss platform_deadline_epoch=%s\n' "$STAGE" "$(date -u +%FT%TZ)" "$LIMIT" "$DEADLINE"
if [[ "$STAGE" == diagnostic ]]; then
 timeout --signal=TERM --kill-after=15s "$LIMIT" python -m experiments.budget_pilot.diagnose_gpu_control --device cuda --size 64 --out "$OUT/gpu64.json"
else
 # Do not bypass or relax the original numerical gate. It may fail; retain logs.
 export STAGE_PRIVATE_OUT="$OUT"
 timeout --signal=TERM --kill-after=30s "$LIMIT" bash -c '
  set -euo pipefail
  python -m experiments.budget_pilot.check_controls --device cuda --size 64 --out "$STAGE_PRIVATE_OUT/p0_controls.json"
  python -m experiments.budget_pilot.run_validation \
    --processed "${PROCESSED:-/root/autodl-tmp/data/totalseg/processed}" \
    --ckpt "${PRIOR:-/root/autodl-tmp/checkpoints/01_diffusion_inverse/prior_256/ema_final.pt}" \
    --out "$STAGE_PRIVATE_OUT/validation" --patients 4 --slices-per-patient 2 --nfe 20 --max-seconds 1100
 '
fi
