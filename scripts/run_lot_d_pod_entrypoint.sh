#!/usr/bin/env bash
# Lot D pod entrypoint — Option C from parametric_insurance issue #14.
#
# Invoked by `parametric_insurance/scripts/runpod_launch.py --with-cerra-download`,
# which forwards all the env vars below. Orchestrates the full Lot B + Lot C
# production for the years covered by [CERRA_START, CERRA_END].
#
# Stages
#   1. Download CERRA atm + CERRA-Land NetCDF (download_cerra_for_pod.py)
#   2. Lot B per year (run_lot_b.py) → Zarr under $CERRA_OUT_LOT_B/<year>.zarr
#   3. Lot C per year (run_lot_c.py) → Zarr under $CERRA_OUT_LOT_C/<year>.zarr
#   4. Push artefacts to kDrive (or S3 when CERRA_PUSH_S3=1) — grids + checkpoints
#
# Inputs (env vars forwarded from runpod_launch.py)
# -------------------------------------------------
#   CDSAPI_URL, CDSAPI_KEY
#   CERRA_START, CERRA_END                    # ISO dates
#   CERRA_BBOX_LAT_MIN/MAX, CERRA_BBOX_LON_MIN/MAX
#   CERRA_OUT_ATM, CERRA_OUT_LAND              # NetCDF download targets
#   CERRA_OUT_LOT_B (default /workspace/data/output/lot_b_grid)
#   CERRA_OUT_LOT_C (default /workspace/data/output/lot_c_grid)
#   CERRA_DEM        (default /workspace/data/dem/bd_alti_drome.nc)
#   CERRA_SENCROP    (default $SENCROP_DATA_ROOT or /workspace/data/sencrop)
#   CERRA_LOT_C_EPOCHS (default 30)
#   CERRA_PUSH_TARGET (default /Users/loicmaurin/kDrive/karpos_datasets/output)
#   WANDB_API_KEY    # optional, enables W&B logging on Lot C
#   PRE_INSTALL=cdsapi   # ensures cdsapi installed
#
# Failure modes
# -------------
#   - Hard-exits non-zero on any stage failure (set -e). Caller propagates.
#   - Skips stages whose outputs already exist (idempotent re-run).

set -euo pipefail

log() { echo "[$(date -u +%FT%TZ)] $*"; }

: "${CERRA_START:?CERRA_START required}"
: "${CERRA_END:?CERRA_END required}"
: "${CERRA_OUT_ATM:?CERRA_OUT_ATM required}"
: "${CERRA_OUT_LAND:?CERRA_OUT_LAND required}"

CERRA_OUT_LOT_B="${CERRA_OUT_LOT_B:-/workspace/data/output/lot_b_grid}"
CERRA_OUT_LOT_C="${CERRA_OUT_LOT_C:-/workspace/data/output/lot_c_grid}"
CERRA_DEM="${CERRA_DEM:-/workspace/data/dem/bd_alti_drome.nc}"
CERRA_SENCROP="${CERRA_SENCROP:-${SENCROP_DATA_ROOT:-/workspace/data/sencrop}}"
CERRA_LOT_C_EPOCHS="${CERRA_LOT_C_EPOCHS:-30}"
CERRA_PUSH_TARGET="${CERRA_PUSH_TARGET:-/Users/loicmaurin/kDrive/karpos_datasets/output}"

YEAR_START=$(date -d "${CERRA_START}" +%Y 2>/dev/null || python3 -c "from datetime import date; print(date.fromisoformat('${CERRA_START}').year)")
YEAR_END=$(date -d "${CERRA_END}" +%Y 2>/dev/null || python3 -c "from datetime import date; print(date.fromisoformat('${CERRA_END}').year)")
YEARS=$(seq "$YEAR_START" "$YEAR_END")

mkdir -p "$CERRA_OUT_ATM" "$CERRA_OUT_LAND" "$CERRA_OUT_LOT_B" "$CERRA_OUT_LOT_C"

# ----------------------------------------------------------------------------
# Stage 1 — CERRA download
# ----------------------------------------------------------------------------
log "Stage 1: CERRA download"
uv run python -m downscaling.scripts.download_cerra_for_pod

# ----------------------------------------------------------------------------
# Stage 2 — Lot B per year
# ----------------------------------------------------------------------------
log "Stage 2: Lot B (statistical + Sencrop)"
for Y in $YEARS; do
  CERRA_ATM="$CERRA_OUT_ATM/cerra_atm_${Y}.nc"
  CERRA_LAND="$CERRA_OUT_LAND/cerra_land_${Y}.nc"
  ZARR_B="$CERRA_OUT_LOT_B/${Y}.zarr"

  if [ -d "$ZARR_B" ]; then
    log "  $Y: skip (Lot B already exists at $ZARR_B)"
    continue
  fi

  log "  $Y: Lot B run"
  uv run python -m downscaling.scripts.run_lot_b \
    --year "$Y" \
    --cerra-atm  "$CERRA_ATM" \
    --cerra-land "$CERRA_LAND" \
    --dem        "$CERRA_DEM" \
    --sencrop    "$CERRA_SENCROP" \
    --out        "$CERRA_OUT_LOT_B"
done

# ----------------------------------------------------------------------------
# Stage 3 — Lot C per year (GPU)
# ----------------------------------------------------------------------------
log "Stage 3: Lot C (DL FiLM + sparse Sencrop)"
for Y in $YEARS; do
  CERRA_ATM="$CERRA_OUT_ATM/cerra_atm_${Y}.nc"
  ZARR_C="$CERRA_OUT_LOT_C/${Y}.zarr"

  if [ -d "$ZARR_C" ]; then
    log "  $Y: skip (Lot C already exists at $ZARR_C)"
    continue
  fi

  log "  $Y: Lot C train + infer"
  uv run python -m downscaling.scripts.run_lot_c \
    --year      "$Y" \
    --cerra-atm "$CERRA_ATM" \
    --dem       "$CERRA_DEM" \
    --sencrop   "$CERRA_SENCROP" \
    --out       "$CERRA_OUT_LOT_C" \
    --epochs    "$CERRA_LOT_C_EPOCHS"
done

# ----------------------------------------------------------------------------
# Stage 4 — Push artefacts to kDrive / S3
# ----------------------------------------------------------------------------
log "Stage 4: push artefacts"
if [ -d "$CERRA_PUSH_TARGET" ]; then
  log "  → $CERRA_PUSH_TARGET"
  rsync -avz --progress "$CERRA_OUT_LOT_B/" "$CERRA_PUSH_TARGET/lot_b_grid/"
  rsync -avz --progress "$CERRA_OUT_LOT_C/" "$CERRA_PUSH_TARGET/lot_c_grid/"
elif [ -n "${CERRA_PUSH_S3:-}" ]; then
  log "  → s3://$CERRA_PUSH_S3"
  uv run python -c "
import boto3, os, glob
s3 = boto3.client('s3')
bucket = os.environ['CERRA_PUSH_S3'].split('/')[0]
prefix = '/'.join(os.environ['CERRA_PUSH_S3'].split('/')[1:])
for src, sub in [(os.environ['CERRA_OUT_LOT_B'], 'lot_b_grid'), (os.environ['CERRA_OUT_LOT_C'], 'lot_c_grid')]:
    for path in glob.glob(f'{src}/**', recursive=True):
        if os.path.isfile(path):
            rel = os.path.relpath(path, src)
            s3.upload_file(path, bucket, f'{prefix}/{sub}/{rel}')
print('S3 push done')
"
else
  log "  no push target (CERRA_PUSH_TARGET unset and CERRA_PUSH_S3 unset) — leaving artefacts in /workspace"
fi

log "Lot D pod entrypoint done."
