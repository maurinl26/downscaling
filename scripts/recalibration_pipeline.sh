#!/usr/bin/env bash
# Recalibration pipeline orchestrator — Option C of parametric_insurance#14.
#
# Invoked by `parametric_insurance/scripts/runpod_launch.py --with-cerra-download`,
# which forwards all the env vars below. Runs the full statistical + DL FiLM
# recalibration on the years covered by [CERRA_START, CERRA_END].
#
# Stages
#   1. Download CERRA atm + CERRA-Land NetCDF
#      (downscaling.scripts.download_cerra_for_recalibration)
#   2. Statistical recalibration per year
#      (downscaling.scripts.recalibrate_statistical) → Zarr under
#      $RECALIB_OUT_STATISTICAL/<year>.zarr
#   3. DL FiLM recalibration per year
#      (downscaling.scripts.recalibrate_dl_film) → Zarr under
#      $RECALIB_OUT_DL_FILM/<year>.zarr
#   4. Push artefacts to kDrive (or S3 when RECALIB_PUSH_S3=1) —
#      grids + checkpoints
#
# Inputs (env vars forwarded from runpod_launch.py)
# -------------------------------------------------
#   CDSAPI_URL, CDSAPI_KEY
#   CERRA_START, CERRA_END                          # ISO dates
#   CERRA_BBOX_LAT_MIN/MAX, CERRA_BBOX_LON_MIN/MAX
#   CERRA_OUT_ATM, CERRA_OUT_LAND                    # NetCDF download targets
#   RECALIB_OUT_STATISTICAL (default /workspace/data/output/recalibrated_statistical)
#   RECALIB_OUT_DL_FILM     (default /workspace/data/output/recalibrated_dl_film)
#   RECALIB_DEM             (default /workspace/data/dem/bd_alti_drome.nc)
#   RECALIB_SENCROP         (default $SENCROP_DATA_ROOT or /workspace/data/sencrop)
#   RECALIB_DL_FILM_EPOCHS  (default 30)
#   RECALIB_PUSH_TARGET     (default /Users/loicmaurin/kDrive/karpos_datasets/output)
#   WANDB_API_KEY           # optional, enables W&B logging on the DL stage
#   PRE_INSTALL=cdsapi      # ensures cdsapi installed
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

RECALIB_OUT_STATISTICAL="${RECALIB_OUT_STATISTICAL:-/workspace/data/output/recalibrated_statistical}"
RECALIB_OUT_DL_FILM="${RECALIB_OUT_DL_FILM:-/workspace/data/output/recalibrated_dl_film}"
RECALIB_DEM="${RECALIB_DEM:-/workspace/data/dem/bd_alti_drome.nc}"
RECALIB_SENCROP="${RECALIB_SENCROP:-${SENCROP_DATA_ROOT:-/workspace/data/sencrop}}"
RECALIB_DL_FILM_EPOCHS="${RECALIB_DL_FILM_EPOCHS:-30}"
RECALIB_PUSH_TARGET="${RECALIB_PUSH_TARGET:-/Users/loicmaurin/kDrive/karpos_datasets/output}"

YEAR_START=$(date -d "${CERRA_START}" +%Y 2>/dev/null || python3 -c "from datetime import date; print(date.fromisoformat('${CERRA_START}').year)")
YEAR_END=$(date -d "${CERRA_END}" +%Y 2>/dev/null || python3 -c "from datetime import date; print(date.fromisoformat('${CERRA_END}').year)")
YEARS=$(seq "$YEAR_START" "$YEAR_END")

mkdir -p "$CERRA_OUT_ATM" "$CERRA_OUT_LAND" "$RECALIB_OUT_STATISTICAL" "$RECALIB_OUT_DL_FILM"

# ----------------------------------------------------------------------------
# Stage 1 — CERRA download
# ----------------------------------------------------------------------------
log "Stage 1: CERRA download"
uv run python -m downscaling.scripts.download_cerra_for_recalibration

# ----------------------------------------------------------------------------
# Stage 2 — Statistical recalibration per year
# ----------------------------------------------------------------------------
log "Stage 2: statistical recalibration (lapse + QDM + Sencrop residual)"
for Y in $YEARS; do
  CERRA_ATM="$CERRA_OUT_ATM/cerra_atm_${Y}.nc"
  CERRA_LAND="$CERRA_OUT_LAND/cerra_land_${Y}.nc"
  ZARR_STAT="$RECALIB_OUT_STATISTICAL/${Y}.zarr"

  if [ -d "$ZARR_STAT" ]; then
    log "  $Y: skip (already at $ZARR_STAT)"
    continue
  fi

  log "  $Y: statistical run"
  uv run python -m downscaling.scripts.recalibrate_statistical \
    --year "$Y" \
    --cerra-atm  "$CERRA_ATM" \
    --cerra-land "$CERRA_LAND" \
    --dem        "$RECALIB_DEM" \
    --sencrop    "$RECALIB_SENCROP" \
    --out        "$RECALIB_OUT_STATISTICAL"
done

# ----------------------------------------------------------------------------
# Stage 3 — DL FiLM recalibration per year (GPU)
# ----------------------------------------------------------------------------
log "Stage 3: DL FiLM recalibration (U-Net FiLM + sparse Sencrop)"
for Y in $YEARS; do
  CERRA_ATM="$CERRA_OUT_ATM/cerra_atm_${Y}.nc"
  ZARR_DL="$RECALIB_OUT_DL_FILM/${Y}.zarr"

  if [ -d "$ZARR_DL" ]; then
    log "  $Y: skip (already at $ZARR_DL)"
    continue
  fi

  log "  $Y: DL FiLM train + infer"
  uv run python -m downscaling.scripts.recalibrate_dl_film \
    --year      "$Y" \
    --cerra-atm "$CERRA_ATM" \
    --dem       "$RECALIB_DEM" \
    --sencrop   "$RECALIB_SENCROP" \
    --out       "$RECALIB_OUT_DL_FILM" \
    --epochs    "$RECALIB_DL_FILM_EPOCHS"
done

# ----------------------------------------------------------------------------
# Stage 4 — Push artefacts to kDrive / S3
# ----------------------------------------------------------------------------
log "Stage 4: push artefacts"
if [ -d "$RECALIB_PUSH_TARGET" ]; then
  log "  → $RECALIB_PUSH_TARGET"
  rsync -avz --progress "$RECALIB_OUT_STATISTICAL/" "$RECALIB_PUSH_TARGET/recalibrated_statistical/"
  rsync -avz --progress "$RECALIB_OUT_DL_FILM/"     "$RECALIB_PUSH_TARGET/recalibrated_dl_film/"
elif [ -n "${RECALIB_PUSH_S3:-}" ]; then
  log "  → s3://$RECALIB_PUSH_S3"
  uv run python -c "
import boto3, os, glob
s3 = boto3.client('s3')
bucket = os.environ['RECALIB_PUSH_S3'].split('/')[0]
prefix = '/'.join(os.environ['RECALIB_PUSH_S3'].split('/')[1:])
for src, sub in [(os.environ['RECALIB_OUT_STATISTICAL'], 'recalibrated_statistical'), (os.environ['RECALIB_OUT_DL_FILM'], 'recalibrated_dl_film')]:
    for path in glob.glob(f'{src}/**', recursive=True):
        if os.path.isfile(path):
            rel = os.path.relpath(path, src)
            s3.upload_file(path, bucket, f'{prefix}/{sub}/{rel}')
print('S3 push done')
"
else
  log "  no push target (RECALIB_PUSH_TARGET unset and RECALIB_PUSH_S3 unset), leaving artefacts in /workspace"
fi

log "Recalibration pipeline done."
