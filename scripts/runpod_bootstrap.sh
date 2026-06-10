#!/usr/bin/env bash
# Bootstrap d'un run downscaling sur un pod RunPod (image PyTorch publique).
# Appelé par runpod_run.py via docker_args : évite le quoting complexe côté API.
#   bash downscaling/scripts/runpod_bootstrap.sh <task>
# Tâches : smoke | caltaw (étage C tail-aware) | unet-train
# Secrets attendus en env : SCW_ACCESS_KEY, SCW_SECRET_KEY, SCW_S3_ENDPOINT.
set -euo pipefail
TASK="${1:-smoke}"
S3="s3://karpos-backtest-data/downscaling"
SYNC="uv run python downscaling/scripts/s3_sync.py"

echo ">>> uv install + sync"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --extra dl --extra cloud

if [ "$TASK" = "smoke" ]; then
  echo ">>> SMOKE : GPU + S3"
  uv run python -c "import torch,sys; open('smoke.txt','w').write(f'cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None} torch={torch.__version__}')"
  cat smoke.txt
  $SYNC push smoke.txt "$S3/_smoke/"
  echo ">>> RUNPOD_DONE"; exit 0
fi

echo ">>> pull data S3"
$SYNC pull "$S3/checkpoints/mv_run/"      checkpoints/mv_run/
$SYNC pull "$S3/cerra_fine_mv_cal/"       data/cerra_fine_mv_cal/
$SYNC pull "$S3/cerra_fine_mv_test/"      data/cerra_fine_mv_test/
$SYNC pull "$S3/sencrop/"                 data/sencrop/
$SYNC pull "$S3/dem/"                     data/training/

case "$TASK" in
  caltaw)
    echo ">>> étage C tail-aware (frost_alpha=6) sur GPU"
    uv run python downscaling/scripts/run_calibration.py cluster=cloud \
      data.dem_attrs=data/training/dem_attributes.nc \
      'dl.met_vars=[t2m,td2m,u10]' 'dl.fine_vars=[t2m]' \
      calibration.cerra_fine_dir=data/cerra_fine_mv_cal \
      calibration.checkpoint=checkpoints/mv_run/best_model.ckpt \
      calibration.stats_file=checkpoints/mv_run/normalization_stats.json \
      calibration.out=checkpoints/mv_run/calibrated_taw.pt \
      calibration.frost_alpha=6 calibration.epochs=30 calibration.min_stations=5
    $SYNC push checkpoints/mv_run/calibrated_taw.pt "$S3/checkpoints/mv_run/"
    ;;
  *) echo "tâche inconnue: $TASK"; exit 1 ;;
esac
echo ">>> RUNPOD_DONE"
