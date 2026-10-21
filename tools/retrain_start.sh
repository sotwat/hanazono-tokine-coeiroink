#!/usr/bin/env bash
set -euo pipefail
source /content/tokine-202211/bin/activate
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
run=/content/drive/MyDrive/MYCOEIROINK_WORK/retrain_180_20261021
cd /content/tokine-retrain
python "$run/retrain_preflight.py" "$run"
python -m pip freeze > "$run/pip-freeze.txt"
python -m espnet2.bin.gan_tts_train --config "$run/train.yaml"
