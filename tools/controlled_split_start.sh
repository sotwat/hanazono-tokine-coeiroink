#!/usr/bin/env bash
set -euo pipefail
root="${1:?Pass the prepared gd_split directory}"
scripts="$(cd "$(dirname "$0")" && pwd)"
source /content/tokine-202211/bin/activate
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
test -f "$root/preparation.json"
exec 9>/tmp/tokine-controlled-training.lock
flock -n 9 || { echo 'Another controlled training launcher is already running'; exit 1; }
run="$root/discriminator_quarter"
epoch="$(python - "$run/model/checkpoint.pth" <<'PY'
import sys, torch
print(torch.load(sys.argv[1], map_location='cpu')['reporter']['epoch'])
PY
)"
if test "$epoch" = 200; then
  echo 'Discriminator-quarter arm already finished; preserving its checkpoint'
  exit 0
fi
(( epoch >= 180 && epoch < 200 )) || { echo "Invalid checkpoint epoch $epoch"; exit 1; }
python -m pip freeze > "$root/pip-freeze.txt"
python "$scripts/controlled_retrain.py" run --run-dir "$run" --max-epoch 200 >> "$run/train.log" 2>&1
echo 'Discriminator-quarter reached 200; untreated speech quality is not yet verified.'
