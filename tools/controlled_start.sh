#!/usr/bin/env bash
set -euo pipefail
root="${1:?Pass the prepared pilot directory}"
target="${2:-182}"
[[ "$target" =~ ^[0-9]+$ ]] && (( target >= 182 && target <= 200 )) || { echo 'Target must be 182..200'; exit 1; }
scripts="$(cd "$(dirname "$0")" && pwd)"
source /content/tokine-202211/bin/activate
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
test -f "$root/preparation.json"
exec 9>/tmp/tokine-controlled-training.lock
flock -n 9 || { echo 'Another paired training launcher is already running'; exit 1; }
python -m pip freeze > "$root/pip-freeze.txt"
for arm in control quarter_lr; do
  epoch="$(python - "$root/$arm/model/checkpoint.pth" <<'PY'
import sys, torch
print(torch.load(sys.argv[1], map_location='cpu')['reporter']['epoch'])
PY
)"
  if test "$epoch" = "$target"; then
    echo "$arm already finished; preserving its checkpoint"
    continue
  fi
  (( epoch < target )) || { echo "Checkpoint $epoch exceeds requested target $target"; exit 1; }
  python "$scripts/controlled_retrain.py" run --run-dir "$root/$arm" --max-epoch "$target" >> "$root/$arm/train.log" 2>&1
done
echo "Paired training reached $target. Synthesis and quality comparison are still required."
