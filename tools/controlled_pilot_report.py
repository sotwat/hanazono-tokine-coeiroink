"""Measure complete pilot outputs and export comparisons with linear gain only."""
import argparse
import csv
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.io import wavfile

from retrain_noise_compare import loudness


def build(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    kinds = [item['name'] for item in manifest]
    assert kinds == ['180_original', '200_retrained', 'control_181', 'control_182',
                     'quarter_lr_181', 'quarter_lr_182']
    rows = json.loads((root / 'results.json').read_text())
    tests = list(dict.fromkeys(row['test'] for row in rows))
    assert len(tests) == 8 and len(rows) == 48
    assert len({(row['name'], row['test']) for row in rows}) == 48
    assert {row['name'] for row in rows} == set(kinds)
    listening = root / 'listening'
    listening.mkdir(exist_ok=True)
    levels = []
    for test in tests:
        sources = [root / 'raw' / kind / (test + '.wav') for kind in kinds]
        before = [loudness(source) for source in sources]
        target = min(-23., *(-1.1 - float(x['input_tp']) + float(x['input_i'])
                            for x in before))
        assert np.isfinite(target)
        parts = []
        for kind, source, measured in zip(kinds, sources, before):
            rate, original = wavfile.read(source)
            assert rate == 44100 and original.dtype == np.int16 and original.ndim == 1
            output = listening / f'{test}_{kind}.wav'
            gain = target - float(measured['input_i'])
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(source),
                            '-af', f'volume={gain}dB', '-c:a', 'pcm_s16le', str(output)], check=True)
            checked = loudness(output)
            out_rate, pcm = wavfile.read(output)
            assert out_rate == rate and pcm.shape == original.shape and pcm.dtype == original.dtype
            assert np.max(np.abs(pcm.astype(np.int32))) < 32767
            assert abs(float(checked['input_i']) - target) < .12
            assert float(checked['input_tp']) < -1
            levels.append({'test': test, 'name': kind, 'gain_db': gain, 'target_lufs': target,
                           'lufs': float(checked['input_i']), 'true_peak_dbtp': float(checked['input_tp'])})
            parts.append(pcm)
        for label, indices in [('all_six', range(6)), ('epoch181_control_quarter', [0, 2, 4]),
                               ('epoch182_control_quarter', [0, 3, 5])]:
            selected = []
            for index in indices:
                if selected:
                    selected.append(np.zeros(44100, dtype=np.int16))
                selected.append(parts[index])
            wavfile.write(listening / f'{test}_{label}.wav', 44100, np.concatenate(selected))
    fields = ['name', 'test', 'comb_mean_db', 'vowel_hf8_rel_db', 'rms_dbfs', 'pcm_clip_count', 'seconds']
    with (root / 'measurements.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    summary = {'listening_order': kinds, 'tests': tests, 'waveforms_measured': len(rows),
               'quality_improved': None, 'listening_verified': False,
               'processing': 'linear gain only; no denoising, EQ, stretching or weight correction',
               'averages': {kind: {field: float(np.mean([row[field] for row in rows if row['name'] == kind]))
                                    for field in ['comb_mean_db', 'vowel_hf8_rel_db', 'rms_dbfs']}
                            for kind in kinds}, 'levels': levels}
    (root / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != 'levels'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    build(parser.parse_args().root)
