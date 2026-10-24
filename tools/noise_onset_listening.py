"""Export matched-loudness original speech without filtering or time changes."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy.io import wavfile

from retrain_noise_compare import loudness

EPOCHS = [100, 140, 170, 180, 182, 184, 200]


def export(root):
    assert (root / 'complete.json').exists()
    folder = root / 'listening'
    folder.mkdir(exist_ok=True)
    sources = [(test, epoch, root / f'raw/{epoch:03}_{test}.wav')
               for test in ('user', 'travel') for epoch in EPOCHS]
    measured = [loudness(path) for _, _, path in sources]
    target = min(-23., *(-1.1 - float(m['input_tp']) + float(m['input_i']) for m in measured))
    report = []
    rendered = {}
    for (test, epoch, path), before in zip(sources, measured):
        rate, original = wavfile.read(path)
        assert rate == 44100 and original.dtype == np.int16
        gain = target - float(before['input_i'])
        output = folder / path.name
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(path), '-af', f'volume={gain}dB',
                        '-c:a', 'pcm_s16le', str(output)], check=True)
        out_rate, pcm = wavfile.read(output)
        assert out_rate == rate and pcm.shape == original.shape and pcm.dtype == original.dtype
        assert np.max(np.abs(pcm.astype(np.int32))) < 32767
        after = loudness(output)
        assert abs(float(after['input_i']) - target) < .12 and float(after['input_tp']) < -1
        rendered[test, epoch] = pcm
        report.append({'test': test, 'epoch': epoch, 'gain_db': gain, 'lufs': float(after['input_i']),
                       'true_peak_dbtp': float(after['input_tp']), 'samples': len(pcm)})
    for test in ('user', 'travel'):
        parts = []
        for epoch in EPOCHS:
            if parts:
                parts.append(np.zeros(44100, dtype=np.int16))
            parts.append(rendered[test, epoch])
        pcm = np.concatenate(parts)
        output = folder / (test + '_100_140_170_180_182_184_200.wav')
        wavfile.write(output, 44100, pcm)
        _, checked = wavfile.read(output)
        assert np.array_equal(pcm, checked)
    (root / 'listening_levels.json').write_text(json.dumps({'epochs': EPOCHS, 'target_lufs': target,
        'processing': 'linear gain only; one-second silence between complete original utterances',
        'levels': report}, indent=2))
    print('LISTENING_EXPORT_VERIFIED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    export(parser.parse_args().root)
