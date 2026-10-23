"""Apply gentle FFT denoising to ordinary sentences, preserving timing and originals."""
import argparse
import json
from pathlib import Path
import re
import subprocess

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, correlate, correlation_lags

from noise_spectral_detail import detail

FILTER = 'afftdn=nr=6:nf=-55:gs=10'

def level(path):
    r = subprocess.run(['ffmpeg', '-hide_banner', '-i', str(path), '-af', 'loudnorm=print_format=json', '-f', 'null', '-'], capture_output=True, text=True, check=True)
    return json.loads(re.search(r'\{\s*"input_i"[\s\S]*?\}', r.stderr).group())

def export(src, dst, filt):
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(src), '-af', filt, '-c:a', 'pcm_s16le', str(dst)], check=True)

def run(root):
    dest = root / 'denoised'
    dest.mkdir(exist_ok=True)
    rows = []
    for name in ['user', 'travel', 'sibilants', 'current']:
        src = root / 'raw/200_retrained' / (name + '.wav')
        dst = dest / (name + '_denoised.wav')
        fs, x = wavfile.read(src)
        # Pad the tail before filtering, then remove the measured FFT-filter latency.
        export(src, dst, f'apad=pad_len=4096,{FILTER},atrim=start_sample=1102:end_sample={1102+len(x)},asetpts=N/SR/TB')
        fs2, y = wavfile.read(dst)
        assert fs == fs2 == 44100 and x.dtype == y.dtype == np.int16
        assert x.shape == y.shape and x.ndim == 1
        xf, yf = x.astype(float)/32768, y.astype(float)/32768
        lag = int(correlation_lags(len(x), len(y))[np.argmax(correlate(xf, yf, method='fft'))])
        assert lag == 0, lag
        assert np.max(np.abs(y.astype(int))) < 32767
        f, _, zx = stft(xf, fs, nperseg=2048, noverlap=1536)
        _, _, zy = stft(yf, fs, nperseg=2048, noverlap=1536)
        bands = {}
        for label, lo, hi in [('speech_300_4000', 300, 4000), ('high_8000_20000', 8000, 20000)]:
            mask = (f >= lo) & (f <= hi)
            bands[label + '_energy_change_db'] = float(10*np.log10(np.sum(abs(zy[mask])**2)/np.sum(abs(zx[mask])**2)))
        pieces = []
        levels = []
        for label, path in [('before', src), ('after', dst)]:
            l = level(path)
            out = dest / (name + '_' + label + '_level.wav')
            export(path, out, 'volume=' + str(-23-float(l['input_i'])) + 'dB')
            checked = level(out)
            assert abs(float(checked['input_i'])+23) < .12 and float(checked['input_tp']) < -1
            _, pcm = wavfile.read(out)
            assert np.max(np.abs(pcm.astype(int))) < 32767
            pieces.append(pcm)
            levels.append(checked)
        wavfile.write(dest / (name + '_before_after.wav'), fs, np.concatenate([pieces[0], np.zeros(fs, dtype=np.int16), pieces[1]]))
        durations = json.loads(src.with_suffix('.json').read_text())['moraDurations']
        rows.append({'sentence': name, 'filter': FILTER, 'samples': len(x), 'lag_samples': lag,
                     'pcm_clip_count': 0, **bands, 'comb_before_db': detail(src, durations)['comb_mean_db'],
                     'comb_after_db': detail(dst, durations)['comb_mean_db'], 'levels': levels})
    (dest / 'verification.json').write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    run(p.parse_args().root)
