"""Measure recovered untreated candidates with the fixed ordinary-eight protocol."""
import argparse
import base64
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import socket
import subprocess
import time
import urllib.request

import torch
import numpy as np
from scipy.io import wavfile
from noise_audit import analyze, post
from noise_spectral_detail import detail
from retrained_decoder_audit import TEXTS, UUID


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root, previous, inventory, models, series):
    root.mkdir(parents=True, exist_ok=True)
    provenance = {'texts': TEXTS, 'speedScale': 1, 'engine_sha256': sha(previous / 'runtime/engine/engine'), 'postprocessing': False}
    if (root / 'provenance.json').exists():
        assert json.loads((root / 'provenance.json').read_text()) == provenance
    with (root.parent / 'ranking_candidate.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with socket.socket() as sock:
            assert sock.connect_ex(('127.0.0.1', 50033)) != 0, 'Audit port occupied'
        (root / 'complete.json').unlink(missing_ok=True)
        reference = torch.load(previous / 'models/180_original.pth', map_location='cpu', weights_only=True)
        speaker = previous / 'runtime/speaker_info/audit'
        meta_path = speaker / 'metas.json'
        meta = json.loads(meta_path.read_text())
        assert meta['speakerUuid'] == UUID
        manifest = []
        for row in json.loads(inventory.read_text())['rows']:
            if row['series'] != series:
                continue
            epoch = row['epoch']
            if epoch == 200 or epoch <= 182:
                continue
            model = models / f'{epoch}epoch.pth'
            if not model.exists():
                continue
            assert model.stat().st_size == row['size'] and sha(model) == row['sha256']
            state = torch.load(model, map_location='cpu', weights_only=True)
            assert len(state) == 1049 and state.keys() == reference.keys()
            assert all(state[k].shape == reference[k].shape and torch.isfinite(state[k]).all() for k in state)
            del state
            sid = {'previous_retrain': 990000, 'control': 991000, 'quarter_lr': 992000}[series] + epoch
            name = f'{series}_{epoch}'
            folder = speaker / 'model' / str(sid)
            folder.mkdir(exist_ok=True)
            target = folder / 'model.pth'
            if target.is_symlink():
                assert target.resolve() == model.resolve()
            else:
                assert not target.exists()
                target.symlink_to(model.resolve())
            shutil.copy2(speaker / 'model/950180/config.yaml', folder / 'config.yaml')
            for directory, suffix in [('icons', '.png'), ('portraits', '.png'), ('voice_samples', '_001.wav'), ('voice_samples', '_002.wav'), ('voice_samples', '_003.wav')]:
                dst = speaker / directory / f'{sid}{suffix}'
                src = (speaker / directory / f'950180{suffix}').resolve()
                if dst.is_symlink():
                    assert dst.resolve() == src
                else:
                    assert not dst.exists()
                    dst.symlink_to(src)
            existing = [s for s in meta['styles'] if s['styleId'] == sid]
            assert not existing or existing == [{'styleId': sid, 'styleName': name}]
            if not existing:
                meta['styles'].append({'styleId': sid, 'styleName': name})
            manifest.append({**row, 'name': name, 'styleId': sid, 'local_path': str(model.resolve())})
        assert manifest, 'No recovered candidates'
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        (root / 'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
        rows = []
        with (root / 'engine.log').open('w') as log:
            engine = subprocess.Popen([str(previous / 'runtime/engine/engine')], cwd=previous / 'runtime', stdout=log, stderr=subprocess.STDOUT)
            try:
                for _ in range(120):
                    assert engine.poll() is None, 'Engine exited'
                    try:
                        urllib.request.urlopen('http://127.0.0.1:50033/v1/speakers', timeout=2).read()
                        break
                    except (OSError, TimeoutError):
                        time.sleep(2)
                else:
                    raise TimeoutError('Engine not ready')
                for item in manifest:
                    raw = root / 'raw' / item['name']
                    raw.mkdir(parents=True, exist_ok=True)
                    for test, text in TEXTS.items():
                        wav = raw / (test + '.wav')
                        metadata = wav.with_suffix('.json')
                        request = {'speakerUuid': UUID, 'styleId': item['styleId'], 'text': text, 'speedScale': 1}
                        assert wav.exists() == metadata.exists(), 'Incomplete cache pair'
                        if not wav.exists():
                            response = json.loads(post(50033, 'predict_with_duration', request))
                            data = base64.b64decode(response.pop('wavBase64'))
                            response.update(request=request, model_sha256=item['sha256'], wav_sha256=hashlib.sha256(data).hexdigest())
                            wav.write_bytes(data)
                            metadata.write_text(json.dumps(response, ensure_ascii=False, indent=2))
                        cached = json.loads(metadata.read_text())
                        assert cached['request'] == request and cached['model_sha256'] == item['sha256'] and cached['wav_sha256'] == sha(wav)
                        fs, pcm = wavfile.read(wav)
                        assert fs == 44100 and pcm.dtype == np.int16 and pcm.ndim == 1
                        duration = cached['moraDurations']
                        metrics = {**analyze(wav, duration), **detail(wav, duration)}
                        metrics = {k: v for k, v in metrics.items() if k not in ('frequencies', 'mean_db', 'median_db', 'hf_psd_db', 'hf_freq_hz')}
                        assert metrics['voiced_frames'] >= 3 and metrics['pcm_clip_count'] == 0
                        assert all(math.isfinite(metrics[k]) for k in ('seconds', 'rms_dbfs', 'comb_mean_db', 'vowel_hf8_rel_db'))
                        rows.append({'series': series, 'epoch': item['epoch'], 'test': test, 'model_sha256': item['sha256'], 'wav_sha256': sha(wav), **metrics})
                        (root / 'results.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
                        print(item['name'], test, len(rows), flush=True)
            finally:
                engine.terminate()
                try:
                    engine.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    engine.kill()
                    engine.wait()
        assert len(rows) == len(manifest) * 8
        (root / 'complete.json').write_text(json.dumps({'models': len(manifest), 'utterances': len(rows), 'results_sha256': sha(root / 'results.json'), 'listening_verified': False}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('root', 'previous', 'inventory', 'models'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--series', choices=['previous_retrain', 'control', 'quarter_lr'], required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    run(**vars(args))
