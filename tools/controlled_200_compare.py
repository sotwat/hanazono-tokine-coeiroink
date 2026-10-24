"""Compare verified, untreated epoch-200 arms with cached ordinary baselines."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import time
import urllib.request
import zipfile

import numpy as np
from scipy.io import wavfile
import torch
import yaml

from retrained_decoder_audit import generate, TEXTS, UUID
from retrain_noise_compare import loudness


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare(root, previous, cache, bundle, external):
    assert bundle.stat().st_size == external['zip_size']
    assert sha(bundle.read_bytes()) == external['zip_sha256']
    root.mkdir(parents=True, exist_ok=True)
    speaker = previous / 'runtime/speaker_info/audit'
    meta_path = speaker / 'metas.json'
    meta = json.loads(meta_path.read_text())
    assert meta['speakerUuid'] == UUID
    config_path = speaker / 'model/950180/config.yaml'
    reference = torch.load(previous / 'models/180_original.pth', map_location='cpu', weights_only=True)
    manifest = [{'name': '180_original', 'styleId': 950180},
                {'name': '200_retrained', 'styleId': 950182}]
    for item in manifest:
        for test in TEXTS:
            for suffix in ('.wav', '.json'):
                src = cache / 'raw' / item['name'] / (test + suffix)
                dst = root / 'raw' / item['name'] / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                data = src.read_bytes()
                if suffix == '.json':
                    req = json.loads(data)['request']
                    assert req == {'speakerUuid': UUID, 'styleId': item['styleId'],
                                   'text': TEXTS[test], 'speedScale': 1}
                if dst.exists():
                    assert dst.read_bytes() == data
                else:
                    shutil.copy2(src, dst)
    with zipfile.ZipFile(bundle) as archive:
        hashes = json.loads(archive.read('manifest.json'))
        assert hashes == external['files']
        assert json.loads(archive.read('checkpoint_audit.json')) == external['audit']
        audit = external['audit']
        assert audit == {'epoch': 200, 'scheduler_epochs': [200, 200],
                         'optimizer_counts': [942, 106], 'optimizer_steps': 200000,
                         'model_tensors': 1049, 'model_matches_checkpoint': True,
                         'postprocessing': False}
        arm = next(iter(hashes)).split('/')[0]
        assert arm in ('control', 'quarter_lr')
        for name, digest in hashes.items():
            assert sha(archive.read(name)) == digest, name
        assert json.loads(archive.read(f'{arm}/training_complete.json'))['epoch'] == 200
        cfg = yaml.safe_load(archive.read(f'{arm}/train.yaml'))
        assert cfg['token_list'] == yaml.safe_load(config_path.read_text())['token_list']
        data = archive.read(f'{arm}/model/200epoch.pth')
        path = root / 'models' / f'{arm}_200.pth'
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            assert path.read_bytes() == data
        else:
            path.write_bytes(data)
        state = torch.load(path, map_location='cpu', weights_only=True)
        assert len(state) == 1049 and state.keys() == reference.keys()
        assert all(state[k].shape == reference[k].shape and torch.isfinite(state[k]).all() for k in state)
        sid = 980200 if arm == 'control' else 980201
        folder = speaker / 'model' / str(sid)
        folder.mkdir(exist_ok=True)
        target = folder / 'model.pth'
        if target.is_symlink():
            assert target.resolve() == path.resolve()
        else:
            assert not target.exists()
            target.symlink_to(path.resolve())
        shutil.copy2(config_path, folder / 'config.yaml')
        for directory, suffix in [('icons', '.png'), ('portraits', '.png'),
                                  ('voice_samples', '_001.wav'), ('voice_samples', '_002.wav'),
                                  ('voice_samples', '_003.wav')]:
            target = speaker / directory / f'{sid}{suffix}'
            src = (speaker / directory / f'950180{suffix}').resolve()
            if target.is_symlink():
                assert target.resolve() == src
            else:
                assert not target.exists()
                target.symlink_to(src)
        name = f'{arm}_200'
        existing = [s for s in meta['styles'] if s['styleId'] == sid]
        assert not existing or existing[0]['styleName'] == name
        if not existing:
            meta['styles'].append({'styleId': sid, 'styleName': name})
        manifest.append({'name': name, 'styleId': sid, 'sha256': sha(data)})
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (root / 'checkpoint_audit.json').write_text(json.dumps(external, indent=2))


def report(root):
    kinds = [m['name'] for m in json.loads((root / 'manifest.json').read_text())]
    rows = json.loads((root / 'results.json').read_text())
    assert len(rows) == 24 and len({(r['name'], r['test']) for r in rows}) == 24
    assert {(r['name'], r['test']) for r in rows} == {(k, t) for k in kinds for t in TEXTS}
    assert all(r['pcm_clip_count'] == 0 for r in rows)
    listening = root / 'listening'
    listening.mkdir(exist_ok=True)
    levels = []
    for test in TEXTS:
        sources = [root / 'raw' / k / f'{test}.wav' for k in kinds]
        before = [loudness(p) for p in sources]
        target = min(-23., *(-1.1 - float(x['input_tp']) + float(x['input_i']) for x in before))
        parts = []
        for name, src, measured in zip(kinds, sources, before):
            rate, original = wavfile.read(src)
            assert rate == 44100 and original.dtype == np.int16 and original.ndim == 1
            out = listening / f'{test}_{name}.wav'
            gain = target - float(measured['input_i'])
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(src), '-af',
                            f'volume={gain}dB', '-c:a', 'pcm_s16le', str(out)], check=True)
            rate2, pcm = wavfile.read(out)
            checked = loudness(out)
            assert rate2 == rate and pcm.shape == original.shape and pcm.dtype == original.dtype
            assert np.max(np.abs(pcm.astype(np.int32))) < 32767
            assert abs(float(checked['input_i']) - target) < .12 and float(checked['input_tp']) < -1
            levels.append({'name': name, 'test': test, 'gain_db': gain, 'target_lufs': target,
                           'lufs': float(checked['input_i']), 'true_peak_dbtp': float(checked['input_tp'])})
            parts.append(pcm)
        for label, indices in [('all_three', [0, 1, 2]), ('180_candidate200', [0, 2])]:
            selected = []
            for i in indices:
                if selected:
                    selected.append(np.zeros(44100, dtype=np.int16))
                selected.append(parts[i])
            joined = np.concatenate(selected)
            output = listening / f'{test}_{label}.wav'
            wavfile.write(output, 44100, joined)
            assert np.array_equal(wavfile.read(output)[1], joined)
    summary = {'order': kinds, 'tests': TEXTS, 'quality_improved': None, 'listening_verified': False,
               'processing': 'linear gain only; no denoising, EQ, stretching or weight correction',
               'averages': {k: {f: float(np.mean([r[f] for r in rows if r['name'] == k]))
                                for f in ('comb_mean_db', 'vowel_hf8_rel_db', 'rms_dbfs')}
                            for k in kinds}, 'levels': levels}
    (root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary['averages']), flush=True)


def run(args):
    torch.set_num_threads(4)
    with socket.socket() as s:
        assert s.connect_ex(('127.0.0.1', 50033)) != 0, 'Preserve an existing audit engine'
    prepare(args.root, args.previous, args.cache, args.bundle, json.loads(args.manifest.read_text()))
    with (args.root / 'engine.log').open('w') as log:
        engine = subprocess.Popen([str(args.previous / 'runtime/engine/engine')],
                                  cwd=args.previous / 'runtime', stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(120):
                assert engine.poll() is None, 'Audit engine exited'
                try:
                    urllib.request.urlopen('http://127.0.0.1:50033/v1/speakers', timeout=2).read()
                    break
                except (OSError, TimeoutError):
                    time.sleep(2)
            else:
                raise TimeoutError('Audit engine not ready')
            generate(args.root, list(TEXTS))
            report(args.root)
        finally:
            engine.terminate()
            try:
                engine.wait(timeout=30)
            except subprocess.TimeoutExpired:
                engine.kill()
                engine.wait()
    print('COMPARISON_COMPLETE_LISTENING_NOT_VERIFIED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'previous', 'cache', 'bundle', 'manifest'):
        parser.add_argument('--' + name, type=Path, required=True)
    run(parser.parse_args())
