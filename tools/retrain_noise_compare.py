"""Compare unmodified epoch 180, old 200 and retrained 200 with one engine."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import wave

import numpy as np
import torch
import yaml

from noise_audit import TEXTS, analyze, post
from noise_spectral_detail import detail

TEXTS = {**{k: v for k, v in TEXTS.items() if k != 'vowels'},
    'sibilants': '静かな朝、涼しい風が吹いています。写真を撮って、一緒に出発しましょう。',
    'current': 'でも、このまま博士がくれた曲を歌うだけじゃ、私の気持ちを伝えきれないと思って'}
KINDS = ['180_original', '200_old', '200_retrained']
UUID = '8cb3363b-0292-4d86-bdf5-52949a66ee01'


def prepare(root, base, retrained):
    root.mkdir(parents=True, exist_ok=True)
    models = root / 'models'
    models.mkdir(exist_ok=True)
    runtime = root / 'runtime'
    engine = runtime / 'engine'
    engine.mkdir(parents=True, exist_ok=True)
    oldengine = base / 'noise_20261021/runtime/engine'
    shutil.copy2(oldengine / 'engine', engine / 'engine')
    if not (engine / '_internal').exists():
        (engine / '_internal').symlink_to((oldengine / '_internal').resolve())
    (engine / 'engine_settings.json').write_text('{"device":"cpu"}')
    speaker = runtime / 'speaker_info/audit'
    speaker.mkdir(parents=True, exist_ok=True)
    oldspeaker = base / 'noise_20261021/runtime/speaker_info/audit'
    for name in ['portrait.png', 'policy.md', 'LICENSE.txt']:
        if not (speaker / name).exists():
            (speaker / name).symlink_to((oldspeaker / name).resolve())
    sources = [base / 'models/180epoch.pth', base / 'models/200epoch.pth', retrained]
    configs = [base / 'models/config.yaml', base / 'models/config_181-200.yaml', base / 'models/config.yaml']
    manifest = []
    styles = []
    for i, (kind, src, conf) in enumerate(zip(KINDS, sources, configs)):
        dst = models / (kind + '.pth')
        if not dst.exists():
            shutil.copy2(src, dst)
        state = torch.load(dst, map_location='cpu', weights_only=True)
        assert len(state) == 1049
        assert all(torch.isfinite(t).all() for t in state.values())
        token = yaml.safe_load(conf.read_text())['token_list']
        assert len(token) == 47
        sid = 950180 + i
        folder = speaker / 'model' / str(sid)
        folder.mkdir(parents=True, exist_ok=True)
        if not (folder / 'model.pth').exists():
            (folder / 'model.pth').symlink_to(dst.resolve())
        shutil.copy2(conf, folder / 'config.yaml')
        for directory, suffix in [('icons', '.png'), ('portraits', '.png'), ('voice_samples', '_001.wav'), ('voice_samples', '_002.wav'), ('voice_samples', '_003.wav')]:
            target = speaker / directory / f'{sid}{suffix}'
            target.parent.mkdir(exist_ok=True)
            if not target.exists():
                target.symlink_to((oldspeaker / directory / f'500100{suffix}').resolve())
        styles.append({'styleId': sid, 'styleName': kind})
        manifest.append({'kind': kind, 'source': str(src), 'sha256': hashlib.sha256(dst.read_bytes()).hexdigest(), 'tensors': len(state), 'token_list': token})
    assert manifest[0]['token_list'] == manifest[2]['token_list']
    (speaker / 'metas.json').write_text(json.dumps({'speakerName': 'Tokine comparison', 'speakerUuid': UUID, 'styles': styles, 'version': '1.0.0'}))
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))


def generate(root):
    rows = []
    for i, kind in enumerate(KINDS):
        folder = root / 'raw' / kind
        folder.mkdir(parents=True, exist_ok=True)
        for test, text in TEXTS.items():
            wav = folder / (test + '.wav')
            metadata = wav.with_suffix('.json')
            if not wav.exists():
                req = {'speakerUuid': UUID, 'styleId': 950180+i, 'text': text, 'speedScale': 1}
                r = json.loads(post(50033, 'predict_with_duration', req))
                wav.write_bytes(base64.b64decode(r.pop('wavBase64')))
                r['request'] = req
                metadata.write_text(json.dumps(r, ensure_ascii=False, indent=2))
            durations = json.loads(metadata.read_text())['moraDurations']
            row = {'kind': kind, 'test': test, **analyze(wav, durations), **detail(wav, durations)}
            rows.append(row)
            (root / 'results.json').write_text(json.dumps(rows, indent=2))
            print(kind, test, 'comb', round(row['comb_mean_db'], 3), flush=True)


def loudness(path):
    r = subprocess.run(['ffmpeg', '-hide_banner', '-i', str(path), '-af', 'loudnorm=print_format=json', '-f', 'null', '-'], capture_output=True, text=True, check=True)
    return json.loads(re.search(r'\{\s*"input_i"[\s\S]*?\}', r.stderr).group())


def listening(root):
    levels = []
    folder = root / 'listening'
    folder.mkdir(exist_ok=True)
    for test in TEXTS:
        parts = []
        for kind in KINDS:
            src = root / 'raw' / kind / (test + '.wav')
            dst = folder / f'{test}_{kind}.wav'
            before = loudness(src)
            gain = -23 - float(before['input_i'])
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(src), '-af', f'volume={gain}dB', '-c:a', 'pcm_s16le', str(dst)], check=True)
            after = loudness(dst)
            assert abs(float(after['input_i']) + 23) < .12
            assert float(after['input_tp']) < -1
            with wave.open(str(dst), 'rb') as w:
                assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 44100)
                data = w.readframes(w.getnframes())
                assert np.max(np.abs(np.frombuffer(data, dtype='<i2').astype(int))) < 32767
                parts.append(data)
            levels.append({'kind': kind, 'test': test, 'linear_gain_db': gain, 'before': before, 'after': after})
        with wave.open(str(folder / f'{test}_180_old200_retrained200.wav'), 'wb') as w:
            w.setparams((1, 2, 44100, 0, 'NONE', 'not compressed'))
            w.writeframes((b'\0' * 88200).join(parts))
    (root / 'listening_levels.json').write_text(json.dumps(levels, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'generate', 'listening'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--base', type=Path)
    p.add_argument('--retrained', type=Path)
    args = p.parse_args()
    if args.action == 'prepare':
        prepare(args.root, args.base, args.retrained)
    elif args.action == 'generate':
        generate(args.root)
    else:
        listening(args.root)
