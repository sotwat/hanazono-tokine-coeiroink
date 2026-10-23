"""Controlled decoder interventions on the verified retrained epoch-200 model."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from balance_upsampler_phases import balance
from noise_audit import analyze, post
from noise_spectral_detail import detail
from retrain_noise_compare import TEXTS, UUID

TEXTS = {**TEXTS,
    'library': 'おはようございます。今日は図書館で本を借りてから、友達と駅前で待ち合わせます。',
    'meeting': '明日の午後三時に、二階の会議室へ来てください。資料は二十五部用意しました。',
    'thanks': '本当？それは嬉しいね。ありがとう。また一緒に遊びに行こう。',
    'park': '夜の公園は静かで、遠くから虫の声が聞こえてきます。'}

DECODER = 'tts.generator.decoder.'
UP1 = DECODER + 'upsamples.1.'

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare(root, previous):
    root.mkdir(parents=True, exist_ok=True)
    out = root / 'models'
    out.mkdir(exist_ok=True)
    a = torch.load(previous / 'models/180_original.pth', map_location='cpu', weights_only=True)
    b = torch.load(previous / 'models/200_retrained.pth', map_location='cpu', weights_only=True)
    stage = lambda k: k.startswith(UP1) or any(k.startswith(DECODER + f'blocks.{j}.') for j in (3,4,5))
    variants = {
        'r200_decoder180': (b, a, lambda k: k.startswith(DECODER)),
        'e180_decoder200': (a, b, lambda k: k.startswith(DECODER)),
        'r200_up1_180': (b, a, lambda k: k.startswith(UP1)),
        'e180_up1_200': (a, b, lambda k: k.startswith(UP1)),
        'r200_stage1_180': (b, a, stage),
    }
    speaker = previous / 'runtime/speaker_info/audit'
    meta = json.loads((speaker / 'metas.json').read_text())
    manifest = [
        {'name': '180_original', 'styleId': 950180, 'path': str(previous / 'models/180_original.pth')},
        {'name': '200_retrained', 'styleId': 950182, 'path': str(previous / 'models/200_retrained.pth')},
    ]
    for name, (base, donor, select) in variants.items():
        state = {k: donor[k] if select(k) else base[k] for k in base}
        output = out / (name + '.pth')
        if not output.exists():
            torch.save(state, output)
        restored = torch.load(output, map_location='cpu', weights_only=True)
        assert all(torch.equal(state[k], restored[k]) for k in state)
        manifest.append({'name': name, 'path': str(output), 'changed_keys': [k for k in base if not torch.equal(base[k], restored[k])]})
    for amount in (.25, .5, 1.):
        state, info = balance(b, amount)
        name = f'r200_balance{amount:g}'
        output = out / (name + '.pth')
        if not output.exists():
            torch.save(state, output)
        manifest.append({'name': name, 'path': str(output), **info})
    for component in ['text_encoder', 'flow', 'duration_predictor']:
        prefix = 'tts.generator.' + component + '.'
        name = 'r200_' + component + '180'
        state = {k: a[k] if k.startswith(prefix) else b[k] for k in b}
        output = out / (name + '.pth')
        if not output.exists():
            torch.save(state, output)
        manifest.append({'name': name, 'path': str(output), 'changed_keys': [k for k in b if k.startswith(prefix)]})
    for i, item in enumerate(manifest):
        path = Path(item['path']).resolve()
        item['path'] = str(path)
        item['sha256'] = digest(path)
        if i < 2:
            continue
        sid = 960200 + i
        item['styleId'] = sid
        folder = speaker / 'model' / str(sid)
        folder.mkdir(exist_ok=True)
        if not (folder / 'model.pth').exists():
            (folder / 'model.pth').symlink_to(path)
        shutil.copy2(speaker / 'model/950180/config.yaml', folder / 'config.yaml')
        for directory, suffix in [('icons','.png'),('portraits','.png'),('voice_samples','_001.wav'),('voice_samples','_002.wav'),('voice_samples','_003.wav')]:
            target = speaker / directory / f'{sid}{suffix}'
            if not target.exists():
                target.symlink_to((speaker / directory / f'950180{suffix}').resolve())
        meta['styles'] = [s for s in meta['styles'] if s['styleId'] != sid] + [{'styleId': sid, 'styleName': item['name']}]
    (speaker / 'metas.json').write_text(json.dumps(meta))
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Prepared', len(manifest), 'models', flush=True)

def generate(root, texts, names=None, result_name='results.json'):
    rows = []
    for item in json.loads((root / 'manifest.json').read_text()):
        if names and item['name'] not in names:
            continue
        folder = root / 'raw' / item['name']
        folder.mkdir(parents=True, exist_ok=True)
        for name in texts:
            wav = folder / (name + '.wav')
            metadata = wav.with_suffix('.json')
            req = {'speakerUuid': UUID, 'styleId': item['styleId'], 'text': TEXTS[name], 'speedScale': 1}
            if not wav.exists():
                r = json.loads(post(50033, 'predict_with_duration', req))
                wav.write_bytes(base64.b64decode(r.pop('wavBase64')))
                r['request'] = req
                metadata.write_text(json.dumps(r, ensure_ascii=False, indent=2))
            durations = json.loads(metadata.read_text())['moraDurations']
            row = {'name': item['name'], 'test': name, **analyze(wav, durations), **detail(wav, durations)}
            rows.append(row)
            (root / result_name).write_text(json.dumps(rows, indent=2))
            print(item['name'], name, 'comb', round(row['comb_mean_db'], 3), 'HF8', round(row['vowel_hf8_rel_db'], 3), 'RMS', round(row['rms_dbfs'], 3), flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'generate'])
    p.add_argument('root', type=Path)
    p.add_argument('--previous', type=Path)
    p.add_argument('--texts', nargs='+', choices=list(TEXTS), default=['user','travel','sibilants','current'])
    p.add_argument('--names', nargs='+')
    p.add_argument('--result-name', default='results.json')
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.action == 'prepare':
        prepare(args.root, args.previous)
    else:
        generate(args.root, args.texts, args.names, args.result_name)
