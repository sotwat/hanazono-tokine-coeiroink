"""Inspect an intermediate full training checkpoint and compare untreated speech."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil

import torch

from controlled_retrain import load
from noise_audit import analyze, post
from noise_spectral_detail import detail
from retrained_decoder_audit import TEXTS, UUID


def prepare(args):
    state = load(args.checkpoint)
    assert state['reporter']['epoch'] == args.epoch
    assert len(state['model']) == 1049
    assert [len(o['state']) for o in state['optimizers']] == [942, 106]
    assert all(s['last_epoch'] == args.epoch for s in state['schedulers'])
    assert all(float(m['step']) == args.epoch * 1000 for o in state['optimizers'] for m in o['state'].values())
    assert all(torch.isfinite(v).all() for v in state['model'].values())
    args.root.mkdir(parents=True, exist_ok=True)
    model_path = args.root / f'{args.epoch}epoch_unmodified.pth'
    assert not model_path.exists(), 'Preserve an existing extracted model'
    torch.save(state['model'], model_path)
    speaker = args.runtime / 'speaker_info/audit'
    folder = speaker / 'model' / str(args.style_id)
    folder.mkdir(exist_ok=False)
    (folder / 'model.pth').symlink_to(model_path.resolve())
    shutil.copy2(args.config, folder / 'config.yaml')
    meta_path = speaker / 'metas.json'
    meta = json.loads(meta_path.read_text())
    assert meta['speakerUuid'] == UUID
    assert not any(s['styleId'] == args.style_id for s in meta['styles'])
    source_id = 950180
    for directory, suffix in [('icons','.png'),('portraits','.png'),('voice_samples','_001.wav'),('voice_samples','_002.wav'),('voice_samples','_003.wav')]:
        (speaker / directory / f'{args.style_id}{suffix}').symlink_to((speaker / directory / f'{source_id}{suffix}').resolve())
    meta['styles'].append({'styleId': args.style_id, 'styleName': f'retrained_{args.epoch}_unmodified'})
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    report = {'epoch': args.epoch, 'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
              'tensors': 1049, 'optimizer_states': [942,106], 'style_id': args.style_id}
    (args.root / 'checkpoint_verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def generate(args):
    folder = args.root / 'raw'
    folder.mkdir(exist_ok=True)
    rows = []
    for name, text in TEXTS.items():
        path = folder / (name + '.wav')
        metadata = path.with_suffix('.json')
        if not path.exists():
            request = {'speakerUuid': UUID, 'styleId': args.style_id, 'text': text, 'speedScale': 1}
            r = json.loads(post(50033, 'predict_with_duration', request))
            path.write_bytes(base64.b64decode(r.pop('wavBase64')))
            r['request'] = request
            metadata.write_text(json.dumps(r, ensure_ascii=False, indent=2))
        durations = json.loads(metadata.read_text())['moraDurations']
        row = {'epoch': args.epoch, 'test': name, **analyze(path, durations), **detail(path, durations)}
        rows.append(row)
        (args.root / 'results.json').write_text(json.dumps(rows, indent=2))
        print(name, round(row['comb_mean_db'],3), round(row['vowel_hf8_rel_db'],3), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'generate'])
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--epoch', type=int, required=True)
    p.add_argument('--style-id', type=int, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--runtime', type=Path)
    p.add_argument('--config', type=Path)
    args = p.parse_args()
    torch.set_num_threads(4)
    (prepare if args.action == 'prepare' else generate)(args)
