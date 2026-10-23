"""Register untreated controlled-pilot weights in the isolated comparison engine."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import torch
import yaml

from retrained_decoder_audit import generate, TEXTS, UUID


def prepare(root, previous, bundle):
    root.mkdir(parents=True, exist_ok=True)
    speaker = previous / 'runtime/speaker_info/audit'
    meta_path = speaker / 'metas.json'
    meta = json.loads(meta_path.read_text())
    assert meta['speakerUuid'] == UUID
    base_config = speaker / 'model/950180/config.yaml'
    reference = torch.load(previous / 'models/180_original.pth', map_location='cpu', weights_only=True)
    manifest = [
        {'name': '180_original', 'styleId': 950180},
        {'name': '200_retrained', 'styleId': 950182},
    ]
    with zipfile.ZipFile(bundle) as archive:
        hashes = json.loads(archive.read('manifest.json'))
        for index, (arm, epoch) in enumerate((a, e) for a in ('control', 'quarter_lr') for e in (181, 182)):
            for rel in ('train.yaml', 'preflight.json', 'training_complete.json', f'model/{epoch}epoch.pth'):
                key = f'{arm}/{rel}'
                data = archive.read(key)
                assert hashlib.sha256(data).hexdigest() == hashes[key], key
            config = yaml.safe_load(archive.read(f'{arm}/train.yaml'))
            assert config['token_list'] == yaml.safe_load(base_config.read_text())['token_list']
            assert json.loads(archive.read(f'{arm}/training_complete.json'))['epoch'] == 182
            name = f'{arm}_{epoch}'
            path = root / 'models' / f'{name}.pth'
            path.parent.mkdir(exist_ok=True)
            data = archive.read(f'{arm}/model/{epoch}epoch.pth')
            if path.exists():
                assert path.read_bytes() == data, 'Preserve existing model'
            else:
                path.write_bytes(data)
            state = torch.load(path, map_location='cpu', weights_only=True)
            assert len(state) == 1049 and state.keys() == reference.keys()
            assert all(state[k].shape == reference[k].shape and torch.isfinite(state[k]).all() for k in state)
            sid = 970181 + index
            folder = speaker / 'model' / str(sid)
            folder.mkdir(exist_ok=True)
            target = folder / 'model.pth'
            if target.is_symlink():
                assert target.resolve() == path.resolve()
            else:
                assert not target.exists()
                target.symlink_to(path.resolve())
            shutil.copy2(base_config, folder / 'config.yaml')
            for directory, suffix in [('icons', '.png'), ('portraits', '.png'), ('voice_samples', '_001.wav'), ('voice_samples', '_002.wav'), ('voice_samples', '_003.wav')]:
                target = speaker / directory / f'{sid}{suffix}'
                source = (speaker / directory / f'950180{suffix}').resolve()
                if target.is_symlink():
                    assert target.resolve() == source
                else:
                    assert not target.exists()
                    target.symlink_to(source)
            existing = [s for s in meta['styles'] if s['styleId'] == sid]
            assert not existing or existing[0]['styleName'] == name
            if not existing:
                meta['styles'].append({'styleId': sid, 'styleName': name})
            manifest.append({'name': name, 'styleId': sid, 'sha256': hashlib.sha256(data).hexdigest(), 'arm': arm, 'epoch': epoch})
            print('VERIFIED', name, manifest[-1]['sha256'], flush=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'generate'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--bundle', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.action == 'prepare':
        prepare(args.root, args.previous, args.bundle)
    else:
        generate(args.root, list(TEXTS))
