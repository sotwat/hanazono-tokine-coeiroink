"""Back up and replace only Tokine style 1200, with its matching phoneme config."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.error
import urllib.request

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def install(root, speaker, config):
    try:
        urllib.request.urlopen('http://127.0.0.1:50032/docs', timeout=2)
    except urllib.error.URLError:
        pass
    else:
        raise RuntimeError('Close the installed COEIROINK before replacing style 1200.')
    source = root / 'final/200epoch_retrained_repaired.pth'
    manifest = json.loads(source.with_suffix('.pth.json').read_text())
    assert digest(source) == manifest['output_sha256']
    backup = root / 'app_backup'
    assert not backup.exists(), 'An existing backup must not be overwritten.'
    backup.mkdir()
    shutil.copy2(speaker / 'metas.json', backup / 'metas.json')
    shutil.copytree(speaker / 'model/1200', backup / 'model1200')
    samples = backup / 'voice_samples'
    samples.mkdir()
    before = []
    for p in sorted((speaker / 'model/1200').iterdir()):
        assert digest(p) == digest(backup / 'model1200' / p.name)
        before.append({'path': str(p), 'sha256': digest(p)})
    for i in (1,2,3):
        shutil.copy2(speaker / f'voice_samples/1200_{i:03}.wav', samples)
    model_target = speaker / 'model/1200/200epoch.pth'
    config_target = speaker / 'model/1200/config.yaml'
    for src, target in [(source, model_target), (config, config_target)]:
        temporary = target.with_suffix(target.suffix + '.new')
        shutil.copy2(src, temporary)
        assert digest(src) == digest(temporary)
        os.replace(temporary, target)
    for i, text in enumerate(['user','sibilants','current'], 1):
        shutil.copy2(root / 'raw/r200_balance0.5' / (text + '.wav'), speaker / f'voice_samples/1200_{i:03}.wav')
    meta = json.loads((speaker / 'metas.json').read_text())
    style = next(x for x in meta['styles'] if x['styleId'] == 1200)
    style['styleName'] = '200epoch（再学習修正）'
    temporary = speaker / 'metas.json.new'
    temporary.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, speaker / 'metas.json')
    report = {'backup': str(backup), 'before': before, 'after_model_sha256': digest(model_target),
              'after_config_sha256': digest(config_target), 'style': style,
              'speaker_uuid': meta['speakerUuid'], 'originals_preserved': True}
    (root / 'installation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('speaker', type=Path)
    p.add_argument('config', type=Path)
    a = p.parse_args()
    install(a.root.resolve(), a.speaker.resolve(), a.config.resolve())
