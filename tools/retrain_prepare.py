"""Rebuild the original audio preprocessing and validate a recovered epoch-180 run."""
from pathlib import Path
import hashlib
import io
import json
import shutil
import tarfile
import urllib.request
import zipfile

import numpy as np
import pyopenjtalk
import soundfile as sf
import torch
import yaml

WORK = Path('/content/drive/MyDrive/MYCOEIROINK_WORK')
RUN = WORK / 'retrain_180_20261021'
LOCAL = Path('/content/tokine-retrain')
EXPECTED_CHECKPOINT = 'ab6925215af0ea111939313749c64bab66743eef90411d9f30165c901687b097'


def main():
    LOCAL.mkdir(exist_ok=True)
    checkpoint = RUN / 'checkpoint180_recovered.pth'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == EXPECTED_CHECKPOINT
    state = torch.load(checkpoint, map_location='cpu')
    assert state['reporter']['epoch'] == 180
    assert len(state['model']) == 1049
    assert [len(o['state']) for o in state['optimizers']] == [942, 106]
    assert all(s['last_epoch'] == 180 for s in state['schedulers'])
    print('CHECKPOINT180_OK', flush=True)

    dictionary = Path(pyopenjtalk.__file__).parent / 'open_jtalk_dic_utf_8-1.11'
    if not dictionary.exists():
        archive = LOCAL / 'dictionary.tar.gz'
        urllib.request.urlretrieve('https://github.com/r9y9/open_jtalk/releases/download/v1.11.1/open_jtalk_dic_utf_8-1.11.tar.gz', archive)
        with tarfile.open(archive) as t:
            assert all(not m.name.startswith('/') and '..' not in Path(m.name).parts for m in t.getmembers())
            t.extractall(dictionary.parent)

    archives = list(WORK.glob('*/花園トキネtalk_full_wavs.zip'))
    assert len(archives) == 1
    wavdir = LOCAL / 'wavs'
    wavdir.mkdir(exist_ok=True)
    audio = {}
    with zipfile.ZipFile(archives[0]) as z:
        members = [n for n in z.namelist() if n.lower().endswith('.wav') and not n.startswith('__MACOSX/')]
        assert len(members) == 652
        assert len({Path(n).stem for n in members}) == 652
        for n in members:
            wav, fs = sf.read(io.BytesIO(z.read(n)))
            assert fs == 44100 and wav.ndim == 1 and np.isfinite(wav).all()
            target = wavdir / Path(n).name
            target.write_bytes(z.read(n))
            audio[target.stem] = {'path': str(target), 'samples': len(wav)}
    print('ORIGINAL_652_UNCHANGED_OK', flush=True)

    corpus = {}
    for line in (WORK / 'marged-corpus.txt').read_text().splitlines():
        key, value = line.split(':', 1)
        corpus[key] = value
    stats = WORK / 'exp/tts_stats_raw_linear_spectrogram_phn_jaconv_pyopenjtalk_prosody'
    config = yaml.safe_load((RUN / 'config180.yaml').read_text())
    assert config['version'] == '202211'
    for split in ['train', 'valid']:
        dest = LOCAL / split
        shutil.copytree(stats / split, dest, dirs_exist_ok=True)
        historical_speech = dict(line.split() for line in (RUN / 'train_speech_shape180').read_text().splitlines())
        historical_text = dict(line.split() for line in (RUN / 'train_text_shape180').read_text().splitlines())
        shapes = [line.split() for line in (dest / 'speech_shape').read_text().splitlines()]
        ids = [parts[0] for parts in shapes]
        (dest / 'speech_shape').write_text(''.join(f'{key} {historical_speech[key]}\n' for key in ids))
        (dest / 'text_shape.phn').write_text(''.join(f'{key} {historical_text[key]}\n' for key in ids))
        shapes = [line.split() for line in (dest / 'speech_shape').read_text().splitlines()]
        assert all(key in audio and key in corpus for key in ids)
        differences = [(key, int(shape.split(',')[0]), audio[key]['samples']) for key, shape in shapes if int(shape.split(',')[0]) != audio[key]['samples']]
        # A changed waveform length means this is not the same preprocessing run.
        assert not differences, differences[:10]
        (dest / 'text').write_text(''.join(f'{key} {corpus[key]}\n' for key in ids))
        (dest / 'wav.scp').write_text(''.join(f'{key} {audio[key]["path"]}\n' for key in ids))
        config[f'{split}_data_path_and_name_and_type'] = [[str(dest / 'text'), 'text', 'text'], [str(dest / 'wav.scp'), 'speech', 'sound']]
        config[f'{split}_shape_file'] = [str(dest / 'text_shape.phn'), str(dest / 'speech_shape')]

    modeldir = RUN / 'model'
    modeldir.mkdir(exist_ok=True)
    if not (modeldir / 'checkpoint.pth').exists():
        shutil.copy2(checkpoint, modeldir / 'checkpoint.pth')
    config.pop('config', None)
    config.pop('version', None)
    config.pop('required', None)
    config.pop('distributed', None)
    config.update(output_dir=str(modeldir), max_epoch=185, resume=True, init_param=[], keep_nbest_models=10)
    (RUN / 'train.yaml').write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    (RUN / 'data_manifest.json').write_text(json.dumps({'source_zip_sha256': hashlib.sha256(archives[0].read_bytes()).hexdigest(), 'audio': audio}, ensure_ascii=False, indent=2))
    print('PREPARED_SAME_LENGTHS_AND_TOKEN_LIST', flush=True)


if __name__ == '__main__':
    main()
