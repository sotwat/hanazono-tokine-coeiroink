"""Survey saved original models using untreated, matched ordinary sentences."""
import argparse
import base64
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import time
import urllib.request

from noise_audit import analyze, post, TEXTS
from noise_spectral_detail import detail

NORMAL_TEXTS = {k: TEXTS[k] for k in ('user', 'travel')}
NORMAL_TEXTS.update(sibilants='静かな朝、涼しい風が吹いています。写真を撮って、一緒に出発しましょう。',
                    current='でも、このまま博士がくれた曲を歌うだけじゃ、私の気持ちを伝えきれないと思って')


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def survey(root, previous, epochs=None, texts=('user', 'travel')):
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'survey.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inventory = json.loads((previous / 'inventory.json').read_text())
        epochs = inventory['epochs'] if epochs is None else epochs
        assert epochs and len(set(epochs)) == len(epochs) and set(epochs) <= set(inventory['epochs'])
        assert texts and len(set(texts)) == len(texts) and set(texts) <= set(NORMAL_TEXTS)
        expected = {x['epoch']: x for x in json.loads((previous / 'model_hashes.json').read_text())}
        sock = socket.socket()
        try:
            assert sock.connect_ex(('127.0.0.1', 50033)) != 0, 'Preserve existing audit engine'
        finally:
            sock.close()
        with (root / 'engine.log').open('a') as log:
            engine = subprocess.Popen([str(previous / 'runtime/engine/engine')],
                                      cwd=previous / 'runtime', stdout=log, stderr=subprocess.STDOUT)
            try:
                for _ in range(120):
                    assert engine.poll() is None, 'Audit engine exited'
                    try:
                        urllib.request.urlopen('http://127.0.0.1:50033/v1/speakers', timeout=2).read()
                        break
                    except Exception:
                        time.sleep(2)
                else:
                    raise TimeoutError('Audit engine startup')
                rows = []
                for epoch in epochs:
                    model = previous / f'runtime/speaker_info/audit/model/{500000 + epoch}/{epoch}epoch.pth'
                    assert model.stat().st_size == expected[epoch]['bytes']
                    assert sha(model) == expected[epoch]['sha256'], epoch
                    for test in texts:
                        request = {'speakerUuid': inventory['uuid'], 'styleId': 500000 + epoch,
                                   'text': NORMAL_TEXTS[test], 'speedScale': 1}
                        folder = root / 'raw'
                        folder.mkdir(exist_ok=True)
                        wav = folder / f'{epoch:03}_{test}.wav'
                        meta = wav.with_suffix('.json')
                        if not wav.exists() and not meta.exists():
                            source = previous / 'original' / wav.name
                            source_meta = source.with_suffix('.json')
                            if source.exists() and source_meta.exists():
                                assert json.loads(source_meta.read_text())['request'] == request
                                shutil.copy2(source, wav)
                                shutil.copy2(source_meta, meta)
                            else:
                                response = json.loads(post(50033, 'predict_with_duration', request))
                                wav.write_bytes(base64.b64decode(response.pop('wavBase64')))
                                response['request'] = request
                                meta.write_text(json.dumps(response, ensure_ascii=False, indent=2))
                        assert wav.exists() and meta.exists(), 'Incomplete existing output: inspect before resuming'
                        metadata = json.loads(meta.read_text())
                        assert metadata['request'] == request
                        durations = metadata['moraDurations']
                        spectrum = detail(wav, durations)
                        spectrum = {k: v for k, v in spectrum.items() if k not in ('frequencies', 'mean_db', 'median_db')}
                        metrics = analyze(wav, durations)
                        metrics = {k: v for k, v in metrics.items() if k not in ('hf_psd_db', 'hf_freq_hz')}
                        row = {'epoch': epoch, 'test': test, 'model_sha256': expected[epoch]['sha256'],
                               'wav_sha256': sha(wav), **metrics, **spectrum}
                        assert row['voiced_frames'] >= 3 and row['pcm_clip_count'] == 0
                        rows.append(row)
                        (root / 'results.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
                        print(epoch, test, 'periodic', round(row['comb_mean_db'], 3),
                              'lines>3dB', row['comb_lines_over_3db'], flush=True)
                assert len(rows) == len(texts) * len(epochs)
                (root / 'complete.json').write_text(json.dumps({'models': len(epochs),
                    'ordinary_texts': list(texts), 'waveforms': len(rows),
                    'missing_epochs': inventory['missing'], 'listening_verified': False,
                    'noise_onset_epoch': None}, indent=2))
                print('SURVEY_COMPLETE_ONSET_REVIEW_REQUIRED', flush=True)
            finally:
                engine.terminate()
                try:
                    engine.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    engine.kill()
                    engine.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--epochs', type=int, nargs='+')
    parser.add_argument('--texts', choices=list(NORMAL_TEXTS), nargs='+', default=['user', 'travel'])
    args = parser.parse_args()
    survey(args.root.resolve(), args.previous.resolve(), args.epochs, args.texts)
