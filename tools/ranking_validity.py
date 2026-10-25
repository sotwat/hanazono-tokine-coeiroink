"""Collect loudness and local ASR screens without changing ranked audio."""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import subprocess

from retrain_noise_compare import loudness


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def run(root, asr_python, asr_script):
    out = root / 'ranking_evidence/validity'
    out.mkdir(exist_ok=True)
    with (out / 'job.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        index = json.loads((root / 'ranking_evidence/index.json').read_text())
        rows = index['rows']
        by_hash = {}
        for row in rows:
            assert sha(Path(row['wav'])) == row['wav_sha256']
            by_hash.setdefault(row['wav_sha256'], []).append(row)
        cache_file = out / 'records.json'
        records = json.loads(cache_file.read_text()) if cache_file.exists() else {}
        # Existing comparison transcripts belong to these exact raw WAV hashes.
        for name in ('pilot_comparison', 'control200_comparison', 'quarter200_comparison'):
            folder = root / name
            for item in json.loads((folder / 'transcripts.json').read_text()):
                wav = Path(item['file']) if 'file' in item else folder / 'raw' / item['name'] / (item['test'] + '.wav')
                digest = sha(wav)
                if digest not in by_hash:
                    continue
                record = records.setdefault(digest, {})
                # Preserve the first existing transcript if independent ASR runs differ.
                if 'transcript' not in record:
                    record.update(transcript=item['transcript'], asr_source=str(folder / 'transcripts.json'),
                                  asr_source_sha256=sha(folder / 'transcripts.json'))
        for i, (digest, matches) in enumerate(by_hash.items()):
            record = records.setdefault(digest, {})
            if 'loudness' not in record:
                value = loudness(Path(matches[0]['wav']))
                fields = {key: float(value[key]) for key in ('input_i', 'input_tp', 'input_lra', 'input_thresh')}
                assert all(math.isfinite(v) for v in fields.values()), matches[0]['wav']
                record['loudness'] = fields
            if (i + 1) % 40 == 0:
                save(cache_file, records)
                print('LOUDNESS', i + 1, '/', len(by_hash), flush=True)
        save(cache_file, records)

        def summary():
            baseline = {r['test']: r for r in rows if r['series'] == 'original' and r['epoch'] == 180}
            assert set(baseline) == set(index['texts'])
            result = []
            for row in rows:
                item = dict(row)
                item.update(records[row['wav_sha256']])
                item['expected_text'] = index['texts'][row['test']]
                reference = baseline[row['test']]
                item['rms_delta_vs_180_db'] = row['rms_dbfs'] - reference['rms_dbfs']
                item['rms_below_180_by_more_than_6db'] = item['rms_delta_vs_180_db'] < -6
                item['duration_ratio_vs_180'] = row['seconds'] / reference['seconds']
                result.append(item)
            value = {'inventory_complete': index['complete'], 'models': len(rows) // 8,
                     'utterances': len(rows), 'loudness_complete': all('loudness' in r for r in result),
                     'asr_complete': all('transcript' in r for r in result),
                     'listening_verified': False, 'asr_is_content_screen_only': True, 'rows': result}
            save(out / 'summary.json', value)
            return value

        summary()
        pending = [Path(matches[0]['wav']) for digest, matches in by_hash.items()
                   if 'transcript' not in records[digest]]
        if asr_python and pending:
            assert asr_script.is_file()
            provenance = {'python': str(asr_python), 'script': str(asr_script),
                          'script_sha256': sha(asr_script), 'audio_processing': 'ASR resampling only; raw WAV unchanged',
                          'service_upload': False, 'wav_sha256': {str(p): sha(p) for p in pending}}
            save(out / 'asr_job.json', provenance)
            with (out / 'asr.stderr.log').open('a') as errors:
                child = subprocess.Popen([str(asr_python), '-u', str(asr_script), *map(str, pending)],
                                         stdout=subprocess.PIPE, stderr=errors, text=True)
                try:
                    for line in child.stdout:
                        if not line.strip():
                            continue
                        try:
                            value = json.loads(line)
                        except json.JSONDecodeError:
                            print('ASR non-JSON output ignored', flush=True)
                            continue
                        path = Path(value['file'])
                        expected = provenance['wav_sha256'][str(path)]
                        assert sha(path) == expected and isinstance(value['transcript'], str)
                        records[expected].update(transcript=value['transcript'], asr_source=str(asr_script),
                                                 asr_source_sha256=provenance['script_sha256'])
                        save(cache_file, records)
                        completed = sum('transcript' in records[h] for h in by_hash)
                        print('ASR', completed, '/', len(by_hash), flush=True)
                    assert child.wait() == 0, 'Local ASR failed; inspect asr.stderr.log and resume'
                finally:
                    if child.poll() is None:
                        child.terminate()
                        child.wait()
                    summary()
        value = summary()
        print('VALIDITY', value['utterances'], 'ASR complete:', value['asr_complete'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--asr-python', type=Path)
    parser.add_argument('--asr-script', type=Path)
    run(**vars(parser.parse_args()))
