"""Aggregate matched artifact proxies without claiming perceptual noise quality."""
import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    assert values and all(math.isfinite(x) for x in values)
    return dict(mean=statistics.mean(values), worst=max(values),
                std=statistics.pstdev(values), minimum=min(values))


def measurements(rows, spectra, prominence, presence, grid):
    peaks = []
    persistent = []
    for row in rows:
        detail = spectra[row['model_sha256'], row['test']]
        local = [p for p in detail['peaks']
                 if p['frame_presence_over_6db_exact'] >= presence]
        persistent.append(max((p['prominence_db_exact'] for p in local), default=0))
        peaks.extend((p['hz'], row['test']) for p in local
                     if p['prominence_db_exact'] >= prominence)
    clusters = []
    for hz, test in sorted(peaks):
        if not clusters or hz - clusters[-1]['min_hz'] > 44100 / 2048:
            clusters.append(dict(min_hz=hz, max_hz=hz, texts=[]))
        cluster = clusters[-1]
        cluster['max_hz'] = hz
        cluster['texts'] = sorted(set(cluster['texts']) | {test})
    fixed = [c for c in clusters if len(c['texts']) >= 4]
    lines = stats([sum(v > grid for v in r['comb_line_scores_db']) for r in rows])
    comb = stats([r['comb_mean_db'] for r in rows])
    local = stats(persistent)
    return dict(fixed_frequency_clusters=fixed, periodic_lines=lines,
                periodic_prominence_db=comb, persistent_peak_db=local,
                order=[len(fixed), lines['worst'], lines['mean'], comb['worst'],
                       comb['mean'], local['worst'], local['mean']])


def ordered_groups(items, stage=0):
    if stage == 7:
        return [sorted(items, key=lambda x: x['model_sha256'])]
    items = sorted(items, key=lambda x: (x['order'][stage], x['model_sha256']))
    buckets = []
    tolerance = 0 if stage == 0 else .01
    for item in items:
        if not buckets or item['order'][stage] - buckets[-1][0]['order'][stage] > tolerance:
            buckets.append([])
        buckets[-1].append(item)
    return [group for bucket in buckets for group in ordered_groups(bucket, stage + 1)]


def ranks(items):
    result = {}
    position = 1
    for group in ordered_groups(items):
        for item in group:
            result[item['model_sha256']] = position
        position += len(group)
    return result


def build(root):
    evidence = root / 'ranking_evidence'
    index_path = evidence / 'index.json'
    spectral_path = evidence / 'spectral.json'
    validity_path = evidence / 'validity/summary.json'
    index = json.loads(index_path.read_text())
    spectra = {(x['model_sha256'], x['test']): x
               for x in json.loads(spectral_path.read_text())}
    validity = json.loads(validity_path.read_text())
    valid = {(x['model_sha256'], x['test']): x for x in validity['rows']}
    groups = defaultdict(list)
    for row in index['rows']:
        groups[row['model_sha256']].append(row)
    expected = set(index['texts'])
    output = []
    scenarios = {}
    for thresholds in itertools.product((3, 4, 5), (.25, .5, .75), (2, 3, 4)):
        scored = []
        for sha, rows in groups.items():
            assert len(rows) == 8 and {r['test'] for r in rows} == expected
            for row in rows:
                detail = spectra[sha, row['test']]
                v = valid[sha, row['test']]
                assert detail['wav_sha256'] == v['wav_sha256'] == row['wav_sha256']
                assert detail['schema'] == 2
            scored.append(dict(model_sha256=sha, **measurements(rows, spectra, *thresholds)))
        scenario = '_'.join(map(str, thresholds))
        scenarios[scenario] = ranks(scored)
        if thresholds == (4, .5, 3):
            output = scored
    base_ranks = scenarios['4_0.5_3']
    for item in output:
        sha = item['model_sha256']
        rows = groups[sha]
        vrows = [valid[sha, r['test']] for r in rows]
        flags = []
        for row in vrows:
            if row['pcm_clip_count'] or row['voiced_frames'] < 3:
                flags.append(row['test'] + ':clip_or_insufficient_voicing')
            if row['rms_below_180_by_more_than_6db']:
                flags.append(row['test'] + ':rms_below_baseline_6db')
            if not row.get('transcript', '').strip():
                flags.append(row['test'] + ':missing_asr')
            if not .5 <= row['duration_ratio_vs_180'] <= 2:
                flags.append(row['test'] + ':duration_ratio_outside_0.5_2_screen')
        item.update(series=rows[0]['series'], epoch=rows[0]['epoch'],
                    provisional_rank=base_ranks[sha], validity_flags=flags,
                    lufs=stats([r['loudness']['input_i'] for r in vrows]),
                    rms_dbfs=stats([r['rms_dbfs'] for r in rows]),
                    hf8_relative_db=stats([r['vowel_hf8_rel_db'] for r in rows]),
                    pcm_clip_count=sum(r['pcm_clip_count'] for r in rows),
                    sensitivity_ranks={k: v[sha] for k, v in scenarios.items()},
                    top_five_scenarios=sum(v[sha] <= 5 for v in scenarios.values()),
                    content_reviewed=False, listening_verified=False,
                    per_text=vrows)
    output.sort(key=lambda x: (x['provisional_rank'], x['model_sha256']))
    result = dict(inventory_complete=index['complete'], missing=index['missing'],
                  measurement_coverage_complete=index['complete'],
                  proxy_ranking_complete=index['complete'] and validity['asr_complete'],
                  perceptual_quality_verified=False,
                  global_top_five_confirmed=False, models=len(output),
                  inputs={str(p): digest(p) for p in (index_path, spectral_path, validity_path)},
                  interpretation='Fixed/persistent/periodic artifact proxies only; not SNR or overall quality.',
                  rows=output)
    target = evidence / 'aggregate_provisional.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    with (evidence / 'aggregate_provisional.csv').open('w', newline='') as f:
        fields = ['rank', 'series', 'epoch', 'model_sha256', 'fixed_clusters',
                  'grid_worst', 'grid_mean', 'comb_worst_db', 'comb_mean_db',
                  'persistent_worst_db', 'persistent_mean_db', 'lufs_mean',
                  'rms_mean_dbfs', 'hf8_mean_db', 'sensitivity_best_rank',
                  'sensitivity_worst_rank', 'flags']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in output:
            writer.writerow(dict(zip(fields, [r['provisional_rank'], r['series'], r['epoch'],
                r['model_sha256'], *r['order'], r['lufs']['mean'], r['rms_dbfs']['mean'],
                r['hf8_relative_db']['mean'], min(r['sensitivity_ranks'].values()),
                max(r['sensitivity_ranks'].values()), '|'.join(r['validity_flags'])])))
    return result


def listening_files(root, result):
    import numpy as np
    from scipy.io import wavfile
    from retrain_noise_compare import loudness

    selected = [next(r for r in result['rows'] if r['series'] == 'original' and r['epoch'] == 180)]
    selected += [r for r in result['rows'] if r['provisional_rank'] <= 5
                 and r['model_sha256'] != selected[0]['model_sha256']]
    assert not any(r['validity_flags'] for r in selected), 'Review flagged candidates first'
    folder = root / 'ranking_evidence/listening_provisional'
    folder.mkdir(exist_ok=True)
    order = [f"{r['series']}_{r['epoch']}" for r in selected]
    levels = []
    tests = {r['test'] for r in selected[0]['per_text']}
    for test in sorted(tests):
        rows = [next(v for v in r['per_text'] if v['test'] == test) for r in selected]
        target = min(-23., *(-1.1 - r['loudness']['input_tp'] + r['loudness']['input_i'] for r in rows))
        chunks = []
        for name, row in zip(order, rows):
            source = Path(row['wav'])
            assert digest(source) == row['wav_sha256']
            rate, original = wavfile.read(source)
            assert rate == 44100 and original.dtype == np.int16 and original.ndim == 1
            output = folder / f'{test}_{name}.wav'
            gain = target - row['loudness']['input_i']
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(source),
                            '-af', f'volume={gain}dB', '-c:a', 'pcm_s16le', str(output)], check=True)
            rate2, pcm = wavfile.read(output)
            measured = loudness(output)
            assert rate2 == rate and pcm.dtype == original.dtype and pcm.shape == original.shape
            assert np.max(np.abs(pcm.astype(np.int32))) < 32767
            assert abs(float(measured['input_i']) - target) < .12 and float(measured['input_tp']) < -1
            if chunks:
                chunks.append(np.zeros(44100, dtype=np.int16))
            chunks.append(pcm)
            levels.append(dict(test=test, name=name, gain_db=gain, target_lufs=target,
                               measured=measured, source_sha256=row['wav_sha256'],
                               output_sha256=digest(output), samples=len(pcm)))
        joined = np.concatenate(chunks)
        path = folder / f'{test}_180_then_provisional_top5.wav'
        wavfile.write(path, 44100, joined)
        assert np.array_equal(wavfile.read(path)[1], joined)
    (folder / 'manifest.json').write_text(json.dumps(dict(order=order, levels=levels,
        inventory_complete=result['inventory_complete'], global_top_five_confirmed=False,
        listening_verified=False, processing='linear gain only; one-second gaps; unchanged utterance lengths'),
        ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--make-listening', action='store_true')
    args = parser.parse_args()
    result = build(args.root)
    if args.make_listening:
        listening_files(args.root, result)
    print(json.dumps({k: result[k] for k in ('models', 'inventory_complete', 'global_top_five_confirmed')}))
