"""Summarize fixed-frequency artifact evidence, without equating HF energy to noise."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def split_summary(points):
    def error(values):
        mean = statistics.mean(values)
        return sum((value - mean) ** 2 for value in values)
    scores = []
    for index in range(12, len(points) - 11):
        before = [p['mean_periodic_db'] for p in points[:index]]
        after = [p['mean_periodic_db'] for p in points[index:]]
        scores.append({'last_before': points[index - 1]['epoch'], 'first_after': points[index]['epoch'],
                       'before_mean': statistics.mean(before), 'after_mean': statistics.mean(after),
                       'squared_error': error(before) + error(after)})
    return min(scores, key=lambda value: value['squared_error'])


def build(root):
    complete = json.loads((root / 'complete.json').read_text())
    rows = json.loads((root / 'results.json').read_text())
    assert len(rows) == complete['waveforms'] == 300
    assert len({(x['epoch'], x['test']) for x in rows}) == 300
    paired = []
    for epoch in sorted({x['epoch'] for x in rows}):
        group = [x for x in rows if x['epoch'] == epoch]
        assert {x['test'] for x in group} == {'user', 'travel'}
        shared = set(range(17))
        for row in group:
            shared &= {i for i, value in enumerate(row['comb_line_scores_db']) if value > 3}
        pair = {'epoch': epoch, 'mean_periodic_db': statistics.mean(x['comb_mean_db'] for x in group),
                'min_periodic_db': min(x['comb_mean_db'] for x in group),
                'shared_lines_over_3db': len(shared), 'shared_line_hz': [(i + 10) * 44100 / 64 for i in sorted(shared)]}
        for row in group:
            for key in ['comb_mean_db', 'comb_lines_over_3db', 'pause_hf8_rel_db', 'hf_flatness']:
                pair[row['test'] + '_' + key] = row[key]
        paired.append(pair)
    groups = []
    for low, high in [(1, 39), (40, 69), (70, 99), (100, 129), (130, 147), (160, 180), (181, 200)]:
        selected = [p for p in paired if low <= p['epoch'] <= high]
        groups.append({'range': [low, high], 'saved_epochs': len(selected),
                       'median_periodic_db': statistics.median(p['mean_periodic_db'] for p in selected),
                       'median_shared_lines_over_3db': statistics.median(p['shared_lines_over_3db'] for p in selected)})
    summary = {'models': 150, 'ordinary_waveforms': 300, 'before_or_at_180_models': sum(x['epoch'] <= 180 for x in paired),
               'missing_epochs': complete['missing_epochs'], 'groups': groups,
               'descriptive_split_40_200': split_summary([p for p in paired if p['epoch'] >= 40]),
               'descriptive_split_40_180': split_summary([p for p in paired if 40 <= p['epoch'] <= 180]),
               'pcm_clips': sum(x['pcm_clip_count'] for x in rows), 'listening_verified': False,
               'limitations': ['Periodic local prominence is an artifact screening metric, not noise SNR or overall quality.',
                              'The fitted split minimizes squared error and is not a proven onset or significance test.',
                              'Neighboring checkpoints are correlated; missing checkpoints are not interpolated.',
                              'Early undertrained voices can contain other defects; the test does not cover every kind of noise.',
                              'Original historical models are separate from the currently running controlled continuations.'],
               'epochs': paired}
    (root / 'onset_summary.json').write_text(json.dumps(summary, indent=2))
    with (root / 'periodic_noise_by_epoch.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(paired)
    print(json.dumps({k: v for k, v in summary.items() if k != 'epochs'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    build(parser.parse_args().root)
