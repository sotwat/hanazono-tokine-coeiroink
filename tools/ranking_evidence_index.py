"""Index matched untreated caches; retain missing candidates explicitly."""
import argparse
import hashlib
import json
from pathlib import Path

from noise_spectral_detail import detail
from retrained_decoder_audit import TEXTS


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build(root, inventory, previous):
    inv = json.loads(inventory.read_text())
    expected = {(x['series'], x['epoch']): x['sha256'] for x in inv['rows']}
    evidence = {}

    def add(key, row, wav):
        model_sha = expected[key]
        metadata = json.loads(wav.with_suffix('.json').read_text())
        test = row['test']
        request = metadata['request']
        assert request['text'] == TEXTS[test] and request['speedScale'] == 1
        wav_sha = digest(wav)
        if 'model_sha256' in row:
            assert row['model_sha256'] == model_sha
        if 'wav_sha256' in row:
            assert row['wav_sha256'] == wav_sha
        item = {'series': key[0], 'epoch': key[1], 'test': test,
                'model_sha256': model_sha, 'wav_sha256': wav_sha, 'wav': str(wav)}
        for k in ('seconds', 'rms_dbfs', 'peak_dbfs', 'pcm_clip_count', 'voiced_frames',
                  'vowel_hf8_rel_db', 'vowel_hf12_rel_db', 'hf_flatness',
                  'comb_mean_db', 'comb_lines_over_3db', 'comb_line_scores_db'):
            item[k] = row[k]
        assert item['pcm_clip_count'] == 0 and item['voiced_frames'] >= 3
        cache_key = (model_sha, test)
        if cache_key in evidence:
            assert evidence[cache_key]['wav_sha256'] == wav_sha, 'Conflicting identical-model cache'
        else:
            evidence[cache_key] = item

    original = root / 'ranking_original'
    for row in json.loads((original / 'results.json').read_text()):
        add(('original', row['epoch']), row,
            original / 'raw' / f"{row['epoch']:03}_{row['test']}.wav")
    assert digest(previous / 'models/180_original.pth') == expected['original', 180]
    assert digest(previous / 'models/200_retrained.pth') == expected['previous_retrain', 200]
    for name in ('pilot_comparison', 'control200_comparison', 'quarter200_comparison'):
        folder = root / name
        manifest = {m['name']: m for m in json.loads((folder / 'manifest.json').read_text())}
        for row in json.loads((folder / 'results.json').read_text()):
            name = row['name']
            if name == '180_original':
                key = ('original', 180)
            elif name == '200_retrained':
                key = ('previous_retrain', 200)
            else:
                series, epoch = name.rsplit('_', 1)
                key = (series, int(epoch))
                assert manifest[name]['sha256'] == expected[key]
            add(key, row, folder / 'raw' / name / (row['test'] + '.wav'))
    folder = root / 'intermediate185'
    assert digest(folder / '185epoch_unmodified.pth') == expected['previous_retrain', 185]
    for row in json.loads((folder / 'results.json').read_text()):
        add(('previous_retrain', 185), row, folder / 'raw' / (row['test'] + '.wav'))
    for name in ('ranking_previous', 'ranking_control', 'ranking_quarter'):
        folder = root / name
        if not (folder / 'results.json').exists():
            continue
        for row in json.loads((folder / 'results.json').read_text()):
            add((row['series'], row['epoch']), row,
                folder / 'raw' / f"{row['series']}_{row['epoch']}" / (row['test'] + '.wav'))
    out = root / 'ranking_evidence'
    out.mkdir(exist_ok=True)
    alias_report = json.loads((out / 'weight_aliases.json').read_text())
    canonical = {}
    for alias in alias_report['aliases']:
        assert alias['all_tensors_exactly_equal'] and alias['tensor_count'] == 1049
        assert expected[alias['series'], alias['epoch']] == alias['sha256']
        canonical[alias['sha256']] = alias_report['reference_sha256']
    assert alias_report['reference_sha256'] == expected['original', 180]
    missing = []
    for candidate in inv['rows']:
        model_sha = canonical.get(candidate['sha256'], candidate['sha256'])
        absent = [t for t in TEXTS if (model_sha, t) not in evidence]
        if absent:
            missing.append({'series': candidate['series'], 'epoch': candidate['epoch'],
                            'model_sha256': candidate['sha256'], 'missing_texts': absent})
    index = {'inventory_sha256': digest(inventory), 'texts': TEXTS,
             'complete': not missing, 'missing': missing, 'canonical_weight_sha': canonical,
             'weight_alias_audit_sha256': digest(out / 'weight_aliases.json'), 'rows': list(evidence.values())}
    (out / 'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2, allow_nan=False))
    # Full peak lists allow the predeclared 3/4/5 dB sensitivity analysis.
    spectral = []
    for i, row in enumerate(evidence.values()):
        wav = Path(row['wav'])
        cache = out / 'spectral' / row['model_sha256'] / (row['test'] + '.json')
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists() and json.loads(cache.read_text()).get('schema') == 2:
            value = json.loads(cache.read_text())
            assert value['wav_sha256'] == row['wav_sha256']
        else:
            durations = json.loads(wav.with_suffix('.json').read_text())['moraDurations']
            value = detail(wav, durations, min_prominence=3, peak_limit=None)
            value = {k: v for k, v in value.items() if k not in ('frequencies', 'mean_db', 'median_db')}
            assert abs(value['comb_mean_db'] - row['comb_mean_db']) < 1e-10
            value.update(schema=2,model_sha256=row['model_sha256'], wav_sha256=row['wav_sha256'], test=row['test'])
            cache.write_text(json.dumps(value, indent=2, allow_nan=False))
        spectral.append(value)
        if (i + 1) % 80 == 0:
            print('spectral', i + 1, flush=True)
    (out / 'spectral.json').write_text(json.dumps(spectral, indent=2, allow_nan=False))
    print('INDEX', len(evidence), 'utterances;', len(missing), 'candidate entries incomplete', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root', 'inventory', 'previous'):
        p.add_argument('--' + name, required=True, type=Path)
    build(**vars(p.parse_args()))
