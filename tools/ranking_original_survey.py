"""Collect matched ordinary-eight evidence for saved untreated original models."""
import argparse
import json
from pathlib import Path
import shutil

import noise_onset_audit as onset
from retrained_decoder_audit import TEXTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    root, previous, cache = (p.resolve() for p in (args.root, args.previous, args.cache))
    inventory = json.loads((previous / 'inventory.json').read_text())
    epochs = [e for e in inventory['epochs'] if 100 <= e <= 200]
    assert len(TEXTS) == 8
    assert all(TEXTS[k] == v for k, v in onset.NORMAL_TEXTS.items())
    hashes = {r['epoch']: r for r in json.loads((previous / 'model_hashes.json').read_text())}
    cached = {(r['epoch'], r['test']): r for r in json.loads((cache / 'results.json').read_text())}
    raw = root / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    # Existing source cache is tied to model hashes revalidated by survey().
    for epoch in epochs:
        for test, text in TEXTS.items():
            source = cache / 'raw' / f'{epoch:03}_{test}.wav'
            metadata = source.with_suffix('.json')
            if not source.exists() or not metadata.exists():
                continue
            evidence = cached[epoch, test]
            assert evidence['model_sha256'] == hashes[epoch]['sha256']
            assert evidence['wav_sha256'] == onset.sha(source)
            request = {'speakerUuid': inventory['uuid'], 'styleId': 500000 + epoch,
                       'text': text, 'speedScale': 1}
            assert json.loads(metadata.read_text())['request'] == request
            for src in (source, metadata):
                target = raw / src.name
                if target.exists():
                    assert onset.sha(target) == onset.sha(src), target
                else:
                    shutil.copy2(src, target)
    provenance = {'texts': TEXTS, 'speedScale': 1,
                  'engine_sha256': onset.sha(previous / 'runtime/engine/engine'),
                  'source_cache': str(cache), 'epochs': epochs,
                  'protocol_sha256': onset.sha(Path(__file__).resolve().parents[1] /
                      'docs/controlled-retrain-20261023/ranking-protocol.md')}
    (root / 'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
    onset.NORMAL_TEXTS = dict(TEXTS)
    onset.survey(root, previous, epochs=epochs, texts=tuple(TEXTS))


if __name__ == '__main__':
    main()
