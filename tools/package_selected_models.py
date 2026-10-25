"""Package the two listener-selected original weights without inference corrections."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def build(output, destination, installed):
    evidence = output / 'controlled_retrain_20261023/ranking_evidence/aggregate_provisional.json'
    ranking = json.loads(evidence.read_text())
    assert ranking['measurement_coverage_complete'] and not ranking['missing']
    selected = [next(r for r in ranking['rows'] if r['series'] == 'original' and r['epoch'] == e) for e in (131, 180)]
    assert [(r['series'], r['epoch']) for r in selected] == [('original', e) for e in (131, 180)]
    assert not destination.exists(), 'Preserve existing packages'
    speaker = destination / 'hanazono-tokine-2.3.0'
    speaker.mkdir(parents=True)
    meta = json.loads((installed / 'metas.json').read_text())
    meta.update(version='2.3.0', styles=[])
    for name in ('portrait.png', 'policy.md', 'LICENSE.txt'):
        shutil.copy2(installed / name, speaker / name)
    report = {'ranking_sha256': sha(evidence), 'postprocessing': False,
              'weight_correction': False, 'selection_basis': 'user listening preference for original 131 and 180; no preference order', 'models': []}
    for rank, row in enumerate(selected, 1):
        epoch = row['epoch']
        sid = 1000 + epoch
        model = output / f'models/{epoch}epoch.pth'
        assert sha(model) == row['model_sha256']
        folder = speaker / f'model/{sid}'
        folder.mkdir(parents=True)
        shutil.copy2(model, folder / model.name)
        shutil.copy2(output / 'models/config.yaml', folder / 'config.yaml')
        meta['styles'].append({'styleName': f'{epoch}epoch', 'styleId': sid})
        for directory, source in [('icons', installed / 'icons/1100.png'),
                                  ('portraits', installed / 'portrait.png')]:
            (speaker / directory).mkdir(exist_ok=True)
            shutil.copy2(source, speaker / directory / f'{sid}.png')
        (speaker / 'voice_samples').mkdir(exist_ok=True)
        for i, test in enumerate(('user', 'sibilants', 'current'), 1):
            item = next(x for x in row['per_text'] if x['test'] == test)
            wav = Path(item['wav'])
            assert sha(wav) == item['wav_sha256']
            shutil.copy2(wav, speaker / f'voice_samples/{sid}_{i:03}.wav')
        report['models'].append({'display_order': rank, 'series': 'original', 'epoch': epoch,
            'styleId': sid, 'model_sha256': sha(model),
            'config_sha256': sha(folder / 'config.yaml'),
            'periodic_prominence_db': row['periodic_prominence_db'],
            'sensitivity_rank_range': [min(row['sensitivity_ranks'].values()), max(row['sensitivity_ranks'].values())]})
    (speaker / 'metas.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    text = ('花園トキネ COEIROINK 131・180epoch版 v1.3.0\n\n'
            '収録スタイル：131epoch、180epoch（元の学習方式・未補正）\n'
            'COEIROINK v2用。各スタイルのモデル・音素設定・画像・サンプル3本を同梱。\n\n'
            '導入：COEIROINKを終了し、ZIPを解凍してできた\n'
            'hanazono-tokine-2.3.0をspeaker_infoフォルダ内へ配置してください。\n'
            'MacではCOEIROINKv2.app/Contents/MacOS/speaker_info内です。\n'
            '旧版を導入済みの場合はCOEIROINKを終了し、旧話者フォルダを\n'
            'speaker_infoの外へ退避してから新版を配置し、再起動してください。\n'
            '同じ話者UUIDの旧版と新版を同時に配置しないでください。\n'
            '既存プロジェクトはバックアップしてから、131epochまたは180epochを選択してください。\n\n'
            '選定：長文を聴き比べた本人が良いと評価した131・180epochを収録しました。\n'
            '2モデル間の優劣や完全無雑音を保証するものではありません。\n'
            'モデルの推論重み補正、EQ、ノイズ除去、音量補正は行っていません。\n'
            '以前の音量補正版とは出力レベルが異なります。音量1.00から調整してください。\n\n'
            'クレジット：COEIROINK:花園トキネ\n'
            '利用前に話者フォルダ内のpolicy.mdとLICENSE.txtをご確認ください。\n')
    (destination / 'README.txt').write_text(text)
    (destination / 'selection.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    files = sorted(p for p in destination.rglob('*') if p.is_file())
    checksums = {str(p.relative_to(destination)): sha(p) for p in files}
    (destination / 'SHA256SUMS.txt').write_text(''.join(f'{h}  {n}\n' for n, h in checksums.items()))
    archive = destination.parent / 'hanazono-tokine-coeiroink-131-180epoch-v1.3.0.zip'
    assert not archive.exists()
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in sorted(destination.rglob('*')):
            if p.is_file():
                z.write(p, str(p.relative_to(destination)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name, expected in checksums.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected
    (archive.with_suffix('.zip.sha256')).write_text(f'{sha(archive)}  {archive.name}\n')
    print(json.dumps({'package': str(destination), 'archive': str(archive), 'bytes': archive.stat().st_size,
                      'sha256': sha(archive), **report}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ('output', 'destination', 'installed'):
        p.add_argument('--' + key, type=Path, required=True)
    build(**vars(p.parse_args()))
