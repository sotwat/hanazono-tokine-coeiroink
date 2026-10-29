"""Add listener-selected raw epoch 70 to the preserved 131/180 distribution."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from package_selected_models import sha


def build(base, installed):
    output = base / 'output/talk_full_20260917'
    previous = base / '配布用パッケージ/selected-131-180-v1.3.0'
    destination = base / '配布用パッケージ/selected-70-131-180-v1.4.0'
    archive = destination.parent / 'hanazono-tokine-coeiroink-70-131-180epoch-v1.4.0.zip'
    assert not destination.exists() and not archive.exists()
    for line in (previous / 'SHA256SUMS.txt').read_text().splitlines():
        expected, name = line.split('  ', 1)
        assert sha(previous / name) == expected
    old_report = json.loads((previous / 'selection.json').read_text())
    assert [r['epoch'] for r in old_report['models']] == [131, 180]
    evidence = output / 'compare_70_54_kotoba_20261030/manifest.json'
    row = next(r for r in json.loads(evidence.read_text())['rows'] if r['epoch'] == 70)
    model = output / 'models/70epoch.pth'
    assert sha(model) == row['model_sha256']
    destination.mkdir()
    speaker = destination / 'hanazono-tokine-2.4.0'
    shutil.copytree(previous / 'hanazono-tokine-2.3.0', speaker)
    folder = speaker / 'model/1070'
    folder.mkdir()
    shutil.copy2(model, folder / model.name)
    shutil.copy2(output / 'models/config.yaml', folder / 'config.yaml')
    assert sha(folder / 'config.yaml') == old_report['models'][0]['config_sha256']
    for kind, source in [('icons', installed / 'icons/1100.png'), ('portraits', speaker / 'portrait.png')]:
        shutil.copy2(source, speaker / kind / '1070.png')
    samples = [(output / 'compare_70_54_kotoba_20261030/70_raw.wav', row['raw_sha256'])]
    onset = json.loads((output / 'noise_onset_20261024/results.json').read_text())
    for test in ('user', 'travel'):
        r = next(x for x in onset if x['epoch'] == 70 and x['test'] == test)
        assert r['model_sha256'] == row['model_sha256']
        samples.append((output / f'noise_onset_20261024/raw/070_{test}.wav', r['wav_sha256']))
    for i, (wav, expected) in enumerate(samples, 1):
        assert sha(wav) == expected
        shutil.copy2(wav, speaker / f'voice_samples/1070_{i:03}.wav')
    meta = json.loads((speaker / 'metas.json').read_text())
    meta.update(version='2.4.0', styles=[{'styleName': f'{e}epoch', 'styleId': 1000+e} for e in (70, 131, 180)])
    (speaker / 'metas.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n')
    report = {'selection_basis': 'User listening selection: add 70, retain 131 and 180',
              'postprocessing': False, 'weight_correction': False,
              'models': [{'epoch': 70, 'series': 'original', 'styleId': 1070,
                          'model_sha256': sha(model), 'config_sha256': sha(folder / 'config.yaml')},
                         *old_report['models']]}
    (destination / 'selection.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    (destination / 'README.txt').write_text('''花園トキネ COEIROINK 70・131・180epoch版 v1.4.0

収録：70epoch、131epoch、180epoch（旧原本・未補正）。音声ライブラリ2.4.0。
本人の聴き比べによる70epoch採用を追加し、従来の131・180epochも収録しています。
数値順位や完全無雑音を保証するものではありません。
各スタイルのモデル・音素設定・画像・原音サンプル3本を同梱しています。
EQ、ノイズ除去、音量補正、推論重み補正は行っていません。

導入：COEIROINKを終了し、解凍したhanazono-tokine-2.4.0をspeaker_infoへ配置します。
MacではCOEIROINKv2.app/Contents/MacOS/speaker_infoです。
旧版はspeaker_infoの外へ退避してください。同じUUIDの版を同時に配置しないでください。
既存プロジェクトはバックアップしてください。131・180のスタイルIDは維持しています。
旧音量補正版とは出力レベルが異なるため、音量1.00から調整してください。

クレジット：COEIROINK:花園トキネ
利用前にpolicy.mdとLICENSE.txtを確認してください。
''')
    checksums = {str(p.relative_to(destination)): sha(p) for p in sorted(destination.rglob('*')) if p.is_file()}
    (destination / 'SHA256SUMS.txt').write_text(''.join(f'{h}  {n}\n' for n,h in checksums.items()))
    assert {p.name for p in (speaker / 'model').iterdir()} == {'1070','1131','1180'}
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in sorted(destination.rglob('*')):
            if p.is_file(): z.write(p,str(p.relative_to(destination)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name, expected in checksums.items(): assert hashlib.sha256(z.read(name)).hexdigest() == expected
    report.update(package=str(destination), archive=str(archive), bytes=archive.stat().st_size, sha256=sha(archive))
    archive.with_suffix('.zip.sha256').write_text(f"{report['sha256']}  {archive.name}\n")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--installed',type=Path,required=True)
    build(**vars(p.parse_args()))
