#!/usr/bin/env python3
"""Render one shared long passage through the installed top three raw models."""
import argparse
import array
import base64
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import urllib.request
import wave

TEXT = ('今日は来てくれてありがとう。駅で君を見つけたとき、少しだけ安心したんだ。'
        'さっきまで雨が降っていたのに、今はすっかり晴れているね。'
        '帰り道、川の向こうにある小さな喫茶店へ寄ってみない？'
        '窓際の席に座って、温かい飲み物を飲みながら、今日あったことを聞かせてほしい。'
        'うまく話そうとしなくていいよ。楽しかったことも、少し困ったことも、僕はちゃんと聞いているから。'
        'そういえば、借りていた本を持ってきたんだ。最後の場面が気になって、昨日は夜遅くまで読んでしまった。'
        '君はあの結末をどう思った？僕は、もう一度最初から読み返してみたいと思ったよ。'
        'それじゃ、そろそろ行こうか。急がなくて大丈夫。今日はまだ、ゆっくり話せる時間があるからね。')
UUID = '6cf172a4-b091-11f1-ace8-0242ac1c000c'
MODEL_SHA = {
    111: 'a04c14436d7a92b7925552457343bd7739c32195ab23b1bb85f743cef2cef35d',
    117: '931b0c633362e7e4a710d0d05d3b8e1a0adeaaa01d5a8e91c8818fa0943856fb',
    134: 'b7f26079aafc5fb051b9016eb6d5add62796c484d55e0a7856086d77e6357672',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(path):
    result = subprocess.run(['ffmpeg', '-hide_banner', '-i', str(path), '-af',
        'loudnorm=print_format=json', '-f', 'null', '-'], capture_output=True, text=True, check=True)
    values = json.loads(re.search(r'\{\s*"input_i"[\s\S]*?\}', result.stderr).group())
    assert all(math.isfinite(float(values[k])) for k in ('input_i', 'input_tp', 'input_lra'))
    return values


def pcm(path):
    with wave.open(str(path)) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 44100)
        data = wav.readframes(wav.getnframes())
    samples = array.array('h', data)
    assert len(samples) > 44100 * 10 and max(abs(v) for v in samples) < 32767
    return data


def main(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / 'text.txt').write_text(TEXT + '\n')
    installed = Path('/Applications/COEIROINKv2.app/Contents/MacOS/speaker_info/hanazono-tokine-2.1.0/model')
    rows = []
    for epoch, expected_sha in MODEL_SHA.items():
        assert digest(installed / str(1000 + epoch) / f'{epoch}epoch.pth') == expected_sha
        request = dict(speakerUuid=UUID, styleId=1000 + epoch, text=TEXT, speedScale=1)
        raw = root / f'{epoch}_raw.wav'
        meta = root / f'{epoch}_request.json'
        if raw.exists():
            assert json.loads(meta.read_text())['request'] == request
        else:
            req = urllib.request.Request('http://127.0.0.1:50032/v1/predict_with_duration',
                data=json.dumps(request).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=600) as response:
                result = json.load(response)
            raw.write_bytes(base64.b64decode(result.pop('wavBase64')))
            meta.write_text(json.dumps(dict(request=request, response=result), ensure_ascii=False, indent=2))
        original = pcm(raw)
        subprocess.run(['python3', '/Users/aaa/.codex/skills/ffmpeg-skill/scripts/probe.py',
                        str(raw), '--compact'], check=True)
        rows.append(dict(epoch=epoch, model_sha256=expected_sha, raw_sha256=digest(raw),
            raw_loudness=measure(raw), samples=len(original) // 2, duration_seconds=len(original) / 88200))
        print(f'Generated {epoch}: {len(original) / 88200:.2f}s', flush=True)
    target = min(-23., *(float(r['raw_loudness']['input_i']) - float(r['raw_loudness']['input_tp']) - 1.2 for r in rows))
    chunks = []
    for row in rows:
        epoch = row['epoch']
        gain = target - float(row['raw_loudness']['input_i'])
        output = root / f'{epoch}_matched.wav'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(root / f'{epoch}_raw.wav'),
            '-af', f'volume={gain}dB', '-c:a', 'pcm_s16le', str(output)], check=True)
        data = pcm(output)
        assert len(data) // 2 == row['samples']
        before = array.array('h', pcm(root / f'{epoch}_raw.wav'))
        after = array.array('h', data)
        error = max(abs(b - a * 10 ** (gain / 20)) for a, b in zip(before, after))
        assert error < .6
        loudness = measure(output)
        assert abs(float(loudness['input_i']) - target) < .12
        assert float(loudness['input_tp']) < -1.
        row.update(gain_db=gain, matched_loudness=loudness, matched_sha256=digest(output),
                   constant_gain_error_lsb=error)
        if chunks:
            chunks.append(b'\0' * 88200)
        chunks.append(data)
    combined = root / 'compare_111_117_134.wav'
    with wave.open(str(combined), 'wb') as wav:
        wav.setparams((1, 2, 44100, 0, 'NONE', 'not compressed'))
        wav.writeframes(b''.join(chunks))
    assert pcm(combined) == b''.join(chunks)
    report = dict(text=TEXT, order=[111, 117, 134], speedScale=1, target_lufs=target,
        processing='constant gain only; full utterances; one-second gaps', listening_verified=False,
        rows=rows, combined_sha256=digest(combined), combined_loudness=measure(combined),
        combined_duration_seconds=sum(r['duration_seconds'] for r in rows) + 2)
    (root / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
