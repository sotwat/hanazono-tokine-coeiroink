"""Apply a measured output gain to an ESPnet VITS checkpoint, preserving the input."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

GAIN_KEY = 'tts.generator.decoder.output_conv.1.weight_g'
BIAS_KEY = 'tts.generator.decoder.output_conv.1.bias'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def calibrate(source, output, gain, expected_sha256=None):
    if not math.isfinite(gain) or not 0 < gain <= 10:
        raise ValueError('Gain must be finite and within (0, 10].')
    if source.resolve() == output.resolve() or output.exists():
        raise ValueError('Output must be a new file, distinct from the original.')
    original_hash = digest(source)
    if expected_sha256 and original_hash != expected_sha256:
        raise ValueError('Input SHA-256 does not match the expected original.')
    original = torch.load(source, map_location='cpu', weights_only=True)
    if GAIN_KEY not in original or BIAS_KEY not in original:
        raise ValueError('Unsupported checkpoint: expected weight-normalized VITS output layer.')
    if original[GAIN_KEY].numel() != 1 or original[BIAS_KEY].numel() != 1:
        raise ValueError('Expected mono output layer.')
    adjusted = original.copy()
    for key in (GAIN_KEY, BIAS_KEY):
        adjusted[key] = original[key] * gain
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(adjusted, output)
    restored = torch.load(output, map_location='cpu', weights_only=True)
    if list(restored) != list(original):
        raise ValueError('Checkpoint keys changed during serialization.')
    changed = []
    for key in original:
        target = original[key] * gain if key in (GAIN_KEY, BIAS_KEY) else original[key]
        if not torch.equal(restored[key], target):
            raise ValueError(f'Unexpected tensor change: {key}')
        if not torch.equal(restored[key], original[key]):
            changed.append(key)
    manifest = dict(input_sha256=original_hash, output_sha256=digest(output),
                    gain_before_tanh=gain, gain_db_before_tanh=20 * math.log10(gain),
                    changed_tensors=changed, tensor_count=len(original))
    output.with_suffix('.pth.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--gain', type=float, required=True)
    parser.add_argument('--expected-sha256')
    args = parser.parse_args()
    print(json.dumps(calibrate(args.input, args.output, args.gain, args.expected_sha256)))
