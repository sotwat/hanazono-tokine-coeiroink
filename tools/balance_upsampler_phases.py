"""Suppress stride-8 DC images in the second ESPnet HiFiGAN upsampler.

This is an inference-only checkpoint derivative, not a training checkpoint.
The original file is never overwritten. Validate speech before installing it.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

PREFIX = "tts.generator.decoder.upsamples.1.1."
V_KEY = PREFIX + "weight_v"
G_KEY = PREFIX + "weight_g"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def balance(state, amount=0.5):
    if not 0 < amount <= 1:
        raise ValueError("Correction amount must be within (0, 1].")
    v, g = state[V_KEY], state[G_KEY]
    if v.ndim != 3 or v.shape[-1] != 16 or g.shape != (v.shape[0], 1, 1):
        raise ValueError("Expected a weight-normalized, 16-tap second upsampler.")
    if not torch.isfinite(v).all() or not torch.isfinite(g).all():
        raise ValueError("Non-finite input weights.")
    norm = torch.linalg.vector_norm(v, dim=(1, 2), keepdim=True)
    if (norm == 0).any():
        raise ValueError("Zero weight-normalization denominator.")
    weight = v * g / norm
    phases = weight.reshape(*weight.shape[:2], 2, 8).sum(2)
    # The minimum-norm correction splits each phase-sum change over its two taps.
    delta = amount * (phases.mean(-1, keepdim=True) - phases) / 2
    corrected = weight + delta.repeat(1, 1, 2)
    corrected_g = torch.linalg.vector_norm(corrected, dim=(1, 2), keepdim=True)
    if (corrected_g == 0).any():
        raise ValueError("Correction would create a degenerate input channel.")
    result = state.copy()
    result[V_KEY], result[G_KEY] = corrected, corrected_g
    effective = corrected * corrected_g / torch.linalg.vector_norm(
        corrected, dim=(1, 2), keepdim=True
    )
    after = effective.reshape(*effective.shape[:2], 2, 8).sum(2)
    expected_phases = phases + amount * (phases.mean(-1, keepdim=True) - phases)
    if not torch.isfinite(effective).all() or not torch.allclose(
        after, expected_phases, atol=1e-6, rtol=1e-5
    ):
        raise ValueError("DC phase balance verification failed.")
    return result, {
        "method": "second_stride8_upsampler_dc_phase_balance_v1",
        "amount": amount,
        "changed_tensors": [V_KEY, G_KEY],
        "effective_weight_relative_change": float((effective - weight).norm() / weight.norm()),
        "dc_phase_std_before": float(phases.std(-1).mean()),
        "dc_phase_std_after": float(after.std(-1).mean()),
        "tensor_count": len(state),
        "inference_only": True,
    }


def correct(source, output, expected_sha256=None, amount=0.5):
    if source.resolve() == output.resolve() or output.exists():
        raise ValueError("Output must be a new file distinct from the input.")
    source_hash = digest(source)
    if expected_sha256 and source_hash != expected_sha256:
        raise ValueError("Input SHA-256 mismatch.")
    state = torch.load(source, map_location="cpu", weights_only=True)
    fixed, manifest = balance(state, amount)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fixed, output)
    restored = torch.load(output, map_location="cpu", weights_only=True)
    if list(restored) != list(state):
        raise ValueError("Checkpoint keys changed.")
    for key in state:
        expected = fixed[key] if key in (V_KEY, G_KEY) else state[key]
        if not torch.equal(restored[key], expected):
            raise ValueError(f"Unexpected tensor change: {key}")
    manifest.update(input_sha256=source_hash, output_sha256=digest(output))
    output.with_suffix(".pth.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--amount", type=float, default=0.5)
    args = parser.parse_args()
    print(json.dumps(correct(args.input, args.output, args.expected_sha256, args.amount)))
