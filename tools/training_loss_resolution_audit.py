"""Inspect the saved mel projection; never modify models or emit altered audio."""
import argparse
import hashlib
import json
from pathlib import Path
import wave

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(model, index):
    state = torch.load(model, map_location="cpu", weights_only=True)
    matrix = state["tts.mel_loss.wav_to_mel.logmel.melmat"].double()
    assert matrix.shape == (1025, 80) and torch.isfinite(matrix).all()
    frequencies = torch.arange(1025, dtype=torch.float64) * 44100 / 2048
    high = (frequencies >= 8000) & (frequencies <= 20000)
    projection = matrix[high].T
    u, singular, vh = torch.linalg.svd(projection, full_matrices=False)
    tolerance = singular.max() * max(projection.shape) * torch.finfo(torch.float64).eps
    rank = int((singular > tolerance).sum())
    seed = torch.cos(2 * torch.pi * frequencies[high] / (44100 / 64))
    row_basis = vh[:rank]
    direction = seed - row_basis.T @ (row_basis @ seed)
    direction /= direction.abs().max()
    spectrum = torch.ones(1025, dtype=torch.float64)
    changed = spectrum.clone()
    changed[high] += 0.5 * direction
    assert changed.min() > 0
    a, b = spectrum @ matrix, changed @ matrix
    null_log_error = float((a.log() - b.log()).abs().max())
    assert null_log_error < 1e-10
    bands = []
    for column in range(80):
        support = frequencies[matrix[:, column] > 0]
        if support.numel():
            bands.append({"band": column, "low_hz": float(support.min()),
                          "high_hz": float(support.max()), "bins": len(support)})

    rows = json.loads(index.read_text())["rows"]
    references = [r for r in rows if r["series"] == "original" and r["epoch"] == 180]
    assert len(references) == 8
    results = []
    window = torch.hann_window(2048, dtype=torch.float64)

    def magnitude(x):
        spec = torch.stft(x, 2048, 512, window=window, center=True,
                          pad_mode="reflect", return_complex=True)
        return spec.abs().square().clamp_min(1e-10).sqrt().T

    for row in references:
        path = Path(row["wav"])
        assert sha(path) == row["wav_sha256"]
        with wave.open(str(path)) as wav:
            assert (wav.getframerate(), wav.getsampwidth(), wav.getnchannels()) == (44100, 2, 1)
            x = torch.from_numpy(np.frombuffer(wav.readframes(wav.getnframes()),
                                              dtype="<i2").astype(np.float64) / 32768)
        baseline = magnitude(x)
        t = torch.arange(x.numel(), dtype=torch.float64) / 44100
        tones = torch.stack([torch.sin(2 * torch.pi * (44100 / 64) * k * t)
                             for k in range(10, 27)]).sum(0)
        generator = torch.Generator().manual_seed(777)
        noise = torch.randn(x.shape, generator=generator, dtype=torch.float64)
        noise_fft = torch.fft.rfft(noise)
        hz = torch.fft.rfftfreq(x.numel(), 1 / 44100)
        noise_fft[(hz < 6890.625) | (hz > 17915.625)] = 0
        noise = torch.fft.irfft(noise_fft, n=x.numel())
        for name, delta in (("periodic_grid", tones), ("band_noise", noise)):
            delta = delta / delta.square().mean().sqrt() * x.square().mean().sqrt()
            for relative_db in (-40, -30):
                altered = x + delta * 10 ** (relative_db / 20)
                generated = magnitude(altered)
                mel_l1 = ((generated @ matrix).clamp_min(1e-10).log()
                          - (baseline @ matrix).clamp_min(1e-10).log()).abs().mean()
                log_l1 = (generated.log() - baseline.log()).abs().mean()
                results.append({"test": row["test"], "wav_sha256": row["wav_sha256"],
                                "perturbation": name, "relative_rms_db": relative_db,
                                "mel_l1": float(mel_l1), "weighted_mel_l1": float(mel_l1 * 45),
                                "linear_bin_log_l1": float(log_l1),
                                "would_clip": bool((altered.abs() >= 1).any())})
    assert all(np.isfinite(r["mel_l1"]) and r["mel_l1"] > 0 for r in results)
    return {"model_sha256": sha(model), "index_sha256": sha(index),
            "mel_matrix_sha256": hashlib.sha256(matrix.numpy().tobytes()).hexdigest(),
            "fft_bin_hz": 44100 / 2048, "bands": bands,
            "high_region": {"low_hz": 8000, "high_hz": 20000,
                            "bins": int(high.sum()), "projection_rank": rank,
                            "null_dimensions": int(high.sum()) - rank},
            "positive_spectral_null_example": {
                "maximum_log_mel_error": null_log_error,
                "linear_bin_log_l1": float((changed.log() - spectrum.log()).abs().mean()),
                "is_consistent_waveform": "not established; magnitude-domain example only"},
            "waveform_sensitivity": results,
            "limits": ["Synthetic perturbations are diagnostic, not retrained output or listening tests.",
                       "Raw losses have different scales; their ratios are not perceptual sensitivity ratios.",
                       "The discriminator and feature-matching losses can still detect mel-null changes.",
                       "No causal attribution or trained quality improvement is established."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = audit(args.model, args.index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"high_region": result["high_region"],
                      "null_example": result["positive_spectral_null_example"],
                      "waveform_tests": len(result["waveform_sensitivity"])}))
