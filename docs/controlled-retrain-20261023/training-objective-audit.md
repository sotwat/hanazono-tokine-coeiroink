# Training objective and next isolated continuation

2026-10-25. The saved-model ranking is complete. This investigation does not re-synthesize those 1,064 utterances or change any deployed model.

## Findings

The inspected ESPnet source is commit `abab1e48c06eddb4bd3bc8762176317d12d2b192`, matching the pinned setup. `espnet2/train/gan_trainer.py` selects discriminator then generator when `generator_first=false`, uses each forward result's optimizer index, and zeros both optimizers after each turn. VITS detaches generated audio on the discriminator turn and reuses the cached generator graph on the generator turn. No swapped optimizer or missing gradient-clear defect was found in this code inspection. This is not a new live gradient trace of a training batch.

The generator objective contains mel, KL, duration, adversarial and feature-matching losses, with weights45/1/1/1/2. The mel loss uses amplitude STFT, a saved1025×80 Slaney projection, natural logarithm and mean L1; 44,100Hz, FFT2048, hop512, Hann window, full Nyquist range. There is no separate linear-frequency STFT reconstruction term. Original180, control200 and quarter200 have exactly identical mel projection buffers; see [training_loss_buffer_validation.json](training_loss_buffer_validation.json).

[training_loss_resolution_audit.json](training_loss_resolution_audit.json) uses the actual saved matrix, without installing another librosa version:

- In8–20kHz,557 FFT bins project to rank20, leaving537 magnitude-domain null directions. A positive artificial spectrum changed while log-mel values stayed identical in double precision. Consistency with a realizable waveform was not established. This is an example of mel projection information loss, not proof that generated speech took this direction.
- For each of the existing eight original180 waveforms, an in-memory diagnostic selects its highest-RMS16,384-sample segment on the512-sample grid, matching training's32-frame segment length and avoiding silent tails. It adds either a689.0625Hz-spaced grid (multiples10–26) or matched-band random noise at−40/−30dB RMS relative to that segment. All32 tests increase mel loss and none clips. No modified WAV is saved. Mean weighted mel increments for the grid are4.526/9.736; for band noise6.718/14.095. The loss therefore is **not completely blind** to this additive grid. These are synthesis segments, not the ground-truth training recordings. Losses on different representations have different scales; raw loss ratios are not perceptual sensitivity ratios.
- The synthetic examples do not reproduce the actual learned degradation. The adversarial and feature-matching losses may also detect changes that mel projection cannot distinguish. Adding a spectral loss is not yet justified as a demonstrated fix.

[phase_deviation_screen.json](phase_deviation_screen.json) compares the second stride8 upsampler's per-channel DC phase sums. Deviation RMS is0.01390444 at180,0.01392646 at control200 and0.01390568 at quarter200. Relative deviation is0.66862/0.66871/0.66858. These aggregate weight statistics alone do not explain the different output artifacts or establish a threshold for audible noise. The previously useful inference correction is not substituted for training.

## Research before the new diagnostic

Dedicated search: `agent skill VITS vocoder training spectral loss SKILL.md`; adjacent search: `github "SKILL.md" "audio" "spectral"`. No directly suitable verified VITS training-loss diagnostic skill was established. The adjacent [songsee skill](https://github.com/NousResearch/hermes-agent/blob/main/skills/media/songsee/SKILL.md) was read: v1.0.0, metadataMIT/macOS, Go CLI, local spectrogram/features. It overlaps the existing FFT/feature tools and supplies neither a training objective nor a validated speech-noise score. It was not installed; CLI license/maintenance/runtime were not independently certified because adoption was rejected for scope/duplication.

The author's [auraloss repository](https://github.com/csteinmetz1/auraloss) and pinned [setup.py](https://github.com/csteinmetz1/auraloss/blob/0853e9e732cd32a7b23f11ef4f0d2975d259e653/setup.py) were inspected. The visible main commit was0853e9e dated2024-02-09; package0.4.0, Apache2.0, Python>=3.6, torch/numpy, optional librosa/scipy/matplotlib. These losses operate locally and require no paid service or new data-upload permission. No platform-specific extension is declared in setup.py, but compatibility with this exact torch1.12.1/CUDA stack was not tested. It is a candidate for a later spectral-objective experiment, not a perceptual-quality evaluator. It was not installed: this audit can use the saved projection and existing PyTorch/NumPy without a redundant dependency or changing the legacy environment.

## Next condition: change only discriminator learning rate

The completed control and quarter_lr conditions changed G and D together. They cannot separate generator update size from discriminator adaptation. The next single condition keeps G at the control rate and divides only D's LR trajectory by four. This tests G/D balance; it does not presume that D is the historical cause or that slowing it will improve audio.

`controlled_retrain.py prepare-split` creates a new sibling `gd_split` experiment from the SHA-verified untreated full180 checkpoint. It reuses the verified652 pilot recordings and metadata. Model tensors and both optimizers' steps/momenta remain exact; only D optimizer LR/initial_lr and D scheduler LR values scale. Configuration differences from pilot/control are only output_dir and optim2_conf. G/D ordering, losses, data, phoneme order and architecture are unchanged. Preparation refuses an existing destination. The existing two-arm launcher is unchanged.

```bash
source /content/tokine-202211/bin/activate
work=/content/drive/MyDrive/MYCOEIROINK_WORK
repo=/content/hanazono-tokine-coeiroink
python "$repo/tools/controlled_retrain.py" prepare-split \
  --pilot "$work/controlled_retrain_20261023/pilot" \
  --reference "$work/retrain_180_20261021" \
  --destination "$work/controlled_retrain_20261023/gd_split"
nohup bash "$repo/tools/controlled_split_start.sh" \
  "$work/controlled_retrain_20261023/gd_split" \
  > "$work/controlled_retrain_20261023/gd_split/launcher.log" 2>&1 < /dev/null &
```

Do not copy Mac absolute-path configurations to Colab. On resume, inspect state and use the launcher directly; do not prepare again. The launcher shares the existing host training lock, resumes only this new arm, preserves every epoch and targets200. Existing control/quarter200 are not restarted. Full strict model load, all652 phoneme lengths,47 phoneme order, both optimizers' state/step/LR and both scheduler states must pass before training. Completion requires actual batch/GPU progress and internal checkpoint epoch, not submission of the command.

After200, evaluate the untreated eight ordinary sentences against cached180/control200/quarter200 and previous proper200. Reuse the exact existing audio for those references. Gain-only listening exports are allowed; vowel stretching, inference weight corrections or denoising do not establish success. Broad-band harshness, pronunciation, listening and actual quality improvement remain unresolved. Preserve the original133-candidate ranking artifacts. An expanded ranking must reuse those caches and measure only the new saved candidates under the same protocol; do not silently mix different coverage or overwrite the earlier evidence.

Local verification: all32 perturbations finite/non-clipping with positive mel increments; matrix-null example residual<1e−10; exact buffer equality across three saved models. Synthetic checkpoint tests exercised old factors1/0.25 and new factors[1,0.25], rejected modified momentum and unintended G LR changes. Python compilation, shell syntax and diff whitespace checks passed. Runtime verification on2026-10-25: the restored real180 checkpoint passed the full preflight; epoch181 batches151–200 and T4 process215120 were observed. See gd_split_start_validation.json for the terminal-observed audit (not an independent binary transfer). The candidate is running toward200; quality improvement is not yet verified.
