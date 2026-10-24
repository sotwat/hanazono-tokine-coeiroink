# Uniform model ranking protocol (2026-10-25)

Frozen before the new eight-text original-series survey and before computing candidate ranks.

## Eligibility and collection

Inventory saved untreated original, prior proper retrain, control and quarter_lr models in epochs 100–200. Record file SHA256, series, epoch, missing epochs and duplicate SHA groups. Corrected weights, EQ and denoising are ineligible. A SHA group occupies one rank. All models must use the exact eight ordinary texts in `retrained_decoder_audit.TEXTS`, speedScale=1 and the existing isolated COEIROINK CPU engine. Never extend vowels. Reuse only matching model, request and engine provenance. Keep raw PCM and per-utterance metadata/hashes.

## Predeclared ordering

This is an artifact-proxy ranking, not a ranking of overall voice quality or true SNR. Lower is preferable. Use lexicographic ordering, without a weighted composite:

1. Number of fixed-frequency peaks reproduced in at least 4/8 texts, with local prominence >=4 dB and frame presence over 6 dB >=0.5. Group frequencies within one 2048-point FFT bin (44100/2048 Hz); retain counts and contributing texts, not only a total.
2. Worst-utterance count of periodic grid lines over 3 dB, then its eight-text mean.
3. Worst-utterance periodic grid mean prominence, then its eight-text mean.
4. Worst persistent local-peak prominence (presence >=0.5), then mean per-text worst prominence; no detected peak is zero.

At numerical ties within 0.01 dB or 0.01 line counts, report tied ranks; SHA is only a deterministic display order and never a quality tie-breaker. Also publish standard deviations, individual text values, 8–18 kHz relative energy and unnormalized RMS. Repeat ordering with peak prominence 3/5 dB, persistence 0.25/0.75 and grid-line thresholds 2/4 dB. Existing detail() detects only peaks >=4 dB, so the 3 dB sensitivity requires recomputing spectral candidates rather than filtering the truncated peak list. Do not silently claim that analysis was performed if it was not.

## Validity checks and limitations

Missing or fewer than eight texts makes a candidate pending, never an overall top-five candidate. Flag nonfinite audio, silence, clipping, duration/voiced-frame abnormalities and RMS below the baseline by >6 dB; inspect flagged cases before eligibility. Report RMS/LUFS, clips and content checks separately. ASR is a content screen, not proof of pronunciation or listening quality. Finalists require equal-gain-only listening files and content review; listening remains unverified unless actually heard.

These measures primarily diagnose fixed tones and periodic artifacts. Broad roughness, sibilance, timbre and naturalness are not fully represented. Do not call low high-frequency energy low noise. A separate validated method is still required to substantiate broad harshness; until then describe any numerical ranking with that limitation. Do not infer a training cause from a ranking.

## Tool investigation status

Dedicated skill search found no verified TTS checkpoint-ranking skill. Adjacent `kunitoki/sonic-skills` audio-artifacts-debug was read; it focuses on realtime DSP diagnostics and does not supply this uniform TTS evaluation, so was not installed.

MoSQITo official docs/repository inspected: version 1.2.1, master commit d990c33 (2024-04-22), API docs dated 2024-05-17. Apache-2.0, Python >=3.5, OS Independent classifier; dependencies NumPy/SciPy/pyuff. Local computation has no service fee or account requirement; installation would add pyuff and overlap existing NumPy/SciPy. No installation, elevated permission or audio upload was performed. Actual current-Mac compatibility was not tested. `sharpness_din_st` requires a pressure signal in Pa and >=48 kHz (otherwise resamples). These synthetic 44.1 kHz PCM files lack playback-pressure calibration. A chosen arbitrary level could support a level-conditional sharpness comparison, but sharpness is not a speech-noise detector and would risk again conflating voice spectrum with noise. Therefore reject it as the primary noise-ranking metric; do not add an uncalibrated acum value as proof of improvement. Maintenance after the inspected 2024 master commit is unverified. Existing local NumPy/SciPy analysis is reused for collection. Broad-harshness evaluation remains pending, not passed.

Sources: https://github.com/kunitoki/sonic-skills/blob/main/skills/audio-artifacts-debug/SKILL.md ; https://github.com/Eomys/MoSQITo ; https://mosqito.readthedocs.io/en/latest/

Primary detail references: https://mosqito.readthedocs.io/en/latest/source/reference/mosqito.sq_metrics.sharpness.sharpness_din.sharpness_din_st.html ; https://github.com/Eomys/MoSQITo/blob/master/LICENSE ; https://github.com/Eomys/MoSQITo/blob/master/setup.py ; https://github.com/Eomys/MoSQITo/blob/master/requirements.txt
