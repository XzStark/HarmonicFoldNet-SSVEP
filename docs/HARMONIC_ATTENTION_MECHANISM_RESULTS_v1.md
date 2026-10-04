# Harmonic-attention mechanism analysis v1

## Contract

Analysis uses all 70 public BETA participants. For each of three training seeds
and five subject-disjoint folds, attention is extracted only from the held-out
participants. Repeated seeds are averaged within participant before Wilcoxon
tests; seeds are not counted as independent samples.

For each trial, only the query corresponding to the true stimulus candidate is
analysed. The same trained checkpoint is evaluated with its harmonic attention
bias present and temporarily masked. This isolates where the attention rule
directs probability mass; it is not a causal performance ablation.

## Results

Participant-averaged values (%):

| Window | Peak within target harmonic +/-0.5 Hz, bias | Peak within, masked | Harmonic-neighborhood mass, bias | Mass, masked |
|---:|---:|---:|---:|---:|
| 0.4 s | 99.58 | 11.96 | 72.78 | 9.87 |
| 0.8 s | 98.92 | 17.93 | 73.53 | 10.22 |
| 1.2 s | 98.72 | 20.89 | 73.79 | 10.15 |

All paired participant-level differences in entropy, peak distance, peak-hit
rate and neighborhood mass are significant (`p <= 3.56e-13`; the exact values
and bootstrap intervals are in the machine-readable summaries).

## Interpretation boundary

The result establishes that the bias makes the late attention stage
frequency-selective in the intended harmonic neighborhood across unseen
participants and random seeds. This is mechanism evidence, not evidence that
the bias always improves classification.

The subsequently completed from-scratch `no_harmonic_bias` ablation used the
same three seeds, five folds and 70 held-out BETA participants. Full minus
no-bias balanced accuracy was -0.85, +0.43, +1.04, +1.57, +1.11 and +1.30
percentage points at 0.4, 0.6, 0.8, 1.0, 1.2 and 1.5 seconds. After Holm
correction, only 1.0 seconds (`p=0.0346`) and 1.5 seconds (`p=0.0163`) remained
significant. The defensible conclusion is therefore window-dependent: the
harmonic bias strongly controls attention location and provides classification
benefit at longer windows, but it is not a universal short-window gain.

Machine-readable evidence:

- `runs/paper_v18/attention_mechanism/beta/0.4s/summary.json`
- `runs/paper_v18/attention_mechanism/beta/0.8s/summary.json`
- `runs/paper_v18/attention_mechanism/beta/1.2s/summary.json`
- `runs/paper_v20/ablations/beta/full_vs_no_harmonic_bias.json`
