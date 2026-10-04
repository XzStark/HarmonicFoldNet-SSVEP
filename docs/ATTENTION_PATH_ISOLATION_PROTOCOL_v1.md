# Attention-path isolation protocol v1

Status: frozen before inspecting either isolated-path result.

## Motivation

On the complete three-seed BETA ablation, v4.1 full attention improves
0.6--1.5-second balanced accuracy but reduces 0.4-second accuracy by 0.66
percentage points relative to `no_attention`. The existing ablation removes
both the candidate-to-spectrum harmonic cross-attention and the subsequent
candidate-to-candidate self-attention, so it cannot identify the harmful path.

## Development comparison

Train two v4.1 variants from scratch on Benchmark grouped fold 0, seed
20260929, with every preprocessing, optimizer, schedule and selection setting
unchanged:

- `no_cross_attention`: retain candidate self-attention, remove harmonic
  cross-attention.
- `no_candidate_attention`: retain harmonic cross-attention, remove candidate
  self-attention.

The candidate advances only if it improves 0.4-second held-out balanced
accuracy and does not materially reduce the mean registered-window selection
score or 0.8/1.2-second accuracy. This is a topology isolation experiment, not
a new novelty claim. Failed variants will not be tuned on the same split.

## Result

The frozen v4.1 baseline scored 29.94/48.33/67.32/75.71/79.29% at
0.4/0.6/0.8/1.0/1.2 seconds.  Removing cross-attention yielded
28.69/45.12/63.69/74.29/77.32%; removing candidate self-attention yielded
28.57/48.51/66.55/76.49/81.31%.

Neither isolated path improved 0.4 seconds.  Candidate self-attention appears
more useful at shorter windows, while removing it can trade short-window
accuracy for longer-window accuracy.  Both variants fail the frozen gate and
are rejected; the result supports retaining both late-attention stages in the
multi-window mainline, without claiming that either stage improves every
individual window.
