# Window-specialized comparison protocol v1

Status: frozen before any window-specialized HarmonicFold result was inspected.

## Why this correction is required

The original HarmonicFold main matrix trained one checkpoint over 0.4--1.2 s
windows and selected it by the mean validation score across those windows.
The external MTSNet reconstruction fixes its input length and therefore trains
and selects a separate checkpoint for each window. Directly calling this a
matched architecture comparison would confound architecture with training
specialization.

## Primary matched comparison

For 0.4, 0.8 and 1.2 s separately:

- train HarmonicFold only on the target window;
- early-stop only on the same target validation window;
- retain the frozen architecture, optimizer, epoch limit, patience, channels,
  preprocessing, participant folds and three random seeds;
- compare against MTSNet trained and selected for that same window;
- merge out-of-fold participants, average each participant across seeds, then
  run paired participant-level tests with Holm correction across windows.

No architecture width, loss, learning rate or window-specific augmentation is
retuned after inspecting the specialized results.

## Separate universal-model track

The original multi-window checkpoint remains relevant as a deployment result:
one set of weights accepts several observation lengths. It will be reported as
a universal-window model, not substituted into the primary matched comparison.
This track measures the accuracy cost, if any, of avoiding one model per
window.

## Interpretation boundary

A positive specialist result supports an architecture-level comparison under
matched specialization. A positive universal result supports operational
flexibility. Neither result alone establishes superiority under a different
channel montage, preprocessing chain, target-participant calibration budget or
hardware runtime.
