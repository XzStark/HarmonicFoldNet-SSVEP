# local-to-global SSVEP architecture freeze v1.3

Frozen after development on the designated Kim2025 development split. No
external grouped-test fold was used to select this revision.

## Fixed model

- architecture revision: `temporal_fusion_v1_3`;
- 1-D depthwise convolutional stem and hierarchical 8x temporal reduction;
- one foldable local-mixing block before and after reduction;
- local updates use learned residual gates initialized at `sigmoid(-2)`;
- temporal self-attention is applied only after temporal reduction;
- candidate queries cross-attend to the reduced temporal tokens, followed by
  compact candidate attention and a shared scoring head;
- candidate-aligned segmented complex demodulation supplies learned tokens;
- regularized analytic reference correlation is an optional residual prior;
- local training branches fold into a single depthwise convolution for the
  deployment graph.

## Frozen training contract

- sample rate: 250 Hz;
- optimization windows: 0.4, 0.6, 0.8, 1.0 and 1.2 seconds;
- AdamW, learning rate 0.0005, weight decay 0.02;
- label smoothing 0.05;
- checkpoint selection: mean validation balanced accuracy over all five short
  windows;
- three registered seeds: 20260929, 20260930 and 20260931;
- five fixed subject-disjoint outer folds per dataset.

## Development finding, not a paper result

The learned local gates increased from about 0.119 to 0.173 and 0.142. The
local branch improved the 0.8--1.2 s held-out development results but did not
improve 0.4--0.6 s, and its five-window mean was approximately tied with the
no-local ablation. The component contribution therefore remains unproven until
the registered grouped cross-validation and participant-level statistics are
complete.
