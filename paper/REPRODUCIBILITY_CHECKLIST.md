# HarmonicFoldNet reproducibility checklist

This checklist describes the frozen manuscript evidence. It is not a claim that
the public datasets or third-party baseline source can be redistributed.

## Experimental contract

- [x] The participant, not the trial or seed, is the independent statistical unit.
- [x] Five outer folds are grouped by participant; a participant never enters fitting,
  validation, and testing in the same fold.
- [x] Three training seeds are reported for the final neural comparisons.
- [x] Window lengths, channels, sampling rate, filtering, optimizer, early stopping,
  and checkpoint selection are specified in the manuscript and frozen configuration.
- [x] Benchmark and BETA are labelled selection-aware because later architecture
  screening consulted them; they are not described as untouched confirmation.
- [x] Wearable dry/wet transfer was run after model retention and is reported as a
  directional acquisition-domain analysis, not validation of a glasses montage.

## Comparators and statistics

- [x] Calibration-free analytic controls include fixed harmonic power, CCA, and FBCCA.
- [x] Neural references include SSVEPformer, FB-SSVEPformer, and a pinned MTSNet
  reconstruction.
- [x] Participant-calibrated comparisons include TRCA, ensemble TRCA, and TDCA with
  the same target-label budget; unequal source pretraining is disclosed.
- [x] Participant-level source data, bootstrap intervals, paired tests, effect sizes,
  and Holm families are machine-readable.
- [x] ITR is explicitly theoretical and is reported under four overhead assumptions.

## Mechanism and deployment

- [x] Local-mixer, combined late-attention, and harmonic-bias ablations are reported.
- [x] Attention allocation is separated from faithful explanation or physiological
  discovery.
- [x] Batch-one CPU/GPU latency states device class, precision, threads, warm-up, and
  repetition count.
- [x] Train-to-deploy folding was checked on 20 preserved checkpoints, five windows,
  and 49,000 held-out paired predictions, with zero label disagreements.

## Public artifact boundary

- [x] Raw EEG is excluded and must be obtained from the original dataset records.
- [x] The unlicensed MTSNet source is excluded; its upstream commit and reconstruction
  protocol are recorded.
- [x] Authored source is licensed under PolyForm Noncommercial 1.0.0; weights and
  documentation are licensed under CC BY-NC 4.0. This is therefore a public
  noncommercial research release, not OSI-approved open source.
- [x] Exact core environment versions are recorded in `paper/environment-lock.txt`.
- [x] Figure PDFs use embedded TrueType fonts, and chart text is at least 8 pt apart
  from mathematical log-axis exponents rendered by Matplotlib.

## Required before public upload

- [x] Insert the final affiliation and corresponding-author email.
- [x] Add the submitting author's verified ORCID iD.
- [x] Exclude the affiliation, ORCID, and correspondence email from the GitHub and Hugging Face release bundles.
- [ ] Insert the permanent GitHub address, release tag, and Hugging Face model address.
- [x] Confirm funding and competing-interest statements.
- [ ] Create a clean tagged release from a reviewed commit; do not publish transient
  experiment caches, raw EEG, credentials, or third-party unlicensed source.
- [ ] If a preprint is deposited, verify its PDF and source archive after upload and
  add the identifier to the journal cover letter and repository citation. A preprint
  is optional for direct JNE submission and for release of the matching code and weights.
