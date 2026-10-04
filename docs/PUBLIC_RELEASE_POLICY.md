# Public release boundary for v0.1

This file separates the reproducible research preview from work reserved for a
future paper. It is an engineering boundary, not legal advice.

## Safe to release in v0.1

- The exact architecture and scripts represented by the frozen checkpoint.
- The fixed participant split and metrics already stored in
  `runs/kim2025_full_v1/evaluation.json` and `metrics.json`.
- Training-form and deploy-form weights, configuration and SHA-256 checksums.
- The current technical report and model card, both labelled research preview.
- Honest limitations: one fixed split, 5-second windows, four test
  participants, no wearable-electrode validation and no end-to-end hardware
  evaluation.

All public numbers must be generated from the same frozen machine-readable
files. Do not manually create a more favorable subset for a figure or post.

## Keep private until the paper plan is fixed

- Unfinished short-window experiments and their implementation details.
- Leave-one-subject-out results until every fold and evaluation rule is frozen.
- New wearable/peri-auricular datasets, self-recorded biometrics and participant
  metadata.
- Unpublished architectural extensions intended to be the paper's main
  contribution.
- Draft manuscripts, reviewer responses, private comparison results and venue
  strategy.

This keeps v0.1 reproducible without prematurely disclosing every future
research contribution.

## Before any public push

1. Choose candidate venues and read their current rules on preprints, public
   code, double-blind review and prior dissemination.
2. Freeze the release commit and rerun tests from a clean environment.
3. Verify that no raw EEG, biometric recordings, credentials, local absolute
   paths or private application materials are tracked.
4. Confirm that dataset, dependency, source and weight licenses are compatible
   with the planned release.
5. Update only placeholders that become real, such as repository URL, model URL
   or DOI. Never invent a DOI or paper acceptance.
6. Publish the same claims in the repository, model card, figure and social
   post; discrepancies should block release.

## Claims policy

- Say “v0.1 fixed-split result” or “research preview”.
- Do not say “state of the art”, “clinically validated”, “real-time BCI”,
  “thought decoding” or “works for everyone”.
- Do not compare accuracy against another paper unless its preprocessing,
  channels, window, split and evaluation have been reproduced under the same
  protocol.
- Report model-forward latency separately from EEG collection and full-system
  latency.

## Authorship and review

Public code and model artifacts use the project handle `XzStark`. Do not place
the author's institutional affiliation, ORCID or correspondence email in the
GitHub or Hugging Face release. Submission-only identity belongs in the private
manuscript source and the journal submission system. If a later venue uses
double-blind review, prepare the manuscript and supplemental files according to
that venue's anonymity rules rather than assuming the public repository alone
is anonymous.
