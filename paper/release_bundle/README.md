# Evidence-input bundle

`HarmonicFoldNet_evidence_inputs.zip` contains the preserved result JSON files
and the one deployment checkpoint referenced by
`paper/SUBMISSION_EVIDENCE_CONFIG.json`. Paths inside the archive are relative
to the repository root.

To rebuild the participant-level tables from a clean source checkout:

1. extract the archive at the repository root, preserving paths;
2. install the locked environment in `paper/environment-lock.txt`;
3. run `python -m scripts.build_submission_evidence`;
4. run `python -m scripts.build_supplement` and the manuscript audits.

This bundle supports exact evidence-table reconstruction from preserved run
outputs. It is not raw-EEG redistribution and does not retrain every model.
Raw EEG must be obtained from the original dataset hosts. The train-to-deploy
equivalence audit is released as derived source data; reproducing that audit
for all 20 checkpoints additionally requires the checkpoint set identified by
its manifest.
