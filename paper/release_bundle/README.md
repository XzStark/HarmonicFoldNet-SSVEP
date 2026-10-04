# Evidence-input bundle

`HarmonicFoldNet_evidence_inputs.zip` contains the preserved result JSON files
and the one deployment checkpoint referenced by
`paper/SUBMISSION_EVIDENCE_CONFIG.json`. Paths inside the archive are relative
to the repository root.

To verify the archive hashes and rebuild the participant-level tables from a
clean source checkout without extracting files into the working tree:

1. install the locked environment in `paper/environment-lock.txt`;
2. run `python -m scripts.rebuild_submission_from_bundle`;
3. run `python -m scripts.build_supplement` and the manuscript audits.

The rebuild command validates every archived input against the manifest,
extracts it into a temporary directory, rebuilds the evidence tables, and
removes the temporary inputs when it exits.

This bundle supports exact evidence-table reconstruction from preserved run
outputs. It is not raw-EEG redistribution and does not retrain every model.
Raw EEG must be obtained from the original dataset hosts. The train-to-deploy
equivalence audit is released as derived source data; reproducing that audit
for all 20 checkpoints additionally requires the checkpoint set identified by
its manifest.
