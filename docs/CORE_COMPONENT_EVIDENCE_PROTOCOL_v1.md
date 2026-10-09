# Core component evidence protocol v1

Frozen on 2026-10-09.  These experiments address three claims that the current
integrated-design ablations do not isolate.

## 1. Spectral-token compression accuracy-efficiency curve

- Dataset and split contract: BETA, the existing five participant-grouped outer
  folds and three seeds.
- Architecture: retained HarmonicFoldNet v4.1; only `spectral_token_stride`
  changes.
- Registered strides: 1 (no reduction), 2 (retained model), and 4 (stronger
  reduction).
- Accuracy endpoints: participant-level accuracy and balanced accuracy at all
  registered BETA windows.  The principal display windows are 0.4, 0.8 and
  1.2 s.
- Efficiency endpoints: token count, deployment parameters, MACs, single-thread
  CPU p50/p95 latency and CUDA p50/p95 latency at batch size 1 on the same
  machine and software environment.
- Statistical unit and aggregation: participant is the independent unit.  The
  three seed predictions are averaged within participant before paired
  comparisons.  Report paired participant bootstrap 95% confidence intervals
  for balanced-accuracy differences and Holm-adjusted two-sided Wilcoxon tests
  across the three principal windows and the two stride contrasts.
- Frozen decision rule: stride 2 is a defensible retained operating point only
  if (a) it is not Pareto-dominated by stride 1 or stride 4 at at least two of
  the three principal windows, (b) its measured token count and MACs are lower
  than stride 1 and its paired CPU or CUDA latency is lower, and (c) stride 4
  does not retain equal-or-better participant performance while also being
  faster.  The three operating points are also tested for a geometric knee in
  normalized accuracy-versus-latency space; this is descriptive support, not a
  significance test.  If stride 1 is both more accurate and no slower, or
  stride 4 is equally accurate and faster, the retained stride-2 choice is
  rejected rather than rationalized after the fact.  No efficiency conclusion
  will be made from token counts alone, and no claim of universal optimality is
  permitted from three stride values.

## 2. Temporal and spectral candidate-path isolation

- Dataset and split contract: identical to section 1.
- Conditions: full model, `no_temporal_candidate`, and
  `no_spectral_candidate`.
- Each ablation is trained from scratch for every fold and seed.  A path is
  replaced by zeros before the learned candidate-fusion layer; the other path,
  late attention and shared scorer remain unchanged.
- Participant is the independent statistical unit.  Seeds are averaged within
  participant.  Full-minus-ablation differences use paired bootstrap intervals
  and two-sided Wilcoxon tests; all windows and both path ablations form one
  Holm family.
- Frozen decision rule: `no_spectral_candidate` is the temporal-only condition
  and `no_temporal_candidate` is the spectral-only condition.  A path is called
  contributory at a window only when the full-minus-corresponding-ablation
  balanced-accuracy estimate is positive, its paired 95% bootstrap interval
  excludes zero, and the Holm-adjusted test is significant.  Complementarity is
  supported only when the full model exceeds both single-path conditions under
  that rule; otherwise the result is reported as window-specific, redundant,
  or unresolved.  Because the removed path is zeroed before fusion while the
  topology is retained, this is a controlled representation ablation, not an
  efficiency ablation.
- Interpretation: these tests estimate whether each evidence path contributes
  in the integrated model.  They do not claim that either path is an isolated
  optimal decoder, and a non-significant difference is not described as proof
  of equivalence.

## 3. Training-graph versus folded-graph runtime

- Checkpoint: the retained BETA seed-20260929, fold-0 checkpoint, selected
  before the timing run.
- Windows: 0.4, 0.8 and 1.2 s; batch size 1; float32.
- For each window, benchmark both the unfused training graph and its numerically
  equivalent folded deployment graph in the same fresh process.
- CPU: one PyTorch thread, 500 warm-up iterations and 5,000 timed iterations.
- CUDA: identical warm-up and iteration counts with synchronization around each
  measurement; record peak allocated memory.
- Repeat the complete process five times.  Report process-level median and
  range of p50/p95 latency, leaf-module inventory, parameter count, peak CUDA
  memory, and the paired relative latency change.  Alternate which graph is
  timed first across the five fresh processes to limit warm-up and thermal-order
  bias.
- Before timing, verify the two graphs with the existing held-out equivalence
  audit.  Prediction labels must be identical and maximum absolute logit error
  must pass the existing numerical tolerance before any speed comparison is
  accepted.  Folding is called faster only if the deployment graph lowers both
  p50 and p95 latency in at least four of five paired process runs on the stated
  backend.  If it only reduces graph/module complexity without a consistent
  measured speed gain, the paper will claim graph simplification and numerical
  equivalence, not acceleration.

These component experiments stay on BETA. Dong2023 remains untouched by
component selection and is reserved for the frozen external-validation
protocol.
