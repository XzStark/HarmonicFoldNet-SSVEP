from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
SOURCE = PAPER / "source_data"
FIGURE_SOURCE = PAPER / "figures" / "source_data"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: str, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def percent(value: str) -> str:
    return number(str(100.0 * float(value)), 2)


def p_value(value: str) -> str:
    numeric = float(value)
    if numeric < 0.0001:
        return f"{numeric:.2e}"
    return f"{numeric:.4f}"


def table(headers: list[str], rows: Iterable[Iterable[object]]) -> str:
    output = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        output.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(output)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    config = json.loads((PAPER / "SUBMISSION_EVIDENCE_CONFIG.json").read_text(encoding="utf-8"))
    main_summary = read_csv(SOURCE / "main_summary.csv")
    main_contrasts = read_csv(SOURCE / "main_contrasts.csv")
    itr_sensitivity_peaks = read_csv(SOURCE / "itr_overhead_sensitivity_peaks.csv")
    ablation_summary = read_csv(SOURCE / "ablation_summary.csv")
    ablation_contrasts = read_csv(SOURCE / "ablation_contrasts.csv")
    attention_mechanism = read_csv(SOURCE / "attention_mechanism_summary.csv")
    analytic_baselines = read_csv(SOURCE / "analytic_baseline_summary.csv")
    strong_filter_bank_summary = read_csv(SOURCE / "strong_filter_bank_summary.csv")
    strong_filter_bank_contrasts = read_csv(SOURCE / "strong_filter_bank_contrasts.csv")
    mtsnet = read_csv(SOURCE / "mtsnet_contrasts_corrected.csv")
    adaptation_summary = read_csv(SOURCE / "adaptation_summary.csv")
    adaptation_contrasts = read_csv(SOURCE / "adaptation_contrasts.csv")
    transfer_summary = read_csv(SOURCE / "electrode_transfer_summary.csv")
    transfer_contrasts = read_csv(SOURCE / "electrode_transfer_contrasts.csv")
    deployment = read_csv(SOURCE / "deployment_measurements.csv")

    lines: list[str] = [
        "# Supplementary information for HarmonicFoldNet",
        "",
        "This supplement accompanies the manuscript *HarmonicFoldNet: A Compact Foldable Local-to-Global Decoder for Cross-Subject SSVEP Recognition*. It is generated from preserved run artifacts rather than transcribed from figures. The participant is the independent statistical unit throughout.",
        "",
        "## Supplementary Methods S1. Evidence reconstruction and validation",
        "",
        "The submission evidence builder reads the original per-fold result files for three training seeds. It validates the expected participant set, averages seed repeats within participant, reconstructs descriptive summaries and paired tests, and compares the reconstructed means with the frozen comparison records. Confidence intervals use 10,000 participant-bootstrap resamples. Paired tests use the two-sided Wilcoxon signed-rank test; effect sizes are rank-biserial correlations. Holm correction follows the multiplicity families declared in the manuscript and evidence configuration.",
        "",
        "Individual audit trails are provided in `main_participant_metrics_by_seed.csv`, `main_participant_metrics_seed_averaged.csv`, and `ablation_participant_balanced_accuracy.csv`. The machine-readable configuration is `SUBMISSION_EVIDENCE_CONFIG.json`; the consolidated record is `submission_evidence.json`.",
        "",
        "## Supplementary Methods S2. Registered dataset contracts",
        "",
    ]

    lines.extend(
        [
            table(
                ["Dataset", "Participants", "Classes", "Role in this study"],
                [
                    ["Kim2025 beta-range", 40, 40, "Initial architecture development and beta-band stress testing"],
                    ["Tsinghua Benchmark", 35, 40, "Subject-disjoint evaluation and later candidate-screening audit; not an untouched confirmatory dataset"],
                    ["BETA", 70, 40, "Subject-disjoint evaluation, later candidate screening, component analysis, and optional participant adaptation; not untouched"],
                    ["Wearable SSVEP", 102, 12, "Frozen directional wet/dry electrode-condition transfer after architecture selection"],
                ],
            ),
            "",
            "Wet and dry Wearable recordings came from the same participants and therefore count once, giving 306 unique participants across five datasets. Benchmark and BETA were later used to accept or reject follow-up variants while retaining the reported architecture; their intervals and tests are therefore selection-aware descriptive evidence rather than untouched confirmatory inference. Dong2023 remained independently reserved until the architecture and analysis contract were frozen, and Wearable remained the external electrode-condition evaluation.",
            "",
        ]
    )

    contract_rows = []
    for dataset, entry in config["main"].items():
        contract_rows.append(
            [
                entry["dataset_label"],
                entry["participants"],
                entry["classes"],
                ", ".join(entry["windows"]),
                ", ".join(entry["display_windows"]),
                "3",
                "5",
            ]
        )
    wearable_entry = config["electrode_transfer"]
    contract_rows.append(
        [
            wearable_entry["dataset_label"],
            wearable_entry["participants"],
            12,
            ", ".join(wearable_entry["windows"]),
            "0.4, 0.6, 0.8, 1.0, 1.2",
            "3",
            "5",
        ]
    )
    lines.extend(
        [
            table(
                ["Dataset", "Participants", "Classes", "Evaluated windows (s)", "Primary figure windows (s)", "Seeds", "Outer folds"],
                contract_rows,
            ),
            "",
            "Benchmark windows of 2.0, 3.0 and 5.0 s were retained as a supplementary long-window audit and were included in the registered Benchmark multiplicity family. They were omitted from the primary figure to keep the wearable operating range legible. BETA used 0.4–1.5 s. The Wearable analysis registered 0.4–2.0 s; 0.4–1.2 s were the primary figure windows and 1.5/2.0 s were secondary transfer-window audits. All seven Wearable windows belong to the declared 21-test Holm family.",
            "",
            "## Supplementary Table S1. Calibration-free balanced accuracy",
            "",
            "Values are participant means with participant-bootstrap 95% confidence intervals. Seed repeats were averaged within participant before summarization.",
            "",
        ]
    )
    rows = []
    for record in main_summary:
        if record["metric"] != "balanced_accuracy":
            continue
        rows.append(
            [
                record["dataset"].title(),
                record["condition"].replace("harmonic_fold", "HarmonicFoldNet").replace("ssvepformer", "Spectral Transformer"),
                record["window_s"],
                record["participants"],
                f"{percent(record['mean'])} [{percent(record['bootstrap_mean_ci95_low'])}, {percent(record['bootstrap_mean_ci95_high'])}]",
            ]
        )
    lines.extend([table(["Dataset", "Method", "Window (s)", "n", "Balanced accuracy, % [95% CI]"], rows), ""])

    lines.extend(["## Supplementary Table S2. Paired main-model contrasts", ""])
    rows = []
    for record in main_contrasts:
        if record["metric"] != "balanced_accuracy":
            continue
        rows.append(
            [
                record["dataset"].title(),
                record["window_s"],
                record["participants"],
                f"{percent(record['mean_difference'])} [{percent(record['bootstrap_difference_ci95_low'])}, {percent(record['bootstrap_difference_ci95_high'])}]",
                number(record["rank_biserial"], 3),
                p_value(record["wilcoxon_p_value"]),
                p_value(record["holm_adjusted_p_value"]),
            ]
        )
    lines.extend(
        [
            table(
                ["Dataset", "Window (s)", "n", "HarmonicFoldNet minus reference, pp [95% CI]", "Rank-biserial", "Raw p", "Holm p"],
                rows,
            ),
            "",
        ]
    )

    lines.extend(["## Supplementary Table S3. Participant-level information transfer rate", ""])
    itr_lookup: dict[tuple[str, str, str], dict[str, str]] = {}
    for record in main_summary:
        if record["metric"] == "itr_bits_per_minute":
            itr_lookup[(record["dataset"], record["condition"], record["window_s"])] = record
    rows = []
    for dataset, entry in config["main"].items():
        for window in entry["display_windows"]:
            left = itr_lookup[(dataset, "harmonic_fold", window)]
            right = itr_lookup[(dataset, "ssvepformer", window)]
            rows.append(
                [
                    entry["dataset_label"],
                    window,
                    f"{number(left['mean'])} [{number(left['bootstrap_mean_ci95_low'])}, {number(left['bootstrap_mean_ci95_high'])}]",
                    f"{number(right['mean'])} [{number(right['bootstrap_mean_ci95_low'])}, {number(right['bootstrap_mean_ci95_high'])}]",
                ]
            )
    lines.extend([table(["Dataset", "Window (s)", "HarmonicFoldNet, bits/min [95% CI]", "Spectral Transformer, bits/min [95% CI]"], rows), ""])

    lines.extend(
        [
            "## Supplementary Table S3b. ITR overhead sensitivity",
            "",
            "Theoretical information transfer rate (ITR) was recomputed per participant and training seed for four assumed non-observation overheads. The table reports the window with the highest participant-mean ITR under each assumption; it shows that the optimal window is not invariant to the overhead model.",
            "",
        ]
    )
    rows = []
    for record in itr_sensitivity_peaks:
        rows.append(
            [
                record["dataset"].title(),
                record["condition"].replace("harmonic_fold", "HarmonicFoldNet").replace("ssvepformer", "SSVEPformer"),
                record["overhead_s"],
                record["peak_window_s"],
                number(record["peak_mean_itr_bits_per_minute"]),
            ]
        )
    lines.extend([table(["Dataset", "Method", "Assumed overhead (s)", "Peak window (s)", "Peak mean ITR (bits/min)"], rows), ""])

    lines.extend(["## Supplementary Table S4. BETA component-ablation accuracy", ""])
    rows = []
    for record in ablation_summary:
        rows.append(
            [
                record["condition"],
                record["window_s"],
                record["participants"],
                f"{percent(record['mean'])} [{percent(record['bootstrap_mean_ci95_low'])}, {percent(record['bootstrap_mean_ci95_high'])}]",
            ]
        )
    lines.extend([table(["Condition", "Window (s)", "n", "Balanced accuracy, % [95% CI]"], rows), ""])

    lines.extend(["## Supplementary Table S5. BETA paired component effects", ""])
    rows = []
    for record in ablation_contrasts:
        rows.append(
            [
                record["right"],
                record["window_s"],
                f"{percent(record['mean_difference'])} [{percent(record['bootstrap_difference_ci95_low'])}, {percent(record['bootstrap_difference_ci95_high'])}]",
                number(record["rank_biserial"], 3),
                p_value(record["holm_adjusted_p_value"]),
            ]
        )
    lines.extend([table(["Removed component", "Window (s)", "Full minus ablation, pp [95% CI]", "Rank-biserial", "Holm p"], rows), ""])

    lines.extend(["## Supplementary Table S6. Harmonic-attention location audit", ""])
    rows = []
    metric_labels = {
        "normalized_entropy": "Normalized entropy",
        "peak_distance_hz": "Peak distance, Hz",
        "peak_within_bias_width_fraction": "Peak within ±0.5 Hz",
        "harmonic_neighborhood_mass": "Harmonic-neighbourhood mass",
    }
    for record in attention_mechanism:
        scale = 100.0 if record["metric"] in {"peak_within_bias_width_fraction", "harmonic_neighborhood_mass"} else 1.0
        rows.append(
            [
                metric_labels[record["metric"]],
                record["window_s"],
                number(str(scale * float(record["with_harmonic_bias_mean"])), 2),
                number(str(scale * float(record["bias_masked_mean"])), 2),
                f"{number(str(scale * float(record['mean_difference'])), 2)} [{number(str(scale * float(record['bootstrap_difference_ci95_low'])), 2)}, {number(str(scale * float(record['bootstrap_difference_ci95_high'])), 2)}]",
                number(record["rank_biserial"], 3),
                p_value(record["holm_adjusted_p_value"]),
            ]
        )
    lines.extend(
        [
            "All rows use the same held-out checkpoints with the fixed bias present and masked. Confidence intervals are participant-bootstrap intervals for the paired difference. One Holm family covers all four metrics and three windows.",
            "",
            table(["Metric", "Window (s)", "Bias", "Masked", "Paired difference [95% CI]", "Rank-biserial", "Holm p"], rows),
            "",
        ]
    )

    lines.extend(
        [
            "## Supplementary Table S7. Calibration-free analytic references",
            "",
            "Fixed harmonic power, CCA, and FBCCA were evaluated on the same causal 6–45 Hz, 250 Hz, eight-channel shards and all participants used by the neural comparisons. They require no participant-specific labels. Values are participant means with participant-bootstrap intervals; these descriptive controls were not included in an additional multiplicity family.",
            "",
        ]
    )
    rows = []
    method_labels = {
        "fixed_harmonic": "Fixed harmonic power",
        "cca": "CCA",
        "fbcca": "FBCCA",
    }
    for record in analytic_baselines:
        if float(record["window_s"]) > (1.5 if record["dataset"] == "beta" else 1.2):
            continue
        rows.append(
            [
                "BETA" if record["dataset"] == "beta" else "Benchmark",
                method_labels[record["condition"]],
                record["window_s"],
                record["participants"],
                f"{percent(record['mean'])} [{percent(record['bootstrap_mean_ci95_low'])}, {percent(record['bootstrap_mean_ci95_high'])}]",
            ]
        )
    lines.extend(
        [
            table(
                ["Dataset", "Method", "Window (s)", "n", "Balanced accuracy, % [95% CI]"],
                rows,
            ),
            "",
        ]
    )

    lines.extend(
        [
            "## Supplementary Table S8. Strong filter-bank Transformer comparison",
            "",
            "The protocol-adapted filter-bank Transformer and HarmonicFoldNet used the same BETA participants, folds, windows and three training seeds. The six paired tests form one Holm family.",
            "",
        ]
    )
    fb_summary = {
        (row["condition"], row["window_s"]): row for row in strong_filter_bank_summary
    }
    rows = []
    for contrast in strong_filter_bank_contrasts:
        window = contrast["window_s"]
        left = fb_summary[("harmonic_fold", window)]
        right = fb_summary[("fb_ssvepformer", window)]
        rows.append(
            [
                window,
                f"{percent(left['mean'])} [{percent(left['bootstrap_mean_ci95_low'])}, {percent(left['bootstrap_mean_ci95_high'])}]",
                f"{percent(right['mean'])} [{percent(right['bootstrap_mean_ci95_low'])}, {percent(right['bootstrap_mean_ci95_high'])}]",
                f"{percent(contrast['mean_difference'])} [{percent(contrast['bootstrap_difference_ci95_low'])}, {percent(contrast['bootstrap_difference_ci95_high'])}]",
                number(contrast["rank_biserial"], 3),
                p_value(contrast["holm_adjusted_p_value"]),
            ]
        )
    lines.extend(
        [
            table(["Window (s)", "HarmonicFoldNet, % [95% CI]", "Filter-bank Transformer, % [95% CI]", "Difference, pp [95% CI]", "Rank-biserial", "Holm p"], rows),
            "",
            "## Supplementary Table S9. MTSNet six-test multiplicity family",
            "",
            "The six registered Benchmark/BETA contrasts form one Holm family. This prevents a separate per-file correction from understating multiplicity.",
            "",
        ]
    )
    rows = []
    for record in mtsnet:
        rows.append(
            [
                record["dataset"].title(),
                record["window_s"],
                record["participants"],
                f"{percent(record['mean_difference'])} [{percent(record['bootstrap_difference_ci95_low'])}, {percent(record['bootstrap_difference_ci95_high'])}]",
                number(record["rank_biserial"], 3),
                p_value(record["wilcoxon_p_value"]),
                p_value(record["holm_adjusted_p_value_six_test_family"]),
            ]
        )
    lines.extend([table(["Dataset", "Window (s)", "n", "Difference, pp [95% CI]", "Rank-biserial", "Raw p", "Holm p"], rows), ""])

    lines.extend(
        [
            "## Supplementary Table S10. Optional participant adaptation",
            "",
            "This is a retrospective cyclic leave-one-block evaluation. For each held-out block, the requested immediately preceding block or blocks were used for calibration, wrapping around at the first block; every block was predicted once. The neural adapter started from a source-trained HarmonicFoldNet checkpoint whose outer-fold training and validation participants excluded the target participant. TRCA, ensemble TRCA, and TDCA were fitted only on the target-user calibration blocks. The comparison therefore equalizes the amount of target-labelled calibration data, not total pretraining supervision, and should not be interpreted as prospective online adaptation.",
            "",
            "TRCA and ensemble TRCA used one spatial component and class-average templates. Ensemble TRCA concatenated the class-specific filters for each class score. TDCA used four harmonics, five delay samples, one spatial component, class-specific sinusoidal-reference projections, and regularized generalized eigendecomposition. All calibrated methods used the same causal 6–45 Hz, 250 Hz, eight-channel shards.",
            "",
        ]
    )
    rows = []
    for record in adaptation_summary:
        rows.append(
            [
                record["condition"],
                record["calibration_blocks"],
                record["window_s"],
                record["participants"],
                f"{percent(record['mean'])} [{percent(record['bootstrap_mean_ci95_low'])}, {percent(record['bootstrap_mean_ci95_high'])}]",
            ]
        )
    lines.extend([table(["Condition", "Calibration blocks", "Window (s)", "n", "Balanced accuracy, % [95% CI]"], rows), ""])

    lines.extend(
        [
            "## Supplementary Table S11. Paired adaptation effects",
            "",
            "TRCA and ensemble TRCA required two calibration blocks in this implementation. TDCA was also evaluated with one block as a diagnostic, yielding a 12-test adapter-versus-classical Holm family; the nine two-block comparisons are shown below, while the complete machine-readable table includes the three one-block TDCA contrasts.",
            "",
        ]
    )
    rows = []
    for record in adaptation_contrasts:
        rows.append(
            [
                record["left"],
                record["right"],
                record["window_s"],
                f"{percent(record['mean_difference'])} [{percent(record['bootstrap_difference_ci95_low'])}, {percent(record['bootstrap_difference_ci95_high'])}]",
                number(record["rank_biserial"], 3),
                p_value(record["holm_adjusted_p_value"]),
            ]
        )
    lines.extend([table(["Left", "Right", "Window (s)", "Paired difference, pp [95% CI]", "Rank-biserial", "Holm p"], rows), ""])

    lines.extend(["## Supplementary Table S12. Directional electrode-condition transfer", ""])
    rows = []
    for record in transfer_summary:
        rows.append(
            [
                record["condition"],
                record["window_s"],
                f"{percent(record['mean'])} [{percent(record['bootstrap_mean_ci95_low'])}, {percent(record['bootstrap_mean_ci95_high'])}]",
            ]
        )
    lines.extend([table(["Train to test", "Window (s)", "Balanced accuracy, % [95% CI]"], rows), ""])

    lines.extend(["## Supplementary Table S13. Paired electrode-transfer effects", ""])
    rows = []
    for record in transfer_contrasts:
        rows.append(
            [
                f"{record['left']} minus {record['right']}",
                record["window_s"],
                f"{percent(record['mean_difference'])} [{percent(record['bootstrap_difference_ci95_low'])}, {percent(record['bootstrap_difference_ci95_high'])}]",
                number(record["rank_biserial"], 3),
                p_value(record["holm_adjusted_p_value"]),
            ]
        )
    lines.extend([table(["Contrast", "Window (s)", "Difference, pp [95% CI]", "Rank-biserial", "Holm p"], rows), ""])

    lines.extend(["## Supplementary Table S14. Deployment measurements", ""])
    rows = []
    deployment_labels = {
        "harmonic_fold_v4_1": "HarmonicFoldNet",
        "ssvepformer": "SSVEPformer",
        "fb_ssvepformer": "FB-SSVEPformer",
        "mtsnet_external_protocol_reconstruction": "MTSNet reconstruction",
    }
    for record in deployment:
        rows.append(
            [
                deployment_labels.get(record["architecture"], record["architecture"]),
                record["window_s"],
                f"{int(record['parameters']):,}",
                number(record["cpu_p50_ms"], 3),
                number(record["cpu_p95_ms"], 3),
                number(record["cuda_p50_ms"], 3),
                number(record["cuda_p95_ms"], 3),
                number(str(float(record["cuda_peak_allocated_bytes"]) / (1024 * 1024)), 2),
            ]
        )
    lines.extend(
        [
            table(
                ["Method", "Window (s)", "Parameters", "CPU median, ms", "CPU p95, ms", "GPU median, ms", "GPU p95, ms", "Peak GPU, MiB"],
                rows,
            ),
            "",
            "These are empirical within-session quantiles from 1,000 batch-one float32 passes after 100 warm-up passes on one laptop, with one timed CPU thread. They do not estimate between-session, power-state, or thermal-state uncertainty. Exact source JSON hashes and the benchmarked checkpoint hash are recorded in the source-data package.",
            "",
            "The fold-equivalence audit covered 20 preserved checkpoints: all 15 BETA fold-seed checkpoints and five Benchmark folds for seed 20260929. At five registered windows, the first trial of every class for every held-out participant yielded 49,000 paired predictions. The folded and training graphs produced zero label disagreements; the maximum absolute logit error was 1.55 × 10^-5. Full checkpoint-window rows are stored in `folding_equivalence.json`.",
            "",
        ]
    )

    lines.extend(
        [
            "## Supplementary Table S15. Comparator implementation provenance",
            "",
            table(
                ["Comparator", "Source anchor", "Local protocol adaptation", "Redistribution"],
                [
                    [
                        "SSVEPformer / FB-SSVEPformer",
                        "Published architecture; public reproduction commit fa21513054c8",
                        "Eight channels; 40 classes; 250 Hz; 0.25 Hz spectral grid; common 6–45 Hz band; three FB branches at 6/14/22 Hz",
                        "Local clean implementation; no third-party source copied",
                    ],
                    [
                        "MTSNet reconstruction",
                        "Upstream model definition commit 890b0a4f93c3",
                        "Common participant folds, windows, optimizer budget, validation selection, channel contract, and one duration-specific model",
                        "Upstream source is not redistributed",
                    ],
                    [
                        "CCA / FBCCA / harmonic power",
                        "Declared analytic definitions in the manuscript and release code",
                        "Same 250 Hz, eight-channel, 6–45 Hz evaluation shards; no participant labels",
                        "Included in the source release",
                    ],
                ],
            ),
            "",
            "Published values obtained under different channel, class, calibration, segmentation, or split contracts were not inserted into the protocol-matched performance tables. DGConformer, SED-xLSTM, and other recent systems remain literature references because a matched local reconstruction was not completed.",
            "",
        ]
    )

    lines.extend(
        [
            "## Supplementary Methods S3. Development record and negative results",
            "",
            "The final architecture was selected after a bounded development sequence. Tested alternatives included increased width, alternative local domains and receptive fields, nested sub-bands, reliability gates, duration conditioning, class-grid alignment, candidate-local mixing, short-window resampling, long-to-short distillation, temporal segment changes, and explicit phase dynamics. These experiments are documented in the repository and are not counted as independent confirmatory tests. They are reported to expose the search history and reduce selective reporting. The final evidence tables use the frozen v4.1 architecture only.",
            "",
            "## Supplementary Methods S4. Reproducibility manifest",
            "",
        ]
    )
    manifest_paths = [
        PAPER / "SUBMISSION_EVIDENCE_CONFIG.json",
        SOURCE / "submission_evidence.json",
        SOURCE / "main_participant_metrics_by_seed.csv",
        SOURCE / "main_participant_metrics_seed_averaged.csv",
        SOURCE / "itr_overhead_sensitivity.csv",
        SOURCE / "itr_overhead_sensitivity_peaks.csv",
        SOURCE / "ablation_participant_balanced_accuracy.csv",
        SOURCE / "attention_mechanism_participant_metrics.csv",
        SOURCE / "attention_mechanism_summary.csv",
        SOURCE / "analytic_baseline_participant.csv",
        SOURCE / "analytic_baseline_summary.csv",
        SOURCE / "strong_filter_bank_participant_by_seed.csv",
        SOURCE / "strong_filter_bank_participant_seed_averaged.csv",
        SOURCE / "strong_filter_bank_summary.csv",
        SOURCE / "strong_filter_bank_contrasts.csv",
        SOURCE / "adaptation_participant_balanced_accuracy.csv",
        SOURCE / "adaptation_summary.csv",
        SOURCE / "adaptation_contrasts.csv",
        SOURCE / "electrode_transfer_participant_by_seed.csv",
        SOURCE / "electrode_transfer_participant_seed_averaged.csv",
        SOURCE / "electrode_transfer_summary.csv",
        SOURCE / "electrode_transfer_contrasts.csv",
        SOURCE / "mtsnet_contrasts_corrected.csv",
        SOURCE / "mtsnet_participant_by_seed.csv",
        SOURCE / "mtsnet_participant_seed_averaged.csv",
        SOURCE / "mtsnet_summary.csv",
        SOURCE / "folding_equivalence.json",
        SOURCE / "split_manifest.csv",
        SOURCE / "split_manifest_audit.json",
        SOURCE / "deployment_measurements.csv",
        SOURCE / "deployment_checkpoint_manifest.csv",
    ]
    manifest_rows = [[path.relative_to(ROOT).as_posix(), path.stat().st_size, sha256(path)] for path in manifest_paths]
    lines.extend(
        [
            table(["File", "Bytes", "SHA-256"], manifest_rows),
            "",
            "The hash values identify the exact source-data files used to generate this supplement. Regenerating the evidence should reproduce the numerical tables; bootstrap intervals are deterministic under seed 20261003.",
            "",
        ]
    )

    output = PAPER / "SUPPLEMENTARY_INFORMATION.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "ok", "output": output.as_posix(), "tables": 16}, ensure_ascii=False))


if __name__ == "__main__":
    main()
