from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .harmonic_fold import (
    TEMPORAL_FUSION_ARCHITECTURE_REVISION,
    FUSION_MODES,
    HARMONIC_FOLD_MODES,
    SPECTRAL_FUSION_ARCHITECTURE_REVISION,
    HARMONIC_FOLD_V4_ARCHITECTURE_REVISION,
    HARMONIC_FOLD_V41_ARCHITECTURE_REVISION,
    HARMONIC_FOLD_V42_ARCHITECTURE_REVISION,
    HARMONIC_FOLD_V43_ARCHITECTURE_REVISION,
    HARMONIC_FOLD_V44_ARCHITECTURE_REVISION,
    HARMONIC_FOLD_V45_ARCHITECTURE_REVISION,
    HARMONIC_FOLD_V46_ARCHITECTURE_REVISION,
    HARMONIC_FOLD_V47_ARCHITECTURE_REVISION,
    SpectralFusionDecoder,
    HarmonicFoldNet,
    TemporalFusionDecoder,
)
from .model import LegacyFusionNet
from .paper_data import class_frequencies, class_phases, grouped_kfold_split, load_subjects_window, loso_split
from .paper_metrics import atomic_write_json, classification_metrics
from .reference_models import EEGNetAdaptive, FBSSVEPFormerReference, SSVEPFormerReference


LEGACY_MODES = ("full", "evidence_only", "fusion_only", "time_only", "frequency_only")
MODES = tuple(dict.fromkeys((*LEGACY_MODES, *FUSION_MODES, *HARMONIC_FOLD_MODES)))
FOLD_ARCHITECTURES = {
    "legacy_fusion", "temporal_fusion", "spectral_fusion", "harmonic_fold_v4",
    "harmonic_fold_v4_1",
    "harmonic_fold_v4_2",
    "harmonic_fold_v4_3",
    "harmonic_fold_v4_4",
    "harmonic_fold_v4_5",
    "harmonic_fold_v4_6",
    "harmonic_fold_v4_7",
}


def parse_trial_filters(values: list[str] | None) -> dict[str, list[str]] | None:
    """Parse repeatable metadata filters such as ``electrode=dry,wet``."""
    if values is None:
        return None
    filters: dict[str, list[str]] = {}
    for value in values:
        key, separator, accepted = value.partition("=")
        key = key.strip()
        choices = [item.strip() for item in accepted.split(",") if item.strip()]
        if not separator or not key or not choices:
            raise ValueError(
                f"invalid trial filter {value!r}; expected metadata=value[,value]"
            )
        filters.setdefault(key, []).extend(choices)
    return filters


def architecture_config_key(architecture: str) -> str:
    if architecture == "harmonic_fold_v4_5":
        return "harmonic_fold_v4_5_model"
    if architecture == "harmonic_fold_v4_6":
        return "harmonic_fold_v4_6_model"
    if architecture == "harmonic_fold_v4_7":
        return "harmonic_fold_v4_7_model"
    if architecture == "harmonic_fold_v4_4":
        return "harmonic_fold_v4_4_model"
    if architecture == "harmonic_fold_v4_3":
        return "harmonic_fold_v4_3_model"
    if architecture == "harmonic_fold_v4_2":
        return "harmonic_fold_v4_2_model"
    if architecture == "harmonic_fold_v4_1":
        return "harmonic_fold_v4_1_model"
    if architecture == "harmonic_fold_v4":
        return "harmonic_fold_v4_model"
    if architecture == "spectral_fusion":
        return "spectral_fusion_model"
    if architecture == "temporal_fusion":
        return "harmonic_fold_model"
    return "model"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def distillation_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("distillation temperature must be positive")
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature * temperature)


def crop_tensor(x: torch.Tensor, seconds: float, sample_rate: int) -> torch.Tensor:
    samples = int(round(seconds * sample_rate))
    if samples <= 0 or samples > x.shape[-1]:
        raise ValueError(f"cannot crop {seconds}s from {x.shape[-1]} samples")
    x = x[..., :samples]
    return (x - x.mean(dim=-1, keepdim=True)) / x.std(
        dim=-1, keepdim=True, correction=0,
    ).clamp_min(1e-6)


def configure_mode(model: LegacyFusionNet, mode: str) -> list[nn.Parameter]:
    if mode not in LEGACY_MODES:
        raise ValueError(mode)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules: list[nn.Module] = []
    if mode in {"full", "fusion_only"}:
        modules.extend([model.time_encoder, model.frequency_encoder, model.fusion, model.head])
    elif mode == "time_only":
        modules.extend([model.time_encoder, model.fusion, model.head])
    elif mode == "frequency_only":
        modules.extend([model.frequency_encoder, model.fusion, model.head])
    if mode in {"full", "evidence_only"}:
        if model.evidence_residual is None or model.evidence_scale is None:
            raise RuntimeError("candidate-frequency evidence is unavailable")
        modules.append(model.evidence_residual)
        model.evidence_scale.requires_grad_(True)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def configure_fold_mode(
    model: TemporalFusionDecoder | SpectralFusionDecoder | HarmonicFoldNet,
    mode: str,
) -> list[nn.Parameter]:
    valid_modes = (
        set(HARMONIC_FOLD_MODES)
        if isinstance(model, HarmonicFoldNet)
        else set(FUSION_MODES)
    )
    if mode not in valid_modes:
        raise ValueError(mode)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if mode == "no_evidence":
        model.prior_scale.requires_grad_(False)
    elif mode == "no_attention":
        if isinstance(model, HarmonicFoldNet):
            modules = (model.fusion_attention, model.candidate_attention)
        else:
            global_attention = getattr(
                model, "temporal_attention", getattr(model, "spectral_attention", None),
            )
            modules = (
                global_attention, model.cross_norm, model.cross_attention,
                model.candidate_attention,
            )
        for module in modules:
            if module is None:
                continue
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    elif mode == "no_cross_attention":
        if not isinstance(model, HarmonicFoldNet):
            raise ValueError("no_cross_attention is supported only by spectral v4")
        for parameter in model.fusion_attention.parameters():
            parameter.requires_grad_(False)
    elif mode == "no_candidate_attention":
        if not isinstance(model, HarmonicFoldNet):
            raise ValueError("no_candidate_attention is supported only by spectral v4")
        for parameter in model.candidate_attention.parameters():
            parameter.requires_grad_(False)
    elif mode == "no_local":
        for module in (
            model.local_stage1, model.local_stage2,
            getattr(model, "input_local_stage1", ()),
            getattr(model, "input_local_stage2", ()),
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


@torch.inference_mode()
def infer_logits(
    model: nn.Module, x: np.ndarray, *, architecture: str, mode: str, window: float,
    sample_rate: int, batch_size: int, device: torch.device,
) -> np.ndarray:
    model.eval()
    output = []
    for begin in range(0, len(x), batch_size):
        batch = torch.from_numpy(x[begin:begin + batch_size]).to(device, non_blocking=True)
        batch = crop_tensor(batch, window, sample_rate)
        logits = (
            model.forward_mode(batch, mode)
            if architecture in FOLD_ARCHITECTURES
            else model(batch)
        )
        output.append(logits.float().cpu().numpy())
    return np.concatenate(output)


def split_metrics(
    y: np.ndarray, subject_ids: np.ndarray, logits: np.ndarray, *, classes: int,
    window: float, cue_seconds: float,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    overall = classification_metrics(
        y, logits, classes=classes, window_seconds=window, cue_seconds=cue_seconds,
    )
    per_subject = {}
    for subject in sorted(set(map(str, subject_ids)), key=lambda value: (len(value), value)):
        mask = subject_ids.astype(str) == subject
        per_subject[subject] = classification_metrics(
            y[mask], logits[mask], classes=classes,
            window_seconds=window, cue_seconds=cue_seconds,
        )
    return overall, per_subject


def run(args: argparse.Namespace) -> dict[str, object]:
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    if args.evidence_mode is not None:
        config["model"]["evidence_mode"] = args.evidence_mode
    if args.attention_depth is not None:
        key = architecture_config_key(args.architecture)
        if args.architecture in {
            "harmonic_fold_v4", "harmonic_fold_v4_1", "harmonic_fold_v4_2",
            "harmonic_fold_v4_3", "harmonic_fold_v4_4", "harmonic_fold_v4_5",
            "harmonic_fold_v4_6",
            "harmonic_fold_v4_7",
        }:
            raise ValueError("v4 has one fixed late-attention stage; attention_depth is not configurable")
        config[key]["attention_depth"] = int(args.attention_depth)
    if args.dropout is not None:
        key = architecture_config_key(args.architecture)
        config[key]["dropout"] = float(args.dropout)
    if getattr(args, "model_width", None) is not None:
        if args.architecture not in FOLD_ARCHITECTURES:
            raise ValueError("model width override is supported only by HarmonicFold architectures")
        width = int(args.model_width)
        if width <= 0:
            raise ValueError("model width must be positive")
        config[architecture_config_key(args.architecture)]["width"] = width
    if getattr(args, "temporal_segments", None) is not None:
        if args.architecture not in {
            "harmonic_fold_v4", "harmonic_fold_v4_1", "harmonic_fold_v4_2",
            "harmonic_fold_v4_3", "harmonic_fold_v4_4", "harmonic_fold_v4_5",
            "harmonic_fold_v4_6",
            "harmonic_fold_v4_7",
        }:
            raise ValueError("temporal_segments is supported only by spectral v4 architectures")
        config[architecture_config_key(args.architecture)]["temporal_segments"] = int(
            args.temporal_segments
        )
    if getattr(args, "temporal_phase_dynamics", False):
        if args.architecture not in {
            "harmonic_fold_v4", "harmonic_fold_v4_1", "harmonic_fold_v4_2",
            "harmonic_fold_v4_3", "harmonic_fold_v4_4", "harmonic_fold_v4_5",
            "harmonic_fold_v4_6", "harmonic_fold_v4_7",
        }:
            raise ValueError(
                "temporal phase dynamics are supported only by spectral v4 architectures"
            )
        config[architecture_config_key(args.architecture)][
            "temporal_phase_dynamics"
        ] = True
    if getattr(args, "align_spectral_grid_to_classes", False):
        if args.architecture not in {
            "harmonic_fold_v4", "harmonic_fold_v4_1", "harmonic_fold_v4_2",
            "harmonic_fold_v4_3", "harmonic_fold_v4_4", "harmonic_fold_v4_5",
            "harmonic_fold_v4_6",
            "harmonic_fold_v4_7",
        }:
            raise ValueError("class-grid alignment is supported only by spectral v4 architectures")
        config[architecture_config_key(args.architecture)][
            "align_spectral_grid_to_classes"
        ] = True
    if getattr(args, "duration_conditioning", False):
        if args.architecture not in {
            "harmonic_fold_v4", "harmonic_fold_v4_1", "harmonic_fold_v4_2",
            "harmonic_fold_v4_3", "harmonic_fold_v4_4", "harmonic_fold_v4_5",
            "harmonic_fold_v4_6",
        }:
            raise ValueError("duration conditioning is supported only by spectral v4 architectures")
        config[architecture_config_key(args.architecture)]["duration_conditioning"] = True
    if getattr(args, "candidate_local_mixing", False):
        if args.architecture not in {
            "harmonic_fold_v4", "harmonic_fold_v4_1", "harmonic_fold_v4_2",
            "harmonic_fold_v4_3", "harmonic_fold_v4_4", "harmonic_fold_v4_5",
            "harmonic_fold_v4_6", "harmonic_fold_v4_7",
        }:
            raise ValueError("candidate local mixing is supported only by spectral v4 architectures")
        config[architecture_config_key(args.architecture)]["candidate_local_mixing"] = True
        placement = getattr(args, "candidate_local_mixing_placement", None)
        if placement is not None:
            config[architecture_config_key(args.architecture)][
                "candidate_local_mixing_placement"
            ] = placement
    dataset_cfg = config["datasets"][args.dataset]
    seed = int(args.seed)
    split_seed = int(config.get("split_seed", 20260929))
    seed_all(seed)
    sample_rate = int(config["sample_rate"])
    windows = [float(value) for value in dataset_cfg["windows"]]
    train_windows = [
        float(value) for value in (
            args.train_windows
            if getattr(args, "train_windows", None) is not None
            else dataset_cfg.get("train_windows", windows)
        )
    ]
    if not train_windows:
        raise ValueError("train_windows cannot be empty")
    invalid_train_windows = [
        value for value in train_windows
        if value <= 0.0 or value > max(windows)
    ]
    if invalid_train_windows:
        raise ValueError(f"invalid train windows: {invalid_train_windows}")
    selection_windows = [
        float(value) for value in (
            args.selection_windows
            if getattr(args, "selection_windows", None) is not None
            else dataset_cfg["selection_windows"]
        )
    ]
    if not selection_windows:
        raise ValueError("selection_windows cannot be empty")
    invalid_selection_windows = [
        value for value in selection_windows
        if value <= 0.0 or value > max(windows)
    ]
    if invalid_selection_windows:
        raise ValueError(f"invalid selection windows: {invalid_selection_windows}")
    max_window = max(windows)
    teacher_checkpoint_arg = getattr(args, "teacher_checkpoint", None)
    distillation_weight = float(getattr(args, "distillation_weight", 0.0))
    distillation_temperature = float(getattr(args, "distillation_temperature", 2.0))
    teacher_window = float(getattr(args, "teacher_window", 1.2))
    distill_windows = [
        float(value) for value in (getattr(args, "distill_windows", None) or [])
    ]
    if distillation_weight < 0.0:
        raise ValueError("distillation_weight cannot be negative")
    if teacher_checkpoint_arg is None and distillation_weight > 0.0:
        raise ValueError("positive distillation_weight requires --teacher-checkpoint")
    if teacher_checkpoint_arg is not None and distillation_weight <= 0.0:
        raise ValueError("--teacher-checkpoint requires a positive distillation_weight")
    if teacher_checkpoint_arg is not None and args.architecture not in FOLD_ARCHITECTURES:
        raise ValueError("long-window distillation currently supports HarmonicFold architectures only")
    if teacher_checkpoint_arg is not None and args.mode != "full":
        raise ValueError("long-window distillation requires mode=full")
    if teacher_window <= 0.0 or teacher_window > max_window:
        raise ValueError(f"invalid teacher window: {teacher_window}")
    invalid_distill_windows = [
        value for value in distill_windows
        if value <= 0.0 or value > max_window or value not in train_windows
    ]
    if invalid_distill_windows:
        raise ValueError(f"invalid distillation windows: {invalid_distill_windows}")
    if teacher_checkpoint_arg is not None and not distill_windows:
        raise ValueError("--teacher-checkpoint requires at least one --distill-windows value")
    subjects = list(map(str, dataset_cfg["subjects"]))
    if args.test_subject is not None and args.fold_index is not None:
        raise ValueError("choose either LOSO or grouped k-fold, not both")
    if args.test_subject is None and args.fold_index is None:
        train_subjects = list(map(str, dataset_cfg["train_subjects"]))
        validation_subjects = list(map(str, dataset_cfg["validation_subjects"]))
        test_subjects = list(map(str, dataset_cfg["test_subjects"]))
        split_name = "development"
    elif args.test_subject is not None:
        train_subjects, validation_subjects, test_subjects = loso_split(
            subjects, str(args.test_subject),
            validation_count=int(dataset_cfg["validation_count_loso"]), split_seed=split_seed,
        )
        split_name = f"loso-{args.test_subject}"
    else:
        train_subjects, validation_subjects, test_subjects = grouped_kfold_split(
            subjects, fold_index=int(args.fold_index), fold_count=int(args.fold_count),
            validation_count=int(dataset_cfg["validation_count_loso"]), split_seed=split_seed,
        )
        split_name = f"group-{args.fold_count}-fold-{args.fold_index}"
    shard_dir = Path(dataset_cfg["shard_dir"])
    channels = list(map(str, dataset_cfg["channels"]))
    configured_filters = dataset_cfg.get("trial_filters")
    trial_filters = {
        "train": parse_trial_filters(args.train_trial_filter)
        if args.train_trial_filter is not None else configured_filters,
        "validation": parse_trial_filters(args.validation_trial_filter)
        if args.validation_trial_filter is not None else configured_filters,
        "test": parse_trial_filters(args.test_trial_filter)
        if args.test_trial_filter is not None else configured_filters,
    }
    load_options = dict(
        window_seconds=max_window, sample_rate=sample_rate, channels=channels,
    )
    train_x, train_y, _ = load_subjects_window(
        shard_dir, train_subjects,
        **load_options, trial_filters=trial_filters["train"],
    )
    validation_x, validation_y, _ = load_subjects_window(
        shard_dir, validation_subjects,
        **load_options, trial_filters=trial_filters["validation"],
    )
    test_x, test_y, test_subject_ids = load_subjects_window(
        shard_dir, test_subjects,
        **load_options, trial_filters=trial_filters["test"],
    )
    classes = int(dataset_cfg["classes"])
    frequencies = class_frequencies(shard_dir, subjects, classes=classes)
    phases = class_phases(shard_dir, subjects, classes=classes)
    if args.architecture == "legacy_fusion":
        model: nn.Module = LegacyFusionNet(
            channels=len(channels), classes=classes, sample_rate=sample_rate,
            class_frequencies=frequencies, class_phases=phases, **config["model"],
        )
    elif args.architecture == "temporal_fusion":
        if args.mode not in FUSION_MODES:
            raise ValueError("HarmonicFold architecture requires a HarmonicFold ablation mode")
        model = TemporalFusionDecoder(
            channels=len(channels), sample_rate=sample_rate,
            class_frequencies=frequencies, class_phases=phases,
            **config["harmonic_fold_model"],
        )
    elif args.architecture == "spectral_fusion":
        if args.mode not in FUSION_MODES:
            raise ValueError("HarmonicFold architecture requires a HarmonicFold ablation mode")
        model = SpectralFusionDecoder(
            channels=len(channels), sample_rate=sample_rate,
            class_frequencies=frequencies, class_phases=phases,
            **config["spectral_fusion_model"],
        )
    elif args.architecture == "harmonic_fold_v4":
        if args.mode not in set(HARMONIC_FOLD_MODES):
            raise ValueError("HarmonicFold spectral v4 supports full/no_attention/no_local")
        model = HarmonicFoldNet(
            channels=len(channels), sample_rate=sample_rate,
            class_frequencies=frequencies, class_phases=phases,
            **config["harmonic_fold_v4_model"],
        )
    elif args.architecture == "harmonic_fold_v4_1":
        if args.mode not in set(HARMONIC_FOLD_MODES):
            raise ValueError("HarmonicFold spectral v4.1 supports full/no_attention/no_local")
        model = HarmonicFoldNet(
            channels=len(channels), sample_rate=sample_rate,
            class_frequencies=frequencies, class_phases=phases,
            **config["harmonic_fold_v4_1_model"],
        )
    elif args.architecture == "harmonic_fold_v4_2":
        if args.mode not in set(HARMONIC_FOLD_MODES):
            raise ValueError("HarmonicFold spectral v4.2 supports full/no_attention/no_local")
        model = HarmonicFoldNet(
            channels=len(channels), sample_rate=sample_rate,
            class_frequencies=frequencies, class_phases=phases,
            **config["harmonic_fold_v4_2_model"],
        )
    elif args.architecture == "harmonic_fold_v4_3":
        if args.mode not in set(HARMONIC_FOLD_MODES):
            raise ValueError("HarmonicFold spectral v4.3 supports full/no_attention/no_local")
        model = HarmonicFoldNet(
            channels=len(channels), sample_rate=sample_rate,
            class_frequencies=frequencies, class_phases=phases,
            **config["harmonic_fold_v4_3_model"],
        )
    elif args.architecture == "harmonic_fold_v4_4":
        if args.mode not in set(HARMONIC_FOLD_MODES):
            raise ValueError("HarmonicFold spectral v4.4 supports full/no_attention/no_local")
        model = HarmonicFoldNet(
            channels=len(channels), sample_rate=sample_rate,
            class_frequencies=frequencies, class_phases=phases,
            **config["harmonic_fold_v4_4_model"],
        )
    elif args.architecture == "harmonic_fold_v4_5":
        if args.mode not in set(HARMONIC_FOLD_MODES):
            raise ValueError("HarmonicFold spectral v4.5 supports full/no_attention/no_local")
        model = HarmonicFoldNet(
            channels=len(channels), sample_rate=sample_rate,
            class_frequencies=frequencies, class_phases=phases,
            **config["harmonic_fold_v4_5_model"],
        )
    elif args.architecture == "harmonic_fold_v4_6":
        if args.mode not in set(HARMONIC_FOLD_MODES):
            raise ValueError("HarmonicFold spectral v4.6 supports full/no_attention/no_local")
        model = HarmonicFoldNet(
            channels=len(channels), sample_rate=sample_rate,
            class_frequencies=frequencies, class_phases=phases,
            **config["harmonic_fold_v4_6_model"],
        )
    elif args.architecture == "harmonic_fold_v4_7":
        if args.mode not in set(HARMONIC_FOLD_MODES):
            raise ValueError("HarmonicFold spectral v4.7 supports registered v4 modes")
        model = HarmonicFoldNet(
            channels=len(channels), sample_rate=sample_rate,
            class_frequencies=frequencies, class_phases=phases,
            **config["harmonic_fold_v4_7_model"],
        )
    elif args.architecture == "eegnet":
        if args.mode != "full":
            raise ValueError("reference architectures support mode=full only")
        model = EEGNetAdaptive(channels=len(channels), classes=classes)
    elif args.architecture == "ssvepformer":
        if args.mode != "full":
            raise ValueError("reference architectures support mode=full only")
        model = SSVEPFormerReference(
            channels=len(channels), classes=classes, sample_rate=sample_rate,
        )
    elif args.architecture == "fb_ssvepformer":
        if args.mode != "full":
            raise ValueError("reference architectures support mode=full only")
        model = FBSSVEPFormerReference(
            channels=len(channels), classes=classes, sample_rate=sample_rate,
        )
    else:
        raise ValueError(args.architecture)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if isinstance(model, LegacyFusionNet):
        trainable = configure_mode(model, args.mode)
    elif isinstance(model, (TemporalFusionDecoder, SpectralFusionDecoder, HarmonicFoldNet)):
        trainable = configure_fold_mode(model, args.mode)
    else:
        trainable = list(model.parameters())
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    teacher: nn.Module | None = None
    teacher_checkpoint_path: Path | None = None
    if teacher_checkpoint_arg is not None:
        teacher_checkpoint_path = Path(teacher_checkpoint_arg).resolve()
        checkpoint = torch.load(teacher_checkpoint_path, map_location="cpu", weights_only=True)
        teacher_result = checkpoint.get("result", {})
        if teacher_result.get("architecture") != args.architecture:
            raise ValueError(
                "teacher architecture mismatch: "
                f"{teacher_result.get('architecture')!r} != {args.architecture!r}"
            )
        if teacher_result.get("mode") != args.mode:
            raise ValueError(
                f"teacher mode mismatch: {teacher_result.get('mode')!r} != {args.mode!r}"
            )
        student_model_config = config[architecture_config_key(args.architecture)]
        if teacher_result.get("model_config") != student_model_config:
            raise ValueError("teacher model configuration does not match the student")
        teacher = copy.deepcopy(model)
        teacher.load_state_dict(checkpoint["state_dict"], strict=True)
        teacher.requires_grad_(False)
        teacher.eval()
    train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_dataset, batch_size=int(config["batch_size"]), shuffle=True,
        num_workers=0, pin_memory=device.type == "cuda", generator=generator,
    )
    if args.architecture in {"ssvepformer", "fb_ssvepformer"}:
        optimizer = torch.optim.SGD(trainable, lr=0.001, momentum=0.9, weight_decay=0.001)
    elif args.architecture == "eegnet":
        optimizer = torch.optim.Adam(trainable, lr=0.001)
    else:
        optimizer = torch.optim.AdamW(
            trainable, lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]),
        )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=float(config["label_smoothing"]))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    epochs = int(args.epochs or config["epochs"])
    patience_limit = int(config["early_stopping_patience"])
    schedule_rng = np.random.default_rng(seed)
    schedule: list[float] = []
    while len(schedule) < epochs:
        schedule.extend(schedule_rng.permutation(train_windows).tolist())
    best_score = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    patience = 0
    history = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_window = float(schedule[epoch - 1])
        model.train()
        losses = []
        distillation_losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            full_batch_x = batch_x
            batch_x = crop_tensor(full_batch_x, train_window, sample_rate)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                if isinstance(model, FBSSVEPFormerReference):
                    branch_logits = model.subnetwork_logits(batch_x)
                    logits = model.fusion(branch_logits).squeeze(1)
                    # The paper stabilizes each subnetwork before learning the
                    # final fusion.  Joint auxiliary supervision is the
                    # multi-window equivalent: all three independent branches
                    # remain directly class-discriminative while the learned
                    # fusion is optimized on the identical minibatches.
                    branch_loss = torch.stack([
                        loss_fn(branch_logits[:, index], batch_y)
                        for index in range(branch_logits.shape[1])
                    ]).mean()
                    loss = loss_fn(logits, batch_y) + branch_loss
                else:
                    logits = (
                        model.forward_mode(batch_x, args.mode)
                        if isinstance(
                            model, (
                                LegacyFusionNet, TemporalFusionDecoder,
                                SpectralFusionDecoder, HarmonicFoldNet,
                            )
                        ) else model(batch_x)
                    )
                    loss = loss_fn(logits, batch_y)
                    if teacher is not None and train_window in distill_windows:
                        with torch.no_grad():
                            teacher_x = crop_tensor(full_batch_x, teacher_window, sample_rate)
                            teacher_logits = teacher.forward_mode(teacher_x, args.mode)
                        distill_loss = distillation_kl_loss(
                            logits, teacher_logits, temperature=distillation_temperature,
                        )
                        loss = loss + distillation_weight * distill_loss
                        distillation_losses.append(float(distill_loss.detach().cpu()))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        validation_scores = {}
        for window in selection_windows:
            logits = infer_logits(
                model, validation_x, architecture=args.architecture, mode=args.mode, window=window,
                sample_rate=sample_rate, batch_size=int(config["eval_batch_size"]), device=device,
            )
            validation_scores[str(window)] = float(
                balanced_accuracy_score(validation_y, logits.argmax(axis=1))
            )
        selection_score = float(np.mean(list(validation_scores.values())))
        row = {
            "epoch": epoch, "train_window_seconds": train_window,
            "loss": float(np.mean(losses)), "selection_score": selection_score,
            "distillation_loss": (
                float(np.mean(distillation_losses)) if distillation_losses else None
            ),
            "validation_balanced_accuracy": validation_scores,
            "seconds": round(time.perf_counter() - started, 3),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if selection_score > best_score + 1e-6:
            best_score = selection_score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= patience_limit:
                break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    results_by_window = {}
    prediction_payload: dict[str, np.ndarray] = {"y": test_y, "subject": test_subject_ids}
    for window in windows:
        logits = infer_logits(
            model, test_x, architecture=args.architecture, mode=args.mode, window=window, sample_rate=sample_rate,
            batch_size=int(config["eval_batch_size"]), device=device,
        )
        overall, per_subject = split_metrics(
            test_y, test_subject_ids, logits, classes=classes, window=window,
            cue_seconds=float(config["cue_seconds_for_itr"]),
        )
        results_by_window[str(window)] = {"overall": overall, "per_subject": per_subject}
        prediction_payload[f"logits_{window:g}s"] = logits.astype(np.float32)
    result = {
        "status": "complete", "protocol_version": int(
            config["harmonic_fold_v4_5_protocol_version"]
            if args.architecture == "harmonic_fold_v4_5"
            else config["harmonic_fold_v4_6_protocol_version"]
            if args.architecture == "harmonic_fold_v4_6"
            else config["harmonic_fold_v4_7_protocol_version"]
            if args.architecture == "harmonic_fold_v4_7"
            else config["harmonic_fold_v4_4_protocol_version"]
            if args.architecture == "harmonic_fold_v4_4"
            else config["harmonic_fold_v4_3_protocol_version"]
            if args.architecture == "harmonic_fold_v4_3"
            else config["harmonic_fold_v4_2_protocol_version"]
            if args.architecture == "harmonic_fold_v4_2"
            else config["harmonic_fold_v4_1_protocol_version"]
            if args.architecture == "harmonic_fold_v4_1"
            else config["harmonic_fold_v4_protocol_version"]
            if args.architecture == "harmonic_fold_v4"
            else config["spectral_fusion_protocol_version"]
            if args.architecture == "spectral_fusion"
            else config["harmonic_fold_protocol_version"]
            if args.architecture == "temporal_fusion"
            else config["protocol_version"]
        ),
        "dataset": args.dataset, "split": split_name, "architecture": args.architecture,
        "architecture_revision": (
            TEMPORAL_FUSION_ARCHITECTURE_REVISION if args.architecture == "temporal_fusion"
            else HARMONIC_FOLD_V45_ARCHITECTURE_REVISION
            if args.architecture == "harmonic_fold_v4_5"
            else HARMONIC_FOLD_V46_ARCHITECTURE_REVISION
            if args.architecture == "harmonic_fold_v4_6"
            else HARMONIC_FOLD_V47_ARCHITECTURE_REVISION
            if args.architecture == "harmonic_fold_v4_7"
            else HARMONIC_FOLD_V44_ARCHITECTURE_REVISION
            if args.architecture == "harmonic_fold_v4_4"
            else HARMONIC_FOLD_V43_ARCHITECTURE_REVISION
            if args.architecture == "harmonic_fold_v4_3"
            else HARMONIC_FOLD_V42_ARCHITECTURE_REVISION
            if args.architecture == "harmonic_fold_v4_2"
            else HARMONIC_FOLD_V41_ARCHITECTURE_REVISION
            if args.architecture == "harmonic_fold_v4_1"
            else HARMONIC_FOLD_V4_ARCHITECTURE_REVISION
            if args.architecture == "harmonic_fold_v4"
            else SPECTRAL_FUSION_ARCHITECTURE_REVISION
            if args.architecture == "spectral_fusion" else None
        ),
        "mode": args.mode, "seed": seed, "split_seed": split_seed,
        "subjects": {"train": train_subjects, "validation": validation_subjects, "test": test_subjects},
        "sample_counts": {"train": len(train_y), "validation": len(validation_y), "test": len(test_y)},
        "trial_filters": trial_filters,
        "channels": channels, "windows": windows, "selection_windows": selection_windows,
        "train_windows": train_windows,
        "distillation": {
            "enabled": teacher is not None,
            "teacher_checkpoint": str(teacher_checkpoint_path) if teacher_checkpoint_path else None,
            "teacher_window_seconds": teacher_window if teacher is not None else None,
            "student_windows_seconds": distill_windows,
            "weight": distillation_weight,
            "temperature": distillation_temperature,
            "inference_graph_unchanged": True,
        },
        "model_config": copy.deepcopy(
            config[architecture_config_key(args.architecture)]
        ),
        "model_parameters": total_parameters, "trainable_parameters": sum(p.numel() for p in trainable),
        "device": str(device), "best_epoch": best_epoch, "best_selection_score": best_score,
        "history": history, "test": results_by_window,
    }
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_dir / "result.json", result)
    temporary_predictions = run_dir / ".predictions.tmp.npz"
    np.savez_compressed(temporary_predictions, **prediction_payload)
    temporary_predictions.replace(run_dir / "predictions.npz")
    if args.save_checkpoint:
        temporary_checkpoint = run_dir / ".model.tmp.pt"
        torch.save({"state_dict": best_state, "config": config, "result": result}, temporary_checkpoint)
        temporary_checkpoint.replace(run_dir / "model.pt")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_multidataset.yaml")
    parser.add_argument("--dataset", required=True, choices=("kim2025", "benchmark", "beta", "wearable"))
    parser.add_argument("--mode", default="full", choices=MODES)
    parser.add_argument(
        "--architecture", default="legacy_fusion",
        choices=(
            "legacy_fusion", "temporal_fusion", "spectral_fusion",
            "harmonic_fold_v4", "harmonic_fold_v4_1",
            "harmonic_fold_v4_2", "harmonic_fold_v4_3",
            "harmonic_fold_v4_4", "harmonic_fold_v4_5",
            "harmonic_fold_v4_6",
            "harmonic_fold_v4_7",
            "eegnet", "ssvepformer", "fb_ssvepformer",
        ),
    )
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--test-subject")
    parser.add_argument("--fold-index", type=int)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--train-windows", nargs="+", type=float,
        help="Optional repeated training-window schedule; duplicates act as weights.",
    )
    parser.add_argument(
        "--selection-windows", nargs="+", type=float,
        help="Optional validation windows used only for early-stopping selection.",
    )
    parser.add_argument(
        "--train-trial-filter", action="append",
        help="Repeatable metadata=value[,value] filter for training trials.",
    )
    parser.add_argument(
        "--validation-trial-filter", action="append",
        help="Repeatable metadata=value[,value] filter for validation trials.",
    )
    parser.add_argument(
        "--test-trial-filter", action="append",
        help="Repeatable metadata=value[,value] filter for test trials.",
    )
    parser.add_argument("--device")
    parser.add_argument("--evidence-mode", choices=("power", "cca_phase", "phase_demod"))
    parser.add_argument("--attention-depth", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--model-width", type=int)
    parser.add_argument("--temporal-segments", type=int)
    parser.add_argument("--temporal-phase-dynamics", action="store_true")
    parser.add_argument("--align-spectral-grid-to-classes", action="store_true")
    parser.add_argument("--duration-conditioning", action="store_true")
    parser.add_argument("--candidate-local-mixing", action="store_true")
    parser.add_argument(
        "--candidate-local-mixing-placement",
        choices=("pre_cross", "post_cross"),
    )
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--teacher-window", type=float, default=1.2)
    parser.add_argument("--distill-windows", nargs="+", type=float)
    parser.add_argument("--distillation-weight", type=float, default=0.0)
    parser.add_argument("--distillation-temperature", type=float, default=2.0)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args()
    result_path = Path(args.run_dir) / "result.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            print(json.dumps({"status": "skipped", "reason": "complete", "path": str(result_path)}))
            return
    run(args)


if __name__ == "__main__":
    main()
