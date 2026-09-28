# FastSSVEPNet Research

这是一个面向可穿戴 SSVEP 识别的独立研究工程。目标不是把图像 FastViT 直接套到脑电上，而是验证同一类工程原则在 EEG 时序上的价值：前段用高效多尺度局部卷积提取稳定频率模式，经过降采样后再使用注意力建模长时关系，从而在准确率、参数量和推理延迟之间取得更好的平衡。

当前状态：

- 已接入 OpenNeuro `ds004745`（CC0）：6 名参与者、8 通道、1/2/4/8 Hz，含静止段和主动头颈/眼动伪迹段。
- 采用被试级隔离切分，训练、验证和测试参与者不重合。
- 同时报告总体、静止段和伪迹段准确率，并与 CCA 基线比较。
- `E:\2` 只作为独立自采验证，不进入公开数据训练集。
- 后续主实验使用许可明确的多被试公开数据，并增加 FBCCA/TRCA、EEGNet、EEG Conformer 与局部/注意力消融。

## 运行

```powershell
Set-Location E:\EEG_FastViT_Research
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.train --config configs\ds004745.yaml
```

## 发布边界

计划发布代码、配置、被试级划分、评测结果和许可允许的模型权重。原始脑电数据不重复上传；用户自采数据默认不公开。使用每个公开数据集时保留其原始许可和引用。

## 引用

- Kumaravel et al., *8-Channel SSVEP EEG Dataset with Artifact Trials*, OpenNeuro ds004745, CC0.
- Wang et al., *A Benchmark Dataset for SSVEP-based Brain-Computer Interfaces*.
- Liu et al., *BETA: A Large Benchmark Database Toward SSVEP-BCI Application*.
- Kim et al., *A 40-class SSVEP speller dataset: beta range stimulation for low-fatigue BCI applications*, CC BY 4.0.
- Waytowich et al., EEGNet.
- Song et al., EEG Conformer.

## Kim2025 main training

The main experiment uses a fixed subject-disjoint 32/4/4 split and eight
posterior channels: PO7, PO3, POz, PO4, PO8, O1, Oz, and O2. Preprocessing is
written as resumable per-subject shards so an interrupted conversion does not
discard completed work.

```powershell
.\.venv\Scripts\python.exe -m src.kim2025_data `
  --root data\public\Kim2025 `
  --metadata-root data\metadata\nm000127 `
  --output-dir data\processed\Kim2025

.\.venv\Scripts\python.exe -m src.train_kim2025 `
  --config configs\kim2025.yaml `
  --run-dir runs\kim2025_initial
```

The model borrows the FastViT research idea of local-first processing and
train/deploy structural reparameterization, but implements new one-dimensional
EEG operators and an EEG-specific time/frequency fusion path. See
`THIRD_PARTY_NOTICES.md` for attribution and release obligations.
