# v0.1 environment record

This file records the environment used for the frozen v0.1 evaluation. It is
not a universal performance guarantee.

## System

- OS build: Windows 10.0.26200 (Windows 11 internal build family)
- Python: 3.11.9
- CPU: AMD Ryzen 9 8945HX with Radeon Graphics
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU
- PyTorch: 2.11.0+cu128
- CUDA runtime reported by PyTorch: 12.8
- cuDNN reported by PyTorch: 91900

## Direct Python dependencies in the evaluation environment

```text
torch==2.11.0+cu128
numpy==2.4.6
scipy==1.16.3
mne==1.13.2
pandas==3.0.0
scikit-learn==1.8.0
pyyaml==6.0.3
huggingface-hub==0.36.2
openneuro-py==2026.9.1
nemar-py==0.3.1
```

The CUDA-specific PyTorch build may require the matching official PyTorch
package index; the list above records the measured environment rather than
promising that every line can be installed from the default Python index.

## Latency protocol

- Batch size: 1
- Input shape: `[1, 8, 1250]`
- Complete 5-second window already available before timing
- 30 untimed warm-up forwards
- 300 timed iterations on CUDA
- 150 timed iterations on CPU
- CPU thread count forced to 1
- CUDA synchronized immediately before and after the timed loop
- PyTorch inference mode enabled

Latency excludes acquisition, preprocessing, data transfer, display and user
feedback. Repeated runs and target-device P50/P95 measurements remain future
work.
