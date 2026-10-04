from __future__ import annotations

import torch

from src.external_mtsnet import ExternalMTSNetAdapter, complex_spectrum_560


def test_complex_spectrum_has_registered_shape_and_is_finite():
    features = complex_spectrum_560(torch.randn(3, 8, 200))
    assert features.shape == (3, 8, 560)
    assert torch.isfinite(features).all()


def test_external_adapter_imports_without_copying_author_source(tmp_path):
    source = tmp_path / "mock_mtsnet.py"
    source.write_text(
        "import torch\n"
        "from torch import nn\n"
        "class ViT_MTFNet(nn.Module):\n"
        "    def __init__(self, class_num, **kwargs):\n"
        "        super().__init__(); self.class_num = class_num\n"
        "    def forward(self, x, x_fre):\n"
        "        assert x_fre.shape[-1] == 560\n"
        "        return torch.zeros((x.shape[0], self.class_num), device=x.device)\n",
        encoding="utf-8",
    )
    model = ExternalMTSNetAdapter(
        source=source, time_points=200, channels=8, classes=12,
    )
    output = model(torch.randn(4, 8, 200))
    assert output.shape == (4, 12)
