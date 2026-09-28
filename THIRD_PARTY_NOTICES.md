# Third-party research attribution

The local-first and train/deploy structural-reparameterization design is based
on the ideas described in:

Pavan Kumar Anasosalu Vasu, James Gabriel, Jeff Zhu, Oncel Tuzel, and Anurag
Ranjan. "FastViT: A Fast Hybrid Vision Transformer using Structural
Reparameterization." ICCV 2023.

Official implementation: https://github.com/apple-aiml-research/ml-fastvit

The implementation in this repository adapts those ideas from two-dimensional
images to one-dimensional multichannel EEG and adds an EEG-specific
time/frequency fusion path. ImageNet model weights are not directly compatible
with this model and are not used.

Apple's official repository is distributed under its accompanying Apple source
license. Before public release, retain the upstream license and acknowledgements
for source copied or adapted from that repository.
