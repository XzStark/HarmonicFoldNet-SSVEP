# Third-party research attribution

The local-first and train/deploy structural-reparameterization design is
informed by the ideas described in:

Pavan Kumar Anasosalu Vasu, James Gabriel, Jeff Zhu, Oncel Tuzel, and Anurag
Ranjan. "FastViT: A Fast Hybrid Vision Transformer using Structural
Reparameterization." ICCV 2023.

Paper: https://openaccess.thecvf.com/content/ICCV2023/html/Vasu_FastViT_A_Fast_Hybrid_Vision_Transformer_Using_Structural_Reparameterization_ICCV_2023_paper.html

Official reference implementation:
https://github.com/apple/ml-fastvit

The implementation in this repository was written as an original
one-dimensional multichannel-EEG model. It uses an EEG-specific
time/frequency fusion path and does not use image-model weights. The citation
above acknowledges the research idea; it does not import the upstream source
license into independently written code.

## Dataset

The v0.1 checkpoint uses Kim2025BetaRange / NEMAR `nm000127` v1.0.2, released
under CC BY 4.0. Raw data is not redistributed.

Official dataset page: https://www.nemar.org/dataexplorer/detail?dataset_id=nm000127

Dataset article: https://doi.org/10.1038/s41597-025-06032-2
