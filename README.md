<p align="center">
  <img src="motivation.png" alt="Motivation for feature-level H&amp;E-to-IHC transformation" width="100%">
</p>

# DCFT / FeatStainDiff

Official repository for **Diffusion-based Cross-Staining Feature Transformation for Whole Slide Image Analysis: From H&amp;E to IHC Representation Learning**.

🎉 **This paper has been accepted by _Medical Image Analysis_ (2026).**

[[Paper](https://doi.org/10.1016/j.media.2026.104138)]

## Overview

FeatStainDiff performs direct feature-level transformation from H&amp;E to IHC representations for whole slide image analysis. Instead of synthesizing IHC pixels, the framework learns diagnostically meaningful cross-staining representations that are naturally compatible with multiple instance learning.

The method contains two core components:

- **Contrastive Semantic Bridging (CSB):** preserves and aligns pathological semantics across H&amp;E and IHC modalities.
- **Frequency-domain Mixture of Experts (FMoE):** adaptively reduces cross-modal distribution shifts through frequency-domain processing.

<p align="center">
  <img src="Fig2.png" alt="Overview of the FeatStainDiff framework" width="100%">
</p>

## Code

The source code and usage instructions will be released in this repository. Please stay tuned.

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhong2026featstaindiff,
  title   = {Diffusion-based cross-staining feature transformation for whole slide image analysis: From H\&E to IHC representation learning},
  author  = {Zhong, Jialong and Zhang, Miao and Liu, Leiye and Liu, Tingwei and Jiang, Jiahong and Piao, Yongri and Xu, Rui and Tian, Feng and Sun, Weibing and Bi, Huan and Lu, Huchuan},
  journal = {Medical Image Analysis},
  volume  = {112},
  pages   = {104138},
  year    = {2026},
  doi     = {10.1016/j.media.2026.104138}
}
```

## License

This repository is released under the MIT License. See `LICENSE` for details.
