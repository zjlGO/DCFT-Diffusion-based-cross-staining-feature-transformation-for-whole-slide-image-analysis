DCFT / FeatStainDiff
=====================

Official repository for "Diffusion-based cross-staining feature
transformation for whole slide image analysis: From H&E to IHC
representation learning," published in Medical Image Analysis 112
(2026), 104138.

Paper: https://doi.org/10.1016/j.media.2026.104138

.. image:: motivation.png
   :alt: Motivation for H&E-to-IHC feature transformation in whole slide analysis
   :width: 100%

Overview
--------

FeatStainDiff maps H&E patch features to IHC-like patch features with a
conditional diffusion model. It works with embeddings from a frozen
feature extractor; it does not synthesize IHC pixels.

* Contrastive Semantic Bridging (CSB) aligns pathological semantics
  across H&E and IHC features.
* Frequency-domain Mixture of Experts (FMoE) adapts the transformation
  across frequency bands.

.. image:: Fig2.png
   :alt: FeatStainDiff pipeline with CSB and frequency-domain experts
   :width: 100%

Release status
--------------

The model implementation, training and inference code, and trained
checkpoints are temporarily withheld while the author completes a
final model review. The paired-feature preparation utility below is
available now. The images, paper citation, and license are included
in this repository.

Data preparation
----------------

Training the method requires spatially registered H&E/IHC patch pairs
and features from the same frozen extractor and preprocessing pipeline.
Matching filenames alone do not establish tissue correspondence.

The available script packs matching per-patch PyTorch .pt tensors
from separate H&E and IHC directory trees into aligned, ID-bearing
.npz archives. Run it with Python 3.10 or newer, NumPy, and PyTorch 2.5
or newer::

    python scripts/pack_pairs.py \
      --he-dir /path/to/he_patches \
      --ihc-dir /path/to/ihc_patches \
      --he-output /path/to/he_features_with_ids.npz \
      --ihc-output /path/to/ihc_features_with_ids.npz

Run python scripts/pack_pairs.py --help for all options. The script
checks path matching, vector shapes, and finite values. Review IDs
for patient identifiers before sharing any output archive. No patient
data, extracted features, pretrained extractor weights, or trained
checkpoints are distributed here.

Repository layout
-----------------

* motivation.png: paper motivation figure.
* Fig2.png: full model pipeline.
* scripts/pack_pairs.py: paired-feature archive preparation.
* CITATION.cff: machine-readable citation.
* LICENSE: MIT license.

Citation
--------

If this work is useful, please cite::

    @article{zhong2026featstaindiff,
      title   = {Diffusion-based cross-staining feature transformation for whole slide image analysis: From H\&E to IHC representation learning},
      author  = {Zhong, Jialong and Zhang, Miao and Liu, Leiye and Liu, Tingwei and Jiang, Jiahong and Piao, Yongri and Xu, Rui and Tian, Feng and Sun, Weibing and Bi, Huan and Lu, Huchuan},
      journal = {Medical Image Analysis},
      volume  = {112},
      pages   = {104138},
      year    = {2026},
      doi     = {10.1016/j.media.2026.104138}
    }

License
-------

This repository is released under the MIT License; see LICENSE.
