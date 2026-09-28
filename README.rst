FeatStainDiff
=============

Official repository for "Diffusion-based cross-staining feature transformation
for whole slide image analysis: From H&E to IHC representation learning"
(*Medical Image Analysis*, 2026).

Paper: https://doi.org/10.1016/j.media.2026.104138

.. image:: motivation.png
   :alt: Motivation for H&E-to-IHC feature transformation
   :width: 100%

Overview
--------

FeatStainDiff translates H&E patch features into IHC-like features with
conditional diffusion; it does not generate IHC images.

* Contrastive Semantic Bridging (CSB) aligns semantics across stains.
* Frequency-domain Mixture of Experts (FMoE) adapts the denoiser.

.. image:: Fig2.png
   :alt: FeatStainDiff model pipeline
   :width: 100%

Release status
--------------

Only the core model file, featstaindiff/model.py, is temporarily withheld
for final review. Training, inference, and test scripts are public but
require that file to run. The paired-feature packing script works now.
Patient data, extracted features, and trained weights are not included.

Quick Start
-----------

1. Create the environment and install the package::

       conda create -n FeatStainDiff python=3.11 -y
       conda activate FeatStainDiff
       python -m pip install -e .

2. Prepare paired features. Each H&E row must match the IHC row at the
   same tissue location; paired IDs are recommended. For matching trees
   of per-patch .pt tensors, run::

       python scripts/pack_pairs.py \
         --he-dir /path/to/he_patches \
         --ihc-dir /path/to/ihc_patches \
         --he-output /path/to/he_features.npz \
         --ihc-output /path/to/ihc_features.npz

   Matching filenames alone do not establish slide registration.

3. Train after the core model file is released::

       HE_FEATURES=/path/to/he_features.npz \
       IHC_FEATURES=/path/to/ihc_features.npz \
       bash train.sh

4. Generate IHC-like features after training::

       python -m featstaindiff.infer \
         --checkpoint runs/default/checkpoints/last.pt \
         --input /path/to/he_features.npz \
         --output /path/to/generated_ihc_features.pt \
         --steps 30 --ensemble 5

See bash train.sh --help and python -m featstaindiff.infer --help for
options. For ID-free arrays, verify row alignment before setting
ASSUME_ALIGNED_ORDER=1.

Repository Layout
-----------------

* featstaindiff/: data, diffusion, training, and inference code;
  model.py is temporarily withheld.
* scripts/pack_pairs.py: prepare aligned feature archives.
* train.sh and tests/: training launcher and regression tests.
* motivation.png and Fig2.png: paper figures.

Citation
--------

Please cite Zhong et al., *Medical Image Analysis* 112 (2026), 104138:
https://doi.org/10.1016/j.media.2026.104138
Machine-readable metadata is in CITATION.cff.

Acknowledgements
----------------

The paper used RegWSI to register paired H&E/IHC slides for IMPRESS and
NSCLC. Please cite Wodzinski et al., "RegWSI: Whole slide image registration
using combined deep feature- and intensity-based methods: Winner of the
ACROBAT 2023 challenge," *Computer Methods and Programs in Biomedicine*
250 (2024), 108187:
https://doi.org/10.1016/j.cmpb.2024.108187
Official implementation: https://github.com/MWod/DeeperHistReg
Registration code is not included in this repository.

The denoiser draws on U-ViT; see THIRD_PARTY_NOTICES.txt and
licenses/U-ViT-LICENSE.txt for attribution.

License
-------

MIT. See LICENSE.
