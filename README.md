# FDRNet-PyTorch

Official PyTorch implementation of **FDRNet**, introduced in:

> Weicheng Liao, Yuhui Ma, Zan Chen, Shijia Zhou, Mingen Zhang,
> Yuanyuan Gu, Meng Wang, and Yitian Zhao. **A frequency-aware dual-domain
> collaborative framework for medical image enhancement**. *Medical Image
> Analysis*, 2026, 104261.
> [https://doi.org/10.1016/j.media.2026.104261](https://doi.org/10.1016/j.media.2026.104261)

FDRNet is a two-stage medical image enhancement framework:

1. **FRED** (FREquency-Decouple deblurring) removes blur and restores detail.
2. **RICE** (Retinex-guided Illumination CompEnsation) corrects illumination
   on the FRED output.

The legacy source files under `FRED/` and `RICE/` are preserved unchanged.
The scripts under `scripts/` and `run_pipeline.py` provide reproducible,
parameterized entry points for training and inference.

> **Important:** use the new entry points documented below. The preserved
> `FRED/main.py` imports a disabled legacy evaluator, and the preserved
> `RICE/test.py` deletes its model before the image loop. They remain in the
> repository only for source provenance and are not the supported interfaces.

## Installation

Python 3.9 or later and an NVIDIA CUDA environment are required by the provided
entry points. RICE contains CUDA-specific tensor creation, while FRED relies on
compiled MMCV operators built for the installed PyTorch and CUDA versions.

```bash
git clone https://github.com/iMED-Lab/FDRNet-PyTorch.git
cd FDRNet-PyTorch
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA toolkit first. FRED also uses
compiled MMCV deformable-convolution operators, so install the matching MMCV
build by following the [MMCV installation guide](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).

## Data preparation

### FRED

FRED is supervised with paired degraded/clean images. In the paper workflow,
`sharp` contains original images and `blur` contains the corresponding images
produced by the stochastic degradation simulation. Paired files must have the
same filename.

```text
fred_data/
├── train/
│   ├── blur/       # synthetically degraded images
│   └── sharp/      # original images
├── valid/
│   ├── blur/
│   └── sharp/
└── test/
    ├── blur/
    └── sharp/
```

The exact degradation-generation program is not included in the supplied code
bundle. Prepare the paired inputs according to the degradation procedure in the
paper before starting FRED training. This repository deliberately does not
invent a substitute degradation pipeline that could change the reported method.

### RICE

RICE is trained directly on original images and does not require paired targets.
Images can be placed recursively under one directory:

```text
rice_data/
├── case_001.png
├── case_002.png
└── subgroup/
    └── case_003.png
```

Supported image extensions are `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, and
`.tiff`. RICE training uses random 256 x 256 crops by default, so both image
dimensions must be at least 256 pixels.

## Training

Train both stages sequentially with one command:

```bash
python run_pipeline.py train \
  --fred-data-dir /path/to/fred_data \
  --rice-data-dir /path/to/rice_data \
  --output-dir ./checkpoints \
  --gpu 0
```

The release entry points default to 100 epochs for FRED and 800 epochs for
RICE. For a smoke test, override them with
`--fred-epochs 1 --rice-epochs 1`.

The resulting final checkpoints are:

```text
checkpoints/fred/Final.pkl
checkpoints/rice/weights_final.pt
```

Use `Final.pkl` for FRED inference. In the preserved training code, validation
PSNR is currently hard-coded to zero, so `Best.pkl` does not identify the best
epoch and should not be used for model selection.

Individual training entry points expose additional hyperparameters:

```bash
python scripts/fred_train.py --data-dir /path/to/fred_data --output-dir ./checkpoints/fred
python scripts/rice_train.py --data-dir /path/to/rice_data --output-dir ./checkpoints/rice --gpu 0
```

FRED training initializes the VGG-19 perceptual network with pretrained
ImageNet weights; the first run may download those weights.

## Testing

Run the complete FRED-to-RICE enhancement pipeline:

```bash
python run_pipeline.py test \
  --input-dir /path/to/test_images \
  --fred-checkpoint ./checkpoints/fred/Final.pkl \
  --rice-checkpoint ./checkpoints/rice/weights_final.pt \
  --output-dir ./outputs
```

FRED outputs are retained in `outputs/fred/`, and final enhanced images are
written to `outputs/final/`. Input subdirectory structure is preserved.

To evaluate either stage independently:

```bash
python run_pipeline.py fred-test --input-dir ./images --checkpoint ./checkpoints/fred/Final.pkl --output-dir ./outputs/fred
python run_pipeline.py rice-test --input-dir ./images --checkpoint ./checkpoints/rice/weights_final.pt --output-dir ./outputs/rice
```

Use `--dry-run` with any `run_pipeline.py` command to inspect the commands
without starting training or inference.


## Acknowledgements

This implementation was developed with substantial reuse and adaptation of
[MIMO-UNet](https://github.com/chosj95/MIMO-UNet) for deblurring and
[SCI](https://github.com/vis-opt-group/SCI) for illumination enhancement. We
sincerely thank Sung-Jin Cho and the MIMO-UNet authors, and Long Ma and the SCI
authors, for providing their valuable code to the research community. Please
cite both upstream works when using this implementation. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for attribution and license
details.

```bibtex
@inproceedings{cho2021rethinking,
  title={Rethinking Coarse-to-Fine Approach in Single Image Deblurring},
  author={Cho, Sung-Jin and Ji, Seo-Won and Hong, Jun-Pyo and Jung, Seung-Won and Ko, Sung-Jea},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2021}
}

@inproceedings{ma2022toward,
  title={Toward Fast, Flexible, and Robust Low-Light Image Enhancement},
  author={Ma, Long and Ma, Tengyu and Liu, Risheng and Fan, Xin and Luo, Zhongxuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5637--5646},
  year={2022}
}
```

## Citation

```bibtex
@article{liao2026fdrnet,
  title={A frequency-aware dual-domain collaborative framework for medical image enhancement},
  author={Liao, Weicheng and Ma, Yuhui and Chen, Zan and Zhou, Shijia and Zhang, Mingen and Gu, Yuanyuan and Wang, Meng and Zhao, Yitian},
  journal={Medical Image Analysis},
  pages={104261},
  year={2026},
  doi={10.1016/j.media.2026.104261}
}
```

## Paper and licensing notes

The publisher's corrected-proof PDF is not redistributed in this repository;
use the DOI link above to access the paper. This also keeps the repository below
GitHub's per-file size limit.

The original SCI MIT license is preserved under `RICE/LICENSE`. MIMO-UNet did
not expose a license file when this repository was prepared, so its code remains
subject to the upstream authors' terms and applicable copyright law. No blanket
repository-wide license is asserted over third-party code.

## Contact

For questions, please contact [liaoweicheng@nimte.ac.cn](mailto:liaoweicheng@nimte.ac.cn).
