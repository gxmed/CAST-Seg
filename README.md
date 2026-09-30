# CAST-Seg

Official PyTorch implementation of **CAST-Seg: Confidence-aware Adaptive Semantic Transfer for Semi-supervised Pneumonia Lesion Segmentation**.

CAST-Seg is a dual-student semi-supervised framework for pneumonia-lesion segmentation under limited pixel-level annotation. It combines confidence-calibrated pseudo-labeling, text-guided spatial priors, synchronized augmentation, adaptive prior refinement, and training-time knowledge transfer from a medical foundation-model expert.

> **Reproduction scope.** This repository contains the complete CAST-Seg segmentation-stage training and evaluation code. Raw datasets, pretrained weights, TSL CAM priors, and MDAA-adapted SAM masks are intentionally not distributed. The current release expects the TSL and MDAA priors to be prepared before `train.py` is run; see [Prior preparation](#prior-preparation) and [Current artifact boundary](#current-artifact-boundary).


## Method overview

| Component | Full name | Role in the implementation |
|---|---|---|
| ACC | Agreement-aware Confidence Calibration | Selects reliable pseudo-label pixels by combining foreground/background decoder agreement and confidence. |
| TSL | Text-guided Semantic Localization | Converts image-report pairs into dense CAM priors. |
| SCA | Synchronized Cross-modal Augmentation | Applies identical spatial transformations to images, labels, CAMs, and SAM masks. |
| ASR | Adaptive Semantic Refinement | Periodically refreshes the shared CAM bank from the improving teacher. |
| MDAA | Medical Domain Adaptation Adapter | Adapts the medical SAM expert; its exported masks supervise Student-B during training. |
| WEMA | Weighted Exponential Moving Average | Updates the teacher from Student-A and Student-B (`0.75 / 0.25` by default). |

Medical SAM is a **training-time expert only**. The deployed segmentation network does not load Medical SAM.

## Repository layout

```text
CAST-Seg/
├── config/
│   ├── qata.yaml                 # QaTa-COV19 paper configuration
│   └── mosmed.yaml               # MosMedData+ paper configuration
├── data/
│   ├── dataset.py                # SSL dataset, priors, and synchronized augmentation
│   └── pretrain_dataset.py       # Labeled-data pretraining dataset
├── engine/
│   ├── pretrain.py               # Supervised initialization stage
│   └── cast_seg.py               # Dual-student/teacher SSL optimization
├── models/
│   └── cast_seg.py               # ConvNeXt segmentation network
├── modules/
│   ├── acc.py                    # ACC
│   ├── tsl.py                    # TSL architecture
│   ├── sca.py                    # SCA
│   ├── asr.py                    # ASR scheduling
│   ├── mdaa.py                   # MDAA building blocks
│   └── wema.py                   # WEMA update
├── train.py                      # Pretraining + semi-supervised training + test
├── evaluate.py                   # Standalone six-metric evaluation
└── requirements.txt
```

## 1. Environment

The supplied entry points currently require an NVIDIA GPU and CUDA. Python 3.10 is recommended.

```bash
conda create -n castseg python=3.10 -y
conda activate castseg
```

Install a CUDA-compatible PyTorch build for your system first, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The model configuration uses these Hugging Face identifiers:

- vision encoder: `facebook/convnext-tiny-224`
- biomedical text encoder: `microsoft/BiomedVLP-CXR-BERT-specialized`

They are downloaded automatically on first use. For an offline machine, download them in advance and replace `vision_type` and `bert_type` in the YAML file with local directories.

## 2. Dataset preparation

Obtain QaTa-COV19 and/or MosMedData+ from their original providers and follow their licenses. The repository does not redistribute either dataset.

Prepare this structure from the repository root:

```text
datasets/
├── qata/
│   ├── prompt/
│   │   ├── Train.csv
│   │   └── Test.csv
│   ├── Train Set/
│   │   ├── Images/
│   │   └── GTs/
│   └── Test Set/
│       ├── Images/
│       └── GTs/
└── mosmed/
    ├── prompt/
    │   ├── Train.csv
    │   └── Test.csv
    ├── Train Set/
    │   ├── Images/
    │   └── GTs/
    └── Test Set/
        ├── Images/
        └── GTs/
```

Each CSV must contain an `Image` column and a diagnostic-text column. The provided configurations use `Description` for QaTa-COV19 and `text` for MosMedData+.

Example:

```csv
Image,Description
mask_case_0001.png,"Bilateral peripheral pulmonary opacity."
```

The filename rule is:

```text
CSV Image       -> mask_case_0001.png
input image     -> Images/case_0001.png
ground truth    -> GTs/mask_case_0001.png
```

Masks may contain `0/1` or `0/255`.


## 3. Prior preparation

CAST-Seg consumes two training-time spatial priors:

1. **TSL CAM prior**: a text-guided, single-channel map in `[0, 1]`.
2. **MDAA SAM expert mask**: a single-channel adapted-SAM probability/binary map in `[0, 1]`.

Prepare the following directories:

```text
priors/
├── qata/
│   ├── tsl/train/
│   ├── tsl/test/
│   ├── mdaa_sam/train/
│   └── mdaa_sam/test/
└── mosmed/
    ├── tsl/train/
    ├── tsl/test/
    ├── mdaa_sam/train/
    └── mdaa_sam/test/
```

Training prior directories must cover every row needed from `Train.csv`, including the validation partition when strict loading is enabled. Test CAM priors must cover every `Test.csv` row for the default CAM-guided evaluation. The standalone evaluator does not use SAM masks unless explicitly extended/configured to do so.

Supported prior formats are:

- NumPy: `.npy`, `.npz` (the loader prefers keys `cam` or `sam`, otherwise the first array)
- Images: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`

For `mask_case_0001.png`, the recommended prior filename is `case_0001.npy`. The loader also accepts the CSV stem and `cam_`, `CAM_`, `sam_`, or `SAM_` prefixes. Priors must be single-channel; they are clipped to `[0, 1]` and resized to the configured image size.

The default YAML files set `strict_cam_prior: True` and `strict_sam_prior: True`, so a missing prior causes an immediate, informative error rather than silently changing the experiment.

## 4. Configuration

Edit only the dataset/prior paths and GPU IDs unless intentionally running an ablation.

Key paper settings in both supplied YAML files are:

| Setting | Value |
|---|---:|
| Input size | `224 x 224` |
| Batch size | `8` |
| Learning rate | `1e-4` |
| Random seed | `0` (set in `train.py`) |
| Validation fraction | `0.2` |
| ACC agreement threshold | `0.40` |
| ACC conflict threshold | `0.80` |
| ACC confidence margin | `0.10` |
| EMA decay | `0.99` |
| WEMA Student-A weight | `0.75` |
| ASR warm-up | `10` epochs |
| ASR update interval | `10` epochs |
| Maximum epochs per stage | `200` |
| Early-stopping patience | `20` |

Relative paths are resolved from the directory in which the command is run. Run all commands from the repository root, or replace them with absolute paths.

## 5. Training

### QaTa-COV19

```bash
python train.py --config config/qata.yaml
```

### MosMedData+

```bash
python train.py --config config/mosmed.yaml
```

Each command performs:

1. supervised initialization on the labeled subset;
2. initialization of Student-A, Student-B, and Teacher from the best supervised checkpoint;
3. dual-student semi-supervised training with ACC, TSL priors, SCA, MDAA expert masks, WEMA, and periodic ASR;
4. testing of the best semi-supervised checkpoint.

Default checkpoints are written to:

```text
outputs/qata/pretrain/cast_seg_pretrain.ckpt
outputs/qata/semi_supervised/cast_seg.ckpt
outputs/mosmed/pretrain/cast_seg_pretrain.ckpt
outputs/mosmed/semi_supervised/cast_seg.ckpt
```

ASR-generated CAM banks and manifests are written below `outputs/<dataset>/asr_cam_priors/`. Lightning logs are written below `lightning_logs/`.

### Resume from an existing supervised checkpoint

Set in the selected YAML:

```yaml
TRAIN:
  pretrain: False
  pretrained_ckpt_path: /absolute/path/to/cast_seg_pretrain.ckpt
```

Then run the same `train.py` command. `pretrained_ckpt_path` must point to a real checkpoint; weights are not included in this repository.

## 6. Evaluation

Evaluate the default best checkpoint and save all metrics as JSON:

```bash
python evaluate.py \
  --config config/qata.yaml \
  --ckpt outputs/qata/semi_supervised/cast_seg.ckpt \
  --output-json outputs/qata/test_metrics.json
```

```bash
python evaluate.py \
  --config config/mosmed.yaml \
  --ckpt outputs/mosmed/semi_supervised/cast_seg.ckpt \
  --output-json outputs/mosmed/test_metrics.json
```

On Windows PowerShell, place the command on one line or replace `\` with the PowerShell continuation character (backtick).

The standalone evaluator reports:

- mean Dice and mean IoU (per-image means);
- global Dice and global IoU (computed from aggregate intersections/unions);
- HD95 and ASSD (in pixels at the resized resolution);
- sensitivity and specificity;
- CAM/decoder conflict statistics.

By default, evaluation retains pixels on which the foreground/background decoders agree and uses the TSL CAM only to arbitrate conflict pixels. Thresholds can be overridden without editing the YAML:

```bash
python evaluate.py --config config/qata.yaml \
  --ckpt outputs/qata/semi_supervised/cast_seg.ckpt \
  --fg-threshold 0.5 --bg-threshold 0.5 --cam-threshold 0.2
```

To evaluate the network without CAM conflict arbitration:

```bash
python evaluate.py --config config/qata.yaml \
  --ckpt outputs/qata/semi_supervised/cast_seg.ckpt \
  --disable-cam-conflict-fusion
```

## 7. Sanity checks before a full run

Check the source tree for syntax errors:

```bash
python -m compileall train.py evaluate.py data engine models modules utils
```

Then verify:

- every CSV path and image/GT directory in the YAML exists;
- the `Image` and text columns match the selected dataset configuration;
- all strict TSL/MDAA prior files resolve correctly;
- CUDA is visible with `python -c "import torch; print(torch.cuda.is_available())"`;
- output directories are unique for the dataset, label ratio, fold, and seed;
- the test checkpoint is the best checkpoint from the corresponding run.

For a quick pipeline check, create a copied YAML with a small temporary dataset, `min_epochs: 1`, `max_epochs: 1`, and unique output directories. Do not use smoke-test metrics as paper results.

## License

No license file is currently included. Add an explicit open-source license before publishing the repository so that downstream users know what reuse is permitted. Dataset and pretrained-model licenses remain governed by their original providers.

---