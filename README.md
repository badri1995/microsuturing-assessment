# Microsuturing image assessment

Research code from Badrinath Padmanabhan's work at AIIMS New Delhi on assessing completed microsutures from images.

The model combines a DINOv2 image backbone, five Normal Inverse Gamma regression heads and a three-class score-group classifier. The indicators are inter-suture distance, slack, position, angulation and width. The code also exposes uncertainty estimates and an experimental deferral rule.

**Research only.** The labels `Unsafe`, `Moderate` and `Safe` are names for annotation score bins, not clinical safety determinations. This repository is not a medical device, a diagnostic tool or evidence of clinical validation.

## Included source

| File                | Purpose                                                               |
| ------------------- | --------------------------------------------------------------------- |
| `dataset_fixed.py`  | Annotation workbook reader and image transformations                  |
| `model_fixed.py`    | DINOv2 backbone, regression heads, classifier and uncertainty outputs |
| `losses_fixed.py`   | Evidential regression, classification and MSE baseline losses         |
| `train_kfold_cv.py` | Stratified cross-validation on a supplied development dataset         |

This is a selected modular experiment snapshot, not the full internal research workspace. It does not include images, annotations, subject identifiers, splits, checkpoints, model weights, experiment logs or measured performance. The older exploratory Drive notebook is not included because it contains unfinished alternative training paths.

## Setup

Use a separate Python 3.11 environment. Install PyTorch and torchvision appropriate to your hardware, then install the remaining dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train_kfold_cv.py --help
```

The backbone loader uses `torch.hub` to obtain code and pretrained weights from `facebookresearch/dinov2`. These are third-party dependencies, not weights produced or distributed by this project. Review the upstream code and terms before allowing downloads. No model download is performed simply by importing these modules.

## Data contract and training

Supply an authorised annotation workbook with columns `Name`, `ISD`, `Slack`, `Position`, `Angulation`, `Width`, and `Overall`. `Name` is an image filename relative to the supplied image directory. Scores follow the original 0 to 10 convention. Score groups use `Overall <= 4`, `4 < Overall <= 7`, and `Overall > 7`.

```bash
python train_kfold_cv.py \
  --annotations /path/to/annotations.xlsx \
  --images /path/to/images \
  --output outputs/cross_validation \
  --folds 5 --epochs 20 --seed 42
```

Each class needs at least as many examples as folds. Duplicate filenames are rejected. The script uses separate training and deterministic validation transforms. It writes checkpoints and per-fold metrics under the output directory, which is ignored by Git.

## Evaluation limits

The supplied splitter is image-level, not subject-level or session-level. If related images, crops or repeated subjects exist, construct grouped splits before using this routine. Duplicate-filename checking alone cannot prevent leakage. An older experiment audit flagged overlap; no historical numbers are presented here as validated results.

Fold metrics select the checkpoint on that fold's validation data. They are not independent held-out test estimates. The code's mean of five indicator predictions is an aggregate proxy, not a separately learned overall score. Fixed class weights and default deferral thresholds are experiment choices, not calibrated recommendations. Calibration, grouped evaluation and out-of-distribution validation require separate studies.

## Provenance and checks

The four modules are derived from the existing modular research source, not reconstructed from a CV. `source-manifest.json` records hashes of those original source files. Publication cleanup removed machine-specific demonstration paths and an unused compatibility wrapper, added command-line paths, corrected scalar prediction shapes, separated validation transforms and corrected the order of optional backbone unfreezing.

Release checks cover source syntax, privacy scanning and small CPU checks with an artificial backbone. They do not establish scientific performance. Full training and evaluation on the original study data were not rerun.

The evidential loss cites Amini et al., _Deep Evidential Regression_ (NeurIPS 2020), in the source. DINOv2 and other dependencies retain their own authorship and licences. No project-wide licence has been added.
