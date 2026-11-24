# barcode-project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Detecting empty shelf space (out-of-stock gaps) on retail shelf photos with deep learning object detectors.

# Retail Shelf Stock & Vacancy Detection

**Team Callsign:** Barcode  
**Team Members:**  
- Qazim Pali   
- Tahir Zogaj 
- Feriha Ozturk  

## Project Overview
Detecting low-stock and out-of-stock products on retail shelves using computer vision models (comparing one-stage, two-stage, and transformer-based detectors).

## Key data findings

From `notebooks/eda/1.0-qp-eda-and-cleaning.ipynb` and `1.1-extra-eda-cleaned-data.ipynb`:

- **Fewer unique photos than files.** The Roboflow export has 4,588 images but only about 1,080 unique photos. Most training photos come with 3-6 augmented copies, often mirrored.
- **The original split leaks.** 88% of the original val/test images have a copy in train, so we re-split 70/15/15 by near-duplicate group (`data/processed/retail-gap/`).
- **Labels cleaned to one class.** Class `"0"` was a batch of gaps under another name (merged into `gap`), and `UROP` had no boxes (dropped).
- **No full shelves in the source data.** Every Roboflow image contains a gap, so 57 fully stocked SKU-110K shelves were added as "no gap" images.
- **Source shortcut.** Image size alone separates those 57 shelves from the gap images (AUC 1.0), so full-shelf scores are not trusted until the SKU-110K images match the Roboflow format.

## Results

Task: detect `gap` (empty shelf space) boxes. The table below is on the held-out **test** split (174 images, grouped so no photo appears in two splits).

| Model | Precision | Recall | mAP50 | mAP50-95 | Tile F1 |
|---|---|---|---|---|---|
| YOLO11n baseline (10 epochs) | 0.864 | 0.894 | 0.930 | 0.648 | 0.891 |
| YOLO11n fine-tuned (+50 epochs) | 0.917 | 0.929 | 0.947 | 0.697 | 0.919 |

Tile F1 is the 32×32 tile-level gap score at a confidence chosen on validation (0.60 baseline, 0.45 fine-tuned). Full tables are in `reports/yolo_finetune_comparison.csv`, and the changes and caveats are written up in [docs/finetune-yolo11n.md](docs/finetune-yolo11n.md).

### Classical baselines vs. YOLO (tile-level task)

The classical models cannot output boxes, so every model is scored on the same tile task: each image is resized to 416×416 and cut into a 13×13 grid of 32×32 tiles, and a tile is "gap" when more than half of it lies inside a gap box (YOLO's predicted boxes are converted the same way). The classical models use hand-made tile features (colour, brightness, edges, HOG, position). **These numbers are on the validation split (172 images).**

| Model | Accuracy | Gap precision | Gap recall | Gap F1 | Empty-share error (lower is better) |
|---|---|---|---|---|---|
| Logistic regression | 0.777 | 0.362 | 0.812 | 0.500 | 0.173 |
| Decision tree | 0.806 | 0.396 | 0.791 | 0.528 | 0.138 |
| Random forest | 0.898 | 0.791 | 0.354 | 0.489 | 0.079 |
| **SVM (RBF), best classical model** | 0.855 | 0.481 | 0.734 | 0.582 | 0.087 |
| YOLO11n baseline (conf 0.25) | 0.959 | 0.801 | 0.929 | 0.860 | 0.032 |
| YOLO11n fine-tuned (conf 0.25) | n/a | 0.861 | 0.912 | 0.886 | 0.023 |

"Empty-share error" is the mean absolute difference between the predicted and labelled share of each shelf that is empty, the input to a restock rule. The classical rows come from `reports/baselines_tiles_val.csv` (`python -m barcode.modeling.baselines --split val`); the fine-tuned row comes from `reports/yolo_finetune_comparison.csv`. The best classical model (SVM, F1 0.58) is far behind even the 10-epoch YOLO11n (F1 0.86). Test-split numbers for the classical models were not computed.

## Reproducing our results

Tested with Python 3.10. Training used an Apple M-series GPU (`mps`); the scripts pick CUDA or CPU automatically when MPS is not available (CPU is much slower).

### 1. Set up the environment

```bash
git clone https://github.com/csci1910-barcode/barcode-project.git
cd barcode-project
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins `opencv-python<5`: OpenCV 5 removed `HOGDescriptor`, which the classical baselines need.

### 2. Data (already in the repo, no download needed)

Where the data comes from:

| Dataset | Used for | Source | License |
|---|---|---|---|
| Retail shelf availability (Roboflow Universe, v2, exported January 2024) | the `gap` labels: 4,588 images, COCO format, resized to 416×416 | https://universe.roboflow.com/srm-university-52jqz/retail-shelf-avialability | CC BY 4.0 |
| SKU-110K (Goldman et al., CVPR 2019) | 57 fully stocked shelves used as "no gap" images | https://platform.ultralytics.com/ultralytics/datasets/sku110k (download used here), original code and data: https://github.com/eg4000/SKU110K_CVPR19, paper: https://arxiv.org/abs/1904.00853 | see the original repository |

The Roboflow link is the one given in the export's own `README.dataset.txt` (note the dataset's spelling "avialability"). Roboflow Universe blocks automated requests, so the link may need to be opened in a browser.

- `data/raw/retail-shelf-availability-coco/`: the Roboflow export, as downloaded.
- `data/interim/retail-gap/`: `manifest.csv` (duplicate groups, scene, brightness, final split) and `sku110k_negatives.csv` (the 57 fully stocked shelves).
- `data/processed/retail-gap/`: the cleaned YOLO-format dataset the models train on (3,239 train / 172 val / 174 test images, one class `gap`), described by `data.yaml`.

To rebuild the processed data, run `notebooks/eda/1.0-qp-eda-and-cleaning.ipynb` top to bottom (fixed `SEED = 42`). The extra checks (split comparison, size shortcut, gap statistics) are in `notebooks/eda/1.1-extra-eda-cleaned-data.ipynb`. Re-running section 7 of notebook 1.0 (the SKU-110K selection) also needs the SKU-110K images, which are not in git (18 GB): `python -m barcode.dataset`. Its image links expire, so see the docstring in `barcode/dataset.py` for getting a fresh export.

### 3. Baseline YOLO11n (optional: the trained weights are committed)

```bash
yolo detect train model=yolo11n.pt data=data/processed/retail-gap/data.yaml \
    epochs=10 imgsz=640 batch=16 device=mps name=baseline-yolo11n seed=0
```

All settings of the original run are in `runs/detect/baseline-yolo11n/args.yaml`. Training on MPS is not bit-exact, so expect small differences.

### 4. Fine-tune and compare with the baseline

```bash
python -m barcode.modeling.finetune_yolo11n train      # 50 epochs, then compares (about 1.8 h on an M-series Mac)
python -m barcode.modeling.finetune_yolo11n compare    # comparison only, using the weights in runs/detect/
```

This writes `runs/detect/finetune-yolo11n/` and `reports/yolo_finetune_comparison.csv`.

### 5. Classical baselines vs. YOLO (tile-level task)

```bash
python -m barcode.modeling.baselines --split val
python -m barcode.modeling.baselines --split test --weights runs/detect/finetune-yolo11n/weights/best.pt --yolo-conf 0.45
```

The first command writes `reports/baselines_tiles_val.csv` (Naive Bayes, logistic regression, decision tree, random forest, SVM and the baseline YOLO11n).

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         barcode and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── barcode   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes barcode a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

