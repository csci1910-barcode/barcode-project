# barcode-project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Detecting low-stock and out-of-stock products on retail shelves using computer vision models. Deep Learning models will be used.

# Retail Shelf Stock & Vacancy Detection

**Team Callsign:** Barcode  
**Team Members:**  
- Qazim Pali   
- Tahir Zogaj 
- Feriha Ozturk  

## Project Overview
Detecting low-stock and out-of-stock products on retail shelves using computer vision models (comparing one-stage, two-stage, and transformer-based detectors).

## Results

Task: detect `gap` (empty shelf space) boxes. All numbers are on the held-out **test** split (174 images, grouped so no photo appears in two splits).

| Model | Precision | Recall | mAP50 | mAP50-95 | Tile F1 |
|---|---|---|---|---|---|
| YOLO11n baseline (10 epochs) | 0.864 | 0.894 | 0.930 | 0.648 | 0.891 |
| YOLO11n fine-tuned (+50 epochs) | 0.917 | 0.929 | 0.947 | 0.697 | 0.919 |

Tile F1 is the 32×32 tile-level gap score at a confidence chosen on validation (0.60 baseline, 0.45 fine-tuned). Full tables are in `reports/yolo_finetune_comparison.csv`, and the changes and caveats are written up in [docs/finetune-yolo11n.md](docs/finetune-yolo11n.md).

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

To rebuild the processed data, run `notebooks/eda/1.0-qp-eda-and-cleaning.ipynb` top to bottom (fixed `SEED = 42`). The extra checks (split comparison, size shortcut, gap statistics) are in `notebooks/eda/1.1-extra-eda-cleaned-data.ipynb`. Re-running step 6 of notebook 1.0 also needs the SKU-110K images, which are not in git (18 GB): `python -m barcode.dataset`. Its image links expire, so see the docstring in `barcode/dataset.py` for getting a fresh export.

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

