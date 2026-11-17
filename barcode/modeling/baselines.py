"""Classical tile-level baselines for gap detection, compared with the YOLO model.

The detection task is turned into a classification task the classical models can do:
every image is resized to 416x416 and cut into a 13x13 grid of 32x32 tiles. A tile is
labelled "gap" when more than half of it lies inside a gap box. Each tile gets simple
hand-made features (colour, brightness, edges, HOG, position).

All models, including YOLO (its predicted boxes are turned into tile labels the same way),
are scored on the same tiles of the chosen split.

    python -m barcode.modeling.baselines            # validation split
    python -m barcode.modeling.baselines --split test
"""

from pathlib import Path
import time

import cv2
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
import typer

from barcode.config import PROCESSED_DATA_DIR, PROJ_ROOT, REPORTS_DIR

app = typer.Typer()

SIZE = 416  # the Roboflow images are 416x416; everything is resized to this
TILE = 32
G = SIZE // TILE  # 13 tiles per side, 169 per image
COVER = 0.5  # a tile is "gap" if more than this fraction is inside a gap box
SEED = 0

HOG = cv2.HOGDescriptor((TILE, TILE), (16, 16), (8, 8), (8, 8), 9)  # 324 values per tile


def image_paths(data_dir: Path, split: str) -> list[Path]:
    return sorted(p for p in (data_dir / "images" / split).iterdir() if p.suffix.lower() in
                  {".jpg", ".jpeg", ".png"})


def label_path(img: Path) -> Path:
    return img.parents[2] / "labels" / img.parent.name / (img.stem + ".txt")


def boxes_to_tiles(boxes_xyxyn: np.ndarray) -> np.ndarray:
    """Normalised xyxy boxes -> (G*G,) bool, True where a tile is mostly covered."""
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    for x1, y1, x2, y2 in boxes_xyxyn:
        c1, r1 = int(round(x1 * SIZE)), int(round(y1 * SIZE))
        c2, r2 = int(round(x2 * SIZE)), int(round(y2 * SIZE))
        mask[max(r1, 0):max(r2, 0), max(c1, 0):max(c2, 0)] = True
    cover = mask.reshape(G, TILE, G, TILE).mean(axis=(1, 3))
    return (cover > COVER).ravel()


def true_tiles(img: Path) -> np.ndarray:
    lp = label_path(img)
    rows = np.loadtxt(lp, ndmin=2) if lp.exists() and lp.stat().st_size else np.zeros((0, 5))
    cx, cy, w, h = rows[:, 1], rows[:, 2], rows[:, 3], rows[:, 4]
    return boxes_to_tiles(np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1))


def tile_stats(arr: np.ndarray) -> np.ndarray:
    """(SIZE, SIZE, C) -> (G*G, 2C): per-tile mean and std of each channel."""
    t = arr.reshape(G, TILE, G, TILE, -1).astype(np.float32)
    return np.concatenate([t.mean(axis=(1, 3)), t.std(axis=(1, 3))], axis=-1).reshape(G * G, -1)


def tile_features(img: Path) -> np.ndarray:
    bgr = cv2.resize(cv2.imread(str(img)), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    edges = cv2.Canny(gray, 100, 200) > 0
    lap = cv2.Laplacian(gray, cv2.CV_32F)

    stacked = np.dstack([bgr, hsv, gray, np.hypot(gx, gy), lap])
    stats = tile_stats(stacked)  # colour, brightness, edge strength, texture
    edge_density = edges.reshape(G, TILE, G, TILE).mean(axis=(1, 3)).reshape(-1, 1)
    hog = HOG.compute(gray, (TILE, TILE), (0, 0)).reshape(G * G, -1)  # row-major windows
    rr, cc = np.divmod(np.arange(G * G), G)
    pos = np.stack([rr / (G - 1), cc / (G - 1)], axis=1)
    return np.hstack([stats, edge_density, hog, pos]).astype(np.float32)


def build(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    imgs = image_paths(data_dir, split)
    X, y = [], []
    for img in tqdm(imgs, desc=f"features {split}"):
        X.append(tile_features(img))
        y.append(true_tiles(img))
    return np.vstack(X), np.concatenate(y), imgs


def yolo_tiles(weights: Path, imgs: list[Path], conf: float) -> np.ndarray:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    out = []
    for i in tqdm(range(0, len(imgs), 16), desc="yolo"):
        for r in model.predict([str(p) for p in imgs[i:i + 16]], imgsz=640, conf=conf,
                               verbose=False):
            out.append(boxes_to_tiles(r.boxes.xyxyn.cpu().numpy()))
    return np.concatenate(out)


def score(name: str, y: np.ndarray, pred: np.ndarray, n_imgs: int, secs: float) -> dict:
    # share of each shelf predicted empty vs. labelled empty: the input to the restock rule
    frac_err = np.abs(pred.reshape(n_imgs, -1).mean(1) - y.reshape(n_imgs, -1).mean(1))
    return {
        "model": name,
        "accuracy": accuracy_score(y, pred),
        "gap_precision": precision_score(y, pred, zero_division=0),
        "gap_recall": recall_score(y, pred, zero_division=0),
        "gap_f1": f1_score(y, pred, zero_division=0),
        "empty_share_MAE": frac_err.mean(),
        "fit_s": round(secs, 1),
    }


@app.command()
def main(
    split: str = "val",
    data_dir: Path = PROCESSED_DATA_DIR / "retail-gap",
    weights: Path = PROJ_ROOT / "runs/detect/baseline-yolo11n/weights/best.pt",
    yolo_conf: float = 0.25,
    train_tiles: int = 150_000,
    svm_tiles: int = 20_000,
):
    rng = np.random.default_rng(SEED)
    X_tr, y_tr, _ = build(data_dir, "train")
    X_ev, y_ev, ev_imgs = build(data_dir, split)
    logger.info(f"train tiles {len(y_tr):,} ({y_tr.mean():.1%} gap), "
                f"{split} tiles {len(y_ev):,} ({y_ev.mean():.1%} gap), {X_tr.shape[1]} features")

    idx = rng.choice(len(y_tr), size=min(train_tiles, len(y_tr)), replace=False)
    svm_idx = idx[:svm_tiles]

    def scaled(model):
        return make_pipeline(StandardScaler(), model)

    models = {
        "Uniform random class": (DummyClassifier(strategy="uniform", random_state=SEED), idx),
        "Always pick one class (no gap)": (DummyClassifier(strategy="most_frequent"), idx),
        "Naive Bayes": (scaled(GaussianNB()), idx),
        "Logistic regression": (scaled(LogisticRegression(max_iter=2000,
                                                          class_weight="balanced")), idx),
        "Decision tree": (DecisionTreeClassifier(max_depth=12, class_weight="balanced",
                                                 random_state=SEED), idx),
        "Random forest": (RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                                 class_weight="balanced_subsample",
                                                 n_jobs=-1, random_state=SEED), idx),
        "SVM (RBF)": (scaled(SVC(kernel="rbf", class_weight="balanced",
                                 random_state=SEED)), svm_idx),
    }

    rows = []
    for name, (model, rows_idx) in models.items():
        t = time.time()
        model.fit(X_tr[rows_idx], y_tr[rows_idx])
        pred = model.predict(X_ev).astype(bool)
        rows.append(score(name, y_ev, pred, len(ev_imgs), time.time() - t))
        logger.info(f"{name}: F1 {rows[-1]['gap_f1']:.3f}")

    t = time.time()
    rows.append(score(f"YOLO11n (deep, conf {yolo_conf})", y_ev,
                      yolo_tiles(weights, ev_imgs, yolo_conf), len(ev_imgs), time.time() - t))

    df = pd.DataFrame(rows)
    out = REPORTS_DIR / f"baselines_tiles_{split}.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    logger.success(f"saved {out}")


if __name__ == "__main__":
    app()
