"""Fine-tune the baseline YOLO11n gap detector and compare it with the baseline.

The baseline (runs/detect/baseline-yolo11n) was trained for only 10 epochs and was still
improving (val mAP50-95 rose every epoch, 0.33 -> 0.67), so it is under-trained. This script
continues from its best.pt with a fresh, longer schedule:

  * lower learning rate (0.001) with a cosine decay, 1 warm-up epoch
  * up to 50 epochs, early stopping after 15 epochs without improvement
  * milder augmentation (scale 0.4, saturation 0.5) and mosaic switched off for the last 10
    epochs; vertical flip and rotation stay off (shelves have a fixed orientation), as in
    the baseline
  * details and results: docs/finetune-yolo11n.md
  * same imgsz (640), data.yaml and seed as the baseline, so the comparison stays fair

    python -m barcode.modeling.finetune_yolo11n train            # train, then compare
    python -m barcode.modeling.finetune_yolo11n train --epochs 80 --name finetune-v2
    python -m barcode.modeling.finetune_yolo11n compare          # compare existing runs

The comparison scores both models on the validation AND test split:
  * standard detection metrics (precision, recall, mAP50, mAP50-95)
  * the tile-level gap metrics used in baselines.py, at the fixed conf 0.25 and at a conf
    threshold tuned for tile F1 on validation only. The test split is never used for
    choosing anything (best.pt is picked on validation), so test is the unbiased column.

The validation split has only ~170 images, so differences of ~0.01 mAP are within noise.
"""

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from tqdm import tqdm
import typer

from barcode.config import PROCESSED_DATA_DIR, PROJ_ROOT, REPORTS_DIR
from barcode.modeling.baselines import boxes_to_tiles, image_paths, score, true_tiles

app = typer.Typer()

RUNS_DIR = PROJ_ROOT / "runs" / "detect"
BASELINE_WEIGHTS = RUNS_DIR / "baseline-yolo11n" / "weights" / "best.pt"
DATA_YAML = PROCESSED_DATA_DIR / "retail-gap" / "data.yaml"
IMGSZ = 640
SEED = 0
CONF_GRID = np.round(np.arange(0.10, 0.76, 0.05), 2)


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@app.command()
def train(
    weights: Path = BASELINE_WEIGHTS,
    data: Path = DATA_YAML,
    name: str = "finetune-yolo11n",
    epochs: int = 50,
    patience: int = 15,
    batch: int = 16,
    imgsz: int = IMGSZ,
    lr0: float = 0.001,
    lrf: float = 0.05,
    close_mosaic: int = 10,
    run_compare: bool = typer.Option(True, "--compare/--no-compare"),
):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    model.train(
        data=str(data),
        epochs=epochs,
        patience=patience,
        batch=batch,
        imgsz=imgsz,
        device=pick_device(),
        project=str(RUNS_DIR),
        name=name,
        seed=SEED,
        deterministic=True,
        # optimisation: gentle fine-tune from an already-trained detector
        optimizer="AdamW",
        lr0=lr0,
        lrf=lrf,
        cos_lr=True,
        warmup_epochs=1,
        warmup_bias_lr=0.0,  # default 0.1 would shock an already-trained detector's biases
        # augmentation tuned for shelf photos
        fliplr=0.5,
        flipud=0.0,
        degrees=0.0,
        scale=0.4,
        translate=0.1,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        mosaic=1.0,
        close_mosaic=close_mosaic,
        mixup=0.0,
        plots=True,
    )
    best = Path(model.trainer.best)
    logger.success(f"best weights: {best}")
    if run_compare:
        compare(baseline=weights, finetuned=best, data=data, imgsz=imgsz)


def predict_boxes(model, imgs: list[Path], imgsz: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """One low-conf pass per image; any conf threshold can then be applied afterwards."""
    out = []
    for i in tqdm(range(0, len(imgs), 16), desc="predict"):
        for r in model.predict([str(p) for p in imgs[i:i + 16]], imgsz=imgsz, conf=0.001,
                               verbose=False):
            out.append((r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy()))
    return out


def tiles_at(preds, conf: float) -> np.ndarray:
    return np.concatenate([boxes_to_tiles(b[c >= conf]) for b, c in preds])


def evaluate(label: str, weights: Path, data: Path, imgsz: int) -> list[dict]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    # a separate instance for predict(): reusing the one val() has touched raises
    # "Inference tensors do not track version counter" on recent torch/ultralytics
    pred_model = YOLO(str(weights))
    data_dir = data.parent
    device = pick_device()
    per_split, preds, truth = {}, {}, {}
    for split in ("val", "test"):
        m = model.val(data=str(data), split=split, imgsz=imgsz, conf=0.001, device=device,
                      plots=False, verbose=False, project=str(RUNS_DIR / "_compare"),
                      name=f"{label}-{split}", exist_ok=True).box
        per_split[split] = {"precision": m.mp, "recall": m.mr, "mAP50": m.map50,
                            "mAP50-95": m.map}
        imgs = image_paths(data_dir, split)
        preds[split] = predict_boxes(pred_model, imgs, imgsz)
        truth[split] = np.concatenate([true_tiles(p) for p in imgs])
        per_split[split]["n_imgs"] = len(imgs)

    # conf for the tile metric is chosen on validation only
    val_f1 = {c: score("", truth["val"], tiles_at(preds["val"], c), per_split["val"]["n_imgs"],
                       0)["gap_f1"] for c in CONF_GRID}
    best_conf = max(val_f1, key=val_f1.get)
    logger.info(f"{label}: tile-F1 tuned conf (val) = {best_conf}")

    rows = []
    for split in ("val", "test"):
        n = per_split[split]["n_imgs"]
        row = {"model": label, "split": split, **{k: v for k, v in per_split[split].items()
                                                  if k != "n_imgs"}}
        for tag, conf in (("conf0.25", 0.25), ("tuned", best_conf)):
            s = score("", truth[split], tiles_at(preds[split], conf), n, 0)
            row.update({f"tile_f1@{tag}": s["gap_f1"], f"tile_prec@{tag}": s["gap_precision"],
                        f"tile_rec@{tag}": s["gap_recall"],
                        f"empty_share_MAE@{tag}": s["empty_share_MAE"]})
        row["tuned_conf"] = best_conf
        rows.append(row)
    return rows


@app.command()
def compare(
    baseline: Path = BASELINE_WEIGHTS,
    finetuned: Path = RUNS_DIR / "finetune-yolo11n" / "weights" / "best.pt",
    data: Path = DATA_YAML,
    imgsz: int = IMGSZ,
):
    rows = evaluate("baseline", baseline, data, imgsz) + evaluate("finetuned", finetuned,
                                                                  data, imgsz)
    df = pd.DataFrame(rows).sort_values(["split", "model"], ascending=[False, True])
    out = REPORTS_DIR / "yolo_finetune_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # headline: change from baseline to finetuned on the held-out test split
    cols = ["precision", "recall", "mAP50", "mAP50-95", "tile_f1@tuned", "empty_share_MAE@tuned"]
    t = df[df.split == "test"].set_index("model")[cols].astype(float)
    logger.info("test split, finetuned minus baseline:\n"
                + (t.loc["finetuned"] - t.loc["baseline"]).round(3).to_string())
    logger.success(f"saved {out}")


if __name__ == "__main__":
    app()
