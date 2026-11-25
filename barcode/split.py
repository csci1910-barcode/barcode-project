"""Build the fixed checkpoint split (v2) from the raw Roboflow export and the saved index.

The split rules come from notebooks/eda/1.0-qp-eda-and-cleaning.ipynb, which wrote the
saved index used here.

Inputs (all in git):
  data/raw/retail-shelf-availability-coco/   Roboflow COCO export (labels and images)
  data/interim/retail-gap/manifest.csv       saved index: duplicate group, copy cluster, scene,
                                             split (seed 42, grouped 70/15/15 per scene)
  data/interim/retail-gap/sku110k_negatives.csv + data/external/sku110k-negatives/
                                             the 57 fully stocked SKU-110K shelves and their split

Split v2 (fixed at the checkpoint, never re-randomized):
  * train / val / test: the store photos (DSC*) keep their manifest split, plus the SKU-110K
    full shelves with empty labels. Val and test keep one file per copy cluster.
  * shift-lab: the lab shelf (235_*, *lowResFull*), held out completely from training and model
    selection as the distribution-shift setting, one file per copy cluster.
  Rule: a stem, copy cluster or near-duplicate group never appears in two of these sets.

    python -m barcode.split
"""

from collections import defaultdict
import json
from pathlib import Path
import shutil

from loguru import logger
import pandas as pd
import typer

from barcode.config import EXTERNAL_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

RF_DIR = RAW_DATA_DIR / "retail-shelf-availability-coco"
RF_FOLDER = {"train": "train", "val": "valid", "test": "test"}
INDEX_DIR = INTERIM_DATA_DIR / "retail-gap"
NEG_DIR = EXTERNAL_DATA_DIR / "sku110k-negatives"
OUT_MAIN = PROCESSED_DATA_DIR / "retail-gap"
OUT_SHIFT = PROCESSED_DATA_DIR / "shift-lab"


def load_yolo_labels() -> dict[str, list[str]]:
    """uid ('train/<file>') -> YOLO label lines; every category becomes class 0 = gap."""
    labels = defaultdict(list)
    for split, folder in RF_FOLDER.items():
        coco = json.loads((RF_DIR / folder / "_annotations.coco.json").read_text())
        images = {im["id"]: im for im in coco["images"]}
        for a in coco["annotations"]:
            im = images[a["image_id"]]
            x, y, w, h = a["bbox"]
            W, H = im["width"], im["height"]
            labels[f"{split}/{im['file_name']}"].append(
                f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}\n"
            )
    return labels


def assign_v2() -> pd.DataFrame:
    man = pd.read_csv(INDEX_DIR / "manifest.csv").sort_values("uid").reset_index(drop=True)
    store, lab = man[man["scene"] == "store"].copy(), man[man["scene"] == "lab"].copy()

    store["split_v2"] = store["split"]
    store = store[store["keep"]]  # train keeps all copies, val/test one file per copy cluster
    lab["split_v2"] = "shift"
    lab = lab[~lab.duplicated("copy_cluster")]  # one file per copy cluster

    rows = pd.concat([store, lab])
    rows["source"] = rows["scene"]
    rows["path"] = [RF_DIR / RF_FOLDER[s] / f for s, f in zip(rows["orig_split"], rows["file"])]

    neg = pd.read_csv(INDEX_DIR / "sku110k_negatives.csv")
    neg = pd.DataFrame({
        "uid": "sku110k/" + neg["file"], "file": "sku110k_" + neg["file"], "source": "sku110k",
        "split_v2": neg["neg_split"], "group": -1, "copy_cluster": -1,
        "path": [NEG_DIR / f"sku110k_{f}" for f in neg["file"]],
    })
    return pd.concat([rows, neg], ignore_index=True)


def check(df: pd.DataFrame) -> None:
    rf = df[df["source"] != "sku110k"]
    for key in ["group", "copy_cluster", "stem"]:
        spans = rf.groupby(key)["split_v2"].nunique()
        assert spans.max() == 1, f"a {key} appears in more than one set"
    assert df["file"].is_unique, "duplicate file names"
    assert all(p.exists() for p in df["path"]), "missing input image"


def write(root: Path, df: pd.DataFrame, labels: dict, yaml: str) -> None:
    if root.exists():
        shutil.rmtree(root)
    for r in df.itertuples():
        img_dir, lbl_dir = root / "images" / r.split_v2, root / "labels" / r.split_v2
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(r.path, img_dir / r.file)
        (lbl_dir / r.file).with_suffix(".txt").write_text("".join(labels.get(r.uid, [])))
    (root / "data.yaml").write_text(yaml)


@app.command()
def main():
    labels = load_yolo_labels()
    df = assign_v2()
    check(df)
    df["gap_boxes"] = [len(labels.get(u, [])) for u in df["uid"]]

    write(OUT_MAIN, df[df["split_v2"] != "shift"], labels,
          "# retail-gap, checkpoint split v2: built by `python -m barcode.split`\n"
          "# store photos + SKU-110K full shelves; the lab shelf is held out in ../shift-lab\n"
          "train: images/train\nval: images/val\ntest: images/test\n\nnames:\n  0: gap\n")
    write(OUT_SHIFT, df[df["split_v2"] == "shift"], labels,
          "# shift-lab: held-out distribution-shift set (lab shelf), built by `python -m barcode.split`\n"
          "# evaluate once at the end with split=test; never train or select models on it\n"
          "train: ../retail-gap/images/train\nval: ../retail-gap/images/val\ntest: images/shift\n\n"
          "names:\n  0: gap\n")

    cols = ["file", "source", "split_v2", "group", "copy_cluster", "gap_boxes"]
    df[cols].rename(columns={"split_v2": "split"}).to_csv(INDEX_DIR / "split_v2.csv", index=False)

    counts = df.pivot_table(index="split_v2", columns="source", values="file", aggfunc="count",
                            fill_value=0).reindex(["train", "val", "test", "shift"])
    counts["images"] = counts.sum(axis=1)
    counts["gap boxes"] = df.groupby("split_v2")["gap_boxes"].sum()
    counts["unique photos"] = df[df["source"] != "sku110k"].groupby("split_v2")["copy_cluster"].nunique()
    print(counts.to_string())
    logger.success(f"wrote {OUT_MAIN}, {OUT_SHIFT} and {INDEX_DIR / 'split_v2.csv'}")


if __name__ == "__main__":
    app()
