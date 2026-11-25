# Data

Two public datasets. Everything under `data/` except the full SKU-110K download is in git for the checkpoint, so the
pipeline runs from a fresh clone without a login (see "Raw data in git" below).

## Sources

| | Retail shelf availability (gap labels) | SKU-110K (fully stocked shelves) |
|---|---|---|
| Used for | all `gap` boxes: 4,588 images, COCO JSON, 1 real class | 57 fully stocked shelves used as "no gap" images (empty labels) |
| URL | https://universe.roboflow.com/srm-university-52jqz/retail-shelf-avialability (the export's own spelling) | https://platform.ultralytics.com/ultralytics/datasets/sku110k; original: https://github.com/eg4000/SKU110K_CVPR19 (Goldman et al., CVPR 2019, https://arxiv.org/abs/1904.00853) |
| Version | version 2, exported by Roboflow 2024-01-09 (`README.roboflow.txt`) | Ultralytics Platform export, version `latest`, updated 2026-10-23 |
| Downloaded | 2026-11-14 by Qazim Pali | 2026-11-14 by Qazim Pali |
| License | CC BY 4.0 | listed as "Research" on the Ultralytics Platform; terms in the original repository |
| Size | 144.5 MB zip, 146 MB unzipped | `sku110k.ndjson` 61 MB (labels + image links); the 11,739 images are 18.5 GB and **not in git**; the 57 we use are 9.4 MB |
| Preprocessing by the source | auto-orient, stretched to 416×416, 3 augmented versions per training image (50% horizontal flip) | none (phone photos, mostly 2448×3264) |

## How to download

**Roboflow export** (already in `data/raw/retail-shelf-availability-coco/`):
1. Open the URL above in a browser (Roboflow blocks scripted requests) and sign in with a free account.
2. Pick version 2, then **Download Dataset → Format: COCO → Download zip to computer**.
3. Unzip into `data/raw/retail-shelf-availability-coco/` so it contains `train/`, `valid/`, `test/`, each with `_annotations.coco.json`.

**SKU-110K** (only needed to redo the full-shelf selection in notebook 1.0, section 7; the 57 selected images are
already in `data/external/sku110k-negatives/`):
- Ultralytics export: on the Platform page click **Download dataset**, save `sku110k.ndjson` to `data/raw/SKU-110K/`, then
  run `python -m barcode.dataset`. The image links in the file expire after about a week, so download a fresh file first.
- Or the original archive (13.6 GB, no login): http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz

## Layout

| Folder | Contents | Built by | Size |
|---|---|---|---|
| `raw/retail-shelf-availability-coco/` | the Roboflow export, unchanged | download | 146 MB |
| `raw/SKU-110K/` | `sku110k.ndjson` + `data.yaml` (images not in git) | download / `python -m barcode.dataset` | 61 MB |
| `external/sku110k-negatives/` | the 57 selected full shelves, resized to at most 640 px | notebook 1.0, section 7 | 9.4 MB |
| `interim/retail-gap/manifest.csv` | **saved index**: every Roboflow image's stem, scene, duplicate group, copy cluster, split | notebook 1.0 | |
| `interim/retail-gap/split_v2.csv` | **the fixed checkpoint split**, one row per image in every set | `python -m barcode.split` | |
| `processed/retail-gap/` | YOLO dataset: train / val / test (`data.yaml`) | `python -m barcode.split` | 122 MB |
| `processed/shift-lab/` | YOLO dataset: the held-out distribution-shift set (`test: images/shift`) | `python -m barcode.split` | 2.8 MB |
| `processed/retail-gap-original-split/` | Roboflow's original split with cleaned labels, kept only for the leakage comparison | notebook 1.0 | 160 MB |

## The fixed split (v2, frozen at the checkpoint on 2026-11-24)

| Set | Images | of which SKU-110K full shelves | Unique photos | Gap boxes |
|---|---|---|---|---|
| train | 2,862 | 40 | 679 | 10,874 |
| val | 154 | 8 | 146 | 530 |
| test | 157 | 9 | 148 | 568 |
| **shift-lab** (distribution shift) | 109 | 0 | 109 | 1,399 |

- **Grouping rule.** Two Roboflow images are in the same group if they share a Roboflow source name (the part before
  `.rf.`) or their flip-aware perceptual-hash distance is ≤ 12 (store photos) or ≤ 4 (lab shelf). A whole group goes to
  one set. Train keeps every augmented copy; val, test and shift-lab keep one file per copy cluster (same source name or
  hash distance 0). The rule and its thresholds are derived in notebook 1.0, section 5.
- **Train / val / test** are the store photos (`DSC*`), split 70/15/15 by unique photo with `SEED = 42` (notebook 1.0,
  section 6), plus the SKU-110K full shelves split 70/15/15 with the same seed. The split is read from
  `manifest.csv`; it is never re-randomized.
- **Distribution-shift setting: the lab shelf** (`235_*`, `*lowResFull*`): one shelf unit in a lab, photographed
  repeatedly with products rearranged, at a different site, camera and lighting from the store photos. It is held out
  completely from training and model selection and evaluated once at the end.
- **No overlap.** `python -m barcode.split` asserts that no stem, copy cluster or duplicate group appears in two sets.
- **Changed from the pre-checkpoint split (v1):** the lab images were removed from train/val/test into shift-lab. The store
  photos keep the same split, so val and test are the same store photos as in v1.

## Raw data in git

The assignment asks for raw data not to be committed. For the checkpoint we keep the Roboflow export (CC BY 4.0, 146 MB)
in git so the one-command pipeline runs from a fresh clone without a Roboflow login. It will be removed before the final
submission, and the pipeline will then start with the download steps above.
