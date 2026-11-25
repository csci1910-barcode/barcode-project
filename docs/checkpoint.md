# Checkpoint: Team Barcode (2026-11-24)

Finding empty shelf space (`gap` boxes) on retail shelf photos.

## Pipeline

One command, `make checkpoint` (about 15 minutes on an M-series laptop CPU), runs in order:

1. **`python -m barcode.split`**: raw Roboflow COCO export + saved index (`data/interim/retail-gap/manifest.csv`, written by
   notebook 1.0) → fixed **split v2**. Labels merged to one class `gap`; near-duplicates grouped (same Roboflow source or
   flip-aware perceptual hash ≤ 12 for store photos / ≤ 4 for the lab shelf); grouped 70/15/15 by unique photo, seed 42.
   Train 2,862 / val 154 / test 157 images (store photos + 57 SKU-110K full shelves); **shift-lab 109** (the lab shelf,
   held out). It asserts that no group, stem or copy cluster appears in two sets. Details: `data/README.md`.
2. **`python -m barcode.modeling.baselines --split val --seeds 0,1,2`**: every image cut into 169 tiles of 32×32,
   hand-made tile features (colour, brightness, edges, HOG, position) → non-deep baselines and a small MLP, 3 seeds; the
   pilot YOLO11n is scored on the same tiles → `reports/checkpoint_baselines_val.csv` and `_summary.csv`.

Run from a fresh clone by: **TODO (name, date)**.

## Numbers

| Model | Seeds | Gap F1 | Gap precision | Gap recall | Empty-share error |
|---|---|---|---|---|---|
| Always "no gap" (dummy) | 3 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.139 ± 0.000 |
| Naive Bayes | 3 | 0.354 ± 0.001 | 0.255 ± 0.001 | 0.582 ± 0.002 | 0.188 ± 0.001 |
| Logistic regression | 3 | 0.554 ± 0.001 | 0.412 ± 0.001 | 0.843 ± 0.001 | 0.147 ± 0.001 |
| Decision tree | 3 | 0.547 ± 0.006 | 0.419 ± 0.011 | 0.786 ± 0.020 | 0.126 ± 0.011 |
| Random forest | 3 | 0.553 ± 0.001 | 0.772 ± 0.003 | 0.431 ± 0.001 | 0.066 ± 0.000 |
| **SVM (RBF), best non-deep** | 3 | 0.608 ± 0.007 | 0.501 ± 0.009 | 0.773 ± 0.006 | 0.089 ± 0.003 |
| **MLP 128-64, simple neural baseline** | 3 | 0.634 ± 0.009 | 0.690 ± 0.009 | 0.587 ± 0.020 | 0.048 ± 0.000 |
| YOLO11n pilot (E1), one run | 1 | 0.856 | 0.794 | 0.929 | 0.034 |

Validation split v2, 154 images = 26,026 tiles (13.9% gap). Mean ± standard deviation over seeds 0, 1, 2 (each seed draws its own 150k-tile training sample and model seed). Accuracy is not used: always answering "no gap" scores 0.861 and finds nothing. Empty-share error = mean absolute error of the share of each shelf that is empty. The YOLO11n row is the pilot from E1 (trained on split v1, whose train set contains none of these validation photos), one run, shown for scale; the deep models are retrained on split v2 next. Run log: `experiments/README.md` (E4–E7).

## Next

| Item | What | Owner |
|---|---|---|
| First deep model | YOLO11n, COCO-pretrained, fine-tuned on split v2 with the E3 schedule, seeds 0/1/2, selected on val | Feriha Ozturk |
| First ablation | pretraining: COCO-pretrained vs. trained from scratch, same schedule, 3 seeds (second factor afterwards: mosaic augmentation on/off) | Tahir Zogaj |
| Responsible-AI evaluation (HW10P4) | shortcut and subgroup audit: re-encode the SKU-110K full shelves to 416×416 like the Roboflow images and compare false alarms; false-alarm rate on ~50 of our own full-shelf phone photos; recall for dark / medium / bright stores (cut points 74.5 / 98.5, notebook 1.0) and on shift-lab. Pass: ≤ 10% false alarms, ≤ 5 points recall gap between lighting groups | Qazim Pali |

Over Thanksgiving: the first deep model and the ablation train on the owners' laptops (YOLO11n: about 2 h for 50 epochs);
RT-DETR-L and Faster R-CNN follow on Colab GPUs.

## Risk

**The source shortcut.** Image size and JPEG statistics identify the 57 SKU-110K full shelves perfectly (AUC 1.0,
notebook 1.1): every gap photo is 416×416, and no full shelf is. A low false-alarm rate on full shelves could therefore
come from recognising the source rather than an intact shelf. **Check:** retrain with the full shelves re-encoded to
416×416 at matching JPEG quality and compare the false-alarm rate, and score full-shelf photos from a different
source (our own phone photos), which the shortcut cannot help with.
