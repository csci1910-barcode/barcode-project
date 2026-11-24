# Fine-tuning YOLO11n for shelf-gap detection

Summary of what was changed between the baseline detector and the fine-tuned one, and how the two compare.

**Result in one line:** the fine-tuned model beats the baseline on every metric on both the validation and the held-out test split. On test, mAP50-95 rises from 0.648 to 0.697 and tile-level gap F1 from 0.891 to 0.919.

| | Baseline | Fine-tuned |
|---|---|---|
| Weights | `runs/detect/baseline-yolo11n/weights/best.pt` | `runs/detect/finetune-yolo11n/weights/best.pt` |
| Code | trained from the `yolo` CLI / notebook | `barcode/modeling/finetune_yolo11n.py` |
| Training | 10 epochs from `yolo11n.pt` | 50 more epochs from the baseline's `best.pt` (60 in total) |
| Test mAP50-95 | 0.648 | **0.697** |
| Best confidence threshold | 0.60 | **0.45** |

## 1. Why fine-tune

The baseline ran for only 10 epochs, and its validation mAP50-95 was still rising at the end (0.33 at epoch 1, 0.674 at epoch 10), with train and validation loss close together. It was under-trained, not over-fitted. Its mAP50 was already 0.93, so it found the gaps, but its boxes were not tight (mAP50-95, which needs a stricter overlap, was only 0.67).

## 2. What changed

The fine-tune continues from the baseline's `best.pt`, not from the generic `yolo11n.pt`. Same data (`data/processed/retail-gap/data.yaml`), same image size (640), same seed (0), same device (Apple MPS).

Values below are read from each run's `args.yaml`.

| Setting | Baseline | Fine-tune | Why |
|---|---|---|---|
| Start weights | `yolo11n.pt` (COCO) | baseline `best.pt` | already a gap detector |
| Epochs | 10 | up to 50 | baseline was still improving |
| Early stopping patience | 100 (never triggered in 10 epochs) | 15 | stop if validation stalls |
| Optimizer | `auto` | AdamW | explicit, so the run is reproducible |
| Peak learning rate (`lr0`) | 0.01 as recorded; `auto` picks its own value at run time | 0.001 | gentle updates to an already-trained model |
| LR schedule | linear decay (`lrf` 0.01) | cosine decay to 5% (`lrf` 0.05) | smooth finish |
| Warm-up | 3 epochs, bias LR 0.1 | 1 epoch, bias LR 0.0 | the default bias warm-up disturbs trained weights |
| Scale augmentation | 0.5 | 0.4 | shelf photos are taken at similar distances |
| Saturation jitter (`hsv_s`) | 0.7 | 0.5 | milder colour shifts |
| Mosaic augmentation | on, but `close_mosaic=10` with 10 epochs means it was effectively off for the whole run | on for epochs 1-40, off for the last 10 | mosaic early for variety, real images at the end |

Not changed, because they were already the baseline defaults: vertical flip off, rotation off, horizontal flip 0.5, translate 0.1, no mixup. Shelves have a fixed orientation, so vertical flip and rotation stay off.

## 3. How to run it

```bash
python -m barcode.modeling.finetune_yolo11n train      # fine-tune, then compare
python -m barcode.modeling.finetune_yolo11n compare    # compare existing runs only
```

Options include `--epochs`, `--name`, `--lr0`, `--patience` and `--no-compare`. The environment is a Python 3.10 virtual environment with `requirements.txt` installed. `opencv-python` must be below version 5, because OpenCV 5 removed `HOGDescriptor`, which `baselines.py` uses.

The full run took about 1.8 hours on an Apple M5 (about 2.3 minutes per epoch).

## 4. How the comparison is done

Both models are scored by the same code, on the validation split (172 images) and the test split (174 images):

- **Detection metrics** from Ultralytics `val()`: precision, recall, mAP50, mAP50-95.
- **Tile-level gap metrics** from `baselines.py`: every image is cut into a 13×13 grid of tiles, and a tile counts as "gap" when more than half of it is inside a box. This is the same scoring used for the classical baselines.
  - At a fixed confidence of 0.25.
  - At a confidence tuned for tile F1 **on validation only**.
- Nothing is chosen on the test split, so the test column is the unbiased one.

The full table is in `reports/yolo_finetune_comparison.csv`.

## 5. Results

### Test split

| Metric | Baseline | Fine-tuned | Change |
|---|---|---|---|
| Precision | 0.864 | 0.917 | +0.053 |
| Recall | 0.894 | 0.929 | +0.036 |
| mAP50 | 0.930 | 0.947 | +0.017 |
| mAP50-95 | 0.648 | 0.697 | +0.049 |
| Tile F1 at conf 0.25 | 0.871 | 0.915 | +0.044 |
| Tile F1 at tuned conf | 0.891 | 0.919 | +0.028 |
| Tile precision at tuned conf | 0.887 | 0.915 | +0.028 |
| Tile recall at tuned conf | 0.896 | 0.924 | +0.028 |
| Empty-share error at tuned conf (lower is better) | 0.019 | 0.014 | −0.005 |

### Validation split

| Metric | Baseline | Fine-tuned | Change |
|---|---|---|---|
| Precision | 0.872 | 0.934 | +0.062 |
| Recall | 0.881 | 0.898 | +0.017 |
| mAP50 | 0.932 | 0.946 | +0.014 |
| mAP50-95 | 0.675 | 0.715 | +0.040 |
| Tile F1 at conf 0.25 | 0.860 | 0.886 | +0.026 |
| Tile F1 at tuned conf | 0.882 | 0.891 | +0.009 |
| Empty-share error at tuned conf | 0.023 | 0.021 | −0.002 |

### Training curve (validation mAP50-95, fine-tune run)

| Epoch | 1 | 5 | 10 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|---|---|
| mAP50-95 | 0.583 | 0.639 | 0.648 | 0.677 | 0.688 | 0.703 | 0.716 |

It starts below the baseline's 0.674 because the optimizer restarts with fresh momentum and a new learning rate. It passes the baseline around epoch 20 and was still improving at epoch 50, which was also the best epoch.

## 6. How to read it

- **Most of the gain is tighter boxes.** mAP50-95 improved about three times as much as mAP50 (+0.049 vs +0.017). The baseline already found the gaps, and the fine-tuned model draws them more accurately.
- **The test gain is at least as large as the validation gain**, even though test was never used for choosing anything. That makes a lucky validation result unlikely.
- **Use confidence 0.45, not 0.25 or 0.60.** The fine-tuned model is more confident on true gaps, so the best threshold moved down from 0.60. At 0.25 it is still better than the baseline, but 0.45 gives the best tile F1.

## 7. Caveats

- **The gain cannot be split between its causes.** The fine-tune differs from the baseline in epoch count, learning rate, warm-up and augmentation, all at once. It also trained for 60 epochs in total, against 10. A control run that only extends the baseline's epochs would show how much each part contributes. It was not run.
- **Small evaluation sets.** Validation has 172 images and test has 174. Differences of about 0.01 are within noise. The mAP50-95 gains (+0.04 and +0.05) are larger than that, but the smaller gains such as tile F1 on validation (+0.009) should not be over-read.
- **One run, one seed.** MPS training is not fully deterministic, so repeating the run may give slightly different numbers.
- **Still improving at epoch 50.** A longer run might go further. It was not tried.
- **`baselines.py` still points at the baseline weights.** Its comparison table against the classical models (`reports/baselines_tiles_val.csv`) was not regenerated for the fine-tuned model. To do that, run `python -m barcode.modeling.baselines --weights runs/detect/finetune-yolo11n/weights/best.pt --yolo-conf 0.45`.
