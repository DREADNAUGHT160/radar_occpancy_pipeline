# Radar 3D Occupancy Pipeline

Deep learning pipeline that predicts a 3D occupancy grid from 4D automotive radar data, trained with LiDAR-derived ground truth. Built for a thesis comparing radar-based perception against CFAR across clear, fog, and rain weather conditions.

**Repository:** `D:\thesis data\radar_occpancy_pipeline\`

---

## Table of Contents

1. [What It Does](#1-what-it-does)
2. [Project Structure](#2-project-structure)
3. [Installation](#3-installation)
4. [Data Preparation](#4-data-preparation)
5. [Training](#5-training)
6. [Evaluation — Basic Mode](#6-evaluation--basic-mode)
7. [Evaluation — Thesis Mode](#7-evaluation--thesis-mode)
8. [Prediction Plots](#8-prediction-plots)
9. [Camera Projection](#9-camera-projection)
10. [Model Architecture](#10-model-architecture)
11. [Loss Functions](#11-loss-functions)
12. [Configuration Reference](#12-configuration-reference)
13. [Critical Rules](#13-critical-rules)

---

## 1. What It Does

```
Raw 4D Radar (.mat)  ──►  extract_mat_to_npy.py  ──►  rad_power/*.npy
                                                        rad_elev/*.npy
                                                              │
Raw LiDAR (.h5)      ──►  generate_lidar_labels.py ──►  labels/*.npy
                                                     (64×256×256 occupancy)
                                                              │
                                                        dataloader.py
                                                              │
                                                   train.py ◄── train_config.yaml
                                                              │
                                                        best_model.pth
                                                              │
                                          ┌───────────────────┼────────────────────┐
                                          │                   │                    │
                                     eval_plots.py      thesis_eval.py      camera_check.py
                               (visual verification)  (thesis metrics)    (camera overlay)
```

**Input:** Raw 4D radar cubes `(512, 256, 256)` — 512 Doppler × 256 range × 256 azimuth  
**Output:** 3D occupancy grid `(64, 256, 256)` — 64 elevation bins × 256 range × 256 azimuth

---

## 2. Project Structure

```
radar_occpancy_pipeline/
│
├── configs/
│   ├── train_config.yaml          Main training config (paths, model, loss, logging)
│   ├── prepare_config.yaml        Minimal config for data preparation only
│   └── eval_config.yaml           Standalone evaluation config (basic + weather + plots)
│
├── dataset/
│   ├── prepare_dataset.py         Unified entry point: radar + LiDAR + camera prep
│   ├── extract_mat_to_npy.py      Converts radar .mat → .npy (power + elevation cubes)
│   ├── generate_lidar_labels.py   Projects LiDAR .h5 points → 3D occupancy .npy
│   ├── check_sync.py              Verifies radar–LiDAR–camera timestamp sync
│   └── dataloader.py              PyTorch Dataset (sync, normalise, augment)
│
├── models/
│   ├── factory.py                 ModelFactory.get_model(config)
│   └── resnet_backbone.py         DopplerHead + RadarResNet-18 architecture
│
├── training/
│   ├── train.py                   Full training loop (AMP, TensorBoard, checkpointing)
│   └── losses.py                  6 loss functions: BCE, Focal, Tversky, Dice variants
│
├── utils/
│   ├── predict.py                 Inference → 3×3 mosaic plots + optional camera projection
│   ├── evaluate.py                Full dataset evaluation with mosaic plots (all frames)
│   ├── eval_plots.py              Equally spaced prediction plots for quick verification
│   ├── camera_check.py            Camera projection overlay for N equally spaced frames
│   ├── thesis_eval.py             Thesis evaluation: CFAR vs DL model, all 4 experiments
│   ├── project_to_image.py        Occupancy grid → camera image overlay (SAVEROAD)
│   ├── config_utils.py            resolve_splits() for auto_split mode
│   ├── logger.py                  Logging setup
│   ├── report.py                  Post-training report generation
│   └── tb_logger.py               TensorBoard image logging
│
├── PIPELINE_OVERVIEW.md           Detailed technical documentation
├── README.md                      This file
└── requirements.txt               Python dependencies
```

---

## 3. Installation

### Conda environment (recommended)

```bash
conda create -n thesis_model python=3.12
conda activate thesis_model
pip install -r requirements.txt
```

### Requirements

- Python 3.12
- PyTorch 2.x + CUDA 12.x
- numpy, scipy, matplotlib, tqdm, pillow, pyyaml, tensorboard, opencv-python

Camera projection additionally requires the **SAVEROAD_DataLoader** toolkit:
```
C:\Users\evinb\Desktop\thesis\SAVEROAD_DataLoader\tools\project_points_v2_withPC.cp312-win_amd64.pyd
```

---

## 4. Data Preparation

Run **once** on each machine before training. Converts raw `.mat` and `.h5` files to training-ready `.npy`.

### Option A — Use prepare_config.yaml (data only)

```yaml
# configs/prepare_config.yaml
raw_data_dir: 'E:/DataCollectionApril2022_11_03'
base_dir:     'D:/dataset'
train: [RC013, RC019]
val:   [RC013]
test:  [RC014]
```

```bash
python dataset/prepare_dataset.py --config configs/prepare_config.yaml
```

### Option B — Use train_config.yaml (data + training in one)

```yaml
# configs/train_config.yaml
dataset:
  raw_data_dir: 'E:/DataCollectionApril2022_11_03'
  base_dir:     'D:/dataset'
```

```bash
python dataset/prepare_dataset.py --config configs/train_config.yaml
```

### What it produces

```
D:/dataset/
  RC013/
    rad_power/   1649671430965.npy   shape: (512, 256, 256) float32
    rad_elev/    1649671430965.npy   shape: (512, 256, 256) float32
    labels/      1649671430965.npy   shape: (64, 256, 256)  uint8
    pco/         *.png               camera images
    calib/       *.txt               calibration files
```

### Timestamp sync check

```bash
python dataset/check_sync.py --src_folder E:/DataCollectionApril2022_11_03 --rc RC013
```

---

## 5. Training

### Configure `configs/train_config.yaml`

```yaml
dataset:
  base_dir: 'D:/dataset'
  train: [RC013, RC019]
  val:   [RC013]
  test:  [RC014]
  batch_size: 1
  num_workers: 4        # set to 0 if multiprocessing causes issues

model:
  doppler_pool: max     # 'max' (default) | 'mean' | 'stride' | 'torch_max'

training:
  epochs: 10
  lr: 0.001
  loss: weighted_bce
  pos_weight: 10.0

logging:
  output_dir: checkpoints
  tensorboard: true
```

### Run training

```bash
# PowerShell — recommended for GPU memory
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
python training/train.py --config configs/train_config.yaml
```

### Terminal output per epoch

```
--- Epoch 3/10 ---
  Train Loss : 0.2841
  Val Loss   : 0.2512  |  Acc: 0.9831
  IoU: 0.1643  |  Precision: 0.4320  |  Recall: 0.4011
  LR         : 0.000872
  *** Best model saved (val_loss=0.2512) ***
```

### CUDA OOM — automatic recovery

If the GPU runs out of memory the batch size is automatically halved and the epoch retried:

```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  CUDA OUT OF MEMORY
  batch_size=6 is too large for this GPU
  Retrying epoch 1 with batch_size=3 ...
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

### Outputs

```
checkpoints/
  20260620_104847/          ← run_id (timestamp)
    best_model.pth          saved when val loss improves (or train loss if no val set)
    final_model.pth         saved at end of last epoch
    training.log            full timestamped log
    tensorboard/            TensorBoard event files
    report.json             metrics summary
```

Training **only trains** — evaluation is run separately (see sections 6–9 below).

### Monitor with TensorBoard

```bash
tensorboard --logdir checkpoints/20260620_104847/tensorboard
```

---

## 6. Evaluation — Basic Mode

**Purpose:** Quickly verify the model is producing meaningful predictions on clear-weather data. Computes per-voxel IoU / Precision / Recall.

### Configure `configs/eval_config.yaml`

```yaml
model:
  name: resnet18_radar
  in_channels: 2
  doppler_depth: 128
  num_classes: 64

checkpoint: 'checkpoints/20260620_104847/best_model.pth'

base_dir: 'D:/dataset'

eval_mode: basic    # ← select basic mode

basic:
  dataset:   RC_clear    # RC folder to evaluate
  threshold: 0.4

out_dir: verification_output/eval
```

### Run

```bash
python utils/thesis_eval.py --config configs/eval_config.yaml
```

### Output

```
──────────────────────────────────────────────
Metric             Global    Per-frame mean
──────────────────────────────────────────────
IoU                 0.2341            0.2187
Precision           0.4812            0.4650
Recall              0.3920            0.3771
──────────────────────────────────────────────
  Model is producing meaningful predictions.

Results saved → verification_output/eval/basic_results.csv
```

**Verdict guide:**
| IoU | Meaning |
|---|---|
| > 0.10 | Model is producing meaningful predictions |
| 0.01 – 0.10 | Detecting something — may need more training |
| < 0.01 | Near zero — check threshold, data paths, or training |

---

## 7. Evaluation — Thesis Mode

**Purpose:** Full 4-experiment thesis evaluation comparing the DL model against CFAR across clear, fog, and rain weather.

### Configure `configs/eval_config.yaml`

```yaml
eval_mode: weather    # ← select weather mode

weather:
  threshold: 0.4
  weather_splits:
    clear: [RC_clear]
    fog:   [RC_fog]
    rain:  [RC_rain]
  cfar_dir: 'D:/cfar_outputs'   # root folder: cfar_dir/<RC_folder>/<timestamp>.npy
                                 # CFAR files: (N,3) XYZ  or  (N,4) XYZ+confidence
```

### Run

```bash
python utils/thesis_eval.py --config configs/eval_config.yaml
```

### Experiments

| # | Name | Metrics | Weather |
|---|---|---|---|
| 1 | Clear accuracy | AP, P_d, P_fa, Chamfer Distance | Clear only |
| 2 | Weather robustness | AP, P_d, P_fa | Fog + Rain |
| 3 | Degradation analysis | `(clear − weather) / clear × 100%` | Fog vs Clear, Rain vs Clear |
| 4 | Point density | Points per m³ inside GT box at 0–10m, 10–15m, 15–20m | All |

### Metric definitions

| Metric | Description |
|---|---|
| **AP** | Average precision — precision-recall curve, point-in-box criterion |
| **P_d** | Detection probability — fraction of frames where ≥1 point is inside GT box |
| **P_fa** | False alarm rate — points outside GT box / total points |
| **Chamfer Distance** | Mean nearest-neighbour distance between DL/CFAR points and LiDAR GT (clear only) |

### Output

```
============================================================
RESULTS TABLE
============================================================
Weather  Method     AP    P_d    P_fa        CD
──────────────────────────────────────────────────
clear    DL      0.412  0.873  0.091    0.0842
clear    CFAR    0.287  0.742  0.214    0.1230
fog      DL      0.356  0.801  0.127       N/A
fog      CFAR    0.198  0.604  0.387       N/A
...

Results saved → verification_output/eval/weather_results.csv
```

---

## 8. Prediction Plots

**Purpose:** Equally spaced visual plots for quick model checking — same mosaic layout as `evaluate.py` but only N frames instead of all of them.

### Configure `configs/eval_config.yaml`

```yaml
eval_plots:
  dataset:   RC_clear
  n_plots:   10         # number of equally spaced frames
  threshold: 0.4
  pco_dir:   ''         # optional: path to camera images for a 4th reference column
```

### Run

```bash
python utils/eval_plots.py --config configs/eval_config.yaml --checkpoint checkpoints/<run_id>/best_model.pth

# Override number of plots from command line
python utils/eval_plots.py --config configs/eval_config.yaml --n_plots 12
```

### Output

Each plot is a **4-row mosaic**:

| Row | Content |
|---|---|
| 1 | Radar input — BEV power, AE map (elevation × azimuth), RE map (elevation × range) |
| 2 | Ground-truth occupancy — BEV, front view, side view |
| 3 | Raw prediction probability (magma colormap) — BEV, front view, side view |
| 4 | Thresholded prediction (binary) — BEV, front view, side view |

```
verification_output/eval/plots/
  plot_000_1649671430965.png
  plot_006_1649671436821.png
  ...
```

---

## 9. Camera Projection

### Full projection (`utils/project_to_image.py`)

Projects predicted occupancy grid onto camera images using the SAVEROAD toolkit. Called automatically after training if `save_inference_images: true`.

```bash
python utils/project_to_image.py \
  --config    configs/train_config.yaml \
  --dataset   RC014 \
  --checkpoint checkpoints/<run_id>/best_model.pth \
  --threshold 0.4
```

### Quick check (`utils/camera_check.py`)

Generates equally spaced side-by-side comparisons (original image vs overlay) for fast visual verification.

```yaml
# configs/eval_config.yaml
camera:
  dataset:      RC_clear
  n_plots:      10
  threshold:    0.4
  saveroad_dir: 'C:/Users/evinb/Desktop/thesis/SAVEROAD_DataLoader'
  pco_dir:      'D:/dataset/RC_clear/pco'
  calib_dir:    ''    # auto-detected as <dataset>/calib if empty
```

```bash
python utils/camera_check.py --config configs/eval_config.yaml --checkpoint checkpoints/<run_id>/best_model.pth
```

Each output image shows:
- **Left:** Original camera image
- **Right:** Radar predictions overlaid (turbo colormap, confidence-coloured)

```
verification_output/eval/camera_check/
  cam_000_1649671430965.png
  cam_006_1649671436821.png
  ...
```

---

## 10. Model Architecture

```
Input: (B, 2, 128, 256, 256)
       └── 2 channels: [power, elevation]
           128 Doppler bins × 256 range × 256 azimuth

┌─ DopplerHead ────────────────────────────────────────────────┐
│  Conv3d(2→16, k=5×3×3)  + BN + ReLU                         │
│  Conv3d(16→64, k=5×3×3) + BN + ReLU                         │
│  MaxPool3d(128×1×1)  →  collapses Doppler entirely           │
└──────────────────────── (B, 64, 256, 256) ───────────────────┘

┌─ ResNet-18 Backbone ─────────────────────────────────────────┐
│  Layer 1: 2× BasicBlock(64→64)   →  (B,  64, 256, 256)      │
│  Layer 2: 2× BasicBlock(64→128)  →  (B, 128, 256, 256)      │
│  Layer 3: 2× BasicBlock(128→256) →  (B, 256, 256, 256)      │
│  Layer 4: 2× BasicBlock(256→512) →  (B, 512, 256, 256)      │
└──────────────────────────────────────────────────────────────┘

1×1 Conv(512→64)  →  (B, 64, 256, 256)

Output: (B, 64, 256, 256)
        64 elevation bins × 256 range × 256 azimuth
        sigmoid → per-voxel occupancy probability
```

**BasicBlock** = Conv2d(3×3) → BN → ReLU → Conv2d(3×3) → BN → residual add → ReLU. No spatial downsampling (stride=1 throughout).

---

## 11. Loss Functions

Configured via `training.loss` in `train_config.yaml`. All use `BCEWithLogitsLoss` internally.

| Loss | Key params | Best for |
|---|---|---|
| `weighted_bce` | `pos_weight: 10.0` | **Default.** Class imbalance (>99.9% empty voxels). Increase `pos_weight` if predictions are too sparse. |
| `bce` | — | Plain BCE, no weighting. Baseline. |
| `focal` | `focal_alpha: 0.25`, `focal_gamma: 2.0` | Down-weights easy background, focuses on hard voxels. |
| `focal_dice` | + `dice_weight: 0.5` | Focal + volumetric Dice overlap. |
| `bce_dice` | `pos_weight: 10.0`, `dice_weight: 0.5` | Weighted BCE + Dice. |
| `tversky` | `tversky_alpha: 0.3`, `tversky_beta: 0.7` | Asymmetric Dice — high `beta` maximises recall (fewer missed voxels). |

**Chosen for this thesis:** `weighted_bce` with `pos_weight: 10.0`.

---

## 12. Configuration Reference

### `configs/train_config.yaml` — training

| Key | Default | Description |
|---|---|---|
| `dataset.raw_data_dir` | `''` | Root folder of raw `.mat`/`.h5` collections |
| `dataset.base_dir` | `''` | Where prepared `.npy` data is written and read from |
| `dataset.train/val/test` | `[]` | Lists of RC folder names for each split |
| `dataset.batch_size` | `1` | Batch size per GPU |
| `dataset.num_workers` | `4` | DataLoader worker processes |
| `dataset.sync_threshold_ms` | `100` | Max radar–LiDAR timestamp delta for a valid match |
| `dataset.normalization.power_log_transform` | `true` | Apply 10·log10 to power values |
| `dataset.normalization.power_min_val` | `-100.0` | Clip range minimum (dB) |
| `dataset.normalization.power_max_val` | `-31.7` | Clip range maximum (dB) |
| `model.in_channels` | `2` | Number of input channels (power + elevation) |
| `model.doppler_depth` | `128` | Doppler bins after downsampling |
| `model.num_classes` | `64` | Elevation bins in output |
| `model.doppler_pool` | `max` | Doppler 4× downsampling method: `max` (numpy), `mean` (numpy), `stride` (subsample), `torch_max` (PyTorch F.max_pool1d) |
| `training.epochs` | `10` | Number of training epochs |
| `training.lr` | `0.001` | Initial learning rate (Adam + CosineAnnealingLR) |
| `training.loss` | `weighted_bce` | Loss function name |
| `training.pos_weight` | `10.0` | Positive class weight for `weighted_bce` |
| `logging.output_dir` | `checkpoints` | Root for checkpoint folders |
| `logging.tensorboard` | `true` | Enable TensorBoard logging |
| `logging.save_inference_images` | — | Unused — evaluation is run separately via `thesis_eval.py` |
| `inference.threshold` | `null` | Occupancy threshold (`null` = dynamic midpoint) |
| `inference.saveroad_dir` | `''` | Path to SAVEROAD toolkit |

### `configs/eval_config.yaml` — evaluation

| Key | Description |
|---|---|
| `checkpoint` | Path to `.pth` file |
| `base_dir` | Same as training `dataset.base_dir` |
| `eval_mode` | `basic` or `weather` |
| `basic.dataset` | RC folder to evaluate (clear weather) |
| `basic.threshold` | Occupancy threshold for binarising predictions |
| `weather.weather_splits` | `clear/fog/rain` → list of RC folders |
| `weather.cfar_dir` | Root folder for CFAR `.npy` point clouds |
| `weather.threshold` | Occupancy threshold for DL model |
| `eval_plots.n_plots` | Number of equally spaced frames to plot |
| `eval_plots.pco_dir` | Camera image folder for optional reference column |
| `camera.saveroad_dir` | Path to SAVEROAD toolkit |
| `camera.pco_dir` | Camera image folder |
| `out_dir` | Output root for all eval scripts |

---

## 13. Critical Rules

These must be preserved in any modification to the pipeline.

### 1. Azimuth flip
GT labels are stored in LiDAR azimuth convention (reversed vs radar). The dataloader flips on load:
```python
label_tensor = torch.flip(label_tensor, [-1])   # dataloader.py
```
Any script loading raw `.npy` labels directly must also apply:
```python
gt_label = gt_label[:, :, ::-1].copy()
```

### 2. Axis order on disk vs in model
- **On disk:** `(D, Azimuth, Range)`
- **After `np.transpose(..., (0, 2, 1))`:** `(D, Range, Azimuth)` — this is what the model sees
- **Model input tensor:** `(Channels, Doppler, Range, Azimuth)`

### 3. Boresight cancellation
`generate_lidar_labels.py` applies `−5°` (align to boresight) then `+5°` (centre the elevation grid) — they cancel. The inverse in `occupancy_to_points()` recovers `phi_raw` directly, **no boresight offset needed**.

### 4. Skip first 2 Doppler bins for AE/RE maps
Bins 0 and 1 contain static/clutter returns. Always start from index 2:
```python
p_mov = power_np[2:]
e_mov = e_bins[2:]
```

### 5. NumPy 2.0 compatibility
`np.fromstring()` was removed. Use:
```python
np.array(match.group(1).strip().split(), dtype=float)
```

### 6. Timestamp formats
| Folder | Format | Example |
|---|---|---|
| `rad_power/`, `labels/` | 13-digit milliseconds | `1649671430965.npy` |
| `calib/` | Decimal seconds | `1649671430.965.txt` |

`_extract_ts_ms()` handles both automatically:
```python
val = float(match.group(0))
return int(val * 1000) if val < 1e11 else int(val)
```

---

## Standard Workflows

### Full pipeline from scratch

```bash
# 1. Prepare data (run once)
python dataset/prepare_dataset.py --config configs/train_config.yaml

# 2. Train — saves best_model.pth + final_model.pth, logs each epoch
python training/train.py --config configs/train_config.yaml

# 3. Basic model sanity check (is it learning?)
#    Set eval_mode: basic and basic.dataset in eval_config.yaml, then:
python utils/thesis_eval.py --config configs/eval_config.yaml \
    --checkpoint checkpoints/<run_id>/best_model.pth

# 4. Quick visual plots
python utils/eval_plots.py --config configs/eval_config.yaml \
    --checkpoint checkpoints/<run_id>/best_model.pth

# 5. Camera overlay check
python utils/camera_check.py --config configs/eval_config.yaml \
    --checkpoint checkpoints/<run_id>/best_model.pth
```

### Full thesis evaluation

```bash
# Edit eval_config.yaml:
#   eval_mode: weather
#   eval_splits:
#     clear: [RC_clear]
#     fog:   [RC_fog]
#     rain:  [RC_rain]

python utils/thesis_eval.py --config configs/eval_config.yaml \
    --checkpoint checkpoints/<run_id>/best_model.pth
```

---

## Environment

| Item | Value |
|---|---|
| Python | 3.12.0 |
| PyTorch | 2.x |
| CUDA | 12.x |
| Conda env | `thesis_model` |
| SAVEROAD toolkit | `C:\Users\evinb\Desktop\thesis\SAVEROAD_DataLoader` |
| Pipeline location | `D:\thesis data\radar_occpancy_pipeline\` |
| GitHub | https://github.com/DREADNAUGHT160/radar_occpancy_pipeline |
