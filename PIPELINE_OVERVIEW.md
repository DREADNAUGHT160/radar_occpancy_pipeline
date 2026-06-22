# Radar 3D Occupancy Pipeline — Complete Overview

## What it does

Takes raw 4D automotive radar data and trains a neural network to predict a 3D occupancy grid — answering "is there something at this (range, azimuth, elevation) position?" — using LiDAR as ground truth during training.

---

## End-to-End Data Flow

```
Raw .mat radar files          Raw .h5 LiDAR point clouds
        │                               │
extract_mat_to_npy.py          generate_lidar_labels.py
        │                               │
  rad_power/*.npy               labels/*.npy
  rad_elev/*.npy            (64×256×256 occupancy grid)
        │                               │
        └──────────── dataloader.py ────┘
                            │
                       train.py  ←── train_config.yaml
                            │
                       predict.py  (inference + plots)
                            │
                   project_to_image.py (camera overlay)
```

---

## 1. Data Preparation (`dataset/`)

### `prepare_dataset.py` — unified entry point
Run once on a new machine before training:
```bash
python dataset/prepare_dataset.py --config configs/train_config.yaml
```
For each RC folder in `train + val + test`:
1. Calls `extract_mat_to_npy` — converts radar `.mat` cubes to `.npy`
2. Calls `generate_lidar_labels` — projects LiDAR points into occupancy grid
3. Copies camera images (`pco/`) and calibration files (`calib/`) for projection
4. Auto-updates `inference.camera` in the config with correct per-dataset paths

### `extract_mat_to_npy.py`
- Reads Matlab `.mat` files containing 4D radar tensors
- Saves two `.npy` cubes per frame: **power** and **elevation**
- Output shape: `(512, 256, 256)` — 512 Doppler bins × 256 range × 256 azimuth
- Filename = timestamp in milliseconds (e.g. `1649671430965.npy`)

### `generate_lidar_labels.py`
- Reads `.h5` LiDAR point clouds (x, y, z in LiDAR frame)
- Converts each point to spherical coordinates relative to the radar:
  - Range: `r = sqrt(x² + y²)`
  - Azimuth: `sin(θ)` linearly mapped to index 0–255
  - Elevation: `sin(φ)` linearly mapped to index 0–63 (with boresight correction applied then cancelled — see critical rules)
- Output: `(64, 256, 256)` uint8 binary occupancy grid
- Syncs radar and LiDAR timestamps within `sync_threshold_ms` (default 100 ms)

---

## 2. Dataset (`dataset/dataloader.py`)

`RadarDataset` loads one frame at a time, returning `(radar_tensor, label_tensor)`.

### Power preprocessing pipeline
1. Transpose `(H, W, D)` → `(D, H, W)` if stored in wrong order
2. 4× Doppler downsampling: `512 → 128` — method controlled by `model.doppler_pool`:
   - `max` (default) — NumPy reshape + `.max(axis=1)`, keeps strongest return per 4-bin window
   - `mean` — NumPy reshape + `.mean(axis=1)`, averages the 4-bin window
   - `stride` — `[::4]` simple subsampling, fastest
   - `torch_max` — `F.max_pool1d(kernel=4, stride=4)`, PyTorch optimised kernel
3. Log transform: `10·log10(x)`, clip to `[power_min_val, power_max_val]`, normalise to `[0, 1]`

### Elevation preprocessing pipeline
1. Doppler downsample always uses `[::4]` stride (elevation is spatial, max not meaningful)
2. Optionally normalise by `elevation_max_angle`

### Axis alignment (critical)
- After Doppler reduction: stored as `(D, Azimuth, Range)` on disk → transposed to `(D, Range, Azimuth)` via `np.transpose(..., (0, 2, 1))`
- Label azimuth is **flipped** at load: `torch.flip(label, [-1])` — LiDAR and radar have opposite azimuth conventions. Any script loading raw labels must replicate this (`gt_label[:, :, ::-1].copy()`)

### Optional per-frame features
| Feature | Config key | What it does |
|---|---|---|
| Range shift | `label_text_dir` | Reads `Translation_Radar_to_Lidar` from calib `.txt`, shifts radar range bins to compensate for sensor offset |
| Elevation mask | `normalization.mask_elevation` | Zeroes elevation channel where radar power < threshold — removes noise |
| Bounding box filter | `dataset.filter_bboxes` | Restricts GT labels to within the calibrated 3D bounding box region |

### Split modes
- **Named folders** (default): explicit `train: [RC013]`, `val: [RC013]`, `test: [RC014]` in config
- **Auto-split** (`auto_split.enabled: true`): `resolve_splits()` in `utils/config_utils.py` scans `base_dir` for RC folders, shuffles with fixed `seed`, splits by `train_ratio`/`val_ratio`. Both `train.py` and `predict.py` call this at startup so they always use the same deterministic split.

---

## 3. Model (`models/`)

### Architecture: `RadarResNet` (ResNet-18 adapted for 5D radar input)

```
Input:  (B, 2, 128, 256, 256)
         2 channels × 128 Doppler × 256 range × 256 azimuth

DopplerHead
├── Conv3d(2→32, k=3) + BN + ReLU
├── Conv3d(32→64, k=3) + BN + ReLU
└── MaxPool3d(128, 1, 1)  →  collapses Doppler entirely

Output after DopplerHead: (B, 64, 256, 256)  ← now a 2D spatial problem

4× BasicBlock stacks (ResNet-18 style):
├── Layer1: 64→64   stride 1  →  256×256
├── Layer2: 64→128  stride 2  →  128×128
├── Layer3: 128→256 stride 2  →  64×64
└── Layer4: 256→512 stride 2  →  32×32

Decoder: bilinear upsample back to 256×256
1×1 Conv → (B, 64, 256, 256)

sigmoid at inference → per-voxel probabilities
```

Output `(64, 256, 256)` = elevation bins × range × azimuth — a full 3D occupancy grid.

---

## 4. Training (`training/train.py`)

### Loop
- Mixed-precision AMP (`torch.amp.autocast`) + gradient scaling (`GradScaler`)
- Adam optimiser + CosineAnnealingLR scheduler
- Best model saved by val loss; falls back to train loss if no val set is configured
- CUDA OOM auto-recovery: halves batch size and retries the epoch automatically
- Clear terminal output each epoch (train loss, val loss, IoU, precision, recall, LR)
- Errors printed with `!!!!` banners so they are immediately visible
- Training only trains — no automatic post-training scripts run
- TensorBoard: loss, IoU, precision, recall, sample prediction images per epoch

### Loss functions (`training/losses.py`)

| Loss | When to use |
|---|---|
| `weighted_bce` | Default. `pos_weight` penalises missed occupancy N× harder than FP. Increase if predictions are too sparse. |
| `bce` | Plain BCE, no weighting. |
| `focal` | Down-weights easy background; focuses training on hard/occupied voxels. `focal_alpha` + `focal_gamma` params. |
| `focal_dice` | Focal + Dice combined. Dice rewards volumetric overlap directly. |
| `bce_dice` | Weighted BCE + Dice. |
| `tversky` | Asymmetric Dice. High `tversky_beta` → maximise recall (fewer missed voxels). |

---

## 5. Inference (`utils/predict.py`)

Runs on `config.dataset.test` by default. Pass `--dataset RC014` to override for a single dataset. Loads the model once, iterates per dataset via `_predict_dataset()`.

### Per frame
1. Load + preprocess radar frame (identical to training pipeline)
2. Forward pass → `sigmoid` → probability grid `(64, 256, 256)`
3. Lookup nearest GT label by timestamp (within 100 ms)
4. Compute metrics at midpoint threshold: IoU, Precision, Recall

### 3×3 plot layout (saved to `prediction_plots/` and `prediction_plots_thresh/`)

| | Col 0 — BEV | Col 1 — Front/AE | Col 2 — Side/RE |
|---|---|---|---|
| **Row 0: Radar input** | Max-power BEV (turbo) | AE map: elev × azimuth | RE map: elev × range |
| **Row 1: GT LiDAR** | Occupancy BEV (gray) | Front view: elev × azimuth | Side view: elev × range |
| **Row 2: Prediction** | Confidence BEV (magma) | Front view | Side view |

`prediction_plots/` = raw confidence values  
`prediction_plots_thresh/` = binarised at dynamic midpoint `(max + min) / 2`

### AE / RE maps
Elevation-indexed mean power maps built from the elevation channel. Skip first 2 Doppler bins (static/clutter). Show where the radar sees in elevation space — useful for debugging alignment with GT.

---

## 6. Camera Projection (`utils/project_to_image.py`)

Projects predicted occupancy voxels onto the camera image for visual verification.

### Steps
1. Find matching calibration `.txt` by timestamp
2. Parse from `.txt`:
   - Camera intrinsic matrix `K`
   - `Lidar_to_PCO` extrinsic `r_t`
   - `Translation_Radar_to_Lidar` offset vector
3. `occupancy_to_points()`: inverts the spherical mapping from label generation
   - Range, azimuth (un-flip the dataloader flip), elevation → Cartesian `(x, y, z)` in radar frame
   - **No boresight correction in the inverse** — the forward transform applies −5° then +5°; they cancel to zero
4. Shift points to LiDAR frame: `pts += radar_to_lidar`
5. SAVEROAD toolkit (`project_points_v2_withPC`) reprojects 3D → 2D using `K` and `r_t`
6. Colour each point by prediction confidence (turbo colormap)

Requires external SAVEROAD_DataLoader toolkit (Cython `.pyd`). Set `inference.saveroad_dir` in config.

---

## 7. Evaluation (`utils/evaluate.py`)

Runs on the full dataset (all splits). Produces a mosaic of prediction plots and computes aggregate metrics (IoU, precision, recall) across all frames.

---

## 8. Configuration (`configs/train_config.yaml`)

All parameters in one file. Key sections:

| Section | Purpose |
|---|---|
| `dataset.raw_data_dir` | Root of raw `.mat`/`.h5` collections |
| `dataset.base_dir` | Where prepared `.npy` data lives |
| `dataset.train/val/test` | Named RC folder lists for each split |
| `dataset.auto_split` | Auto-discover RC folders and split by ratio |
| `dataset.subfolders` | Override subfolder names if your layout differs |
| `dataset.normalization` | Log transform, min/max clipping, elevation normalisation |
| `dataset.sync_threshold_ms` | Max radar–LiDAR timestamp delta for a valid match |
| `model` | `in_channels`, `doppler_depth`, `num_classes`, `doppler_pool` (max/mean/stride/torch_max) |
| `training` | Epochs, lr, loss function + its params |
| `logging.output_dir` | Checkpoint output root |
| `inference.checkpoint` | Path to `.pth` for prediction (blank = latest run) |
| `inference.threshold` | Occupancy threshold (null = dynamic midpoint) |
| `inference.saveroad_dir` | Path to SAVEROAD toolkit |
| `inference.camera.<RC>` | Per-dataset `pco_dir` and `label_txt_dir` |

---

## Critical Rules (never break these)

### 1. Azimuth flip
GT labels are stored in LiDAR convention. The dataloader flips the azimuth axis on load:
```python
label_tensor = torch.flip(label_tensor, [-1])   # in dataloader.py
gt_label = gt_label[:, :, ::-1].copy()           # in any script loading raw labels directly
```

### 2. Axis order after preprocessing
Stored on disk: `(D, Azimuth, Range)`  
After `np.transpose(..., (0, 2, 1))`: `(D, Range, Azimuth)`  
Model input tensor axis order: `(Channels, Doppler, Range, Azimuth)`

### 3. Boresight cancellation in elevation
The forward transform in `generate_lidar_labels.py` applies:
- `phi_aln = phi_raw - 5°` (align to boresight)
- `phi_map = phi_aln + 5°` (centre the grid)

These cancel: `phi_map = phi_raw`. The inverse in `occupancy_to_points()` therefore recovers `phi_raw` directly — **no boresight offset needed**.

### 4. Doppler bins to skip for AE/RE maps
Skip the first 2 Doppler channels — they contain static/clutter returns. Start from index 2:
```python
p_mov = power_np[2:]
e_mov = e_bins[2:]
```

### 5. NumPy 2.0 compatibility
`np.fromstring()` was removed. All calibration file parsing uses:
```python
np.array(match.group(1).strip().split(), dtype=float)
```

### 6. Timestamp formats
- `rad_power/` and `labels/`: 13-digit millisecond integers (`1649671430965.npy`)
- `calib/`: decimal seconds (`1649671430.965.txt`)

`_extract_ts_ms()` handles both:
```python
val = float(match.group(0))
return int(val * 1000) if val < 1e11 else int(val)
```

---

## Project Structure

```
configs/
  train_config.yaml          all configuration
dataset/
  prepare_dataset.py         unified data preparation entry point
  extract_mat_to_npy.py      radar .mat → .npy
  generate_lidar_labels.py   LiDAR .h5 → occupancy grid .npy
  dataloader.py              PyTorch Dataset
models/
  factory.py                 ModelFactory.get_model(config)
  resnet_backbone.py         DopplerHead + ResNet-18 architecture
training/
  train.py                   training loop
  losses.py                  all loss functions
utils/
  predict.py                 inference → 3×3 plots + camera projection
  project_to_image.py        occupancy grid → camera image overlay
  evaluate.py                batch evaluation with metrics
  config_utils.py            resolve_splits() for auto_split support
  logger.py / report.py      logging utilities
CLAUDE.md                    Claude Code project instructions
PIPELINE_OVERVIEW.md         this file
```

---

## Standard Workflow

```bash
# 1. Configure paths in configs/train_config.yaml
#    Set: base_dir, train/val/test lists, doppler_pool (optional)

# 2. Prepare data (run once)
python dataset/prepare_dataset.py --config configs/train_config.yaml

# 3. Train — saves best_model.pth and final_model.pth
python training/train.py --config configs/train_config.yaml

# 4. Basic model check (set eval_mode: basic in eval_config.yaml)
python utils/thesis_eval.py --config configs/eval_config.yaml \
    --checkpoint checkpoints/<run_id>/best_model.pth

# 5. Full thesis evaluation (set eval_mode: weather in eval_config.yaml)
python utils/thesis_eval.py --config configs/eval_config.yaml \
    --checkpoint checkpoints/<run_id>/best_model.pth

# 6. Camera projection
python utils/project_to_image.py --config configs/train_config.yaml --dataset RC014
```

---

## Environment

- Python 3.12, PyTorch 2.x, CUDA 12.x
- Conda environment: `thesis_model`
- Camera projection requires SAVEROAD_DataLoader toolkit (Cython `.pyd`, must match Python version)
- See `requirements.txt` for full dependency list
