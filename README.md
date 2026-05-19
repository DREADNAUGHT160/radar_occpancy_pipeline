# Radar 3D Occupancy Pipeline

Predicts a 3D occupancy grid from 4D automotive radar data using a deep learning model trained with LiDAR-derived ground truth.

## Architecture

**Input:** Radar cube `(B, 2, 128, 256, 256)` — channel 0 = power, channel 1 = elevation; 128 Doppler bins × 256 range × 256 azimuth.

**Model:** `RadarResNet`
- `DopplerHead` — two 3D conv layers + MaxPool3d collapses Doppler → `(B, 64, 256, 256)`
- Four ResNet-18-style `BasicBlock` layers (64 → 128 → 256 → 512 channels)
- 1×1 conv output head → `(B, 64, 256, 256)` (64 elevation bins × 256 range × 256 azimuth)

**Output:** Logits → `sigmoid` + threshold at inference.

---

## Quick Start (new machine)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure paths
Edit `configs/train_config.yaml`:
```yaml
dataset:
  raw_data_dir: E:/DataCollectionApril2022_11_03   # raw MAT/H5/PNG data
  base_dir: D:/dataset                              # where prepared data will be written
  train: [RC013, RC019]
  val:   [RC013]
  test:  [RC014]
```

### 3. Prepare dataset (run once)
```bash
python dataset/prepare_dataset.py --config configs/train_config.yaml
```
Extracts radar `.mat` → `.npy`, generates LiDAR occupancy labels, copies camera images and calibration files.

### 4. Train
```bash
python training/train.py --config configs/train_config.yaml
```
Checkpoints saved to `<logging.output_dir>/<timestamp>/`.

### 5. Predict + visualise
```bash
python utils/predict.py \
    --config  configs/train_config.yaml \
    --dataset RC014 \
    --out_dir verification_output/RC014_run1
```
Produces per-frame 3×3 plots (radar input | LiDAR GT | prediction) and optional camera projection images.

---

## Dataset layout (after prepare_dataset.py)

```
<base_dir>/
  RC013/
    rad_power/   <timestamp_ms>.npy   (512, 256, 256)
    rad_elev/    <timestamp_ms>.npy   (512, 256, 256)
    labels/      <timestamp_ms>.npy   (64, 256, 256) uint8
    pco/         *.png   camera images
    calib/       *.txt   calibration files (PCO intrinsics + Lidar_to_PCO)
```

---

## Configuration reference

All parameters live in `configs/train_config.yaml`.

| Key | Description |
|---|---|
| `dataset.raw_data_dir` | Root of raw collections (subfolders `<N>_<RCNAME>/`) |
| `dataset.base_dir` | Where prepared data is written and read from |
| `dataset.subfolders.*` | Override subfolder names (rad_power, rad_elev, labels, pco, calib) |
| `dataset.train/val/test` | Lists of RC folder names for each split |
| `dataset.auto_split.enabled` | Auto-discover RC folders and split frames randomly |
| `dataset.sync_threshold_ms` | Max radar–LiDAR timestamp delta for a valid match (default 100 ms) |
| `model.doppler_depth` | Doppler bins after downsampling (128 default) |
| `model.num_classes` | Elevation bins in output (must match label shape, default 64) |
| `training.loss` | `weighted_bce`, `bce`, `focal`, `focal_dice`, `bce_dice`, `tversky` |
| `training.pos_weight` | Foreground weight for `weighted_bce` / `bce_dice` |
| `logging.output_dir` | Where checkpoints are written |
| `inference.saveroad_dir` | Path to SAVEROAD_DataLoader toolkit (enables camera projection) |
| `inference.threshold` | Occupancy threshold (null = dynamic midpoint) |

---

## Project structure

```
configs/
  train_config.yaml        main config
  smoke_test_config.yaml   quick 2-epoch test
dataset/
  prepare_dataset.py       unified data preparation entry point
  extract_mat_to_npy.py    radar .mat → .npy
  generate_lidar_labels.py LiDAR .h5 → occupancy grid .npy
  dataloader.py            PyTorch Dataset
models/
  factory.py               ModelFactory.get_model(config)
  resnet_backbone.py       DopplerHead + ResNet-18 architecture
training/
  train.py                 training loop
  losses.py                all loss functions
utils/
  predict.py               inference → 3×3 plots + camera projection
  project_to_image.py      occupancy grid → camera image overlay
  evaluate.py              batch evaluation with metrics
  logger.py / report.py    logging utilities
CLAUDE.md                  Claude Code project instructions
```

---

## Requirements

Python 3.12, PyTorch 2.x, CUDA 12.x recommended. See `requirements.txt`.

Camera projection additionally requires the [SAVEROAD_DataLoader](https://github.com/yourorg/SAVEROAD_DataLoader) toolkit compiled for your Python version.
