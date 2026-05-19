# Radar 3D Occupancy Pipeline — Claude Code Instructions

## Project Overview

Thesis project: predict a 3D occupancy grid from a 4D radar cube using deep learning.
- **Input**: 4D radar cube `(Doppler, Range, Azimuth, Elevation)` stored as power + elevation `.npy` files
- **Output**: 3D occupancy grid `(64, 256, 256)` matching LiDAR ground truth
- **Pipeline**: Raw MAT files → extracted `.npy` → trained model → prediction plots + camera projections

---

## Repository Layout

```
E:\radar_occupancy_pipeline\
  configs/
    train_config.yaml        # main config — single source of truth
    smoke_test_config.yaml   # quick 2-epoch smoke test (RC013+RC014, no augment)
  dataset/
    prepare_dataset.py       # ONE-TIME setup: extracts radar, generates labels, copies camera/calib
    extract_mat_to_npy.py    # radar .mat → .npy (called by prepare_dataset.py)
    generate_lidar_labels.py # LiDAR .h5 → occupancy .npy (called by prepare_dataset.py)
    dataloader.py            # PyTorch dataset; reads rad_power/ rad_elev/ labels/
  models/
    factory.py               # ModelFactory.get_model(config)
    resnet_backbone.py       # ResNet-18 + DopplerHead architecture
  training/
    train.py                 # main training loop
    losses.py                # weighted_bce, focal, focal_dice, bce_dice, tversky
  utils/
    predict.py               # inference → 3×3 prediction plots + camera projections
    project_to_image.py      # project occupancy grid onto camera image
    evaluate.py              # metrics (IoU, Precision, Recall)
    logger.py / tb_logger.py / report.py
```

---

## Prepared Dataset Layout

`prepare_dataset.py` writes to `base_dir/<RC_NAME>/`:

```
D:/dataset/
  RC013/
    rad_power/   # <timestamp_ms>.npy  shape (512, 256, 256) — raw radar power cubes
    rad_elev/    # <timestamp_ms>.npy  shape (512, 256, 256) — radar elevation channel
    labels/      # <timestamp_ms>.npy  shape (64, 256, 256) uint8 — LiDAR occupancy GT
    pco/         # *.png  camera images
    calib/       # *.txt  calibration files (PCO_Intrinsic + Lidar_to_PCO transforms)
  RC014/
    ...
```

Subfolder names are configurable in `configs/train_config.yaml` under `dataset.subfolders`.

---

## Raw Data Layout Expected

```
<raw_data_dir>/                      # e.g. E:/DataCollectionApril2022_11_03
  1_RC002/
    Radar/                           # *.mat radar cube files
    data/
      pcd/                           # *.h5 LiDAR point cloud files
      pco/                           # *.png camera images
      labels_new2/                   # *.txt calibration files (bboxes + transforms)
  12_RC013/
    Radar/ data/pcd/ pco/ labels_new2/
  18_RC019/
    ...
```

Subfolder numbering pattern: `<N>_<RCNAME>` (e.g. `12_RC013`). `prepare_dataset.py` scans by regex `*_<RCNAME>`.

---

## Model Architecture

- **DopplerHead**: Conv3D → MaxPool3D — compresses Doppler 512 → 128 bins
- **Encoder**: ResNet-18 backbone, modified for radar input
- **Decoder**: Upsample + skip connections
- **Input tensor**: `(B, 2, 128, 256, 256)` — channel 0 = power, channel 1 = elevation
- **Output tensor**: `(B, 64, 256, 256)` — 64 height bins × 256 range × 256 azimuth

---

## Critical Technical Rules

### 1. GT Label Azimuth Flip — DO NOT REMOVE
Raw label files on disk have the **opposite azimuth convention** to the radar.
`dataloader.py` corrects this with `torch.flip(label_tensor, [-1])` on every load.
Any code that loads label `.npy` files directly (e.g. `predict.py`, evaluation scripts)
**must** also flip the azimuth: `gt_label[:, :, ::-1].copy()`
Without this the GT grid is mirrored left-right against the radar input and predictions.

### 2. Coordinate Transpose — DO NOT REMOVE
Raw `.npy` shape after loading: `(Doppler, Azimuth, Range)`
Model needs: `(Doppler, Range, Azimuth)`
Fix applied in `dataloader.py`:
```python
power = np.transpose(preprocess(power), (0, 2, 1))
elev  = np.transpose(preprocess(elev),  (0, 2, 1))
```
Removing or changing this breaks all camera projections and GT matching.

### 2. Doppler Downsampling
Raw data has 512 Doppler bins. Model uses 128. Downsampling in `load_frame()`:
- Power: `max_pool3d(kernel=(4,1,1), stride=(4,1,1))`
- Elevation: `data[::4, :, :]` (stride slicing)

### 3. Normalization (Power)
Log-dB transform then min-max to `[0, 1]`:
```python
data = np.clip(10 * np.log10(data + 1e-10), power_min_val, power_max_val)
data = (data - power_min_val) / (power_max_val - power_min_val)
```
Default range: `power_min_val=-100.0`, `power_max_val=-31.7`

### 4. Timestamp Formats
- `rad_power/` and `labels/` filenames: **13-digit millisecond integer** (e.g. `1649671430965.npy`)
- `calib/` filenames: **seconds.subseconds** (e.g. `1649671430.1594195.txt`)
- Sync tolerance: `sync_threshold_ms: 100` (default 100 ms)
- Helper used everywhere: `_extract_ts_ms(path)` — if value < 1e11 multiply by 1000, else keep as-is

### 5. AE / RE Map Computation
`compute_ae_re_maps(power_np, elev_np, config)` in `predict.py`:
- Skip first 2 Doppler channels (static/clutter)
- AE map: mean power binned by elevation × azimuth — shape `(64, 256)`
- RE map: mean power binned by elevation × range — shape `(64, 256)`
- Used in prediction plot Row 0

---

## Prediction Plot Layout (3×3 grid)

Matches `evaluate_predictions_views.py` in the development model pipeline exactly:

| | Col 0 | Col 1 | Col 2 |
|---|---|---|---|
| **Row 0** | Radar BEV (turbo) | AE Map — Elev×Az (turbo) | RE Map — Elev×Range (turbo) |
| **Row 1** | GT LiDAR BEV (gray) | GT Front View — Elev×Az (gray) | GT Side View — Elev×Range (gray) |
| **Row 2** | Pred BEV (magma) | Pred Front View (magma) | Pred Side View (magma) |

- Azimuth x-axis ticks: `np.linspace(0, 255, 7)` → degrees via `arcsin(linspace(-1,1,7))`
- Suptitle: `<dataset> | ts=<timestamp>\nIoU=X.XXX  Prec=X.XXX  Rec=X.XXX`
- Two plots per frame: `full_<ts>.png` (confidence) and `thresh_<ts>.png` (binary at midpoint threshold)
- Saved to: `<out_dir>/prediction_plots/` and `<out_dir>/prediction_plots_thresh/`

---

## Standard Workflows

### First-time data preparation
```bash
python dataset/prepare_dataset.py --config configs/train_config.yaml
```
Runs per RC folder: (1) extract radar .mat → .npy, (2) generate LiDAR labels, (3) copy camera images + calib files.

### Training
```bash
python training/train.py --config configs/train_config.yaml
```
Checkpoints saved to `logging.output_dir/<timestamp>/`. Best and final model saved as `.pth`.

### Prediction / Inference
```bash
python utils/predict.py \
  --config    configs/train_config.yaml \
  --dataset   RC014 \
  --out_dir   verification_output/RC014_run1
```
Auto-finds latest checkpoint from `logging.output_dir` if `inference.checkpoint` is empty.

### Smoke test (quick sanity check)
```bash
python training/train.py --config configs/smoke_test_config.yaml
python utils/predict.py  --config configs/smoke_test_config.yaml --dataset RC014 --out_dir smoke_test_output/RC014
```

---

## Dataset Split Options

### Manual split (explicit RC folders)
```yaml
dataset:
  train: [RC013, RC019]
  val:   [RC013]
  test:  [RC014]
```

### Auto-split (random per-folder frame-level split)
```yaml
dataset:
  auto_split:
    enabled: true
    train_ratio: 0.70
    val_ratio: 0.15
    seed: 42       # remainder goes to test
```
Each RC folder's frames are shuffled with `random.Random(seed)` and split via `torch.utils.data.Subset`. The seed must stay `42` for reproducibility.

---

## Loss Functions

| Key | Description | Extra params |
|---|---|---|
| `weighted_bce` | BCE with positive-class weight — good default | `pos_weight: 10.0` |
| `bce` | Plain BCE, no weighting | — |
| `focal` | Focal loss — down-weights easy background | `focal_alpha: 0.25`, `focal_gamma: 2.0` |
| `focal_dice` | Focal + Dice combined | above + `dice_weight: 0.5` |
| `bce_dice` | Weighted BCE + Dice combined | `pos_weight`, `dice_weight: 0.5` |
| `tversky` | Asymmetric Dice — push recall > precision | `tversky_alpha: 0.3`, `tversky_beta: 0.7` |

Increase `pos_weight` if predictions are too sparse. Use `tversky` if you want fewer missed occupancy voxels.

---

## Config Reference (train_config.yaml)

```yaml
dataset:
  raw_data_dir: E:/DataCollectionApril2022_11_03  # source of raw MAT/H5/PNG/TXT files
  base_dir: D:/dataset                             # where prepared data lives
  subfolders:                                      # customise if your folder names differ
    rad_power: rad_power
    rad_elev:  rad_elev
    labels:    labels
    pco:       pco
    calib:     calib
  auto_split: { enabled: false, train_ratio: 0.70, val_ratio: 0.15, seed: 42 }
  train: [RC013, RC014]
  val:   [RC013]
  test:  [RC014]
  batch_size: 1
  num_workers: 0
  sync_threshold_ms: 100

model:
  name: resnet18_radar
  in_channels: 2
  doppler_depth: 128
  encoder_out_channels: 64
  num_classes: 64                # = number of height bins in output

training:
  epochs: 10
  lr: 0.001
  optimizer: Adam
  weight_decay: 1e-4
  scheduler: CosineAnnealingLR
  loss: weighted_bce
  pos_weight: 10.0

logging:
  output_dir: E:/checkpoints     # checkpoints written here
  tensorboard: true
  save_inference_images: true
  visualization_split: test

inference:
  checkpoint: ''                 # leave empty → auto-use latest checkpoint
  threshold: null                # leave null → dynamic midpoint threshold
  out_dir: verification_output
  saveroad_dir: ''               # path to SAVEROAD toolkit (optional, for camera projection)
```

---

## Session Rules

### When asked to quit / exit / stop
Kill **every** background process first:
```powershell
Stop-Process -Name python* -Force -ErrorAction SilentlyContinue
Stop-Process -Name conda*  -Force -ErrorAction SilentlyContinue
```
Confirm all processes are gone, then say goodbye.

### Always use absolute paths
All `out_dir` values passed to `predict.py` are normalized with `os.path.abspath()` at startup. Never leave paths as mixed-separator relative strings — PIL on Windows will throw `OSError: [Errno 22]`.

### conda environment
All Python commands use: `conda run -n thesis_model python ...`

### Output always goes under `out_dir/<dataset>/`
Never hardcode a dataset name into an output path. Always append `args.dataset` so multiple datasets can be predicted into the same parent directory.
