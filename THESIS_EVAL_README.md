# Thesis Evaluation Program — README

## Overview

`utils/thesis_eval.py` is the main evaluation script for the radar 4D occupancy prediction thesis.
It loads a trained deep learning (DL) model, runs inference on one or more RC datasets, and
produces all quantitative metrics and thesis figures in one command.

It runs in two modes:

| Mode | Purpose |
|------|---------|
| `basic` | Quick sanity check — per-voxel IoU / Precision / Recall on one dataset |
| `weather` | Full thesis evaluation — AP, P_d, P_fa, Chamfer Distance, degradation, range-band analysis, figures |

---

## What the Program Produces

### Metrics (printed to terminal and saved to CSV)

**1. Results Table**
Average Precision (AP), Detection Probability (P_d), False Alarm Rate (P_fa), and
Chamfer Distance (CD) for each weather condition and method.

```
Weather  Method     AP    P_d   P_fa        CD
------------------------------------------------
clear    DL      0.784  1.000  0.163    0.2722
clear    CFAR    0.281  0.473  0.966    0.9407
```

**2. Degradation Table**
How much each metric drops from clear weather to fog/rain (lower % = more robust).

```
Weather  Method  AP deg%  P_d deg%  P_fa deg%
fog      DL        12.3%      0.0%     +45.2%
```

**3. Point Density Table**
Average predicted points per m^3 inside the GT bounding box, split by range band.

```
Weather  Method     0-5m    5-10m   10-15m   15-20m
clear    DL      165.106   35.302    0.000    0.000
```

**4. Range-Band Detection Table**
AP, P_d, P_fa broken down by how far the target is from the sensor.

```
Weather  Method Band      n     AP    P_d   P_fa
clear    DL     0-5m     16  0.844  1.000  0.121
clear    DL     5-10m    39  0.714  1.000  0.230
```

If CFAR data is present, all tables include a CFAR row for comparison.
If CFAR data is absent, only DL rows appear — no errors.

---

### Thesis Figures (saved as PNG)

**Prediction plots** — 3-row mosaic for 5 equally spaced frames per RC folder:
- Row 1: Radar input — Power BEV, AE Map (elevation x azimuth), RE Map (elevation x range)
- Row 2: GT LiDAR occupancy — BEV, Front View, Side View
- Row 3: DL prediction probability — BEV, Front View, Side View

**Camera projection** — camera image with radar occupancy points projected onto it
(only generated if the `pco/` folder exists inside the RC dataset folder).

---

## Required Folder Structure

The dataset root (`base_dir` in config) must contain one subfolder per RC recording:

```
base_dir/
  RC019/
    rad_power/          <- radar power cubes (.npy files, one per frame)
    rad_elev/           <- radar elevation cubes (.npy files, one per frame)
    labels/             <- LiDAR occupancy labels (.npy files, 64x256x256)
    calib/              <- bounding box annotation files (.txt from labels_new2)
    cfar/               <- CFAR radar point clouds (.txt, optional)
    pco/                <- camera images (.png, optional — for camera projection)
  RC020/
    rad_power/
    rad_elev/
    ...
```

The `rad_power/`, `rad_elev/`, `labels/`, `calib/`, and `pco/` folders are created
automatically by `prepare_dataset.py`.

The `cfar/` folder is NOT created by `prepare_dataset.py`. To add it, run:

```
python dataset/copy_cfar.py --config configs/prepare_config.yaml
```

This copies the SAVEROAD radar `.txt` files from the raw data source into each RC folder.
If `cfar/` is absent, the evaluation runs without CFAR comparison (DL only).

---

## Output Folder Structure

All outputs are written to `out_dir` (set in `eval_config.yaml`, default: `verification_output/eval`):

```
verification_output/eval/
  weather_results.csv                          <- all metrics in one CSV file
  thesis_figures/
    RC019/
      prediction_plots/
        frame_005_1649675877.203.png           <- 3-row mosaic plot
        frame_027_1649675879.114.png
        frame_054_...png
        frame_081_...png
        frame_103_...png
      camera_projection/                       <- only if pco/ exists
        frame_005_1649675877.203.png
        frame_027_...png
        ...
    RC020/
      prediction_plots/
        ...
```

---

## Setup — Edit eval_config.yaml

Open `configs/eval_config.yaml` and set these three things:

### 1. Dataset root folder

```yaml
base_dir: 'D:/dataset'    # path to the folder containing RC subfolders
```

### 2. Model checkpoint

```yaml
checkpoint: 'D:/path/to/best_model.pth'
```

### 3. Which RC folders are clear / fog / rain

```yaml
weather:
  weather_splits:
    clear: [RC019, RC020]   # list all RC folders for clear weather
    fog:   [RC021, RC022]   # list all RC folders for fog
    rain:  [RC023]          # list all RC folders for rain
```

Leave a list empty (`[]`) if you have no data for that condition.

### Optional — Thesis figures settings

```yaml
thesis_plots:
  enable:    true    # set false to skip figure generation entirely
  n_plots:   5       # number of frames per RC folder (equally spaced)
  threshold: 0.4     # confidence threshold for camera projection
```

---

## How to Run

### Full thesis evaluation (weather mode)

```
python utils/thesis_eval.py --config configs/eval_config.yaml
```

### Override checkpoint without editing the config

```
python utils/thesis_eval.py --config configs/eval_config.yaml --checkpoint path/to/best_model.pth
```

### Quick sanity check (basic mode)

Set `eval_mode: basic` in `eval_config.yaml`, then:

```
python utils/thesis_eval.py --config configs/eval_config.yaml
```

### Add CFAR data to an existing prepared dataset

```
python dataset/copy_cfar.py --config configs/prepare_config.yaml
```

---

## Metric Definitions

| Metric | Definition |
|--------|-----------|
| **AP** | Average Precision — area under the precision-recall curve. Points inside the GT bounding box are true positives; points outside are false positives. Higher is better. |
| **P_d** | Detection probability — fraction of frames where at least one predicted point falls inside the GT bounding box. Higher is better. |
| **P_fa** | False alarm rate — fraction of all predicted points that fall outside the GT bounding box. Lower is better. |
| **CD** | Chamfer Distance — mean nearest-neighbour distance between predicted points and GT LiDAR points (meters). Lower is better. Only computed for clear weather. |
| **Density** | Average number of predicted points per m^3 inside the GT bounding box per range band. |

CFAR points use a Y-axis sign correction automatically (SAVEROAD exports have Y inverted relative to the LiDAR coordinate frame).

---

## File Summary

| File | Purpose |
|------|---------|
| `utils/thesis_eval.py` | Main evaluation script |
| `configs/eval_config.yaml` | All settings — edit this before running |
| `dataset/copy_cfar.py` | One-time script to add CFAR data to prepared dataset folders |
| `configs/prepare_config.yaml` | Settings for copy_cfar.py (raw_data_dir, base_dir, RC lists) |

---

## Dependencies

```
Python >= 3.8
torch
numpy
yaml (pyyaml)
tqdm
matplotlib
scipy
cv2 (opencv-python)   -- only needed for camera projection figures
```
