# Thesis Evaluation Program — README

## Overview

`utils/thesis_eval.py` evaluates a trained 4D radar occupancy prediction model
across weather conditions and generates all thesis figures in one command.

Two modes:

| Mode | Config | Purpose |
|------|--------|---------|
| `basic` | `eval_basic.yaml` | Quick sanity check — IoU / Precision / Recall on one RC folder |
| `weather` | `eval_weather.yaml` | Full thesis evaluation — AP, P_d, P_fa, CD, degradation, range-band, figures |

---

## Step-by-Step Workflow

### Step 1 — Prepare the dataset

If not already done, run `prepare_dataset.py` on each raw RC folder.
This creates `rad_power/`, `rad_elev/`, `labels/`, `calib/`, `pco/`, and `cfar/`
inside each RC folder under `base_dir`.

```
python dataset/prepare_dataset.py --config configs/train_config.yaml
```

### Step 2 — Update calibration and CFAR (if calibration was updated)

If the professor has updated calibration files in the raw SAVEROAD data,
or if CFAR is missing from the prepared dataset, run:

```
python dataset/update_calib_cfar.py --config configs/eval_weather.yaml
```

This copies:
- `data/labels_new2/*.txt` → `calib/`  *(overwrites — picks up updated calibration)*
- `data/radar/*.txt`       → `cfar/`   *(skips files already present)*

Requires `raw_data_dir` to be set in the config.
To update a single RC folder only:

```
python dataset/update_calib_cfar.py --config configs/eval_weather.yaml --rc RC019
```

### Step 3 — Edit the config

Open `configs/eval_weather.yaml` and set:

```yaml
checkpoint: 'checkpoints/20260626_125729/best_model.pth'   # trained model weights
base_dir:   '/media/SSD2/radar_dataset/'                    # prepared dataset root

eval_splits:
  clear: [RC019, RC031, RC032, RC033, RC036]   # assign each RC to its condition
  fog:   []
  rain:  []
```

Leave any condition empty (`[]`) if you have no data for it.

### Step 4 — Quick sanity check (basic mode)

Run on one RC folder first to confirm the model loads and predicts correctly.

```
python utils/thesis_eval.py --config configs/eval_basic.yaml
```

Output → `verification_output/basic/`
Check that prediction plots appear and metrics look reasonable before running the full eval.

### Step 5 — Full thesis evaluation (weather mode)

Runs across all RC folders in `eval_splits`. Produces all metrics and thesis figures.

```
python utils/thesis_eval.py --config configs/eval_weather.yaml
```

Output → `verification_output/weather/`

---

## What the Program Produces

### Metrics — `verification_output/weather/weather_results.csv`

**Results Table** — AP, P_d, P_fa, Chamfer Distance per condition:
```
Weather  Method     AP    P_d   P_fa        CD
clear    DL      0.897  0.944  0.048    0.2194
clear    CFAR    0.281  0.473  0.966    0.9407
fog      DL      0.701  0.923  0.218       N/A
rain     DL      0.654  0.871  0.251       N/A
```
CD is clear-only — fog/rain LiDAR is unreliable as geometric ground truth.

**Degradation Table** — % drop from clear to fog/rain (lower = more robust):
```
Weather  Method  AP deg%  P_d deg%  P_fa deg%
fog      DL        12.3%      0.0%     +45.2%
```

**Point Density Table** — predicted points/m³ inside GT box by range band:
```
Weather  Method     0-5m    5-10m   10-15m   15-20m
clear    DL       56.425   36.960    9.506    1.921
```

**Range-Band Detection** — AP / P_d / P_fa split by target distance:
```
Weather  Method Band      n     AP    P_d   P_fa
clear    DL     0-5m     57  0.888  1.000  0.057
clear    DL     5-10m   239  0.916  1.000  0.040
clear    DL     10-15m  180  0.913  1.000  0.017
clear    DL     15-20m  180  0.731  0.917  0.166
```

If `cfar/` exists for an RC folder, CFAR rows are added to every table automatically.
If `cfar/` is absent, only DL rows appear — no errors.

### Thesis Figures — `verification_output/weather/thesis_figures/`

**Prediction plots** — 5 equally spaced frames per RC folder, 3-row mosaic:
- Row 1: Radar input — Power BEV | AE Map (elev × azimuth) | RE Map (elev × range)
- Row 2: GT LiDAR occupancy — BEV | Front View | Side View
- Row 3: DL prediction probability — BEV | Front View | Side View

**Camera projection** — camera image with radar points projected onto it
(only if `pco/` folder and calibration exist, and `camera_projection: true` in config).

If any individual frame fails during plot generation, a `[WARN]` is printed and
the remaining frames continue — plots are always generated no matter what.

```
verification_output/weather/
  weather_results.csv
  thesis_figures/
    RC019/
      prediction_plots/
        frame_005_1649675877.203.png
        frame_027_1649675879.114.png
        frame_054_...png
        frame_081_...png
        frame_103_...png
      camera_projection/         <- only if camera_projection: true in config
        frame_005_...png
        ...
    RC031/
      prediction_plots/
        ...
```

---

## Required Folder Structure

```
base_dir/
  RC019/
    rad_power/     <- radar power cubes (.npy, one per frame)
    rad_elev/      <- radar elevation cubes (.npy, one per frame)
    labels/        <- LiDAR occupancy labels (.npy, 64×256×256)
    calib/         <- calibration + annotation files (.txt from labels_new2)
    pco/           <- camera images (.png)         [optional]
    cfar/          <- CFAR point clouds (.txt)     [optional — for DL vs CFAR table]
  RC031/
    ...
```

---

## Metric Definitions

| Metric | Definition |
|--------|-----------|
| **AP** | Average Precision — area under the precision-recall curve. Points inside the GT bounding box = true positives; outside = false positives. Higher is better. |
| **P_d** | Detection probability — fraction of frames where at least one predicted point falls inside the GT bounding box. Higher is better. |
| **P_fa** | False alarm rate — fraction of all predicted points outside the GT bounding box. Lower is better. |
| **CD** | Chamfer Distance — mean nearest-neighbour distance to GT LiDAR points (meters). Lower is better. Clear weather only. |
| **Density** | Average predicted points per m³ inside the GT bounding box, by range band. |

CFAR points have Y-axis sign corrected automatically (SAVEROAD exports have Y inverted vs LiDAR frame).

---

## File Summary

| File | Purpose |
|------|---------|
| `utils/thesis_eval.py` | Main evaluation script |
| `configs/eval_basic.yaml` | Config for basic mode (Step 4) |
| `configs/eval_weather.yaml` | Config for weather mode (Step 5) |
| `dataset/update_calib_cfar.py` | Update calibration + copy CFAR from raw source (Step 2) |
| `dataset/copy_cfar.py` | Legacy: copy CFAR only (use update_calib_cfar.py instead) |

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
