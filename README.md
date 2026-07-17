# Radar 3D Occupancy Pipeline

Predicts a 3D occupancy grid from 4D radar data, trained against LiDAR ground truth.

> For full technical details see [PIPELINE_OVERVIEW.md](PIPELINE_OVERVIEW.md)

---

## Setup

```bash
conda create -n thesis_model python=3.12
conda activate thesis_model
pip install -r requirements.txt
```

> **Always activate `thesis_model` before running any script.** The base conda env has CPU-only PyTorch and will silently train on CPU.

---

## Step 1 — Prepare the dataset (run once)

```bash
python dataset/prepare_dataset.py --config configs/train_config.yaml
```

Converts raw `.mat` radar and `.h5` LiDAR files into `.npy` training files.  
Set `dataset.raw_data_dir` and `dataset.base_dir` in `configs/train_config.yaml` first.

---

## Step 1b — Check data (optional but recommended)

Generates 2-row plots (radar input + GT label) for 5 random frames from every train RC folder. No model needed — use this to verify the dataset is loaded and aligned correctly before training.

```bash
# Default: train split, 5 frames per folder
python utils/check_data.py --config configs/train_config.yaml

# Choose a different split
python utils/check_data.py --config configs/train_config.yaml --split val
python utils/check_data.py --config configs/train_config.yaml --split test

# Check more frames per folder
python utils/check_data.py --config configs/train_config.yaml --n_plots 10

# Check a single folder (ignores --split)
python utils/check_data.py --config configs/train_config.yaml --rc RC019
```

Or set the defaults in `configs/train_config.yaml`:
```yaml
check_data:
  split:   train    # train | val | test
  n_plots: 5
```

Each plot shows:
- **Row 0 — Radar input:** Power BEV / AE map (elev × azimuth) / RE map (elev × range)
- **Row 1 — GT LiDAR:** BEV / Front view / Side view

Output saved to `verification_output/data_check/<RC_folder>/`.

---

## Step 1c — Pre-pool Doppler (recommended before training)

Pools the Doppler axis (512 → 128) once and saves to disk. The dataloader auto-detects the pre-pooled folders and uses them directly — no config change needed. If skipped, the dataloader falls back to runtime pooling as before.

Run the two scripts **in order** (elevation after power finishes):

```bash
# Power cube — run first
python utils/prepool_doppler.py --config configs/train_config.yaml

# Elevation cube — run after power finishes
python utils/elev_pool.py --config configs/train_config.yaml
```

| Script | Input | Output folder | Method |
|---|---|---|---|
| `prepool_doppler.py` | `rad_power/` | `rad_power_pooled/` | Max pool — GPU: `F.max_pool3d` / CPU: numpy max |
| `elev_pool.py` | `rad_elev/` | `rad_elev_pooled/` | Controlled by `model.elev_pool` in config (default: `max`) |

Each script prints the device it is using at startup. Both fall back to CPU numpy automatically if no GPU is available.

**Elevation pooling method** (`model.elev_pool` in train config):

| Value | Behaviour |
|---|---|
| `argmax_gather` *(default for eval, see `configs/eval_config.yaml`)* | Elevation is read from the same Doppler bin where **power** is maximal, instead of independently max-pooling each channel. `RAD_elev` is a sparse, signed channel (~99.6% exactly zero); plain max-pooling always loses to the zero background whenever the real reading is negative, so it silently erases every target below sensor height. `argmax_gather` keeps power and elevation physically consistent voxel-for-voxel. Uses a precomputed `rad_elev_argmax/` folder if present, else computes it live per-frame from raw `rad_power`/`rad_elev`; falls back to `max` with a `[WARN]` log if neither is available. |
| `max` *(legacy default for training configs)* | `F.max_pool3d` kernel=(4,1,1) — takes the numerically largest elevation value per 4-bin window, independent of power. Systematically erases negative elevation readings (see above). |
| `stride` | `arr[::4]` — samples every 4th Doppler bin directly, preserving the true signed angle without power dependency, but risks missing the peak if it doesn't land on a sampled index. |

> If you change `elev_pool`, re-run `prepool_elev.py --force` so the pre-pooled files match, then retrain from scratch. `argmax_gather` does not require `prepool_elev.py` — it computes/loads its own data independently (see `dataset/dataloader.py::_pool_argmax_gather`).

**`elev_pool.py` (recommended precompute tool):** a standalone script (repo root)
that precomputes elevation pooling directly, without going through
`prepool_elev.py`/`model.elev_pool` config at all:

```bash
python elev_pool.py --base_dir /path/to/radar_dataset
```

For each 4-bin Doppler group, it gathers elevation from the bin with the largest
**|elevation| magnitude** (`np.abs(e_blocks).argmax(axis=1)`) rather than from
power's peak bin — `RAD_elev` is sparse enough (~99.6% exactly zero) that
elevation's own magnitude reliably identifies the real detection without needing
power at all. This is a **different algorithm** from `argmax_gather` above (which
uses power's argmax instead) — both solve the same negative-elevation-erasure
problem, via different signals. Output goes to `rad_elev_pooled/`, the same
folder `prepool_elev.py` writes to — **running this replaces (deletes then
rewrites) whatever pooling method was there before**, so check which method
last populated `rad_elev_pooled/` for a given RC folder before assuming its
contents. Works on either a single RC folder directly (pass its path) or a
root directory containing multiple `RC*` folders.

---

## Step 2 — Train

### Method 1 — Max-pool elevation (default)

```bash
python training/train.py --config configs/train_config.yaml
```

Each epoch prints train loss, val loss, IoU, precision, recall, and learning rate.  
Best model is saved automatically to `checkpoints/<run_id>/best_model.pth`.

**TensorBoard** — the exact command and URL are printed at training startup:
```
  TensorBoard:
    Run : tensorboard --logdir "checkpoints/<run_id>/tensorboard"
    Open: http://localhost:6006
```

If the GPU runs out of memory, the batch size is automatically halved and the epoch retried.

---

### Method 2 — Stride elevation (fallback if Method 1 training fails)

If the model is not learning with max-pool elevation (loss not decreasing, IoU stays near zero), try stride elevation instead. Stride samples the true elevation angle at each Doppler velocity rather than always picking the most positive angle, which can give the model a cleaner geometric signal.

**Step 1 — Re-pool elevation with stride:**
```bash
python utils/prepool_elev.py --config configs/train_config_stride_elev.yaml --force
```

**Step 2 — Train with the stride config:**
```bash
python training/train.py --config configs/train_config_stride_elev.yaml
```

Checkpoints go to `checkpoints_stride_elev/` so the Method 1 run is untouched and both can be compared side by side.

> To switch back to max-pool: run `prepool_elev.py --config configs/train_config.yaml --force` and retrain with `train_config.yaml`.

---

## Step 3 — Evaluate (basic — is the model working?)

Edit `configs/eval_config.yaml`:
```yaml
eval_mode: basic
checkpoint: 'checkpoints/<run_id>/best_model.pth'
base_dir: '/media/SSD2/radar_dataset/'
eval_splits:
  clear: [RC019]
  fog:   []
  rain:  []
basic:
  threshold: 0.4
```

```bash
python utils/thesis_eval.py --config configs/eval_config.yaml
```

Prints IoU, Precision, Recall. IoU > 0.10 means the model is learning.

---

## Step 4 — Evaluate (thesis — full weather comparison)

Edit `configs/eval_config.yaml`:
```yaml
eval_mode: weather
eval_metrics: true
checkpoint: 'checkpoints/<run_id>/best_model.pth'
base_dir: '/media/SSD2/radar_dataset/'

eval_splits:
  clear: [RC019, RC032, RC033]   # RC folders recorded in clear weather
  fog:   [RC031]                 # fog conditions
  rain:  [RC036]                 # rain conditions
                                 # any empty list is silently skipped

weather:
  threshold: 0.4
```

```bash
python utils/thesis_eval.py --config configs/eval_config.yaml
```

Produces AP, P_d, P_fa, Chamfer Distance, and point density per range band for both the DL model and CFAR baseline. Results saved to `<out_dir>/weather_results.csv`.

> Set `eval_metrics: false` to skip metrics and only run figure generation — useful when you want to regenerate plots without the full evaluation pass.

### What the CFAR baseline does

CFAR detections are read from the `cfar/` subfolder inside each RC folder. Each detection file has columns `[x, y, z, doppler_velocity]`. The evaluation applies a **Doppler gate** (`velocity < -1.8 m/s`) to keep only approaching targets before computing metrics — this matches the physical expectation that the ego vehicle is approaching a stationary obstacle.

CFAR frames are collected independently from all calib files, not just the frames in the prepared dataset. This means CFAR evaluation covers the full range of target distances including 10–20 m frames.

---

## Step 5 — CFAR-only evaluation (no model needed)

If you want CFAR metrics without loading the DL model:

```bash
python utils/cfar_only_eval.py --config configs/eval_config.yaml
```

This runs the same CFAR evaluation pipeline as `thesis_eval.py` but skips model loading entirely. Useful for verifying CFAR metrics in isolation or on machines without a GPU.

---

## Step 6 — Camera projection figures

Generates 2-panel camera overlay images for visual inspection:

- **Left panel** — DL model occupancy projected onto the camera image (green dots)
- **Right panel** — CFAR Doppler-filtered detections projected onto the camera image (orange dots)
- Both panels show the GT bounding box (red) and range in the title

### Option A — Combined run (figures generated after eval)

In `configs/eval_config.yaml`:
```yaml
thesis_plots:
  enable:            true
  camera_projection: true   # set this
  n_plots:           5      # frames per RC folder
  threshold:         0.4
```

```bash
python utils/thesis_eval.py --config configs/eval_config.yaml
```

### Option B — Standalone (if the combined run fails or you only need figures)

```bash
# All RC folders from eval_splits:
python utils/gen_camera_proj.py --config configs/eval_config.yaml

# Only specific RC folders:
python utils/gen_camera_proj.py --config configs/eval_config.yaml --rc RC019 RC031

# Override number of frames and checkpoint:
python utils/gen_camera_proj.py --config configs/eval_config.yaml --n_plots 10 \
    --checkpoint checkpoints/<run_id>/best_model.pth
```

**Camera images (pco):** by default, looked up at `<base_dir>/<RC>/pco/`. Override with a
top-level `pco_dir` key in the config yaml, or pass `--pco_dir` on the command line
(standalone script only) — useful when camera images live outside the prepared dataset
folder structure.

Output: `<out_dir>/camera_projection/<rc_name>/frame_01_range8.3m.png`, etc.

---

## Outputs

```
checkpoints/
  <run_id>/
    best_model.pth      best checkpoint (lowest val loss)
    final_model.pth     last epoch checkpoint
    training.log        full training log
    tensorboard/        TensorBoard events

verification_output/eval/
  weather_results.csv         AP / P_d / P_fa / CD / density, DL vs CFAR
  frame_match_log.csv         per-frame sync audit trail: DL<->CFAR/calib
                               timestamps, gaps, matched/extended type, range band
  thesis_figures/
    <RC_folder>/
      prediction_plots/       3-row mosaic PNGs (radar input, prediction, GT)
  camera_projection/
    <RC_folder>/
      frame_01_range8.3m.png  2-panel DL-vs-CFAR camera overlay
      frame_02_range12.1m.png
      ...
```

---

## Key config options (`configs/train_config.yaml`)

| Option | What it does |
|---|---|
| `dataset.base_dir` | Path to prepared `.npy` dataset |
| `dataset.train / val / test` | Lists of RC folder names |
| `dataset.batch_size` | Start at 6; auto-halved on OOM |
| `model.doppler_pool` | `max` (default) / `mean` / `stride` / `torch_max` |
| `model.elev_pool` | `max` (default for training) / `stride` / `argmax_gather` — elevation Doppler pooling method, see Step 1c |
| `training.epochs` | Number of training epochs |
| `training.loss` | `weighted_bce` (default) — see [PIPELINE_OVERVIEW.md](PIPELINE_OVERVIEW.md) for others |
| `training.pos_weight` | Weight for occupied voxels (default 10.0) |
| `logging.tensorboard` | `true` to enable TensorBoard |

## Key config options (`configs/eval_config.yaml`)

| Option | What it does |
|---|---|
| `checkpoint` | Path to `best_model.pth` to evaluate |
| `base_dir` | Root of the prepared dataset |
| `model.elev_pool` | `argmax_gather` (default) / `max` / `stride` — must match what the checkpoint was trained with, see Step 1c |
| `eval_mode` | `basic` (quick IoU check) / `weather` (full thesis eval) |
| `eval_metrics` | `true` to compute AP/P_d/P_fa/CD; `false` to skip and only generate figures |
| `eval_splits.clear/fog/rain` | RC folder lists per weather condition; empty lists are skipped |
| `weather.threshold` | Occupancy threshold for DL predictions (default 0.4) |
| `thesis_plots.enable` | `true` to generate BEV prediction mosaic PNGs |
| `thesis_plots.camera_projection` | `true` to generate 2-panel DL-vs-CFAR camera overlay PNGs |
| `thesis_plots.n_plots` | Frames per RC folder for figure generation (default 5) |
| `out_dir` | Root output directory for all results and figures |
