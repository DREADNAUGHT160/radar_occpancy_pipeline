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
python utils/prepool_elev.py --config configs/train_config.yaml
```

| Script | Input | Output folder | Method |
|---|---|---|---|
| `prepool_doppler.py` | `rad_power/` | `rad_power_pooled/` | Max pool — GPU: `F.max_pool3d` / CPU: numpy max |
| `prepool_elev.py` | `rad_elev/` | `rad_elev_pooled/` | Max pool — GPU: `F.max_pool3d` / CPU: numpy max |

Each script prints the device it is using at startup. Both fall back to CPU numpy automatically if no GPU is available.

---

## Step 2 — Train

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

## Step 3 — Evaluate (basic — is the model working?)

Edit `configs/eval_config.yaml`:
```yaml
eval_mode: basic
checkpoint: 'checkpoints/<run_id>/best_model.pth'
base_dir: '/media/SSD2/radar_dataset/'
basic:
  dataset: RC019     # any RC folder from your test set
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
checkpoint: 'checkpoints/<run_id>/best_model.pth'
base_dir: '/media/SSD2/radar_dataset/'
weather:
  threshold: 0.4
  weather_splits:
    clear: [RC019, RC032, RC033]
    fog:   [RC031]
    rain:  [RC036]
```

```bash
python utils/thesis_eval.py --config configs/eval_config.yaml
```

Produces AP, P_d, P_fa per weather condition. Chamfer Distance is clear weather only.  
Results saved to `verification_output/eval/weather_results.csv`.

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
  basic_results.csv
  weather_results.csv
  thesis_figures/
    <RC_folder>/
      prediction_plots/   3-row mosaic PNGs
```

---

## Key config options (`configs/train_config.yaml`)

| Option | What it does |
|---|---|
| `dataset.base_dir` | Path to prepared `.npy` dataset |
| `dataset.train / val / test` | Lists of RC folder names |
| `dataset.batch_size` | Start at 6; auto-halved on OOM |
| `model.doppler_pool` | `max` (default) / `mean` / `stride` / `torch_max` |
| `training.epochs` | Number of training epochs |
| `training.loss` | `weighted_bce` (default) — see [PIPELINE_OVERVIEW.md](PIPELINE_OVERVIEW.md) for others |
| `training.pos_weight` | Weight for occupied voxels (default 10.0) |
| `logging.tensorboard` | `true` to enable TensorBoard |
