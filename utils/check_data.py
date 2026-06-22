"""
Visualise raw input + ground-truth labels for any dataset split.

No model required.  For each RC folder in the chosen split, picks N random
frames and saves a 2-row mosaic:
  Row 0 — Radar input : Power BEV  |  AE map  |  RE map
  Row 1 — GT LiDAR   : BEV        |  Front view  |  Side view

Usage:
  python utils/check_data.py --config configs/train_config.yaml
  python utils/check_data.py --config configs/train_config.yaml --split val
  python utils/check_data.py --config configs/train_config.yaml --split test
  python utils/check_data.py --config configs/train_config.yaml --n_plots 10
  python utils/check_data.py --config configs/train_config.yaml --rc RC019

Or set in train_config.yaml:
  check_data:
    split:   train       # train | val | test
    n_plots: 5
"""
import os
import sys
import argparse
import random
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataset.dataloader import RadarDataset


def _build_ds_config(train_cfg, rc_dir):
    ds = train_cfg.get('dataset', {})
    sf = ds.get('subfolders', {})
    return {
        'model':   train_cfg.get('model', {}),
        'dataset': {
            'radar_dir':         rc_dir,
            'lidar_path':        os.path.join(rc_dir, sf.get('labels', 'labels')),
            'sync_threshold_ms': ds.get('sync_threshold_ms', 100),
            'subfolders':        sf,
            'normalization':     ds.get('normalization', {}),
            'filter_bboxes':     False,
            'label_text_dir':    '',
        },
    }


def _build_ae_re(radar_tensor, norm_cfg):
    """Build AE and RE maps from a preprocessed radar tensor (D, R, A)."""
    p_np  = radar_tensor[0].numpy()
    e_np  = radar_tensor[1].numpy() if radar_tensor.shape[0] > 1 else np.zeros_like(p_np)
    max_a = norm_cfg.get('elevation_max_angle', 0.7854)
    e_bins = ((np.clip(e_np / max_a, -1.0, 1.0) + 1.0) / 2.0 * 63).clip(0, 63).astype(int)

    n_e, n_r, n_a = 64, p_np.shape[1], p_np.shape[2]
    p_m = p_np[2:]    # skip static Doppler bins
    e_b = e_bins[2:]
    d, r, a = p_m.shape

    a_grid = np.tile(np.arange(a)[np.newaxis, np.newaxis, :], (d, r, 1))
    r_grid = np.tile(np.arange(r)[np.newaxis, :, np.newaxis], (d, 1, a))

    ae_sum = np.zeros((n_e, n_a)); ae_cnt = np.zeros((n_e, n_a))
    np.add.at(ae_sum, (e_b.flatten(), a_grid.flatten()), p_m.flatten())
    np.add.at(ae_cnt, (e_b.flatten(), a_grid.flatten()), 1)
    with np.errstate(invalid='ignore'):
        ae_map = np.where(ae_cnt > 0, ae_sum / ae_cnt, 0)
    if ae_map.max() > 0: ae_map /= ae_map.max()

    re_sum = np.zeros((n_e, n_r)); re_cnt = np.zeros((n_e, n_r))
    np.add.at(re_sum, (e_b.flatten(), r_grid.flatten()), p_m.flatten())
    np.add.at(re_cnt, (e_b.flatten(), r_grid.flatten()), 1)
    with np.errstate(invalid='ignore'):
        re_map = np.where(re_cnt > 0, re_sum / re_cnt, 0)
    if re_map.max() > 0: re_map /= re_map.max()

    bev = p_np.max(axis=0)
    return bev, ae_map, re_map


def plot_frame(frame_idx, radar_tensor, label_tensor, norm_cfg, rc_name, timestamp, out_dir):
    import torch
    bev, ae_map, re_map = _build_ae_re(radar_tensor, norm_cfg)

    label     = label_tensor
    label_bev = label.max(dim=0)[0].numpy()
    label_fv  = label.max(dim=1)[0].numpy()   # elevation × azimuth
    label_sv  = label.max(dim=2)[0].numpy()   # elevation × range

    xticks     = np.linspace(0, 255, 7)
    xlabels_az = np.round(np.degrees(np.arcsin(np.linspace(-1, 1, 7)))).astype(int)

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle(f"{rc_name}  |  frame {frame_idx:03d}  |  ts={timestamp}", fontsize=13)

    # Row 0 — Radar input
    axes[0, 0].imshow(bev,    cmap='turbo', origin='lower', aspect='auto', vmin=0, vmax=1)
    axes[0, 0].set_title('Input: Power BEV')
    axes[0, 0].set_xticks(xticks); axes[0, 0].set_xticklabels(xlabels_az)
    axes[0, 0].set_xlabel('Azimuth (deg)'); axes[0, 0].set_ylabel('Range bin')

    axes[0, 1].imshow(ae_map, cmap='turbo', origin='lower', aspect='auto')
    axes[0, 1].set_title('Input: AE Map (elev × azimuth)')
    axes[0, 1].set_xticks(xticks); axes[0, 1].set_xticklabels(xlabels_az)
    axes[0, 1].set_xlabel('Azimuth (deg)'); axes[0, 1].set_ylabel('Elevation bin')

    axes[0, 2].imshow(re_map, cmap='turbo', origin='lower', aspect='auto')
    axes[0, 2].set_title('Input: RE Map (elev × range)')
    axes[0, 2].set_xlabel('Range bin'); axes[0, 2].set_ylabel('Elevation bin')

    # Row 1 — Ground truth
    axes[1, 0].imshow(label_bev, cmap='gray', origin='lower', aspect='auto')
    axes[1, 0].set_title('GT: BEV (max over elevation)')
    axes[1, 0].set_xticks(xticks); axes[1, 0].set_xticklabels(xlabels_az)
    axes[1, 0].set_xlabel('Azimuth (deg)'); axes[1, 0].set_ylabel('Range bin')

    axes[1, 1].imshow(label_fv, cmap='gray', origin='lower', aspect='auto')
    axes[1, 1].set_title('GT: Front View (elev × azimuth)')
    axes[1, 1].set_xticks(xticks); axes[1, 1].set_xticklabels(xlabels_az)
    axes[1, 1].set_xlabel('Azimuth (deg)'); axes[1, 1].set_ylabel('Elevation bin')

    axes[1, 2].imshow(label_sv, cmap='gray', origin='lower', aspect='auto')
    axes[1, 2].set_title('GT: Side View (elev × range)')
    axes[1, 2].set_xlabel('Range bin'); axes[1, 2].set_ylabel('Elevation bin')

    plt.tight_layout()
    fname = os.path.join(out_dir, f'frame_{frame_idx:03d}_{timestamp}.png')
    plt.savefig(fname, bbox_inches='tight', dpi=110)
    plt.close(fig)
    return fname


def check_rc(rc_name, rc_dir, n_plots, train_cfg, out_dir):
    ds_cfg = _build_ds_config(train_cfg, rc_dir)
    try:
        ds = RadarDataset(rc_dir, augment=False, config=ds_cfg)
    except FileNotFoundError as e:
        print(f"  [SKIP] {rc_name}: {e}")
        return 0

    total = len(ds)
    if total == 0:
        print(f"  [SKIP] {rc_name}: 0 matched frames")
        return 0

    k       = min(n_plots, total)
    indices = sorted(random.sample(range(total), k))
    norm_cfg = train_cfg.get('dataset', {}).get('normalization', {})

    rc_out = os.path.join(out_dir, rc_name)
    os.makedirs(rc_out, exist_ok=True)

    for idx in indices:
        radar_tensor, label_tensor = ds[idx]
        ts = os.path.basename(ds.matched_data[idx]['power']).replace('.npy', '')
        plot_frame(idx, radar_tensor, label_tensor, norm_cfg, rc_name, ts, rc_out)

    return k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',   default='configs/train_config.yaml',
                        help='Path to train_config.yaml')
    parser.add_argument('--split',    default=None,
                        choices=['train', 'val', 'test'],
                        help='Which split to visualise (default: from config or train)')
    parser.add_argument('--n_plots',  type=int, default=None,
                        help='Random frames to sample per RC folder (default 5)')
    parser.add_argument('--rc',       default=None,
                        help='Check only this RC folder (e.g. RC019); overrides --split')
    parser.add_argument('--out_dir',  default='verification_output/data_check',
                        help='Output root directory')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        train_cfg = yaml.safe_load(f)

    check_cfg = train_cfg.get('check_data', {})
    base_dir  = train_cfg.get('dataset', {}).get('base_dir', '')

    split   = args.split   or check_cfg.get('split',   'train')
    n_plots = args.n_plots or check_cfg.get('n_plots', 5)

    if args.rc:
        rc_list = [args.rc]
    else:
        rc_list = train_cfg.get('dataset', {}).get(split, [])

    if not rc_list:
        print(f"No RC folders found in dataset.{split} — pass --rc <name> or populate dataset.{split} in config.")
        return

    label = f"--rc {args.rc}" if args.rc else f"split={split}"
    print(f"\nData check: {label}  |  {len(rc_list)} folder(s)  |  {n_plots} random frames each")
    print(f"Output: {os.path.abspath(args.out_dir)}\n")

    total_saved = 0
    for rc_name in tqdm(rc_list, desc="RC folders"):
        # If rc_name is already an absolute path, use it directly
        if os.path.isabs(rc_name) or os.path.exists(rc_name):
            rc_dir = rc_name
            rc_name = os.path.basename(rc_name.rstrip('/\\'))
        else:
            rc_dir = os.path.join(base_dir, rc_name) if base_dir else rc_name
        n = check_rc(rc_name, rc_dir, n_plots, train_cfg, args.out_dir)
        total_saved += n
        if n:
            print(f"  {rc_name}: saved {n} plots -> {os.path.join(args.out_dir, rc_name)}")

    print(f"\nDone. {total_saved} plots saved to: {os.path.abspath(args.out_dir)}")


if __name__ == '__main__':
    main()
