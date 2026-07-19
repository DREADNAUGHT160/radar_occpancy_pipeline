"""
Visualise RC011 pooled radar frames in the exact same style as the
eval script's prediction_plots — Input row only (no model needed):
  Col 0: Power BEV        (max over Doppler -> range x azimuth)
  Col 1: AE Map           (elevation x azimuth, from moving Doppler bins)
  Col 2: RE Map           (elevation x range,   from moving Doppler bins)

Normalization is done via RadarDataset (same code path as thesis_eval.py),
so the plots are guaranteed to match what the model receives.

Usage:
  python scripts/visualize_pooled.py --config configs/eval_rc011_pool.yaml
"""

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataset.dataloader import RadarDataset


# ── Helpers (copied from thesis_eval.py) ──────────────────────────────────────

def _build_ds_config(base_cfg, rc_dir):
    sf = base_cfg.get('subfolders', {})
    return {
        'model':   base_cfg.get('model', {}),
        'dataset': {
            'normalization':    base_cfg.get('normalization', {}),
            'sync_threshold_ms': 100,
            'label_text_dir':   os.path.join(rc_dir, sf.get('calib', 'calib')),
            'subfolders':       sf,
        },
    }


def _compute_ae_re_maps(power_np, elev_np, config):
    """Elevation-indexed AE and RE maps — identical to thesis_eval.py."""
    num_e   = 64
    norm    = config.get('normalization', {})
    max_ang = norm.get('elevation_max_angle', 0.7854)
    is_norm = norm.get('normalize_elevation', False)

    e_norm = elev_np if is_norm else np.clip(elev_np / (max_ang + 1e-9), -1.0, 1.0)
    e_bins = ((e_norm + 1.0) / 2.0 * (num_e - 1)).clip(0, num_e - 1).astype(int)

    p_mov = power_np[2:]
    e_mov = e_bins[2:]
    d, r, a = p_mov.shape

    # AE map: elevation x azimuth
    ae  = np.zeros((num_e, a), dtype=np.float64)
    ac  = np.zeros((num_e, a), dtype=np.float64)
    ag  = np.broadcast_to(np.arange(a)[np.newaxis, np.newaxis, :], (d, r, a))
    np.add.at(ae, (e_mov.ravel(), ag.ravel()), p_mov.ravel())
    np.add.at(ac, (e_mov.ravel(), ag.ravel()), 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        ae_map = np.where(ac > 0, ae / ac, 0).astype(np.float32)
    if ae_map.max() > 0:
        ae_map /= ae_map.max()

    # RE map: elevation x range
    re  = np.zeros((num_e, r), dtype=np.float64)
    rc2 = np.zeros((num_e, r), dtype=np.float64)
    rg  = np.broadcast_to(np.arange(r)[np.newaxis, :, np.newaxis], (d, r, a))
    np.add.at(re, (e_mov.ravel(), rg.ravel()), p_mov.ravel())
    np.add.at(rc2, (e_mov.ravel(), rg.ravel()), 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        re_map = np.where(rc2 > 0, re / rc2, 0).astype(np.float32)
    if re_map.max() > 0:
        re_map /= re_map.max()

    return ae_map, re_map


def _save_input_plot(rc_name, ts_str, frame_idx, power_np, elev_np, config, out_path):
    """Save the Input row exactly as _save_pred_plot does in thesis_eval.py."""
    ae_map, re_map = _compute_ae_re_maps(power_np, elev_np, config)

    # Max over Doppler -> (range, azimuth)
    radar_bev = torch.from_numpy(power_np).max(dim=0)[0].numpy()

    # Azimuth tick labels in degrees (same as eval script)
    xt = np.linspace(0, 255, 7)
    xl = np.round(np.degrees(np.arcsin(np.linspace(-1, 1, 7)))).astype(int)

    fig = plt.figure(figsize=(18, 5))
    gs  = fig.add_gridspec(1, 3, wspace=0.32)

    def _ishow(ax, data, title, xlabel, ylabel, vmax=1.0, az_ticks=False):
        im = ax.imshow(data, cmap='turbo', origin='lower', aspect='auto',
                       vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        if az_ticks:
            ax.set_xticks(xt)
            ax.set_xticklabels(xl, fontsize=7)
        plt.colorbar(im, ax=ax)

    bev_max = float(radar_bev.max()) if radar_bev.max() > 0 else 1.0
    _ishow(fig.add_subplot(gs[0, 0]), radar_bev,
           'Input: Power BEV (max over Doppler)',
           'Azimuth (deg)', 'Range (Bins)', vmax=bev_max, az_ticks=True)

    ae_max = float(ae_map.max()) if ae_map.max() > 0 else 1.0
    _ishow(fig.add_subplot(gs[0, 1]), ae_map,
           'Input: AE Map',
           'Azimuth (deg)', 'Elevation (Bins)', vmax=ae_max, az_ticks=True)

    re_max = float(re_map.max()) if re_map.max() > 0 else 1.0
    _ishow(fig.add_subplot(gs[0, 2]), re_map,
           'Input: RE Map',
           'Range (Bins)', 'Elevation (Bins)', vmax=re_max)

    fig.suptitle(f"{rc_name}  |  Frame {frame_idx:03d}  |  ts={ts_str}", fontsize=12)
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='configs/eval_rc011_pool.yaml',
                        help='eval config (used for normalization settings)')
    parser.add_argument('--out_dir',    default='verification_output/eval_rc011_pool/pooled_check')
    parser.add_argument('--n',          type=int, default=5,
                        help='Number of equally-spaced frames to plot')
    parser.add_argument('--all_frames', action='store_true',
                        help='Plot every frame (ignores --n)')
    args = parser.parse_args()

    import yaml
    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    base_dir = config.get('base_dir', '').strip()
    splits   = config.get('eval_splits', {})
    rc_names = [rc for rcs in splits.values() for rc in rcs]
    if not rc_names:
        print("ERROR: no RC folders in eval_splits")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    for rc_name in rc_names:
        rc_dir = os.path.join(base_dir, rc_name) if base_dir else rc_name
        print(f"\n{'='*54}")
        print(f"  {rc_name}  ->  {rc_dir}")
        print(f"{'='*54}")

        try:
            ds = RadarDataset(rc_dir, augment=False,
                              config=_build_ds_config(config, rc_dir))
        except Exception as e:
            print(f"  [SKIP] {e}")
            continue

        n = len(ds)
        if n == 0:
            print("  [SKIP] 0 frames")
            continue

        if args.all_frames:
            indices = list(range(n))
        else:
            start   = min(5, n - 1)
            end     = max(start, n - 6)
            indices = list(dict.fromkeys(
                np.linspace(start, end, args.n).astype(int).tolist()
            ))

        plot_dir = os.path.join(args.out_dir, rc_name)
        os.makedirs(plot_dir, exist_ok=True)

        print(f"  Frames: {n}  |  Plotting indices: {indices}")

        for idx in indices:
            sample   = ds.matched_data[idx]
            ts_str   = os.path.basename(sample['power']).replace('.npy', '')
            radar_tensor, _ = ds[idx]

            power_np = radar_tensor[0].numpy()   # (D, range, azimuth) normalized
            elev_np  = radar_tensor[1].numpy()

            out_path = os.path.join(plot_dir, f'frame_{idx:03d}_{ts_str}.png')
            _save_input_plot(rc_name, ts_str, idx, power_np, elev_np, config, out_path)

    print(f"\nDone. Plots saved to {args.out_dir}")


if __name__ == '__main__':
    main()
