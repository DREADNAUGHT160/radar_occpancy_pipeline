#!/usr/bin/env python3
"""
plot_near_predictions.py
Generate input + prediction plots for 0-5m calib frames in a rain/fog dataset.

Produces a 2-row figure per frame:
  Row 0 — Radar input  : Power BEV | AE map | RE map
  Row 1 — DL prediction: BEV       | Front View | Side View

No GT / LiDAR overlay.

Usage:
    conda run -n thesis_model python utils/plot_near_predictions.py \
        --config configs/eval_rain98_argmax_Fdrive.yaml \
        --range_max 5.0 \
        --out_dir "D:/thesis data/plots/20260717_rain98_near_predictions"
"""

import argparse
import glob
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataset.dataloader import RadarDataset
from models.factory import ModelFactory
from utils.project_to_image import occupancy_to_points

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── helpers (mirrors thesis_eval.py) ─────────────────────────────────────────

def _ts_ms(path):
    stem = re.sub(r'\.[a-zA-Z]+$', '', os.path.basename(path))
    m    = re.search(r'(\d+\.\d+|\d+)', stem)
    val  = float(m.group(1))
    return int(val * 1000) if val < 1e11 else int(val)


def _parse_range(txt_path):
    try:
        with open(txt_path) as f:
            text = f.read()
        m = re.search(r'"Center_front"\s*:\s*([\d.\-e+]+)\s+([\d.\-e+]+)', text)
        if not m:
            return None
        return float(np.sqrt(float(m.group(1))**2 + float(m.group(2))**2))
    except Exception:
        return None


def _build_ds_config(base_cfg, rc_dir):
    sf = base_cfg.get('subfolders', {})
    return {
        'model':   base_cfg.get('model', {}),
        'dataset': {
            'radar_dir':         rc_dir,
            'lidar_path':        '',   # no labels — all power frames available
            'sync_threshold_ms': base_cfg.get('sync_threshold_ms', 100),
            'subfolders':        sf,
            'normalization':     base_cfg.get('normalization', {}),
            'filter_bboxes':     False,
            'label_text_dir':    '',
        },
    }


def _compute_ae_re_maps(power_np, elev_np, config):
    num_e   = 64
    norm    = config.get('normalization', {})
    max_ang = norm.get('elevation_max_angle', 0.7854)
    is_norm = norm.get('normalize_elevation', False)

    e_norm = elev_np if is_norm else np.clip(elev_np / (max_ang + 1e-9), -1.0, 1.0)
    e_bins = ((e_norm + 1.0) / 2.0 * (num_e - 1)).clip(0, num_e - 1).astype(int)

    p_mov = power_np[2:]
    e_mov = e_bins[2:]
    d, r, a = p_mov.shape

    ae  = np.zeros((num_e, a), dtype=np.float64)
    ac  = np.zeros((num_e, a), dtype=np.float64)
    ag  = np.broadcast_to(np.arange(a)[np.newaxis, np.newaxis, :], (d, r, a))
    np.add.at(ae, (e_mov.ravel(), ag.ravel()), p_mov.ravel())
    np.add.at(ac, (e_mov.ravel(), ag.ravel()), 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        ae_map = np.where(ac > 0, ae / ac, 0).astype(np.float32)
    if ae_map.max() > 0:
        ae_map /= ae_map.max()

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


def _pred_views(pred_np, threshold, raw_prediction=False):
    """BEV / Front-View / Side-View of prediction."""
    import torch
    pred_t = torch.from_numpy(pred_np)            # (H, R, A)
    if not raw_prediction:
        pred_t = (pred_t >= threshold).float()

    bev = pred_t.max(dim=0).values                # (R, A)
    fv  = pred_t.max(dim=1).values                # (H, A)
    sv  = pred_t.max(dim=2).values                # (H, R)
    return bev, fv, sv


def _save_plot(title, power_np, elev_np, pred_np, config,
               threshold, out_path, raw_prediction=False):
    import torch

    xt = np.linspace(0, 255, 7)
    xl = np.round(np.degrees(np.arcsin(np.linspace(-1, 1, 7)))).astype(int)
    yt_e = np.linspace(0, 63, 5)
    yl_e = np.round(np.linspace(-45, 45, 5)).astype(int)

    ae_map, re_map = _compute_ae_re_maps(power_np, elev_np, config)

    # Power BEV: max over Doppler then max over range-elevation → (R, A)
    bev_power = torch.from_numpy(power_np).max(dim=0).values.numpy()

    pred_bev, pred_fv, pred_sv = _pred_views(pred_np, threshold, raw_prediction)

    fig = plt.figure(figsize=(18, 8))
    gs  = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)

    def _ishow(ax, data, t, cmap, xl_, yl_, az_ticks=False, elev_ticks=False,
               vmin=0, vmax=None):
        vmax = vmax or (float(data.max()) if data.max() > 0 else 1.0)
        im = ax.imshow(data, cmap=cmap, origin='lower', aspect='auto',
                       vmin=vmin, vmax=vmax)
        ax.set_title(t, fontsize=10)
        ax.set_xlabel(xl_, fontsize=8)
        ax.set_ylabel(yl_, fontsize=8)
        if az_ticks:
            ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=7)
        if elev_ticks:
            ax.set_yticks(yt_e); ax.set_yticklabels(yl_e, fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 0 — Input
    _ishow(fig.add_subplot(gs[0, 0]), bev_power,
           'Input: Power BEV (max-Doppler)', 'turbo',
           'Azimuth (°)', 'Range (bins)', az_ticks=True)
    _ishow(fig.add_subplot(gs[0, 1]), ae_map,
           'Input: AE Map', 'turbo',
           'Azimuth (°)', 'Elevation (°)', az_ticks=True, elev_ticks=True)
    _ishow(fig.add_subplot(gs[0, 2]), re_map,
           'Input: RE Map', 'turbo',
           'Range (bins)', 'Elevation (bins)')

    # Row 1 — Prediction
    pred_label = 'Raw score' if raw_prediction else f'Thresholded (≥{threshold})'
    _ishow(fig.add_subplot(gs[1, 0]), pred_bev.numpy(),
           f'Pred BEV ({pred_label})', 'Reds',
           'Azimuth (°)', 'Range (bins)', az_ticks=True, vmax=1)
    _ishow(fig.add_subplot(gs[1, 1]), pred_fv.numpy(),
           f'Pred Front View ({pred_label})', 'Reds',
           'Azimuth (°)', 'Height (bins)', az_ticks=True, vmax=1)
    _ishow(fig.add_subplot(gs[1, 2]), pred_sv.numpy(),
           f'Pred Side View ({pred_label})', 'Reds',
           'Range (bins)', 'Height (bins)', vmax=1)

    fig.suptitle(title, fontsize=12)
    plt.savefig(out_path, bbox_inches='tight', dpi=150, facecolor='white')
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config',    required=True)
    ap.add_argument('--range_max', type=float, default=5.0,
                    help='Only plot frames whose calib range < this value (m)')
    ap.add_argument('--out_dir',   default='')
    ap.add_argument('--threshold', type=float, default=0.4)
    ap.add_argument('--raw',       action='store_true',
                    help='Plot raw sigmoid scores instead of thresholded')
    ap.add_argument('--sync_ms',   type=int, default=100)
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    base_dir    = config.get('base_dir', '')
    subs        = config.get('subfolders', {})
    calib_sf    = subs.get('calib', 'calib')
    eval_splits = config.get('eval_splits', {})
    ckpt        = config.get('checkpoint', '')
    threshold   = args.threshold

    all_rcs = []
    for weather, rcs in eval_splits.items():
        for rc in (rcs or []):
            all_rcs.append(rc)

    out_dir = args.out_dir or os.path.join(
        'D:/thesis data/plots',
        datetime.now().strftime('%Y%m%d') + '_rain98_near_predictions')
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output -> {out_dir}")

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    model_cfg = config.get('model', {})
    model     = ModelFactory.get_model({'model': model_cfg}).to(device)
    sd        = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    print(f"Loaded : {ckpt}")

    total_plotted = 0

    for rc in all_rcs:
        rc_dir    = os.path.join(base_dir, rc)
        calib_dir = os.path.join(rc_dir, calib_sf)
        rc_out    = os.path.join(out_dir, rc.replace('/', '_'))
        os.makedirs(rc_out, exist_ok=True)

        calib_files = sorted(glob.glob(os.path.join(calib_dir, '*.txt')))
        if not calib_files:
            print(f"  [{rc}] no calib files — skip")
            continue

        # Find 0-5m calib files
        near_calib = [(cf, _parse_range(cf)) for cf in calib_files]
        near_calib = [(cf, r) for cf, r in near_calib
                      if r is not None and r < args.range_max]
        if not near_calib:
            print(f"  [{rc}] no calib frames within {args.range_max} m — skip")
            continue

        # Build no-labels dataset so all power frames are available
        ds = RadarDataset(rc_dir, augment=False,
                          config=_build_ds_config(config, rc_dir))
        if len(ds) == 0:
            print(f"  [{rc}] empty dataset — skip")
            continue

        power_ts_all = np.array([_ts_ms(s['power']) for s in ds.matched_data])

        print(f"\n  [{rc}]  {len(near_calib)} near-range calib frames  "
              f"({len(ds)} total power frames)")

        seen = set()
        for cf, rng in near_calib:
            ct  = _ts_ms(cf)
            bi  = int(np.argmin(np.abs(power_ts_all - ct)))
            gap = abs(power_ts_all[bi] - ct)
            if gap > args.sync_ms or bi in seen:
                continue
            seen.add(bi)

            radar_tensor, _ = ds[bi]
            sample   = ds.matched_data[bi]
            ts_str   = os.path.basename(sample['power']).replace('.npy', '')

            with torch.no_grad():
                pred_np = torch.sigmoid(
                    model(radar_tensor.unsqueeze(0).to(device))
                )[0].cpu().numpy()

            # Extract raw numpy arrays for AE/RE maps
            power_np = radar_tensor[0].numpy()   # (D, R, A)
            elev_np  = radar_tensor[1].numpy()   # (D, R, A)

            title    = f"{rc}  |  range={rng:.2f}m  |  ts={ts_str}  |  Δ={gap}ms"
            fname    = f"range{rng:.2f}m_ts{ts_str[:13]}.png"
            out_path = os.path.join(rc_out, fname)

            _save_plot(title, power_np, elev_np, pred_np, config,
                       threshold, out_path, raw_prediction=args.raw)
            print(f"    range={rng:.2f}m  frame={bi:03d}  -> {fname}")
            total_plotted += 1

    print(f"\nDone. {total_plotted} plots saved -> {out_dir}")


if __name__ == '__main__':
    main()
