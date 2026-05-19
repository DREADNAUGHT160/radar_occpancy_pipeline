"""
Run inference on a dataset — generates prediction plots and (optionally) camera projections.

All paths can be set in the config under the inference: section, so no extra CLI
arguments are required beyond --config, --dataset, and --checkpoint.

Usage:
  python utils/predict.py \
    --config   configs/train_config.yaml \
    --dataset  RC002 \
    [--checkpoint  E:/checkpoints/20260417_090255/best_model.pth] \
    [--out_dir     verification_output/RC002_inference] \
    [--threshold   0.4]
"""
import os
import sys
import re
import glob
import argparse
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.factory import ModelFactory
from utils.project_to_image import parse_calibration, occupancy_to_points, project_to_image


def _extract_ts_ms(path):
    """Extract timestamp in milliseconds from a filename.
    Handles both 13-digit ms integers (1649671430965.npy)
    and decimal-second filenames (1649671430.965.npy).
    Mirrors dataloader._extract_timestamp for consistency.
    """
    match = re.search(r'(\d+\.\d+|\d+)', os.path.basename(path))
    if match:
        val = float(match.group(0))
        return int(val * 1000) if val < 1e11 else int(val)
    return 0


def compute_ae_re_maps(power_np, elev_np, config):
    """Compute elevation-indexed AE (azimuth-elevation) and RE (range-elevation) maps.
    Replicates compare_gt_radar.py using the elevation channel to bin each voxel.
    power_np / elev_np: (D, R, A) float32, already preprocessed and normalised.
    Returns (ae_map, re_map) each shape (64, 256), values in [0, 1].
    """
    num_e_bins = 64
    norm       = config.get('dataset', {}).get('normalization', {})
    max_angle  = norm.get('elevation_max_angle', 0.7854)
    is_norm    = norm.get('normalize_elevation', False)

    e_norm = elev_np if is_norm else np.clip(elev_np / (max_angle + 1e-9), -1.0, 1.0)
    e_bins = ((e_norm + 1.0) / 2.0 * (num_e_bins - 1)).clip(0, num_e_bins - 1).astype(int)

    # Skip first 2 Doppler channels (static/clutter)
    p_mov = power_np[2:]
    e_mov = e_bins[2:]
    d_dim, r_dim, a_dim = p_mov.shape

    # AE map: elevation × azimuth  (mean power over range & Doppler)
    ae_sum = np.zeros((num_e_bins, a_dim), dtype=np.float64)
    ae_cnt = np.zeros((num_e_bins, a_dim), dtype=np.float64)
    a_grid = np.broadcast_to(np.arange(a_dim)[np.newaxis, np.newaxis, :], (d_dim, r_dim, a_dim))
    np.add.at(ae_sum, (e_mov.ravel(), a_grid.ravel()), p_mov.ravel())
    np.add.at(ae_cnt, (e_mov.ravel(), a_grid.ravel()), 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        ae_map = np.where(ae_cnt > 0, ae_sum / ae_cnt, 0).astype(np.float32)
    if ae_map.max() > 0:
        ae_map /= ae_map.max()

    # RE map: elevation × range  (mean power over azimuth & Doppler)
    re_sum = np.zeros((num_e_bins, r_dim), dtype=np.float64)
    re_cnt = np.zeros((num_e_bins, r_dim), dtype=np.float64)
    r_grid = np.broadcast_to(np.arange(r_dim)[np.newaxis, :, np.newaxis], (d_dim, r_dim, a_dim))
    np.add.at(re_sum, (e_mov.ravel(), r_grid.ravel()), p_mov.ravel())
    np.add.at(re_cnt, (e_mov.ravel(), r_grid.ravel()), 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        re_map = np.where(re_cnt > 0, re_sum / re_cnt, 0).astype(np.float32)
    if re_map.max() > 0:
        re_map /= re_map.max()

    return ae_map, re_map


def load_frame(power_path, elev_path, config):
    """Load and preprocess a single radar frame into a model-ready tensor."""
    try:
        power = np.load(power_path).astype(np.float32)
    except Exception:
        return None
    try:
        elev = np.load(elev_path).astype(np.float32) if os.path.exists(elev_path) else np.zeros_like(power)
    except Exception:
        elev = np.zeros_like(power)

    def preprocess(data, is_power=True):
        if data.ndim == 3 and data.shape[2] > data.shape[0]:
            data = data.transpose(2, 0, 1)
        if data.shape[0] == 512:
            if is_power:
                t    = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
                t    = torch.nn.functional.max_pool3d(t, kernel_size=(4, 1, 1), stride=(4, 1, 1))
                data = t.squeeze(0).squeeze(0).numpy()
            else:
                data = data[::4, :, :]
        norm = config['dataset']['normalization']
        if norm.get('enable', False) and is_power and norm.get('power_log_transform', False):
            p_min = norm.get('power_min_val', -100.0)
            p_max = norm.get('power_max_val', -31.7)
            data  = np.clip(10 * np.log10(data + 1e-10), p_min, p_max)
            data  = (data - p_min) / (p_max - p_min + 1e-6)
        return data

    power = np.transpose(preprocess(power, is_power=True),  (0, 2, 1))
    elev  = np.transpose(preprocess(elev,  is_power=False), (0, 2, 1))
    return torch.from_numpy(np.stack([power, elev], axis=0)).float()


def _get_latest_checkpoint(output_dir):
    runs = sorted(glob.glob(os.path.join(output_dir, '*')), key=os.path.getctime, reverse=True)
    for run in runs:
        for name in ('best_model.pth', 'final_model.pth'):
            ckpt = os.path.join(run, name)
            if os.path.exists(ckpt):
                return ckpt
    raise FileNotFoundError(f"No checkpoint found in {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     required=True,
                        help='Path to config YAML')
    parser.add_argument('--dataset',    required=True,
                        help='Dataset folder name (e.g. RC002). Must be inside config.dataset.base_dir')
    parser.add_argument('--checkpoint', default=None,
                        help='Path to .pth checkpoint. Defaults to config.inference.checkpoint or latest run.')
    parser.add_argument('--out_dir',    default=None,
                        help='Output directory. Defaults to config.inference.out_dir/<dataset>')
    parser.add_argument('--threshold',  type=float, default=None,
                        help='Occupancy threshold. Defaults to config.inference.threshold or dynamic midpoint.')
    parser.add_argument('--saveroad_dir',  default=None,
                        help='Override config.inference.saveroad_dir')
    parser.add_argument('--pco_dir',       default=None,
                        help='Override config.inference.camera.<dataset>.pco_dir')
    parser.add_argument('--label_txt_dir', default=None,
                        help='Override config.inference.camera.<dataset>.label_txt_dir')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    inf        = config.get('inference', {})
    cam_cfg    = inf.get('camera', {}).get(args.dataset, {})
    base_dir   = config['dataset'].get('base_dir', '')
    radar_dir  = os.path.join(base_dir, args.dataset) if base_dir else args.dataset

    # Resolve all paths: CLI arg > config > empty
    checkpoint    = args.checkpoint    or inf.get('checkpoint') or _get_latest_checkpoint(config['logging']['output_dir'])
    out_dir       = args.out_dir       or os.path.join(inf.get('out_dir', 'verification_output'), args.dataset)
    threshold     = args.threshold     if args.threshold is not None else inf.get('threshold')
    saveroad_dir  = args.saveroad_dir  or inf.get('saveroad_dir', '')
    sf = config['dataset'].get('subfolders', {})
    pco_dir       = args.pco_dir       or cam_cfg.get('pco_dir', '')       or os.path.join(radar_dir, sf.get('pco',   'pco'))
    label_txt_dir = args.label_txt_dir or cam_cfg.get('label_txt_dir', '') or os.path.join(radar_dir, sf.get('calib', 'calib'))

    out_dir = os.path.abspath(out_dir)

    print(f"Dataset  : {radar_dir}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Output   : {out_dir}")

    # Optional camera projection
    project_points = None
    if saveroad_dir and pco_dir and label_txt_dir:
        sys.path.insert(0, saveroad_dir)
        from tools import project_points_v2_withPC as project_points
        print("Camera projection: enabled")
    else:
        missing = [k for k, v in [('saveroad_dir', saveroad_dir), ('pco_dir', pco_dir), ('label_txt_dir', label_txt_dir)] if not v]
        print(f"Camera projection: disabled (missing in config.inference: {', '.join(missing)})")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = ModelFactory.get_model(config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    power_files = sorted(glob.glob(os.path.join(radar_dir, sf.get('rad_power', 'rad_power'), '*.npy')))
    print(f"Found {len(power_files)} radar frames")

    # Pre-index label files by timestamp (ms) for GT overlay
    label_dir   = os.path.join(radar_dir, sf.get('labels', 'labels'))
    label_files = sorted(glob.glob(os.path.join(label_dir, '*.npy')))
    label_ts    = np.array([_extract_ts_ms(p) for p in label_files]) \
                  if label_files else np.array([])
    print(f"Found {len(label_files)} label files for GT overlay")

    # Pre-index calibration files by timestamp
    txt_ts_arr, txt_files = np.array([]), []
    if project_points and label_txt_dir:
        txt_files  = sorted(glob.glob(os.path.join(label_txt_dir, '*.txt')))
        txt_ts_arr = np.array([_extract_ts_ms(p) for p in txt_files])
        print(f"Found {len(txt_files)} calibration files")

    plot_dir        = os.path.join(out_dir, 'prediction_plots')
    thresh_plot_dir = os.path.join(out_dir, 'prediction_plots_thresh')
    proj_dir        = os.path.join(out_dir, 'camera_projection')
    os.makedirs(plot_dir,        exist_ok=True)
    os.makedirs(thresh_plot_dir, exist_ok=True)
    if project_points:
        os.makedirs(proj_dir, exist_ok=True)

    xticks     = np.linspace(0, 255, 7)
    xlabels_az = np.round(np.degrees(np.arcsin(np.linspace(-1, 1, 7)))).astype(int)

    for power_path in tqdm(power_files):
        fname     = os.path.basename(power_path)
        ts        = fname.replace('.npy', '')
        elev_path = os.path.join(radar_dir, sf.get('rad_elev', 'rad_elev'), fname)

        tensor = load_frame(power_path, elev_path, config)
        if tensor is None:
            continue

        with torch.no_grad():
            pred_prob = torch.sigmoid(model(tensor.unsqueeze(0).to(device)))[0].cpu()

        # ── Lookup GT label (nearest within 100 ms) ───────────────────────────
        gt_label = None
        if len(label_ts) > 0:
            ts_ms = _extract_ts_ms(power_path)
            i = int(np.argmin(np.abs(label_ts - ts_ms)))
            if abs(label_ts[i] - ts_ms) < 100:
                try:
                    gt_label = np.load(label_files[i]).astype(np.float32)
                    # Dataloader flips azimuth (dim=-1) to align LiDAR→radar conventions.
                    # Replicate that here so GT matches the model's training view.
                    gt_label = gt_label[:, :, ::-1].copy()
                except Exception:
                    gt_label = None

        # ── Data setup (mirrors evaluate_predictions_views.py) ────────────────
        # tensor[0] = power (D, R, A),  tensor[1] = elev (D, R, A)
        radar_power = tensor[0]
        radar_elev  = tensor[1]
        p_np = radar_power.numpy()
        e_np = radar_elev.numpy()

        # ── AE / RE maps (elevation-indexed, from compare_gt_radar.py logic) ──
        radar_ae, radar_re = compute_ae_re_maps(p_np, e_np, config)

        # ── BEV / FV / SV projections ──────────────────────────────────────────
        radar_bev  = torch.max(radar_power, dim=0)[0]        # (R, A)
        pred_bev   = pred_prob.max(dim=0)[0]                 # (R, A)
        pred_fv    = pred_prob.max(dim=1)[0]                 # (Elev, A)
        pred_sv    = pred_prob.max(dim=2)[0]                 # (Elev, R)

        # Thresholded binary prediction (midpoint threshold)
        p_max, p_min   = pred_prob.max(), pred_prob.min()
        threshold_val  = (p_max + p_min) / 2.0
        pred_binary    = (pred_prob > threshold_val).float()
        pred_bev_th    = pred_binary.max(dim=0)[0]
        pred_fv_th     = pred_binary.max(dim=1)[0]
        pred_sv_th     = pred_binary.max(dim=2)[0]

        # GT projections (only if label available)
        if gt_label is not None:
            gt  = torch.from_numpy(gt_label)
            gt_bev = gt.max(dim=0)[0]
            gt_fv  = gt.max(dim=1)[0]
            gt_sv  = gt.max(dim=2)[0]

            # Metrics
            pred_b = (pred_prob > threshold_val).bool()
            label_b = (gt > 0).bool()
            tp = (pred_b & label_b).sum().item()
            fp = (pred_b & ~label_b).sum().item()
            fn = (~pred_b & label_b).sum().item()
            iou       = tp / (tp + fp + fn + 1e-8)
            precision = tp / (tp + fp + 1e-8)
            recall    = tp / (tp + fn + 1e-8)
            metric_str = f"IoU={iou:.3f}  Prec={precision:.3f}  Rec={recall:.3f}"
        else:
            gt_bev = gt_fv = gt_sv = None
            metric_str = "(no GT label)"

        xticks_az      = np.linspace(0, 255, 7)
        xticklabels_az = np.round(np.degrees(np.arcsin(np.linspace(-1, 1, 7)))).astype(int)

        def _create_plot(p_bev, p_fv, p_sv, out_dir, prefix, title_suffix):
            fig = plt.figure(figsize=(18, 12))
            gs  = fig.add_gridspec(3, 3)

            # Row 0 — Radar input
            ax = fig.add_subplot(gs[0, 0])
            im = ax.imshow(radar_bev.numpy(), cmap='turbo', origin='lower', aspect='auto', vmin=0, vmax=1)
            ax.set_title('Input: Radar Power (BEV)')
            ax.set_xlabel('Azimuth (°)'); ax.set_ylabel('Range (Bins)')
            ax.set_xticks(xticks_az); ax.set_xticklabels(xticklabels_az)
            plt.colorbar(im, ax=ax)

            ax = fig.add_subplot(gs[0, 1])
            im = ax.imshow(radar_ae, cmap='turbo', origin='lower', aspect='auto', vmin=0, vmax=1)
            ax.set_title('Input: Radar AE Map (Mean Power)')
            ax.set_xlabel('Azimuth (°)'); ax.set_ylabel('Elevation (Bins)')
            ax.set_xticks(xticks_az); ax.set_xticklabels(xticklabels_az)
            plt.colorbar(im, ax=ax)

            ax = fig.add_subplot(gs[0, 2])
            im = ax.imshow(radar_re, cmap='turbo', origin='lower', aspect='auto', vmin=0, vmax=1)
            ax.set_title('Input: Radar RE Map (Mean Power)')
            ax.set_xlabel('Range (Bins)'); ax.set_ylabel('Elevation (Bins)')
            plt.colorbar(im, ax=ax)

            # Row 1 — GT LiDAR
            ax = fig.add_subplot(gs[1, 0])
            im = ax.imshow(gt_bev.numpy() if gt_bev is not None else np.zeros((256,256)),
                           cmap='gray', origin='lower', aspect='auto')
            ax.set_title('GT: LiDAR Occupancy (BEV)' if gt_bev is not None else 'GT: (no matched label)')
            ax.set_xlabel('Azimuth (°)'); ax.set_ylabel('Range (Bins)')
            ax.set_xticks(xticks_az); ax.set_xticklabels(xticklabels_az)
            plt.colorbar(im, ax=ax)

            ax = fig.add_subplot(gs[1, 1])
            im = ax.imshow(gt_fv.numpy() if gt_fv is not None else np.zeros((64,256)),
                           cmap='gray', origin='lower', aspect='auto')
            ax.set_title('GT: LiDAR Occupancy (Front View)')
            ax.set_xlabel('Azimuth (°)'); ax.set_ylabel('Height (Bins)')
            ax.set_xticks(xticks_az); ax.set_xticklabels(xticklabels_az)
            plt.colorbar(im, ax=ax)

            ax = fig.add_subplot(gs[1, 2])
            im = ax.imshow(gt_sv.numpy() if gt_sv is not None else np.zeros((64,256)),
                           cmap='gray', origin='lower', aspect='auto')
            ax.set_title('GT: LiDAR Occupancy (Side View)')
            ax.set_xlabel('Range (Bins)'); ax.set_ylabel('Height (Bins)')
            plt.colorbar(im, ax=ax)

            # Row 2 — Prediction
            ax = fig.add_subplot(gs[2, 0])
            im = ax.imshow(p_bev.numpy(), cmap='magma', origin='lower', aspect='auto', vmin=0, vmax=1)
            ax.set_title(f'Prediction: BEV ({title_suffix})')
            ax.set_xlabel('Azimuth (°)'); ax.set_ylabel('Range (Bins)')
            ax.set_xticks(xticks_az); ax.set_xticklabels(xticklabels_az)
            plt.colorbar(im, ax=ax)

            ax = fig.add_subplot(gs[2, 1])
            im = ax.imshow(p_fv.numpy(), cmap='magma', origin='lower', aspect='auto', vmin=0, vmax=1)
            ax.set_title(f'Prediction: Front View ({title_suffix})')
            ax.set_xlabel('Azimuth (°)'); ax.set_ylabel('Height (Bins)')
            ax.set_xticks(xticks_az); ax.set_xticklabels(xticklabels_az)
            plt.colorbar(im, ax=ax)

            ax = fig.add_subplot(gs[2, 2])
            im = ax.imshow(p_sv.numpy(), cmap='magma', origin='lower', aspect='auto', vmin=0, vmax=1)
            ax.set_title(f'Prediction: Side View ({title_suffix})')
            ax.set_xlabel('Range (Bins)'); ax.set_ylabel('Height (Bins)')
            plt.colorbar(im, ax=ax)

            plt.suptitle(
                f"{args.dataset} | ts={ts}\n{metric_str}",
                fontsize=14
            )
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'{prefix}_{ts}.png'), bbox_inches='tight', dpi=150)
            plt.close(fig)

        _create_plot(pred_bev,    pred_fv,    pred_sv,    plot_dir,       'full',   'Confidence')
        _create_plot(pred_bev_th, pred_fv_th, pred_sv_th, thresh_plot_dir, 'thresh', f'Thresh={threshold_val:.3f}')

        # ── Camera projection ─────────────────────────────────────────────────
        if project_points is None or len(txt_ts_arr) == 0:
            continue

        ts_ms = int(ts)
        diffs = np.abs(txt_ts_arr - ts_ms)
        best  = np.argmin(diffs)
        if diffs[best] > 100:
            continue

        img_name, K, r_t, radar_to_lidar = parse_calibration(txt_files[best])
        if not img_name:
            continue
        img_path = os.path.join(pco_dir, img_name)
        if not os.path.exists(img_path):
            continue

        pred_np   = pred_prob.numpy()
        thr       = threshold if threshold is not None else (pred_np.max() + pred_np.min()) / 2.0
        pts_3d, probs = occupancy_to_points(pred_np, thr)
        if len(pts_3d) == 0:
            continue

        pts_3d += radar_to_lidar
        if len(pts_3d) > 150_000:
            top    = np.argsort(probs)[-150_000:]
            pts_3d = pts_3d[top]; probs = probs[top]

        img = cv2.imread(img_path)
        if img is None:
            continue

        img_drawn = project_to_image(pts_3d, probs, img, K, r_t, project_points)
        img_rgb   = cv2.cvtColor(img_drawn, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(figsize=(16, 9))
        ax.imshow(img_rgb)
        cbar = plt.colorbar(ax.scatter([], [], c=[], cmap='turbo', vmin=0, vmax=1),
                            ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Prediction Confidence')
        ax.set_title(f"{args.dataset} Camera Projection | ts={ts}", fontsize=14)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(proj_dir, f'proj_{ts}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"\nDone.\n  Plots: {plot_dir}")
    if project_points:
        print(f"  Projections: {proj_dir}")


if __name__ == '__main__':
    main()
