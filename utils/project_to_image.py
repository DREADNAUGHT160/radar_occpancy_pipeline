"""
Project radar occupancy predictions onto camera images.

Requires the SAVEROAD_DataLoader toolkit for the reprojection functions.
Pass its path via --saveroad_dir.

Usage:
  python utils/project_to_image.py \
    --config       configs/train_config.yaml \
    --checkpoint   E:/checkpoints/20260417_090255/best_model.pth \
    --label_txt_dir  E:/dataset/RC019/calib \
    --pco_dir      E:/dataset/RC019/pco \
    --saveroad_dir "D:/path/to/SAVEROAD_DataLoader" \
    --out_dir      verification_output/my_run/camera_projection \
    --threshold    0.4
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
from tqdm import tqdm
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.factory import ModelFactory
from dataset.dataloader import RadarDataset

# -- Geometry constants -- must match generate_lidar_labels.py -----------------
R_BINS          = 256
A_BINS          = 256
NUM_HEIGHT_BINS = 64
MAX_RANGE       = 25.6
FOV_RAD         = np.deg2rad(180.0)
BORESIGHT_DEG   = 5.0
PHI_MAX_RAD     = 0.5236  # ~30 deg


def parse_calibration(txt_path):
    """Parse camera intrinsics and extrinsics from a SAVEROAD .txt label file."""
    with open(txt_path) as f:
        content = f.read()

    match = re.search(r'"PCO_frame":"([^"]+)"', content)
    img_name = match.group(1) if match else None

    K = np.eye(3)
    match = re.search(r'"PCO_Intrinsic":\s*([\d\s.-]+),', content)
    if match:
        vals = np.array(match.group(1).split(), dtype=float)
        if len(vals) >= 9:
            K = vals[:9].reshape(3, 3)

    r_t = np.zeros(6)
    match = re.search(r'"Lidar_to_PCO":\s*([\d\s.-]+),', content)
    if match:
        vals = np.array(match.group(1).split(), dtype=float)
        if len(vals) >= 6:
            r_t = vals[:6]

    radar_to_lidar = np.zeros(3)
    match = re.search(r'"Translation_Radar_to_Lidar":\s*([-\d\s.]+)\s*,', content)
    if match:
        vals = np.array(match.group(1).strip().split(), dtype=float)
        if len(vals) >= 3:
            radar_to_lidar = vals[:3]
        elif len(vals) >= 1:
            radar_to_lidar[0] = vals[0]

    return img_name, K, r_t, radar_to_lidar


def occupancy_to_points(pred_prob, threshold):
    """
    Convert a predicted occupancy grid to 3D Cartesian points (radar frame).

    Inverts the spherical mapping used in generate_lidar_labels.py.
    The dataloader flips the azimuth axis on load, so we un-flip here first.
    """
    z_idx, r_idx, a_idx = np.where(pred_prob > threshold)
    probs = pred_prob[z_idx, r_idx, a_idx]

    # Range
    r = (r_idx + 0.5) / R_BINS * MAX_RANGE

    # Azimuth -- un-flip the dataloader's torch.flip(label, [-1])
    a_orig    = (A_BINS - 1) - a_idx
    sin_half  = np.sin(FOV_RAD / 2.0)
    sin_norm  = (a_orig + 0.5) / A_BINS
    sin_theta = np.clip(sin_norm * 2 * sin_half - sin_half, -1.0, 1.0)
    theta     = np.arcsin(sin_theta)

    # Elevation -- invert spherical mapping.
    # Forward transform applies phi_aln = phi_raw - BORESIGHT, then phi_map = phi_aln - GRID_CTR
    # (-5deg + +5deg). The two corrections cancel so phi_map = phi_raw. Inverse recovers phi_raw
    # directly with no boresight offset needed.
    sin_max = np.sin(PHI_MAX_RAD)
    elev_n  = (z_idx + 0.5) / NUM_HEIGHT_BINS
    sin_phi = np.clip(elev_n * 2 * sin_max - sin_max, -1.0, 1.0)
    phi     = np.arcsin(sin_phi)   # = phi_raw (boresight + grid-centre cancel in forward pass)

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = r * np.tan(phi)

    return np.vstack((x, y, z)).T, probs


def project_to_image(pts, probs, img, K, r_t, project_points):
    """Project 3D points onto the camera image using the SAVEROAD reprojection tool.

    reprojection_opt filters points (z > 0.2, inside image bounds) but returns no
    indices, so naively mapping colors[0:N_proj] picks wrong confidences.
    Fix: use reprojection_data_loader which carries extra columns through the same
    filters via pcd_ground — column 3 gives the correctly matched probabilities.
    """
    if len(pts) == 0:
        return img
    # Append probs as column 3 so the filter indices inside reprojection_data_loader
    # are applied to both xyz and confidence simultaneously.
    pts_with_probs = np.column_stack([pts.astype(np.float64),
                                      probs.astype(np.float64)])
    proj_pts, _, pcd_ground = project_points.reprojection_data_loader(
        pts_with_probs, img, K, r_t)
    if len(proj_pts) == 0:
        return img
    kept_probs = np.clip(pcd_ground[:, 3], 0.0, 1.0)
    colors     = (plt.cm.turbo(kept_probs)[:, :3] * 255).astype(np.int32)
    colors     = np.fliplr(colors)   # RGB -> BGR for OpenCV
    img_drawn  = project_points.project_points(img.copy(), proj_pts, colors, size=3)
    return img_drawn


def _get_latest_checkpoint(output_dir):
    runs = sorted(glob.glob(os.path.join(output_dir, '*')), key=os.path.getctime, reverse=True)
    for run in runs:
        for name in ('best_model.pth', 'final_model.pth'):
            ckpt = os.path.join(run, name)
            if os.path.exists(ckpt):
                return ckpt
    raise FileNotFoundError(f"No checkpoint found in {output_dir}")


def main():
    import cv2
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',        required=True)
    parser.add_argument('--dataset',       default=None,
                        help='Dataset folder name to run projection on (e.g. RC019). '
                             'Uses config.dataset.base_dir/<dataset> as radar_dir.')
    parser.add_argument('--checkpoint',    default=None,
                        help='Path to .pth checkpoint. Defaults to config.inference.checkpoint or latest run.')
    parser.add_argument('--label_txt_dir', default=None,
                        help='Override config.inference.camera.<dataset>.label_txt_dir')
    parser.add_argument('--pco_dir',       default=None,
                        help='Override config.inference.camera.<dataset>.pco_dir')
    parser.add_argument('--saveroad_dir',  default=None,
                        help='Override config.inference.saveroad_dir')
    parser.add_argument('--out_dir',       default=None,
                        help='Override config.inference.out_dir')
    parser.add_argument('--threshold',     type=float, default=None,
                        help='Occupancy threshold. Default: config.inference.threshold or dynamic midpoint.')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    inf       = config.get('inference', {})
    ds_name   = args.dataset or ''
    cam_cfg   = inf.get('camera', {}).get(ds_name, {})
    base_dir  = config['dataset'].get('base_dir', '')

    # Resolve radar_dir first -- used by pco_dir / label_txt_dir fallbacks below
    if ds_name and base_dir:
        radar_dir = os.path.join(base_dir, ds_name)
    elif ds_name:
        radar_dir = ds_name
    else:
        radar_dir = config['dataset'].get('radar_dir', '')

    # Resolve all paths: CLI arg > config > auto-derive from radar_dir
    checkpoint    = args.checkpoint    or inf.get('checkpoint') or _get_latest_checkpoint(config['logging']['output_dir'])
    saveroad_dir  = args.saveroad_dir  or inf.get('saveroad_dir', '')
    sf = config['dataset'].get('subfolders', {})
    pco_dir       = args.pco_dir       or cam_cfg.get('pco_dir', '')       or os.path.join(radar_dir, sf.get('pco',   'pco'))
    label_txt_dir = args.label_txt_dir or cam_cfg.get('label_txt_dir', '') or os.path.join(radar_dir, sf.get('calib', 'calib'))
    out_dir       = args.out_dir       or os.path.join(inf.get('out_dir', 'verification_output'), ds_name, 'camera_projection')
    threshold_cfg = args.threshold     if args.threshold is not None else inf.get('threshold')

    if not saveroad_dir:
        print("ERROR: saveroad_dir not set. Add it to config.inference.saveroad_dir or pass --saveroad_dir.")
        return

    sys.path.insert(0, saveroad_dir)
    from tools import project_points_v2_withPC as project_points

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = ModelFactory.get_model(config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    # Build a temporary config pointing at the chosen dataset
    ds_config = {**config, 'dataset': {**config['dataset'], 'radar_dir': radar_dir,
                                        'lidar_path': os.path.join(radar_dir, 'labels')}}
    full_ds = RadarDataset(radar_dir, augment=False, config=ds_config)
    print(f"Running camera projection for {len(full_ds)} frames in {radar_dir}")

    os.makedirs(out_dir, exist_ok=True)

    if not label_txt_dir:
        print("WARNING: label_txt_dir not set -- skipping camera projection. "
              "Set config.inference.camera.<dataset>.label_txt_dir")
        return
    if not pco_dir:
        print("WARNING: pco_dir not set -- skipping camera projection. "
              "Set config.inference.camera.<dataset>.pco_dir")
        return

    # Pre-index calibration files by timestamp
    txt_files  = sorted(glob.glob(os.path.join(label_txt_dir, '*.txt')))
    def _txt_ts(p):
        stem = os.path.basename(p).replace('.txt', '')
        try: return int(float(stem) * 1000)
        except ValueError: return 0
    txt_ts_arr = np.array([_txt_ts(p) for p in txt_files]) if txt_files else np.array([])

    for idx in tqdm(range(len(full_ds))):
        radar_tensor, _ = full_ds[idx]
        sample_info     = full_ds.matched_data[idx]

        if len(txt_ts_arr) == 0:
            continue
        ts_ms   = int(os.path.basename(sample_info['power']).replace('.npy', ''))
        diffs   = np.abs(txt_ts_arr - ts_ms)
        best    = np.argmin(diffs)
        if diffs[best] > 100:
            continue
        txt_path = txt_files[best]

        img_name, K, r_t, radar_to_lidar = parse_calibration(txt_path)
        if not img_name:
            continue
        img_path = os.path.join(pco_dir, img_name)
        if not os.path.exists(img_path):
            continue

        with torch.no_grad():
            pred_np = torch.sigmoid(model(radar_tensor.unsqueeze(0).to(device)))[0].cpu().numpy()

        threshold = threshold_cfg if threshold_cfg is not None else \
            (pred_np.max() + pred_np.min()) / 2.0

        pts_3d, probs = occupancy_to_points(pred_np, threshold)
        if len(pts_3d) == 0:
            continue

        pts_3d += radar_to_lidar   # shift from radar frame to LiDAR frame

        if len(pts_3d) > 150_000:
            top = np.argsort(probs)[-150_000:]
            pts_3d, probs = pts_3d[top], probs[top]

        img = cv2.imread(img_path)
        if img is None:
            continue

        img_drawn = project_to_image(pts_3d, probs, img, K, r_t, project_points)
        img_rgb   = cv2.cvtColor(img_drawn, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(figsize=(16, 9))
        ax.imshow(img_rgb)
        sc   = ax.scatter([], [], c=[], cmap='turbo', vmin=0, vmax=1)
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Prediction Confidence')
        ax.set_title(f"Radar Occupancy Projection -- Sample {idx:03d} | ts={ts_ms}", fontsize=14)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'proj_{idx:03d}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"\nProjections saved to: {os.path.abspath(out_dir)}")


if __name__ == '__main__':
    main()
