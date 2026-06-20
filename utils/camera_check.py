"""
Camera projection check — overlays model predictions on camera images.

Converts the predicted occupancy grid to 3D points and projects them
onto camera images using calibration from the calib .txt files.
Generates equally spaced frames for quick visual verification.

Requires the SAVEROAD DataLoader toolkit (set saveroad_dir in eval_config.yaml
or pass --saveroad_dir).

Usage:
  python utils/camera_check.py --config configs/eval_config.yaml --checkpoint checkpoints/<run_id>/best_model.pth
"""
import os
import sys
import re
import glob
import argparse
import yaml
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.factory import ModelFactory
from dataset.dataloader import RadarDataset
from utils.project_to_image import occupancy_to_points, parse_calibration


def _extract_ts_ms(path):
    match = re.search(r'(\d+\.\d+|\d+)', os.path.basename(path))
    if match:
        val = float(match.group(0))
        return int(val * 1000) if val < 1e11 else int(val)
    return 0


def _build_ds_config(base_cfg, rc_dir):
    sf = base_cfg.get('subfolders', {})
    return {
        'model':   base_cfg.get('model', {}),
        'dataset': {
            'radar_dir':         rc_dir,
            'lidar_path':        os.path.join(rc_dir, sf.get('labels', 'labels')),
            'sync_threshold_ms': 100,
            'subfolders':        sf,
            'normalization':     base_cfg.get('normalization', {}),
            'filter_bboxes':     False,
            'label_text_dir':    '',
        },
    }


def _find_nearest_file(files, ts_arr, ts_ms, threshold_ms=100):
    if len(ts_arr) == 0:
        return None
    diffs = np.abs(ts_arr - ts_ms)
    best  = int(np.argmin(diffs))
    return files[best] if diffs[best] <= threshold_ms else None


def project_and_save(idx, pts_3d, probs, img_path, K, r_t,
                     radar_to_lidar, project_points, out_dir, ts, threshold):
    """Project 3D points onto the camera image and save side-by-side with original."""
    if img_path is None or not os.path.exists(img_path):
        print(f"  Frame {idx}: camera image not found, skipping.")
        return

    img = cv2.imread(img_path)
    if img is None:
        return

    # Shift from radar frame to LiDAR frame
    pts_world = pts_3d + radar_to_lidar

    # Limit to highest-confidence points to keep overlay readable
    if len(pts_world) > 100_000:
        top = np.argsort(probs)[-100_000:]
        pts_world = pts_world[top]
        probs_plot = probs[top]
    else:
        probs_plot = probs

    # Project via SAVEROAD toolkit
    try:
        proj_pts, _, _, _ = project_points.reprojection_opt(pts_world, img, K, r_t)
    except Exception as e:
        print(f"  Frame {idx}: projection error — {e}")
        return

    if len(proj_pts) == 0:
        print(f"  Frame {idx}: no points projected onto image.")
        return

    colors     = (plt.cm.turbo(probs_plot[:len(proj_pts)])[:, :3] * 255).astype(np.int32)
    colors_bgr = np.fliplr(colors)
    img_drawn  = project_points.project_points(img.copy(), proj_pts, colors_bgr, size=3)

    img_rgb_orig  = cv2.cvtColor(img,       cv2.COLOR_BGR2RGB)
    img_rgb_drawn = cv2.cvtColor(img_drawn, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    axes[0].imshow(img_rgb_orig);  axes[0].set_title('Camera Image');              axes[0].axis('off')
    axes[1].imshow(img_rgb_drawn); axes[1].set_title('Radar Predictions Overlaid'); axes[1].axis('off')

    sm = plt.cm.ScalarMappable(cmap='turbo', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[1], fraction=0.03, pad=0.02)
    cbar.set_label('Prediction Confidence')

    plt.suptitle(f"Frame {idx:03d}  |  ts={ts}  |  thresh={threshold}  |  {len(proj_pts)} pts projected",
                 fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'cam_{idx:03d}_{ts}.png')
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',      default='configs/eval_config.yaml')
    parser.add_argument('--checkpoint',  default=None)
    parser.add_argument('--dataset',     default=None,
                        help='RC folder name (overrides eval_config camera.dataset)')
    parser.add_argument('--saveroad_dir', default=None)
    parser.add_argument('--n_plots',     type=int, default=None)
    parser.add_argument('--threshold',   type=float, default=None)
    parser.add_argument('--out_dir',     default=None)
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    cam_cfg      = config.get('camera', {})
    basic_cfg    = config.get('basic', {})

    ckpt         = args.checkpoint   or config.get('checkpoint', '')
    rc_name      = args.dataset      or cam_cfg.get('dataset')  or basic_cfg.get('dataset', '')
    saveroad_dir = args.saveroad_dir or cam_cfg.get('saveroad_dir', '')
    n_plots      = args.n_plots      or cam_cfg.get('n_plots', 10)
    threshold    = args.threshold    or cam_cfg.get('threshold', 0.4)
    base_dir     = config.get('base_dir', '')
    out_dir      = args.out_dir or os.path.join(config.get('out_dir', 'verification_output/eval'), 'camera_check')

    if not ckpt:
        print("ERROR: set checkpoint in eval_config.yaml or pass --checkpoint"); return
    if not rc_name:
        print("ERROR: set camera.dataset in eval_config.yaml or pass --dataset"); return
    if not saveroad_dir:
        print("ERROR: set camera.saveroad_dir in eval_config.yaml or pass --saveroad_dir"); return

    sys.path.insert(0, saveroad_dir)
    try:
        from tools import project_points_v2_withPC as project_points
    except ImportError as e:
        print(f"ERROR: could not import SAVEROAD toolkit from {saveroad_dir}: {e}"); return

    rc_dir    = os.path.join(base_dir, rc_name) if base_dir else rc_name
    sf        = config.get('subfolders', {})
    calib_dir = cam_cfg.get('calib_dir') or os.path.join(rc_dir, sf.get('calib', 'calib'))
    pco_dir   = cam_cfg.get('pco_dir',   '')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = ModelFactory.get_model(config).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    ds = RadarDataset(rc_dir, augment=False, config=_build_ds_config(config, rc_dir))
    total = len(ds)
    print(f"\nDataset : {rc_name}  ({total} frames)")
    print(f"Plotting : {n_plots} equally spaced frames  →  {out_dir}")

    # Index calib files
    txt_files = sorted(glob.glob(os.path.join(calib_dir, '*.txt')))
    txt_ts    = np.array([_extract_ts_ms(f) for f in txt_files]) if txt_files else np.array([])

    # Pick equally spaced indices
    indices = np.linspace(0, total - 1, min(n_plots, total)).astype(int)
    os.makedirs(out_dir, exist_ok=True)

    for idx in tqdm(indices, desc="Camera projection"):
        radar_tensor, _ = ds[int(idx)]
        sample  = ds.matched_data[int(idx)]
        ts_ms   = _extract_ts_ms(sample['power'])
        ts      = os.path.basename(sample['power']).replace('.npy', '')

        # Find calibration file
        txt_path = _find_nearest_file(txt_files, txt_ts, ts_ms)
        if txt_path is None:
            print(f"  Frame {idx}: no calib file within 100ms, skipping.")
            continue

        img_name, K, r_t, radar_to_lidar = parse_calibration(txt_path)

        # Find camera image
        img_path = None
        if img_name and pco_dir:
            candidate = os.path.join(pco_dir, img_name)
            img_path  = candidate if os.path.exists(candidate) else None
        if img_path is None and pco_dir:
            # Try matching by timestamp
            for ext in ('.jpg', '.jpeg', '.png'):
                candidate = os.path.join(pco_dir, ts + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break

        # Inference
        with torch.no_grad():
            pred_np = torch.sigmoid(
                model(radar_tensor.unsqueeze(0).to(device))
            )[0].cpu().numpy()

        pts_3d, probs = occupancy_to_points(pred_np, threshold=float(threshold))

        if len(pts_3d) == 0:
            print(f"  Frame {idx}: no points above threshold {threshold}.")
            continue

        project_and_save(int(idx), pts_3d, probs, img_path,
                         K, r_t, radar_to_lidar,
                         project_points, out_dir, ts, threshold)

    print(f"\nSaved {len(indices)} camera projection images to: {os.path.abspath(out_dir)}")


if __name__ == '__main__':
    main()
