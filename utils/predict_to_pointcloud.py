"""
Run model inference on a prepared RC folder and save predicted occupancy
as 3D point clouds.

Outputs the full raw probability distribution — no threshold applied.
Every voxel with probability > 0.01 (background floor) is saved with its
confidence score so the complete prediction shape is preserved.

Output formats:
  .txt  -- CSV x,y,z,confidence  (default, opens in CloudCompare / Excel)
  .ply  -- ASCII PLY              (opens in CloudCompare / MeshLab)
  .npy  -- numpy array (N,4): x,y,z,confidence

Coordinate frames:
  radar  -- default, origin at radar sensor
  lidar  -- adds Translation_Radar_to_Lidar from calib .txt
            USE THIS for camera projection (camera calib expects LiDAR frame)

Usage:
  # All frames, radar frame
  python utils/predict_to_pointcloud.py --config configs/eval_config.yaml --rc RC019

  # Single frame, LiDAR frame (for camera projection)
  python utils/predict_to_pointcloud.py --config configs/eval_config.yaml --rc RC019 --frame 5 --frame_coord lidar

  # Custom output directory
  python utils/predict_to_pointcloud.py --config configs/eval_config.yaml --rc RC019 --out_dir my_clouds/
"""

import os
import sys
import re
import glob
import argparse
import yaml
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.factory import ModelFactory
from dataset.dataloader import RadarDataset
from utils.project_to_image import occupancy_to_points, parse_calibration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _find_r2l(calib_dir, ts_ms, threshold_ms=100):
    """Return radar-to-lidar translation for the nearest calib file, or zeros."""
    txt_files = sorted(glob.glob(os.path.join(calib_dir, '*.txt')))
    if not txt_files:
        return np.zeros(3)
    ts_arr = np.array([_extract_ts_ms(f) for f in txt_files])
    diffs  = np.abs(ts_arr - ts_ms)
    best   = int(np.argmin(diffs))
    if diffs[best] > threshold_ms:
        return np.zeros(3)
    _, _, _, r2l = parse_calibration(txt_files[best])
    return r2l


def _save_txt(path, pts, confidences):
    """Save point cloud as CSV: x,y,z,confidence"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = np.column_stack([pts, confidences])
    np.savetxt(path, data, fmt='%.4f', delimiter=',', header='x,y,z,confidence', comments='')


def _save_ply(path, pts, confidences):
    """Save an ASCII PLY point cloud with x,y,z,confidence."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = len(pts)
    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float confidence\n")
        f.write("end_header\n")
        for (x, y, z), c in zip(pts, confidences):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {c:.4f}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Predict radar occupancy -> 3D point cloud (.ply + .npy)')
    parser.add_argument('--config',      default='configs/eval_config.yaml')
    parser.add_argument('--checkpoint',  default=None,
                        help='Override checkpoint in config')
    parser.add_argument('--rc',          required=True,
                        help='RC folder name, e.g. RC019')
    parser.add_argument('--frame',       type=int, default=None,
                        help='Single frame index (0-based). Default: all frames')
    parser.add_argument('--frame_coord', choices=['radar', 'lidar'], default='radar',
                        help='Coordinate frame: radar (default) or lidar')
    parser.add_argument('--fmt',         choices=['ply', 'npy', 'txt', 'both'], default='txt',
                        help='Output format: txt (default), ply, npy, both (ply+npy)')
    parser.add_argument('--out_dir',     default=None,
                        help='Output directory (default: verification_output/pointclouds/<rc>)')
    args = parser.parse_args()

    # -- Load config -----------------------------------------------------------
    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    ckpt     = args.checkpoint or config.get('checkpoint', '')
    base_dir = config.get('base_dir', '').strip()
    sf       = config.get('subfolders', {})

    if not ckpt:
        print("ERROR: set checkpoint in config or pass --checkpoint")
        return

    rc_dir = os.path.join(base_dir, args.rc) if base_dir else args.rc
    if not os.path.isdir(rc_dir):
        print(f"ERROR: RC folder not found: {rc_dir}")
        return

    out_dir = args.out_dir or os.path.join(
        config.get('out_dir', 'verification_output/eval'),
        'pointclouds', args.rc
    )
    os.makedirs(out_dir, exist_ok=True)

    # -- Load model ------------------------------------------------------------
    # Background floor — skip voxels the sigmoid assigns near-zero probability.
    # Keeps file sizes reasonable while preserving the full probability distribution.
    BG_FLOOR = 0.01

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device     : {device}")
    print(f"Checkpoint : {ckpt}")
    print(f"RC folder  : {rc_dir}")
    print(f"Frame coord: {args.frame_coord}")
    print(f"Output     : {out_dir}\n")

    model = ModelFactory.get_model(config).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    # -- Dataset ---------------------------------------------------------------
    ds = RadarDataset(rc_dir, augment=False, config=_build_ds_config(config, rc_dir))
    print(f"Frames in {args.rc}: {len(ds)}")

    calib_dir = os.path.join(rc_dir, sf.get('calib', 'calib'))
    use_lidar_frame = (args.frame_coord == 'lidar')

    # Determine which frames to process
    if args.frame is not None:
        if args.frame < 0 or args.frame >= len(ds):
            print(f"ERROR: --frame {args.frame} out of range [0, {len(ds)-1}]")
            return
        indices = [args.frame]
    else:
        indices = list(range(len(ds)))

    # -- Inference loop --------------------------------------------------------
    saved = 0
    for idx in tqdm(indices, desc='Predicting'):
        radar_tensor, _ = ds[idx]
        sample = ds.matched_data[idx]
        ts_ms  = _extract_ts_ms(sample['power'])
        ts_str = os.path.basename(sample['power']).replace('.npy', '')

        with torch.no_grad():
            pred_prob = torch.sigmoid(
                model(radar_tensor.unsqueeze(0).to(device))
            )[0].cpu().numpy()

        pts, confidences = occupancy_to_points(pred_prob, BG_FLOOR)

        if len(pts) == 0:
            print(f"  [WARN] Frame {idx}: no predicted points (model output is flat background)")
            continue

        # Shift to LiDAR frame if requested
        if use_lidar_frame:
            r2l  = _find_r2l(calib_dir, ts_ms)
            pts  = pts + r2l

        fname = f"frame_{idx:03d}_{ts_str}"

        if args.fmt == 'txt':
            _save_txt(os.path.join(out_dir, f"{fname}.txt"), pts, confidences)

        if args.fmt in ('ply', 'both'):
            _save_ply(os.path.join(out_dir, f"{fname}.ply"), pts, confidences)

        if args.fmt in ('npy', 'both'):
            np.save(os.path.join(out_dir, f"{fname}.npy"),
                    np.column_stack([pts, confidences]).astype(np.float32))

        saved += 1

    print(f"\nDone. Saved {saved}/{len(indices)} point clouds -> {os.path.abspath(out_dir)}")
    print(f"\nOpen in CloudCompare:")
    print(f"  File -> Open -> select the .txt or .ply files")
    print(f"  Set scalar field to 'confidence' to colour by prediction strength")
    print(f"\nFor camera projection use --frame_coord lidar to align with camera calib.")


if __name__ == '__main__':
    main()
