"""
Step 2 — Convert LiDAR point clouds to 3D occupancy grid labels.

Syncs LiDAR timestamps to radar timestamps and converts each matched
LiDAR frame to a binary occupancy grid saved as <radar_timestamp>.npy.

The occupancy grid dimensions must match the model output:
  (NUM_HEIGHT_BINS, GRID_R_BINS, GRID_A_BINS) = (64, 256, 256)

Note: the dataloader applies torch.flip(label, [-1]) on load to align
the azimuth axis with the radar convention, so no azimuth flip is
applied here during label generation.

Usage:
  python dataset/generate_lidar_labels.py \
    --radar_npy_dir  "E:/dataset/RC019" \
    --lidar_pcd_dir  "E:/raw/RC019/pcd" \
    --out_dir        "E:/dataset/RC019/labels" \
    --sync_threshold 100
"""
import os
import re
import glob
import argparse
import numpy as np
import h5py
from tqdm import tqdm

# ── Grid parameters — must match model config ─────────────────────────────────
# Defaults; overridden at runtime by --height_bins / --grid_r_bins / --grid_a_bins
GRID_R_BINS     = 256
GRID_A_BINS     = 256
MAX_RANGE_M     = 25.6
FOV_DEG         = 180.0
NUM_HEIGHT_BINS = 64

# ── Spatial crop (metres) ────────────────────────────────────────────────────
CROP_X_MIN = 0.0
CROP_X_MAX = 30.0
CROP_Y_MIN = -5.0
CROP_Y_MAX = 7.5
CROP_Z_MIN = -0.65
CROP_Z_MAX = 3.0
Z_RANGE    = (-2.0, 10.0)

# ── Elevation mapping parameters ──────────────────────────────────────────────
# Spherical elevation mapping — must match the inverse in utils/project_to_image.py
BORESIGHT_DEG = 5.0    # physical sensor tilt upward
GRID_CTR_DEG  = -5.0   # elevation angle that maps to the grid centre bin
PHI_MAX_RAD   = 0.5236  # ~30 degrees — controls the elevation bin spread


def extract_ts_s(filepath):
    stem = os.path.basename(filepath)
    stem = re.sub(r'\.(npy|h5)$', '', stem)
    val  = float(re.search(r'[\d.]+', stem).group(0))
    # LiDAR .h5 stems are already decimal seconds — keep as-is
    # Radar .npy stems are integer ms — divide by 1000 to get seconds
    return val if val < 1e11 else val / 1000.0


def sync_timestamps(radar_files, lidar_files, threshold_ms):
    """
    Radar-centric sync: for each radar frame find the nearest LiDAR frame.
    threshold_ms > 0  — only accept matches within this delta.
    threshold_ms == 0 — no threshold, always assign the nearest LiDAR frame.
    """
    matched     = []
    no_threshold = threshold_ms == 0
    threshold_s  = threshold_ms / 1000.0 if not no_threshold else None
    lidar_ts     = np.array([extract_ts_s(f) for f in lidar_files])
    for rf in radar_files:
        r_ts_s = extract_ts_s(rf)
        diffs  = np.abs(lidar_ts - r_ts_s)
        idx    = int(np.argmin(diffs))
        if no_threshold or diffs[idx] <= threshold_s:
            matched.append({'radar': rf, 'lidar': lidar_files[idx],
                            'r_ts': r_ts_s, 'dt_ms': round(diffs[idx] * 1000)})
    return matched


def process_lidar_h5(file_path):
    """Load a .h5 LiDAR file and convert it to a binary occupancy grid."""
    try:
        with h5py.File(file_path, 'r') as f:
            points = np.array(f['lidar'] if 'lidar' in f else f[list(f.keys())[0]])
    except Exception as e:
        print(f"  [ERROR] {os.path.basename(file_path)}: {e}")
        return None

    if points.shape[1] < 3:
        return None

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Spatial crop
    mask = ((x > CROP_X_MIN) & (x < CROP_X_MAX) &
            (y > CROP_Y_MIN) & (y < CROP_Y_MAX) &
            (z > CROP_Z_MIN) & (z < CROP_Z_MAX))
    x, y, z = x[mask], y[mask], z[mask]

    # Polar conversion
    r        = np.sqrt(x**2 + y**2)
    theta    = np.arctan2(y, x)
    half_fov = np.deg2rad(FOV_DEG) / 2.0
    mask2    = ((r > 0) & (r < MAX_RANGE_M) &
                (theta >= -half_fov) & (theta <= half_fov) &
                (z >= Z_RANGE[0]))
    r, theta, z = r[mask2], theta[mask2], z[mask2]

    if len(r) == 0:
        return np.zeros((NUM_HEIGHT_BINS, GRID_R_BINS, GRID_A_BINS), dtype=np.uint8)

    # Range bin
    r_idx = np.clip(np.floor(r / MAX_RANGE_M * GRID_R_BINS).astype(int),
                    0, GRID_R_BINS - 1)

    # Azimuth bin (sine-based mapping)
    sin_theta    = np.sin(theta)
    sin_half_fov = np.sin(half_fov)
    sin_norm     = (sin_theta + sin_half_fov) / (2.0 * sin_half_fov)
    a_idx        = np.clip(np.floor(sin_norm * GRID_A_BINS).astype(int),
                           0, GRID_A_BINS - 1)

    # Elevation bin (spherical mapping with boresight and grid-centre correction)
    phi_raw  = np.arctan2(z, r)
    phi_aln  = phi_raw - np.deg2rad(BORESIGHT_DEG)
    phi_map  = phi_aln  - np.deg2rad(GRID_CTR_DEG)
    phi_norm = (np.sin(phi_map) + np.sin(PHI_MAX_RAD)) / (2.0 * np.sin(PHI_MAX_RAD))
    z_idx    = np.clip(np.floor(phi_norm * NUM_HEIGHT_BINS).astype(int),
                       0, NUM_HEIGHT_BINS - 1)

    # Fill occupancy grid
    # Note: dataloader applies torch.flip(label, [-1]) on load to align azimuth with radar
    grid   = np.zeros((NUM_HEIGHT_BINS, GRID_R_BINS, GRID_A_BINS), dtype=np.uint8)
    coords = np.unique(np.stack((z_idx, r_idx, a_idx), axis=1), axis=0)
    grid[coords[:, 0], coords[:, 1], coords[:, 2]] = 1

    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--radar_npy_dir',  required=True,
                        help='Dataset root containing rad_power/ subfolder')
    parser.add_argument('--lidar_pcd_dir',  required=True,
                        help='Folder with .h5 LiDAR point cloud files')
    parser.add_argument('--out_dir',        required=True,
                        help='Output folder for .npy occupancy label files')
    parser.add_argument('--sync_threshold', type=int, default=100,
                        help='Maximum timestamp delta accepted as a match (ms). '
                             'Set to 0 to disable threshold — always assigns nearest LiDAR frame.')
    parser.add_argument('--height_bins',    type=int, default=64,
                        help='Number of height bins in the occupancy grid (default 64). '
                             'Use 128 for higher vertical resolution. '
                             'Must match model num_classes in train config.')
    parser.add_argument('--grid_r_bins',    type=int, default=256,
                        help='Range bins (default 256, must match radar cube spatial size).')
    parser.add_argument('--grid_a_bins',    type=int, default=256,
                        help='Azimuth bins (default 256, must match radar cube spatial size).')
    args = parser.parse_args()

    # Override module-level grid constants from CLI args
    global NUM_HEIGHT_BINS, GRID_R_BINS, GRID_A_BINS
    NUM_HEIGHT_BINS = args.height_bins
    GRID_R_BINS     = args.grid_r_bins
    GRID_A_BINS     = args.grid_a_bins

    os.makedirs(args.out_dir, exist_ok=True)

    radar_files = sorted(glob.glob(os.path.join(args.radar_npy_dir, 'rad_power', '*.npy')))
    lidar_files = sorted(glob.glob(os.path.join(args.lidar_pcd_dir, '*.h5')))

    print(f"Grid         : {NUM_HEIGHT_BINS}h × {GRID_R_BINS}r × {GRID_A_BINS}a")
    print(f"Radar frames : {len(radar_files)}")
    print(f"LiDAR frames : {len(lidar_files)}")

    matched = sync_timestamps(radar_files, lidar_files, args.sync_threshold)
    thresh_str = "no threshold" if args.sync_threshold == 0 else f"threshold {args.sync_threshold} ms"
    print(f"Synced       : {len(matched)} pairs ({thresh_str})")
    if not matched:
        print("No matches found — check paths and sync_threshold.")
        return

    skipped = 0
    for m in tqdm(matched, desc='Generating labels'):
        out_path = os.path.join(args.out_dir, f"{m['r_ts']}.npy")
        if os.path.exists(out_path):
            skipped += 1
            continue
        grid = process_lidar_h5(m['lidar'])
        if grid is not None:
            np.save(out_path, grid)

    saved = len([f for f in os.listdir(args.out_dir) if f.endswith('.npy')])
    print(f"\nDone. Labels saved to: {args.out_dir}")
    print(f"  Total .npy files : {saved}")
    if skipped:
        print(f"  Skipped (already exist) : {skipped}")

    occ = sum(np.load(os.path.join(args.out_dir, f)).mean()
              for f in os.listdir(args.out_dir) if f.endswith('.npy')) / max(saved, 1)
    print(f"  Mean occupancy : {occ * 100:.3f}%")


if __name__ == '__main__':
    main()
