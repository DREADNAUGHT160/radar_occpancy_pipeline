"""
Generate camera projections + BEV range plots of CFAR detections for all DL-synced frames.

Three-way sync chain (identical to thesis_eval):
  rad_power ts  ->  calib (Radar_frame match, <=100 ms)
                ->  cfar  (Radar_frame ts, <=5 ms)
                ->  camera image (from calib PCO_frame field)

Output per frame: three-panel figure
  Left        -- camera image with CFAR points projected (coloured by score) + GT box
  Top-right   -- radar power BEV; heading = radar frame filename
  Bottom-right -- CFAR detections top-down in metres; heading = CFAR filename

Run:
  conda activate thesis_model
  python utils/cfar_camera_proj.py --config configs/cfar_cam_all.yaml
"""
import sys
import os
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon as MplPolygon
from tqdm import tqdm
import yaml

from dataset.dataloader import RadarDataset
from utils.thesis_eval import (
    _extract_ts_ms,
    _load_cfar,
    _cfar_to_lidar,
    _project_pts_cam,
    _in_image,
    _draw_bbox_cam,
    _build_calib_index,
    _find_calib_by_radar_frame,
)
from utils.project_to_image import parse_calibration


def _build_ds_config(config, rc_dir):
    sf   = config.get('subfolders', {})
    norm = config.get('normalization', config.get('dataset', {}).get('normalization', {}))
    return {
        'model':      config.get('model', {}),
        'dataset':    {'normalization': norm, 'label_text_dir': ''},
        'subfolders': sf,
    }


def _build_cfar_index(cfar_dir):
    """Pre-load CFAR file list and timestamps for fast per-frame lookup."""
    files = (sorted(glob.glob(os.path.join(cfar_dir, '*.npy'))) +
             sorted(glob.glob(os.path.join(cfar_dir, '*.txt'))))
    if not files:
        return [], np.array([])
    ts_arr = np.array([_extract_ts_ms(f) for f in files], dtype=float)
    return files, ts_arr


def _find_cfar_fname(cfar_files, cfar_ts_arr, ts_ms, threshold_ms=5):
    """Return basename of the CFAR file closest to ts_ms, or 'N/A' if none within threshold."""
    if len(cfar_ts_arr) == 0:
        return 'N/A'
    diffs = np.abs(cfar_ts_arr - ts_ms)
    best  = int(np.argmin(diffs))
    if diffs[best] > threshold_ms:
        return 'N/A'
    return os.path.basename(cfar_files[best])


def _box_bev_corners(corners_3d):
    """Return convex-hull XY footprint of the bounding box for BEV display."""
    if corners_3d is None:
        return None
    xy = corners_3d[:, :2]
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(xy)
        return xy[hull.vertices]
    except Exception:
        return xy


def _save_combined(pts_lidar, scores, corners, K, r_t,
                   radar_tensor, img_path,
                   rc_name, radar_fname, cfar_fname, frame_idx, out_path):
    """
    Three-panel figure:
      Left        -- camera + projected CFAR (coloured by score) + GT box
      Top-right   -- radar power BEV; title = radar frame filename
      Bottom-right -- CFAR detections top-down in metres; title = CFAR filename
    """
    s_arr  = scores if (scores is not None and len(scores) > 0) else np.ones(len(pts_lidar))
    s_min  = float(s_arr.min())
    s_max  = float(s_arr.max())
    norm_s = Normalize(vmin=s_min, vmax=s_max)
    cmap   = plt.cm.turbo

    # ---- Camera panel -------------------------------------------------------
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return False
    h_img, w_img = img_bgr.shape[:2]

    px, front = _project_pts_cam(pts_lidar.astype(np.float64), K, r_t)
    in_img    = _in_image(px, h_img, w_img)

    img_draw = img_bgr.copy()
    if in_img.any():
        sc_s = s_arr[front][in_img]
        for (x, y), s in zip(px[in_img], sc_s):
            bgr = tuple(int(c * 255) for c in reversed(cmap(norm_s(float(s)))[:3]))
            cv2.circle(img_draw, (int(x), int(y)), 4, bgr, -1)
    if corners is not None:
        _draw_bbox_cam(img_draw, corners, K, r_t, color=(0, 0, 220), thickness=2)
    img_rgb = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)

    # ---- Radar power BEV (bins) ---------------------------------------------
    power_np = radar_tensor[0].numpy()   # (D, R, A)
    bev      = power_np.max(axis=0)      # (R, A)

    # ---- CFAR in metres (LiDAR frame: X=lateral, Y=depth) -------------------
    cfar_x = pts_lidar[:, 0]
    cfar_y = pts_lidar[:, 1]

    # ---- Figure: left (spans 2 rows) + 2 stacked right ----------------------
    fig = plt.figure(figsize=(22, 9), facecolor='white')
    gs  = fig.add_gridspec(2, 2, width_ratios=[3, 2], hspace=0.42, wspace=0.28)

    # Left: camera
    ax_cam = fig.add_subplot(gs[:, 0])
    ax_cam.imshow(img_rgb)
    sm = ScalarMappable(cmap='turbo', norm=norm_s)
    sm.set_array([])
    plt.colorbar(sm, ax=ax_cam, fraction=0.025, pad=0.02).set_label('CFAR Score')
    ax_cam.set_title(f"Camera  |  {rc_name}  Frame {frame_idx:03d}", fontsize=10)
    ax_cam.axis('off')

    # Top-right: radar power BEV — title is the radar frame filename
    ax_pwr = fig.add_subplot(gs[0, 1])
    ax_pwr.imshow(bev, cmap='turbo', origin='lower', aspect='auto',
                  extent=[0, bev.shape[1], 0, bev.shape[0]])
    ax_pwr.set_title(f"Radar Frame: {radar_fname}", fontsize=9)
    ax_pwr.set_xlabel('Azimuth (bins)', fontsize=8)
    ax_pwr.set_ylabel('Range (bins)',   fontsize=8)

    # Bottom-right: CFAR in metres — title is the CFAR filename
    ax_m = fig.add_subplot(gs[1, 1])

    y_max = max(float(cfar_y.max()) + 5.0, 25.0) if len(cfar_y) else 30.0
    x_lo  = float(cfar_x.min()) if len(cfar_x) else -20.0
    for r_ring in np.arange(5, y_max + 1, 5):
        ax_m.axhline(r_ring, color='grey', linewidth=0.5, linestyle='--', alpha=0.4)
        ax_m.text(x_lo - 0.5, r_ring + 0.3, f'{r_ring:.0f} m',
                  fontsize=6, color='grey', va='bottom')

    sc_m = ax_m.scatter(cfar_x, cfar_y, c=s_arr, cmap='turbo', norm=norm_s, s=18, zorder=3)
    plt.colorbar(sc_m, ax=ax_m, fraction=0.046, pad=0.04).set_label('CFAR Score')

    if corners is not None:
        bv = _box_bev_corners(corners)
        if bv is not None:
            poly = MplPolygon(bv, closed=True, edgecolor='red',
                              facecolor='none', linewidth=2, zorder=4)
            ax_m.add_patch(poly)
            ax_m.annotate('GT box', (bv[:, 0].mean(), bv[:, 1].mean()),
                          color='red', fontsize=7, ha='center', va='bottom')

    ax_m.set_xlabel('Lateral (m)',       fontsize=9)
    ax_m.set_ylabel('Range / Depth (m)', fontsize=9)
    ax_m.set_title(f"CFAR: {cfar_fname}", fontsize=9)
    ax_m.grid(True, linestyle=':', alpha=0.3)
    if len(cfar_x):
        pad_x = max((float(cfar_x.max()) - x_lo) * 0.15, 2.0)
        ax_m.set_xlim(x_lo - pad_x, float(cfar_x.max()) + pad_x)
        ax_m.set_ylim(max(float(cfar_y.min()) - 2.0, 0.0), y_max)

    fig.suptitle(f"{rc_name}  |  Frame {frame_idx:03d}", fontsize=11)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return True


def run(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    base_dir = config.get('base_dir', '')
    out_base = os.path.join(
        config.get('out_dir', 'verification_output/cfar_cam'),
        'cfar_camera',
    )
    sf = config.get('subfolders', {})

    splits     = config.get('eval_splits', {})
    rc_folders = (splits.get('clear', []) +
                  splits.get('fog',   []) +
                  splits.get('rain',  []))

    for rc_name in rc_folders:
        rc_dir    = os.path.join(base_dir, rc_name) if base_dir else rc_name
        calib_dir = os.path.join(rc_dir, sf.get('calib', 'calib'))
        cfar_dir  = os.path.join(rc_dir, sf.get('cfar',  'cfar'))
        pco_dir   = os.path.join(rc_dir, sf.get('pco',   'pco'))

        try:
            ds = RadarDataset(rc_dir, augment=False,
                              config=_build_ds_config(config, rc_dir))
        except Exception as e:
            print(f"  [SKIP] {rc_name}: {e}")
            continue
        if len(ds) == 0:
            print(f"  [SKIP] {rc_name}: 0 frames in dataset")
            continue

        calib_files = sorted(glob.glob(os.path.join(calib_dir, '*.txt')))
        if not calib_files:
            print(f"  [SKIP] {rc_name}: no calib files in {calib_dir}")
            continue
        calib_index = _build_calib_index(calib_files)

        # Pre-index CFAR files once per RC for fast filename lookup
        cfar_files, cfar_ts_arr = _build_cfar_index(cfar_dir)

        rc_out = os.path.join(out_base, rc_name.replace('/', os.sep))
        os.makedirs(rc_out, exist_ok=True)

        saved = skipped = 0

        for idx in tqdm(range(len(ds)), desc=rc_name):
            sample     = ds.matched_data[idx]
            ts_ms      = _extract_ts_ms(sample['power'])
            radar_fname = os.path.basename(sample['power'])  # e.g. 1649675876.820941.npy

            # Step 1: radar ts -> calib Radar_frame (<=100 ms)
            corners, r2l, R_r2l, _gap, radar_frame_ts = \
                _find_calib_by_radar_frame(calib_index, ts_ms, threshold_ms=100)
            if radar_frame_ts is None:
                skipped += 1
                continue

            # Step 2: Radar_frame ts -> CFAR (<=5 ms)
            cfar_fname = _find_cfar_fname(cfar_files, cfar_ts_arr, radar_frame_ts, threshold_ms=5)
            pts_c, scores_c = _load_cfar(cfar_dir, radar_frame_ts, threshold_ms=5)
            if pts_c is None or len(pts_c) == 0:
                skipped += 1
                continue
            pts_c = _cfar_to_lidar(pts_c, R_r2l, r2l)

            # Step 3: matched calib -> camera image
            rf_ts_arr  = calib_index['radar_frame_ts']
            valid_mask = ~np.isnan(rf_ts_arr)
            diffs      = np.abs(rf_ts_arr[valid_mask] - radar_frame_ts)
            best_i     = int(np.argmin(diffs))
            valid_idx  = np.where(valid_mask)[0]
            matched_cf = calib_files[valid_idx[best_i]]

            img_name, K, r_t, _ = parse_calibration(matched_cf)
            if not img_name:
                skipped += 1
                continue
            img_path = os.path.join(pco_dir, img_name)
            if not os.path.exists(img_path):
                skipped += 1
                continue

            # Step 4: load radar tensor for BEV panel
            radar_tensor, _ = ds[idx]

            # Output filename uses radar frame name (without .npy) for easy lookup
            ts_str   = radar_fname.replace('.npy', '')
            out_path = os.path.join(rc_out, f'frame_{idx:04d}_{ts_str}.png')

            ok = _save_combined(
                pts_lidar    = pts_c,
                scores       = scores_c,
                corners      = corners,
                K            = K,
                r_t          = r_t,
                radar_tensor = radar_tensor,
                img_path     = img_path,
                rc_name      = rc_name,
                radar_fname  = radar_fname,
                cfar_fname   = cfar_fname,
                frame_idx    = idx,
                out_path     = out_path,
            )
            if ok:
                saved += 1
            else:
                skipped += 1

        print(f"  {rc_name}: {saved} saved, {skipped} skipped  ->  {rc_out}")

    print(f"\nDone. Output root: {os.path.abspath(out_base)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True,
                        help='YAML config (same as thesis_eval)')
    args = parser.parse_args()
    run(args.config)
