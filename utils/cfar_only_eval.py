"""
CFAR-only evaluation for RC019 (no DL model needed).

Usage:
  conda activate thesis_model
  cd D:\thesis data\radar_occpancy_pipeline
  python utils/cfar_only_eval.py
"""
import os, sys, re, glob
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.thesis_eval import (
    compute_weather_metrics, _extract_ts_ms, _find_calib,
    _cfar_to_lidar, occupancy_to_points, points_in_box
)
from utils.project_to_image import parse_calibration

DATASET_DIR = r'D:\dataset\RC019'
WEATHER     = 'clear'
CFAR_THRESH = 0.5


def load_cfar(cfar_dir, ts_ms, R_r2l, t_r2l, thr_ms=200):
    files = sorted(glob.glob(os.path.join(cfar_dir, '*.npy')) +
                   glob.glob(os.path.join(cfar_dir, '*.txt')))
    if not files:
        return None, None
    ts_arr = np.array([_extract_ts_ms(f) for f in files])
    best   = int(np.argmin(np.abs(ts_arr - ts_ms)))
    if abs(ts_arr[best] - ts_ms) > thr_ms:
        return None, None
    f = files[best]
    try:
        data = np.load(f) if f.endswith('.npy') else np.loadtxt(f, delimiter=',')
    except Exception:
        return None, None
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if len(data) == 0:
        return None, None
    # Apply Doppler filter: keep only approaching targets (velocity < -1.8 m/s).
    # Column 3 is Doppler velocity, NOT a detection score, so we must filter here
    # rather than relying on a score threshold in compute_pd_pfa.
    DOPPLER_THR = -1.8
    if data.shape[1] > 3:
        data = data[data[:, 3] < DOPPLER_THR]
    if len(data) == 0:
        return None, None
    pts    = data[:, :3].astype(np.float32)
    scores = np.ones(len(pts), dtype=np.float32)   # binary: passed Doppler gate
    pts    = _cfar_to_lidar(pts, R_r2l, t_r2l)
    return pts, scores


def main():
    calib_dir = os.path.join(DATASET_DIR, 'calib')
    cfar_dir  = os.path.join(DATASET_DIR, 'cfar')
    label_dir = os.path.join(DATASET_DIR, 'labels')

    calib_files = sorted(glob.glob(os.path.join(calib_dir, '*.txt')))

    # Index label files by timestamp for optional Chamfer distance
    label_files = sorted(glob.glob(os.path.join(label_dir, '*.npy')))
    label_ts    = np.array([_extract_ts_ms(f) for f in label_files]) if label_files else np.array([])

    print(f'Calib files : {len(calib_files)}')
    print(f'Label files : {len(label_files)}')

    cfar_frames = []
    no_cfar     = 0

    for cf in calib_files:
        ts_ms = _extract_ts_ms(cf)

        # Parse corners and transforms directly from this calib file
        img_name, K, r_t, r2l = parse_calibration(cf)
        with open(cf) as fh:
            content = fh.read()
        import re as _re
        m = _re.search(r'"BoundingBox":([\d\s.-]+),', content)
        if not m:
            continue
        corners    = np.array(m.group(1).split(), dtype=float).reshape(-1, 3)
        R_r2l      = np.eye(3)
        m_rot = _re.search(r'"Rotation_Radar_to_Lidar":\s*([-\d\s.e+]+),', content)
        if m_rot:
            vals = np.array(m_rot.group(1).strip().split(), dtype=float)
            if len(vals) == 9:
                R_r2l = vals.reshape(3, 3)
        dims       = corners.max(0) - corners.min(0)
        box_volume = float(dims[0] * dims[1] * dims[2])
        center     = (corners.max(0) + corners.min(0)) / 2
        box_range  = float(np.sqrt(center[0]**2 + center[1]**2))

        # LiDAR GT points (Chamfer) — only if a matching label exists within 100ms
        lidar_pts_in_box = None
        if WEATHER == 'clear' and len(label_ts) > 0:
            diffs = np.abs(label_ts - ts_ms)
            best  = int(np.argmin(diffs))
            if diffs[best] <= 100:
                lbl     = np.load(label_files[best])
                lbl_pts, _ = occupancy_to_points(lbl, threshold=0.5)
                lbl_pts    = lbl_pts + r2l
                if len(lbl_pts) > 0:
                    lidar_pts_in_box = lbl_pts[points_in_box(lbl_pts, corners)]

        pts, scores = load_cfar(cfar_dir, ts_ms, R_r2l, r2l)
        if pts is None:
            no_cfar += 1
            continue

        cfar_frames.append({
            'pts':         pts,
            'scores':      scores,
            'box_corners': corners,
            'lidar_pts':   lidar_pts_in_box,
            'box_volume':  box_volume,
            'box_range':   box_range,
        })

    print(f'Frames used : {len(cfar_frames)}  skipped (no CFAR match): {no_cfar}\n')

    if not cfar_frames:
        print('No frames to evaluate.')
        return

    m = compute_weather_metrics(cfar_frames, CFAR_THRESH, WEATHER)

    print('=' * 50)
    print('CFAR-ONLY RESULTS  (RC019)')
    print('=' * 50)
    cd_s = f"{m['CD']:.4f}" if not np.isnan(m['CD']) else 'N/A'
    print(f"  AP   : {m['AP']:.4f}")
    print(f"  P_d  : {m['P_d']:.4f}")
    print(f"  P_fa : {m['P_fa']:.4f}")
    print(f"  CD   : {cd_s}")

    print('\nDensity (pts/m³ inside GT box):')
    for band, val in m['density'].items():
        v = f'{val:.4f}' if not np.isnan(val) else 'N/A'
        print(f'  {band}: {v}')

    print('\nRange-band breakdown:')
    print(f"  {'Band':<10} {'n':>5} {'AP':>7} {'P_d':>7} {'P_fa':>7}")
    print('  ' + '-' * 40)
    for band in ['0-5m', '5-10m', '10-15m', '15-20m']:
        bm = m['range_metrics'].get(band)
        if bm:
            print(f"  {band:<10} {bm['n']:>5} {bm['AP']:>7.4f} {bm['P_d']:>7.4f} {bm['P_fa']:>7.4f}")
        else:
            print(f"  {band:<10}   N/A")


if __name__ == '__main__':
    main()
