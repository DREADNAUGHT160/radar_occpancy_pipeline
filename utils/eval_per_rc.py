"""
Per-RC-folder side evaluation -- no camera projection, no pooling across RC folders.

Computes AP/P_d/P_fa per RC folder (not blended across folders in the same
weather category), plus a pooled summary row per weather category for
comparison against thesis_eval.py's own weather_results.csv.

Also explicitly reports n_zero_point_frames per RC/band/method -- matched
frames where the method produced zero points above the eval threshold still
count in the frame totals (as they already do for P_d), and their count is
surfaced directly instead of silently vanishing into a lower AP number.

RC folders are processed strictly one at a time; GPU memory is released
between folders, and a CUDA OOM on any single frame falls back to CPU for
that frame only rather than aborting the run.

Usage:
  python utils/eval_per_rc.py --config configs/eval_datamasters_dry.yaml
"""
import os
import sys
import glob
import csv
import argparse
import yaml
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataset.dataloader import RadarDataset
from utils.thesis_eval import (
    _load_model, _build_ds_config, _extract_ts_ms, _build_calib_index,
    _find_calib_by_radar_frame, _load_cfar, _cfar_to_lidar, _box_stats,
    _range_band, occupancy_to_points, points_in_box, compute_ap, compute_pd_pfa,
)


def _predict(model, radar_tensor, device):
    """Run inference with GPU OOM fallback to CPU for this one frame."""
    try:
        with torch.no_grad():
            return torch.sigmoid(model(radar_tensor.unsqueeze(0).to(device)))[0].cpu().numpy()
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            torch.cuda.empty_cache()
            with torch.no_grad():
                return torch.sigmoid(model(radar_tensor.unsqueeze(0).cpu()))[0].numpy()
        raise


def collect_rc_frames(rc_name, rc_dir, config, model, device, threshold, weather):
    """Collect DL + CFAR-matched frame dicts for a single RC folder only."""
    sf = config.get('subfolders', {})
    calib_dir = os.path.join(rc_dir, sf.get('calib', 'calib'))
    cfar_dir  = os.path.join(rc_dir, sf.get('cfar', 'cfar'))

    calib_files = sorted(glob.glob(os.path.join(calib_dir, '*.txt')))
    calib_index = _build_calib_index(calib_files)

    try:
        ds = RadarDataset(rc_dir, augment=False, config=_build_ds_config(config, rc_dir))
    except Exception as e:
        print(f"  [SKIP] {rc_name}: {e}")
        return [], []

    dl_frames, cfar_frames = [], []

    for idx in tqdm(range(len(ds)), desc=rc_name):
        radar_tensor, label_tensor = ds[idx]
        sample = ds.matched_data[idx]
        ts_ms  = _extract_ts_ms(sample['power'])

        corners, r2l, R_r2l, gap, radar_frame_ts = _find_calib_by_radar_frame(calib_index, ts_ms)
        box_volume, box_range = _box_stats(corners) if corners is not None else (0.0, np.nan)

        pred_np = _predict(model, radar_tensor, device)

        pts_dl, scores_dl = occupancy_to_points(pred_np, threshold=0.05)
        pts_dl = pts_dl + r2l

        if corners is not None and len(pts_dl) > 0:
            in_box_dl     = points_in_box(pts_dl, corners)
            mask_thresh   = scores_dl >= threshold
            pts_thresh    = pts_dl[mask_thresh]
            ib_thresh     = points_in_box(pts_thresh, corners) if len(pts_thresh) else np.zeros(0, dtype=bool)
            pts_in_thresh = pts_thresh[ib_thresh] if len(ib_thresh) else np.empty((0, 3), dtype=np.float32)
        else:
            in_box_dl     = np.zeros(len(pts_dl), dtype=bool)
            pts_thresh    = np.empty((0, 3), dtype=np.float32)
            pts_in_thresh = np.empty((0, 3), dtype=np.float32)

        lidar_pts_in_box = None
        if weather == 'clear':
            lbl_pts, _ = occupancy_to_points(label_tensor.numpy(), threshold=0.5)
            lbl_pts    = lbl_pts + r2l
            if corners is not None and len(lbl_pts) > 0:
                lidar_pts_in_box = lbl_pts[points_in_box(lbl_pts, corners)]

        dl_frames.append({
            'scores':            scores_dl,
            'in_box':            in_box_dl,
            'pts_in_box_thresh': pts_in_thresh,
            'box_corners':       corners,
            'lidar_pts':         lidar_pts_in_box,
            'box_volume':        box_volume,
            'box_range':         box_range,
            'zero_points':       len(pts_thresh) == 0,   # matched frame, but 0 pts above eval threshold
            'dl_ts_ms':          ts_ms,
            'n_pts_thresh':      len(pts_thresh),
            'n_pts_in_box':      len(pts_in_thresh),
        })

        if radar_frame_ts is not None:
            pts_c, scores_c = _load_cfar(cfar_dir, radar_frame_ts, threshold_ms=5)
        else:
            pts_c, scores_c = _load_cfar(cfar_dir, ts_ms)
        cfar_zero = pts_c is None or len(pts_c) == 0
        n_cfar_pts = 0 if cfar_zero else len(pts_c)
        n_cfar_in_box = 0
        cfar_in_box_arr = np.zeros(0, dtype=bool)
        if not cfar_zero and corners is not None:
            pts_c = _cfar_to_lidar(pts_c, R_r2l, r2l)
            cfar_in_box_arr = points_in_box(pts_c, corners)
            n_cfar_in_box = int(cfar_in_box_arr.sum())
        cfar_frames.append({
            'pts':         pts_c,
            'scores':      scores_c,
            'in_box':      cfar_in_box_arr,
            'box_corners': corners,
            'lidar_pts':   lidar_pts_in_box,
            'box_volume':  box_volume,
            'box_range':   box_range,
            'zero_points': cfar_zero,
            'dl_ts_ms':    ts_ms,
            'n_pts_thresh': n_cfar_pts,
            'n_pts_in_box': n_cfar_in_box,
        })

    del ds
    torch.cuda.empty_cache()
    return dl_frames, cfar_frames


def _frame_ap(scores, in_box):
    """Step-function AP (same formula as compute_ap) scoped to one frame's own points."""
    if scores is None or len(scores) == 0 or in_box is None or len(in_box) == 0:
        return float('nan')
    if not in_box.any():
        return 0.0
    order  = np.argsort(-np.asarray(scores))
    ib     = np.asarray(in_box)[order]
    tp_cum = np.cumsum(ib)
    fp_cum = np.cumsum(~ib)
    prec   = tp_cum / (tp_cum + fp_cum + 1e-8)
    rec    = tp_cum / ib.sum()
    rec_prev = np.concatenate(([0.0], rec[:-1]))
    return float(np.sum((rec - rec_prev) * prec))


def _per_frame_rows(weather, rc_name, method, frames):
    """One row per individual matched frame -- full metrics, no aggregation."""
    rows = []
    for fd in frames:
        if fd['box_corners'] is None:
            continue
        n_thresh = fd['n_pts_thresh']
        n_in_box = fd['n_pts_in_box']
        p_fa = (n_thresh - n_in_box) / n_thresh if n_thresh > 0 else float('nan')
        density = n_in_box / (fd['box_volume'] + 1e-9) if fd['box_volume'] else float('nan')
        ap = _frame_ap(fd.get('scores'), fd.get('in_box'))
        rows.append({
            'weather':       weather,
            'rc':            rc_name,
            'method':        method,
            'dl_ts_ms':      fd['dl_ts_ms'],
            'band':          _range_band(fd['box_range']),
            'box_range_m':   f"{fd['box_range']:.2f}",
            'n_pts_thresh':  n_thresh,
            'n_pts_in_box':  n_in_box,
            'P_d':           int(n_in_box > 0),
            'P_fa':          f'{p_fa:.4f}' if not np.isnan(p_fa) else 'N/A',
            'AP':            f'{ap:.4f}' if not np.isnan(ap) else 'N/A',
            'density':       f'{density:.4f}' if not np.isnan(density) else 'N/A',
        })
    return rows


def _zero_point_counts(frames):
    """Return {band: n_zero_point_frames} among frames that have a box."""
    counts = {}
    for fd in frames:
        if fd['box_corners'] is None:
            continue
        band = _range_band(fd['box_range'])
        if fd.get('zero_points'):
            counts[band] = counts.get(band, 0) + 1
    return counts


def _band_frame_counts(frames):
    counts = {}
    for fd in frames:
        if fd['box_corners'] is None:
            continue
        band = _range_band(fd['box_range'])
        counts[band] = counts.get(band, 0) + 1
    return counts


BANDS = ['0-5m', '5-10m', '10-15m', '15-20m']


def _rows_for(weather, rc_name, method, frames, threshold):
    """Build per-band + overall CSV rows for one (rc, method) pair."""
    rows = []
    zero_counts  = _zero_point_counts(frames)
    band_counts  = _band_frame_counts(frames)

    ap  = compute_ap(frames)
    pd, pfa = compute_pd_pfa(frames, threshold)
    n_total = sum(band_counts.values())
    n_zero_total = sum(zero_counts.values())
    rows.append({
        'weather': weather, 'rc': rc_name, 'method': method, 'band': 'ALL',
        'n_frames': n_total, 'n_zero_point_frames': n_zero_total,
        'AP': f'{ap:.4f}', 'P_d': f'{pd:.4f}', 'P_fa': f'{pfa:.4f}',
    })

    for band in BANDS:
        band_frames = [fd for fd in frames
                       if fd['box_corners'] is not None and _range_band(fd['box_range']) == band]
        if not band_frames:
            continue
        b_ap = compute_ap(band_frames)
        b_pd, b_pfa = compute_pd_pfa(band_frames, threshold)
        rows.append({
            'weather': weather, 'rc': rc_name, 'method': method, 'band': band,
            'n_frames': band_counts.get(band, 0),
            'n_zero_point_frames': zero_counts.get(band, 0),
            'AP': f'{b_ap:.4f}', 'P_d': f'{b_pd:.4f}', 'P_fa': f'{b_pfa:.4f}',
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/eval_config.yaml')
    parser.add_argument('--checkpoint', default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    ckpt = args.checkpoint or config.get('checkpoint', '')
    base_dir = config.get('base_dir', '')
    out_dir  = config.get('out_dir', 'verification_output/eval_per_rc')
    threshold = float(config.get('weather', {}).get('threshold', 0.4))
    eval_splits = config.get('eval_splits', {})

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading checkpoint: {ckpt}")
    print(f"Device: {device}")
    model = _load_model(config, ckpt, device)

    all_rows = []
    per_frame_rows = []

    for weather, rc_folders in eval_splits.items():
        if not rc_folders:
            continue
        print(f"\n{'='*60}\n  {weather.upper()} -- {rc_folders}\n{'='*60}")

        pooled_dl, pooled_cfar = [], []

        for rc_name in rc_folders:
            rc_dir = os.path.join(base_dir, rc_name) if base_dir else rc_name
            try:
                dl_frames, cfar_frames = collect_rc_frames(
                    rc_name, rc_dir, config, model, device, threshold, weather)
            except Exception as e:
                print(f"  [ERROR] {rc_name} failed, skipping: {e}")
                torch.cuda.empty_cache()
                continue

            if not dl_frames:
                continue

            all_rows.extend(_rows_for(weather, rc_name, 'DL',   dl_frames,   threshold))
            all_rows.extend(_rows_for(weather, rc_name, 'CFAR', cfar_frames, threshold))
            per_frame_rows.extend(_per_frame_rows(weather, rc_name, 'DL',   dl_frames))
            per_frame_rows.extend(_per_frame_rows(weather, rc_name, 'CFAR', cfar_frames))
            pooled_dl.extend(dl_frames)
            pooled_cfar.extend(cfar_frames)

        if pooled_dl:
            all_rows.extend(_rows_for(weather, 'ALL_POOLED', 'DL',   pooled_dl,   threshold))
            all_rows.extend(_rows_for(weather, 'ALL_POOLED', 'CFAR', pooled_cfar, threshold))

    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'per_rc_results.csv')
    fields = ['weather', 'rc', 'method', 'band', 'n_frames', 'n_zero_point_frames', 'AP', 'P_d', 'P_fa']
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nResults saved -> {os.path.abspath(out_csv)}")

    frame_csv = os.path.join(out_dir, 'per_frame_results.csv')
    frame_fields = ['weather', 'rc', 'method', 'dl_ts_ms', 'band', 'box_range_m',
                     'n_pts_thresh', 'n_pts_in_box', 'P_d', 'P_fa', 'AP', 'density']
    with open(frame_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=frame_fields)
        writer.writeheader()
        writer.writerows(per_frame_rows)
    print(f"Per-frame audit trail ({len(per_frame_rows)} rows) -> {os.path.abspath(frame_csv)}")


if __name__ == '__main__':
    main()
