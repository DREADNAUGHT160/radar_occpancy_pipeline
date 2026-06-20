"""
Thesis evaluation: CFAR vs DL model across weather conditions.

Implements the four experiments from the thesis evaluation plan:
  Exp 1 — Clear weather accuracy  (AP, P_d, P_fa, Chamfer Distance)
  Exp 2 — Weather robustness      (AP, P_d, P_fa in fog and rain)
  Exp 3 — Degradation analysis    ((clear - weather) / clear × 100)
  Exp 4 — Point density per distance band inside GT box

Usage:
  python utils/thesis_eval.py \
    --config   configs/train_config.yaml \
    --checkpoint checkpoints/<run_id>/best_model.pth \
    --threshold 0.4
"""
import os
import sys
import re
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

from models.factory import ModelFactory
from dataset.dataloader import RadarDataset
from utils.project_to_image import occupancy_to_points, parse_calibration


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _extract_ts_ms(path):
    match = re.search(r'(\d+\.\d+|\d+)', os.path.basename(path))
    if match:
        val = float(match.group(0))
        return int(val * 1000) if val < 1e11 else int(val)
    return 0


def load_gt_box(txt_path):
    """Parse BoundingBox corners from calib .txt → (N,3) array or None."""
    with open(txt_path) as f:
        content = f.read()
    match = re.search(r'"BoundingBox":([\d\s.-]+),', content)
    if not match:
        return None
    corners = np.array(match.group(1).split(), dtype=float).reshape(-1, 3)
    return corners


def points_in_box(pts, corners):
    """Boolean mask: which points (N,3) lie inside the AABB of box corners."""
    mn = corners.min(axis=0)
    mx = corners.max(axis=0)
    return np.all((pts >= mn) & (pts <= mx), axis=1)


def chamfer_distance(pts_a, pts_b):
    """Symmetric Chamfer distance between two point sets (mean nearest-neighbour)."""
    if len(pts_a) == 0 or len(pts_b) == 0:
        return np.nan
    diff = np.linalg.norm(pts_a[:, None, :] - pts_b[None, :, :], axis=-1)
    return (diff.min(axis=1).mean() + diff.min(axis=0).mean()) / 2.0


def point_density_by_band(pts_in_box, box_volume, bands=((0, 10), (10, 15), (15, 20))):
    """Points per m³ inside GT box broken down by horizontal range band."""
    r = np.sqrt(pts_in_box[:, 0] ** 2 + pts_in_box[:, 1] ** 2) if len(pts_in_box) > 0 else np.array([])
    result = {}
    for d_min, d_max in bands:
        n = int(((r >= d_min) & (r < d_max)).sum()) if len(r) > 0 else 0
        result[f"{d_min}-{d_max}m"] = n / (box_volume + 1e-9)
    return result


# ── Metric computation ────────────────────────────────────────────────────────

def compute_ap(frames_data):
    """
    AP via precision-recall curve (point-in-box criterion).

    frames_data: list of dicts with 'pts' (N,3), 'scores' (N,), 'box_corners' (M,3).
    A point is TP if it lies inside the GT box, FP otherwise.
    """
    all_scores, all_in_box = [], []

    for fd in frames_data:
        pts     = fd['pts']
        scores  = fd['scores']
        corners = fd['box_corners']
        if pts is None or len(pts) == 0 or corners is None:
            continue
        if scores is None:
            scores = np.ones(len(pts))
        in_box = points_in_box(pts, corners)
        all_scores.extend(scores.tolist())
        all_in_box.extend(in_box.tolist())

    if not all_scores or sum(all_in_box) == 0:
        return 0.0, np.array([]), np.array([])

    scores_arr  = np.array(all_scores)
    in_box_arr  = np.array(all_in_box, dtype=bool)
    order       = np.argsort(-scores_arr)
    in_box_arr  = in_box_arr[order]

    tp_cum  = np.cumsum(in_box_arr)
    fp_cum  = np.cumsum(~in_box_arr)
    total_p = in_box_arr.sum()

    precision = tp_cum / (tp_cum + fp_cum + 1e-8)
    recall    = tp_cum / total_p

    ap = float(np.trapz(precision, recall))
    return ap, recall, precision


def compute_pd_pfa(frames_data, threshold):
    """
    P_d: fraction of frames where ≥1 point above threshold is inside GT box.
    P_fa: points above threshold outside GT box / total points above threshold.
    """
    n_frames = n_detected = total = outside = 0

    for fd in frames_data:
        pts     = fd['pts']
        scores  = fd['scores']
        corners = fd['box_corners']
        if pts is None or corners is None:
            continue
        n_frames += 1

        if scores is not None:
            mask = scores >= threshold
            pts_t = pts[mask]
        else:
            pts_t = pts

        if len(pts_t) == 0:
            continue

        in_box = points_in_box(pts_t, corners)
        total   += len(pts_t)
        outside += int((~in_box).sum())
        if in_box.any():
            n_detected += 1

    p_d  = n_detected / n_frames if n_frames > 0 else 0.0
    p_fa = outside / total        if total    > 0 else 0.0
    return p_d, p_fa


# ── CFAR loader ───────────────────────────────────────────────────────────────

def load_cfar_for_ts(cfar_rc_dir, ts_ms, threshold_ms=200):
    """
    Load CFAR point cloud nearest to ts_ms from cfar_rc_dir.

    Expected formats:
      .npy — (N,3) XYZ  or  (N,4) XYZ+confidence
      .txt — space-separated rows, first 3 cols = XYZ, optional 4th = confidence
    """
    files = (sorted(glob.glob(os.path.join(cfar_rc_dir, '*.npy'))) +
             sorted(glob.glob(os.path.join(cfar_rc_dir, '*.txt'))))
    if not files:
        return None, None

    ts_arr = np.array([_extract_ts_ms(f) for f in files])
    diffs  = np.abs(ts_arr - ts_ms)
    best   = int(np.argmin(diffs))
    if diffs[best] > threshold_ms:
        return None, None

    f = files[best]
    try:
        data = np.load(f) if f.endswith('.npy') else np.loadtxt(f)
    except Exception:
        return None, None

    if data.ndim == 1:
        data = data.reshape(1, -1)
    if len(data) == 0:
        return None, None

    pts    = data[:, :3].astype(np.float32)
    scores = data[:, 3].astype(np.float32) if data.shape[1] > 3 else None
    return pts, scores


# ── Per-weather data collection ───────────────────────────────────────────────

def collect_frames(rc_folders, base_dir, config, model, device, threshold, cfar_dir, weather):
    """Run inference + load CFAR for all frames in a list of RC folders."""
    sf      = config['dataset'].get('subfolders', {})
    dl_frames, cfar_frames = [], []

    for rc_name in rc_folders:
        rc_dir    = os.path.join(base_dir, rc_name) if base_dir else rc_name
        calib_dir = os.path.join(rc_dir, sf.get('calib', 'calib'))
        lbl_dir   = os.path.join(rc_dir, sf.get('labels', 'labels'))

        ds_cfg = {**config, 'dataset': {**config['dataset'],
                   'radar_dir': rc_dir, 'lidar_path': lbl_dir}}
        try:
            ds = RadarDataset(rc_dir, augment=False, config=ds_cfg)
        except Exception as e:
            print(f"  Skipping {rc_name}: {e}")
            continue

        txt_files = sorted(glob.glob(os.path.join(calib_dir, '*.txt')))
        txt_ts    = np.array([_extract_ts_ms(f) for f in txt_files]) if txt_files else np.array([])

        cfar_rc_dir = os.path.join(cfar_dir, rc_name) if cfar_dir else ''

        for idx in tqdm(range(len(ds)), desc=rc_name):
            radar_tensor, label_tensor = ds[idx]
            sample = ds.matched_data[idx]
            ts_ms  = _extract_ts_ms(sample['power'])

            # ── Calib / GT box ────────────────────────────────────────────────
            box_corners    = None
            radar_to_lidar = np.zeros(3)
            box_volume     = 0.0

            if len(txt_ts) > 0:
                diffs = np.abs(txt_ts - ts_ms)
                best  = int(np.argmin(diffs))
                if diffs[best] <= 100:
                    txt_path = txt_files[best]
                    box_corners = load_gt_box(txt_path)
                    _, _, _, radar_to_lidar = parse_calibration(txt_path)
                    if box_corners is not None:
                        dims       = box_corners.max(axis=0) - box_corners.min(axis=0)
                        box_volume = float(dims[0] * dims[1] * dims[2])

            # ── DL inference ──────────────────────────────────────────────────
            with torch.no_grad():
                pred_np = torch.sigmoid(
                    model(radar_tensor.unsqueeze(0).to(device))
                )[0].cpu().numpy()

            pts_dl, scores_dl = occupancy_to_points(pred_np, threshold=0.0)
            pts_dl = pts_dl + radar_to_lidar

            # ── LiDAR GT points (clear weather only, from label grid) ─────────
            lidar_pts_in_box = None
            if weather == 'clear':
                lbl_np = label_tensor.numpy()
                pts_lbl, _ = occupancy_to_points(lbl_np, threshold=0.5)
                pts_lbl = pts_lbl + radar_to_lidar
                if box_corners is not None and len(pts_lbl) > 0:
                    lidar_pts_in_box = pts_lbl[points_in_box(pts_lbl, box_corners)]

            dl_frames.append({
                'pts':        pts_dl,
                'scores':     scores_dl,
                'box_corners': box_corners,
                'lidar_pts':  lidar_pts_in_box,
                'box_volume': box_volume,
            })

            # ── CFAR ──────────────────────────────────────────────────────────
            if cfar_rc_dir:
                pts_c, scores_c = load_cfar_for_ts(cfar_rc_dir, ts_ms)
                cfar_frames.append({
                    'pts':        pts_c,
                    'scores':     scores_c,
                    'box_corners': box_corners,
                    'lidar_pts':  lidar_pts_in_box,
                    'box_volume': box_volume,
                })

    return dl_frames, cfar_frames


# ── Experiment runners ────────────────────────────────────────────────────────

def compute_all_metrics(frames, threshold, weather):
    """AP, P_d, P_fa, Chamfer Distance, point density for a set of frames."""
    ap, _, _   = compute_ap(frames)
    p_d, p_fa  = compute_pd_pfa(frames, threshold)

    # Chamfer Distance (clear only)
    cd = np.nan
    if weather == 'clear':
        cd_vals = []
        for fd in frames:
            if (fd['pts'] is None or fd['box_corners'] is None
                    or fd['lidar_pts'] is None or len(fd['lidar_pts']) == 0):
                continue
            scores = fd['scores']
            mask   = scores >= threshold if scores is not None else np.ones(len(fd['pts']), dtype=bool)
            pts_in = fd['pts'][mask]
            if len(pts_in) == 0:
                continue
            pts_in = pts_in[points_in_box(pts_in, fd['box_corners'])]
            if len(pts_in) == 0:
                continue
            val = chamfer_distance(pts_in, fd['lidar_pts'])
            if not np.isnan(val):
                cd_vals.append(val)
        cd = float(np.mean(cd_vals)) if cd_vals else np.nan

    # Point density per distance band
    bands       = ((0, 10), (10, 15), (15, 20))
    density_acc = {f"{a}-{b}m": [] for a, b in bands}
    for fd in frames:
        if fd['pts'] is None or fd['box_corners'] is None:
            continue
        scores = fd['scores']
        mask   = scores >= threshold if scores is not None else np.ones(len(fd['pts']), dtype=bool)
        pts_t  = fd['pts'][mask]
        if len(pts_t) == 0:
            continue
        pts_in = pts_t[points_in_box(pts_t, fd['box_corners'])]
        if len(pts_in) == 0:
            continue
        bv  = fd['box_volume']
        den = point_density_by_band(pts_in, bv, bands)
        for k, v in den.items():
            density_acc[k].append(v)

    density = {k: float(np.mean(v)) if v else np.nan for k, v in density_acc.items()}

    return {'AP': ap, 'P_d': p_d, 'P_fa': p_fa, 'CD': cd, 'density': density}


def degradation(clear_val, weather_val):
    if clear_val and clear_val > 0:
        return (clear_val - weather_val) / clear_val * 100
    return np.nan


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     required=True)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--threshold',  type=float, default=0.4)
    parser.add_argument('--out_dir',    default='verification_output/thesis_eval')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    eval_cfg = config.get('thesis_eval', {})
    weather_splits = eval_cfg.get('weather_splits', {})
    cfar_dir       = eval_cfg.get('cfar_dir', '')
    base_dir       = config['dataset'].get('base_dir', '')

    if not weather_splits:
        print("ERROR: thesis_eval.weather_splits not set in config.")
        print("Add this to your config:\n"
              "thesis_eval:\n"
              "  weather_splits:\n"
              "    clear: [RC_clear]\n"
              "    fog:   [RC_fog]\n"
              "    rain:  [RC_rain]\n"
              "  cfar_dir: ''")
        return

    ckpt = args.checkpoint or config.get('inference', {}).get('checkpoint', '')
    if not ckpt:
        print("ERROR: provide --checkpoint or set inference.checkpoint in config.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = ModelFactory.get_model(config).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    print(f"Model loaded from: {ckpt}")
    print(f"Device: {device}  |  Threshold: {args.threshold}")

    os.makedirs(args.out_dir, exist_ok=True)

    all_results = {}   # weather → {dl: metrics, cfar: metrics}

    for weather, rc_folders in weather_splits.items():
        print(f"\n{'='*60}")
        print(f"  {weather.upper()} — {rc_folders}")
        print('='*60)

        dl_frames, cfar_frames = collect_frames(
            rc_folders, base_dir, config, model, device,
            args.threshold, cfar_dir, weather)

        print(f"  Collected {len(dl_frames)} frames")

        dl_metrics = compute_all_metrics(dl_frames, args.threshold, weather)
        all_results[weather] = {'dl': dl_metrics}

        if cfar_frames and any(f['pts'] is not None for f in cfar_frames):
            cfar_metrics = compute_all_metrics(cfar_frames, 0.5, weather)
            all_results[weather]['cfar'] = cfar_metrics

    # ── Experiment 1 & 2: Results table ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESULTS TABLE")
    print(f"{'='*60}")
    print(f"{'Weather':<8} {'Method':<6} {'AP':>6} {'P_d':>6} {'P_fa':>6} {'CD':>8}")
    print('-' * 46)

    for weather in ['clear', 'fog', 'rain']:
        if weather not in all_results:
            continue
        for method in ['dl', 'cfar']:
            if method not in all_results[weather]:
                continue
            m = all_results[weather][method]
            cd_str = f"{m['CD']:8.4f}" if not np.isnan(m['CD']) else '     N/A'
            print(f"{weather:<8} {method.upper():<6} {m['AP']:6.3f} {m['P_d']:6.3f} {m['P_fa']:6.3f} {cd_str}")

    # ── Experiment 3: Degradation ─────────────────────────────────────────────
    if 'clear' in all_results:
        print(f"\n{'='*60}")
        print("DEGRADATION ANALYSIS  (clear → weather)")
        print(f"{'='*60}")
        print(f"{'Weather':<8} {'Method':<6} {'AP deg%':>8} {'P_d deg%':>9} {'P_fa deg%':>10}")
        print('-' * 45)

        for weather in ['fog', 'rain']:
            if weather not in all_results:
                continue
            for method in ['dl', 'cfar']:
                if method not in all_results.get('clear', {}) or \
                   method not in all_results.get(weather, {}):
                    continue
                c = all_results['clear'][method]
                w = all_results[weather][method]
                ap_deg  = degradation(c['AP'],  w['AP'])
                pd_deg  = degradation(c['P_d'], w['P_d'])
                pfa_deg = degradation(c['P_fa'], w['P_fa'])

                def fmt(v):
                    return f"{v:8.1f}%" if not np.isnan(v) else "     N/A"

                print(f"{weather:<8} {method.upper():<6} {fmt(ap_deg)} {fmt(pd_deg)} {fmt(pfa_deg)}")

    # ── Experiment 4: Point density ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("POINT DENSITY  (pts/m³ inside GT box)")
    print(f"{'='*60}")
    print(f"{'Weather':<8} {'Method':<6} {'0-10m':>8} {'10-15m':>8} {'15-20m':>8}")
    print('-' * 45)

    for weather in ['clear', 'fog', 'rain']:
        if weather not in all_results:
            continue
        for method in ['dl', 'cfar']:
            if method not in all_results[weather]:
                continue
            d = all_results[weather][method]['density']
            def fmt_d(v):
                return f"{v:8.3f}" if not np.isnan(v) else "     N/A"
            print(f"{weather:<8} {method.upper():<6} "
                  f"{fmt_d(d.get('0-10m', np.nan))} "
                  f"{fmt_d(d.get('10-15m', np.nan))} "
                  f"{fmt_d(d.get('15-20m', np.nan))}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_csv = os.path.join(args.out_dir, 'thesis_results.csv')
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Weather', 'Method', 'AP', 'P_d', 'P_fa', 'Chamfer_Distance',
                         'Density_0-10m', 'Density_10-15m', 'Density_15-20m'])
        for weather in ['clear', 'fog', 'rain']:
            if weather not in all_results:
                continue
            for method in ['dl', 'cfar']:
                if method not in all_results[weather]:
                    continue
                m = all_results[weather][method]
                d = m['density']
                writer.writerow([
                    weather, method.upper(),
                    f"{m['AP']:.4f}", f"{m['P_d']:.4f}", f"{m['P_fa']:.4f}",
                    f"{m['CD']:.4f}" if not np.isnan(m['CD']) else 'N/A',
                    f"{d.get('0-10m', np.nan):.4f}" if not np.isnan(d.get('0-10m', np.nan)) else 'N/A',
                    f"{d.get('10-15m', np.nan):.4f}" if not np.isnan(d.get('10-15m', np.nan)) else 'N/A',
                    f"{d.get('15-20m', np.nan):.4f}" if not np.isnan(d.get('15-20m', np.nan)) else 'N/A',
                ])

    print(f"\nResults saved → {os.path.abspath(out_csv)}")


if __name__ == '__main__':
    main()
