"""
Unified evaluation script — reads configs/eval_config.yaml.

eval_mode: basic
    Load model on clear-weather data → per-voxel IoU / Precision / Recall.
    Use this to verify the model is working before running thesis eval.

eval_mode: weather
    Full thesis evaluation across clear / fog / rain:
      Exp 1 — AP, P_d, P_fa, Chamfer Distance  (clear)
      Exp 2 — AP, P_d, P_fa                    (fog, rain)
      Exp 3 — Degradation %                    (clear → fog/rain)
      Exp 4 — Point density per distance band  (0-10m, 10-15m, 15-20m)
    Compares DL model vs CFAR baseline.

Usage:
  python utils/thesis_eval.py --config configs/eval_config.yaml
  python utils/thesis_eval.py --config configs/eval_config.yaml --checkpoint checkpoints/<run_id>/best_model.pth
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


# ── Shared helpers ────────────────────────────────────────────────────────────

def _extract_ts_ms(path):
    match = re.search(r'(\d+\.\d+|\d+)', os.path.basename(path))
    if match:
        val = float(match.group(0))
        return int(val * 1000) if val < 1e11 else int(val)
    return 0


def _build_ds_config(base_cfg, rc_dir):
    """Wrap eval_config into the shape RadarDataset expects."""
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


def _load_model(config, ckpt, device):
    model = ModelFactory.get_model(config).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    return model


def _find_calib(txt_files, txt_ts, ts_ms, threshold_ms=100):
    """Return (box_corners, radar_to_lidar) for the nearest calib file, or (None, zeros)."""
    if len(txt_ts) == 0:
        return None, np.zeros(3)
    diffs = np.abs(txt_ts - ts_ms)
    best  = int(np.argmin(diffs))
    if diffs[best] > threshold_ms:
        return None, np.zeros(3)
    txt_path = txt_files[best]
    with open(txt_path) as f:
        content = f.read()
    match = re.search(r'"BoundingBox":([\d\s.-]+),', content)
    corners = np.array(match.group(1).split(), dtype=float).reshape(-1, 3) if match else None
    _, _, _, r2l = parse_calibration(txt_path)
    return corners, r2l


# ── Geometry ──────────────────────────────────────────────────────────────────

def points_in_box(pts, corners):
    mn = corners.min(axis=0)
    mx = corners.max(axis=0)
    return np.all((pts >= mn) & (pts <= mx), axis=1)


def chamfer_distance(pts_a, pts_b):
    if len(pts_a) == 0 or len(pts_b) == 0:
        return np.nan
    diff = np.linalg.norm(pts_a[:, None, :] - pts_b[None, :, :], axis=-1)
    return (diff.min(axis=1).mean() + diff.min(axis=0).mean()) / 2.0


def point_density_by_band(pts_in_box, box_volume, bands=((0, 10), (10, 15), (15, 20))):
    r = np.sqrt(pts_in_box[:, 0] ** 2 + pts_in_box[:, 1] ** 2) if len(pts_in_box) > 0 else np.array([])
    result = {}
    for d_min, d_max in bands:
        n = int(((r >= d_min) & (r < d_max)).sum()) if len(r) > 0 else 0
        result[f"{d_min}-{d_max}m"] = n / (box_volume + 1e-9)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# BASIC MODE
# ─────────────────────────────────────────────────────────────────────────────

def run_basic(config, ckpt, out_dir):
    """Per-voxel IoU / Precision / Recall on a single clear-weather dataset."""
    basic_cfg  = config.get('basic', {})
    rc_name    = basic_cfg.get('dataset', '')
    threshold  = float(basic_cfg.get('threshold', 0.4))
    base_dir   = config.get('base_dir', '')

    if not rc_name:
        print("ERROR: set basic.dataset in eval_config.yaml (e.g. RC_clear)")
        return

    rc_dir = os.path.join(base_dir, rc_name) if base_dir else rc_name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = _load_model(config, ckpt, device)

    ds = RadarDataset(rc_dir, augment=False, config=_build_ds_config(config, rc_dir))
    print(f"\nBasic eval — {rc_name}  ({len(ds)} frames)  threshold={threshold}")

    tp_total = fp_total = fn_total = 0
    per_frame = []

    for idx in tqdm(range(len(ds)), desc="Evaluating"):
        radar_tensor, label_tensor = ds[idx]

        with torch.no_grad():
            pred_prob = torch.sigmoid(
                model(radar_tensor.unsqueeze(0).to(device))
            )[0].cpu()

        pred_bin  = (pred_prob  > threshold).float()
        label_bin = (label_tensor > 0.5).float()

        tp = (pred_bin * label_bin).sum().item()
        fp = (pred_bin * (1 - label_bin)).sum().item()
        fn = ((1 - pred_bin) * label_bin).sum().item()

        tp_total += tp
        fp_total += fp
        fn_total += fn

        iou   = tp / (tp + fp + fn + 1e-8)
        prec  = tp / (tp + fp + 1e-8)
        rec   = tp / (tp + fn + 1e-8)
        per_frame.append({'frame': idx, 'IoU': iou, 'Precision': prec, 'Recall': rec})

    iou_g  = tp_total / (tp_total + fp_total + fn_total + 1e-8)
    prec_g = tp_total / (tp_total + fp_total + 1e-8)
    rec_g  = tp_total / (tp_total + fn_total + 1e-8)

    iou_m  = float(np.mean([f['IoU']       for f in per_frame]))
    prec_m = float(np.mean([f['Precision'] for f in per_frame]))
    rec_m  = float(np.mean([f['Recall']    for f in per_frame]))

    print(f"\n{'─'*46}")
    print(f"{'Metric':<18} {'Global':>10} {'Per-frame mean':>14}")
    print(f"{'─'*46}")
    print(f"{'IoU':<18} {iou_g:10.4f} {iou_m:14.4f}")
    print(f"{'Precision':<18} {prec_g:10.4f} {prec_m:14.4f}")
    print(f"{'Recall':<18} {rec_g:10.4f} {rec_m:14.4f}")
    print(f"{'─'*46}")

    # Verdict
    if iou_g > 0.1:
        print("\n  Model is producing meaningful predictions.")
    elif iou_g > 0.01:
        print("\n  Model is detecting something — IoU is low, may need more training.")
    else:
        print("\n  WARNING: IoU near zero — model may not be learning or threshold is wrong.")

    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'basic_results.csv')
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame', 'IoU', 'Precision', 'Recall'])
        for fd in per_frame:
            writer.writerow([fd['frame'], f"{fd['IoU']:.4f}",
                             f"{fd['Precision']:.4f}", f"{fd['Recall']:.4f}"])
        writer.writerow([])
        writer.writerow(['GLOBAL', f"{iou_g:.4f}", f"{prec_g:.4f}", f"{rec_g:.4f}"])
        writer.writerow(['MEAN',   f"{iou_m:.4f}", f"{prec_m:.4f}", f"{rec_m:.4f}"])

    print(f"\n  Results saved → {os.path.abspath(out_csv)}")


# ─────────────────────────────────────────────────────────────────────────────
# WEATHER MODE — metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_cfar(cfar_rc_dir, ts_ms, threshold_ms=200):
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


def compute_ap(frames_data):
    all_scores, all_in_box = [], []
    for fd in frames_data:
        pts, scores, corners = fd['pts'], fd['scores'], fd['box_corners']
        if pts is None or len(pts) == 0 or corners is None:
            continue
        if scores is None:
            scores = np.ones(len(pts))
        in_box = points_in_box(pts, corners)
        all_scores.extend(scores.tolist())
        all_in_box.extend(in_box.tolist())

    if not all_scores or sum(all_in_box) == 0:
        return 0.0

    scores_arr = np.array(all_scores)
    in_box_arr = np.array(all_in_box, dtype=bool)
    order      = np.argsort(-scores_arr)
    in_box_arr = in_box_arr[order]

    tp_cum = np.cumsum(in_box_arr)
    fp_cum = np.cumsum(~in_box_arr)
    prec   = tp_cum / (tp_cum + fp_cum + 1e-8)
    rec    = tp_cum / in_box_arr.sum()

    return float(np.trapz(prec, rec))


def compute_pd_pfa(frames_data, threshold):
    n_frames = n_detected = total = outside = 0
    for fd in frames_data:
        pts, scores, corners = fd['pts'], fd['scores'], fd['box_corners']
        if pts is None or corners is None:
            continue
        n_frames += 1
        pts_t = pts[scores >= threshold] if scores is not None else pts
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


def compute_weather_metrics(frames, threshold, weather):
    ap        = compute_ap(frames)
    p_d, p_fa = compute_pd_pfa(frames, threshold)

    cd = np.nan
    if weather == 'clear':
        cd_vals = []
        for fd in frames:
            if (fd['pts'] is None or fd['box_corners'] is None
                    or fd['lidar_pts'] is None or len(fd['lidar_pts']) == 0):
                continue
            scores = fd['scores']
            mask   = (scores >= threshold) if scores is not None else np.ones(len(fd['pts']), dtype=bool)
            pts_in = fd['pts'][mask]
            pts_in = pts_in[points_in_box(pts_in, fd['box_corners'])] if len(pts_in) else pts_in
            if len(pts_in) == 0:
                continue
            val = chamfer_distance(pts_in, fd['lidar_pts'])
            if not np.isnan(val):
                cd_vals.append(val)
        cd = float(np.mean(cd_vals)) if cd_vals else np.nan

    bands       = ((0, 10), (10, 15), (15, 20))
    density_acc = {f"{a}-{b}m": [] for a, b in bands}
    for fd in frames:
        if fd['pts'] is None or fd['box_corners'] is None:
            continue
        scores = fd['scores']
        mask   = (scores >= threshold) if scores is not None else np.ones(len(fd['pts']), dtype=bool)
        pts_t  = fd['pts'][mask]
        if len(pts_t) == 0:
            continue
        pts_in = pts_t[points_in_box(pts_t, fd['box_corners'])]
        if len(pts_in) == 0:
            continue
        den = point_density_by_band(pts_in, fd['box_volume'], bands)
        for k, v in den.items():
            density_acc[k].append(v)

    density = {k: float(np.mean(v)) if v else np.nan for k, v in density_acc.items()}
    return {'AP': ap, 'P_d': p_d, 'P_fa': p_fa, 'CD': cd, 'density': density}


def collect_frames(rc_folders, base_dir, config, model, device, threshold, cfar_dir, weather):
    sf         = config.get('subfolders', {})
    dl_frames  = []
    cfar_frames = []

    for rc_name in rc_folders:
        rc_dir    = os.path.join(base_dir, rc_name) if base_dir else rc_name
        calib_dir = os.path.join(rc_dir, sf.get('calib', 'calib'))

        try:
            ds = RadarDataset(rc_dir, augment=False, config=_build_ds_config(config, rc_dir))
        except Exception as e:
            print(f"  Skipping {rc_name}: {e}")
            continue

        txt_files = sorted(glob.glob(os.path.join(calib_dir, '*.txt')))
        txt_ts    = np.array([_extract_ts_ms(f) for f in txt_files]) if txt_files else np.array([])
        cfar_rc   = os.path.join(cfar_dir, rc_name) if cfar_dir else ''

        for idx in tqdm(range(len(ds)), desc=rc_name):
            radar_tensor, label_tensor = ds[idx]
            sample = ds.matched_data[idx]
            ts_ms  = _extract_ts_ms(sample['power'])

            corners, r2l = _find_calib(txt_files, txt_ts, ts_ms)
            box_volume   = 0.0
            if corners is not None:
                dims       = corners.max(axis=0) - corners.min(axis=0)
                box_volume = float(dims[0] * dims[1] * dims[2])

            with torch.no_grad():
                pred_np = torch.sigmoid(
                    model(radar_tensor.unsqueeze(0).to(device))
                )[0].cpu().numpy()

            pts_dl, scores_dl = occupancy_to_points(pred_np, threshold=0.0)
            pts_dl = pts_dl + r2l

            lidar_pts_in_box = None
            if weather == 'clear':
                lbl_pts, _ = occupancy_to_points(label_tensor.numpy(), threshold=0.5)
                lbl_pts    = lbl_pts + r2l
                if corners is not None and len(lbl_pts) > 0:
                    lidar_pts_in_box = lbl_pts[points_in_box(lbl_pts, corners)]

            dl_frames.append({
                'pts': pts_dl, 'scores': scores_dl,
                'box_corners': corners, 'lidar_pts': lidar_pts_in_box,
                'box_volume': box_volume,
            })

            if cfar_rc:
                pts_c, scores_c = _load_cfar(cfar_rc, ts_ms)
                cfar_frames.append({
                    'pts': pts_c, 'scores': scores_c,
                    'box_corners': corners, 'lidar_pts': lidar_pts_in_box,
                    'box_volume': box_volume,
                })

    return dl_frames, cfar_frames


# ─────────────────────────────────────────────────────────────────────────────
# WEATHER MODE
# ─────────────────────────────────────────────────────────────────────────────

def run_weather(config, ckpt, out_dir):
    w_cfg          = config.get('weather', {})
    threshold      = float(w_cfg.get('threshold', 0.4))
    weather_splits = w_cfg.get('weather_splits', {})
    cfar_dir       = w_cfg.get('cfar_dir', '')
    base_dir       = config.get('base_dir', '')

    if not weather_splits or not any(weather_splits.values()):
        print("ERROR: set weather.weather_splits in eval_config.yaml")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = _load_model(config, ckpt, device)
    print(f"\nWeather eval — threshold={threshold}  device={device}")

    all_results = {}

    for weather, rc_folders in weather_splits.items():
        if not rc_folders:
            continue
        print(f"\n{'='*60}")
        print(f"  {weather.upper()} — {rc_folders}")
        print('='*60)

        dl_frames, cfar_frames = collect_frames(
            rc_folders, base_dir, config, model, device,
            threshold, cfar_dir, weather)

        print(f"  {len(dl_frames)} frames collected")
        dl_m = compute_weather_metrics(dl_frames, threshold, weather)
        all_results[weather] = {'DL': dl_m}

        if cfar_frames and any(f['pts'] is not None for f in cfar_frames):
            cfar_m = compute_weather_metrics(cfar_frames, 0.5, weather)
            all_results[weather]['CFAR'] = cfar_m

    # ── Results table ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESULTS TABLE")
    print(f"{'='*60}")
    print(f"{'Weather':<8} {'Method':<6} {'AP':>6} {'P_d':>6} {'P_fa':>6} {'CD':>9}")
    print('─' * 48)

    for weather in ['clear', 'fog', 'rain']:
        if weather not in all_results:
            continue
        for method, m in all_results[weather].items():
            cd_s = f"{m['CD']:9.4f}" if not np.isnan(m['CD']) else '      N/A'
            print(f"{weather:<8} {method:<6} {m['AP']:6.3f} {m['P_d']:6.3f} {m['P_fa']:6.3f} {cd_s}")

    # ── Degradation ───────────────────────────────────────────────────────────
    if 'clear' in all_results:
        print(f"\n{'='*60}")
        print("DEGRADATION  (clear → weather)  lower % = more robust")
        print(f"{'='*60}")
        print(f"{'Weather':<8} {'Method':<6} {'AP deg%':>8} {'P_d deg%':>9} {'P_fa deg%':>10}")
        print('─' * 45)

        def _deg(c, w):
            return f"{(c - w) / c * 100:8.1f}%" if c > 0 else '     N/A'

        for weather in ['fog', 'rain']:
            if weather not in all_results:
                continue
            for method in ['DL', 'CFAR']:
                if method not in all_results.get('clear', {}) or \
                   method not in all_results.get(weather, {}):
                    continue
                c = all_results['clear'][weather]    if False else all_results['clear'][method]
                w = all_results[weather][method]
                print(f"{weather:<8} {method:<6} "
                      f"{_deg(c['AP'], w['AP'])} "
                      f"{_deg(c['P_d'], w['P_d'])} "
                      f"{_deg(c['P_fa'], w['P_fa'])}")

    # ── Point density ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("POINT DENSITY  (pts/m³ inside GT box)")
    print(f"{'='*60}")
    print(f"{'Weather':<8} {'Method':<6} {'0-10m':>8} {'10-15m':>8} {'15-20m':>8}")
    print('─' * 45)

    def _fd(v):
        return f"{v:8.3f}" if not np.isnan(v) else '     N/A'

    for weather in ['clear', 'fog', 'rain']:
        if weather not in all_results:
            continue
        for method, m in all_results[weather].items():
            d = m['density']
            print(f"{weather:<8} {method:<6} "
                  f"{_fd(d.get('0-10m', np.nan))} "
                  f"{_fd(d.get('10-15m', np.nan))} "
                  f"{_fd(d.get('15-20m', np.nan))}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'weather_results.csv')
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Weather', 'Method', 'AP', 'P_d', 'P_fa', 'Chamfer_Distance',
                         'Density_0-10m', 'Density_10-15m', 'Density_15-20m'])
        for weather in ['clear', 'fog', 'rain']:
            if weather not in all_results:
                continue
            for method, m in all_results[weather].items():
                d = m['density']
                writer.writerow([
                    weather, method,
                    f"{m['AP']:.4f}", f"{m['P_d']:.4f}", f"{m['P_fa']:.4f}",
                    f"{m['CD']:.4f}" if not np.isnan(m['CD']) else 'N/A',
                    f"{d.get('0-10m', np.nan):.4f}"  if not np.isnan(d.get('0-10m',  np.nan)) else 'N/A',
                    f"{d.get('10-15m', np.nan):.4f}" if not np.isnan(d.get('10-15m', np.nan)) else 'N/A',
                    f"{d.get('15-20m', np.nan):.4f}" if not np.isnan(d.get('15-20m', np.nan)) else 'N/A',
                ])

    print(f"\n  Results saved → {os.path.abspath(out_csv)}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='configs/eval_config.yaml')
    parser.add_argument('--checkpoint', default=None,
                        help='Override eval_config.yaml checkpoint path')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    ckpt = args.checkpoint or config.get('checkpoint', '')
    if not ckpt:
        print("ERROR: set checkpoint in eval_config.yaml or pass --checkpoint")
        return

    out_dir  = config.get('out_dir', 'verification_output/eval')
    mode     = config.get('eval_mode', 'basic')

    print(f"Config : {args.config}")
    print(f"Mode   : {mode}")
    print(f"Ckpt   : {ckpt}")

    if mode == 'basic':
        run_basic(config, ckpt, out_dir)
    elif mode == 'weather':
        run_weather(config, ckpt, out_dir)
    else:
        print(f"ERROR: unknown eval_mode '{mode}' — use 'basic' or 'weather'")


if __name__ == '__main__':
    main()
