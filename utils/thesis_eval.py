"""
Unified evaluation script -- reads configs/eval_config.yaml.

eval_mode: basic
    Load model on clear-weather data -> per-voxel IoU / Precision / Recall.
    Use this to verify the model is working before running thesis eval.

eval_mode: weather
    Full thesis evaluation across clear / fog / rain:
      Exp 1 -- AP, P_d, P_fa, Chamfer Distance  (clear)
      Exp 2 -- AP, P_d, P_fa                    (fog, rain)
      Exp 3 -- Degradation %                    (clear -> fog/rain)
      Exp 4 -- Point density per distance band  (0-10m, 10-15m, 15-20m)
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

try:
    import cv2
except ImportError:
    cv2 = None

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.factory import ModelFactory
from dataset.dataloader import RadarDataset
from utils.project_to_image import occupancy_to_points, parse_calibration


# -- Shared helpers ------------------------------------------------------------

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


def _parse_calib_file(txt_path):
    """Parse one calib .txt → (corners, t_r2l, R_r2l). corners=None if no BoundingBox."""
    with open(txt_path) as f:
        content = f.read()
    match   = re.search(r'"BoundingBox":([\d\s.-]+),', content)
    corners = np.array(match.group(1).split(), dtype=float).reshape(-1, 3) if match else None
    _, _, _, r2l = parse_calibration(txt_path)
    R_r2l = np.eye(3)
    m = re.search(r'"Rotation_Radar_to_Lidar":\s*([-\d\s.e+]+),', content)
    if m:
        vals = np.array(m.group(1).strip().split(), dtype=float)
        if len(vals) == 9:
            R_r2l = vals.reshape(3, 3)
    return corners, r2l, R_r2l


def _find_calib(txt_files, txt_ts, ts_ms, threshold_ms=100):
    """Return (corners, t_r2l, R_r2l) for nearest calib within threshold, else (None, zeros, eye)."""
    if len(txt_ts) == 0:
        return None, np.zeros(3), np.eye(3)
    diffs = np.abs(txt_ts - ts_ms)
    best  = int(np.argmin(diffs))
    if diffs[best] > threshold_ms:
        return None, np.zeros(3), np.eye(3)
    return _parse_calib_file(txt_files[best])


def _cfar_to_lidar(pts, R_r2l, t_r2l):
    """Transform raw SAVEROAD CFAR points to LiDAR frame.
    Matches cfar_all_frames.py: flip X then apply inverse of Radar_to_Lidar."""
    p = pts.copy().astype(np.float64)
    p[:, 0] *= -1
    T   = np.vstack([np.hstack([R_r2l.T, (-R_r2l.T @ t_r2l).reshape(-1, 1)]),
                     [0, 0, 0, 1]])
    hom = np.hstack([p, np.ones((len(p), 1))])
    return (T @ hom.T).T[:, :3].astype(np.float32)


# -- Geometry ------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# BASIC MODE
# -----------------------------------------------------------------------------

def run_basic(config, ckpt, out_dir):
    """Per-voxel IoU / Precision / Recall across all RC folders in eval_splits."""
    basic_cfg = config.get('basic', {})
    threshold = float(basic_cfg.get('threshold', 0.4))
    base_dir  = config.get('base_dir', '')

    # Collect all RC folders from eval_splits (all conditions)
    splits   = config.get('eval_splits', {})
    rc_names = [rc for folders in splits.values() for rc in folders]
    if not rc_names:
        print("ERROR: no RC folders found in eval_splits in eval_config.yaml")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = _load_model(config, ckpt, device)

    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'basic_results.csv')
    rows    = []

    for rc_name in rc_names:
        rc_dir = os.path.join(base_dir, rc_name) if base_dir else rc_name
        try:
            ds = RadarDataset(rc_dir, augment=False, config=_build_ds_config(config, rc_dir))
        except Exception as e:
            print(f"\n[SKIP] {rc_name}: {e}")
            continue

        print(f"\nBasic eval -- {rc_name}  ({len(ds)} frames)  threshold={threshold}")

        tp_total = fp_total = fn_total = 0
        per_frame = []

        for idx in tqdm(range(len(ds)), desc=rc_name):
            radar_tensor, label_tensor = ds[idx]
            with torch.no_grad():
                pred_prob = torch.sigmoid(
                    model(radar_tensor.unsqueeze(0).to(device))
                )[0].cpu()

            pred_bin  = (pred_prob    > threshold).float()
            label_bin = (label_tensor > 0.5).float()

            tp = (pred_bin * label_bin).sum().item()
            fp = (pred_bin * (1 - label_bin)).sum().item()
            fn = ((1 - pred_bin) * label_bin).sum().item()
            tp_total += tp; fp_total += fp; fn_total += fn

            iou  = tp / (tp + fp + fn + 1e-8)
            prec = tp / (tp + fp + 1e-8)
            rec  = tp / (tp + fn + 1e-8)
            per_frame.append({'frame': idx, 'IoU': iou, 'Precision': prec, 'Recall': rec})

        iou_g  = tp_total / (tp_total + fp_total + fn_total + 1e-8)
        prec_g = tp_total / (tp_total + fp_total + 1e-8)
        rec_g  = tp_total / (tp_total + fn_total + 1e-8)
        iou_m  = float(np.mean([f['IoU']       for f in per_frame]))
        prec_m = float(np.mean([f['Precision'] for f in per_frame]))
        rec_m  = float(np.mean([f['Recall']    for f in per_frame]))

        print(f"\n{'-'*46}")
        print(f"{'Metric':<18} {'Global':>10} {'Per-frame mean':>14}")
        print(f"{'-'*46}")
        print(f"{'IoU':<18} {iou_g:10.4f} {iou_m:14.4f}")
        print(f"{'Precision':<18} {prec_g:10.4f} {prec_m:14.4f}")
        print(f"{'Recall':<18} {rec_g:10.4f} {rec_m:14.4f}")
        print(f"{'-'*46}")

        rows.append({'RC': rc_name, 'IoU': iou_g, 'Precision': prec_g, 'Recall': rec_g,
                     'per_frame': per_frame})

    if not rows:
        return

    # Summary table
    print(f"\n{'='*54}")
    print(f"{'RC':<10} {'IoU':>8} {'Precision':>12} {'Recall':>10}")
    print(f"{'-'*54}")
    for r in rows:
        print(f"{r['RC']:<10} {r['IoU']:8.4f} {r['Precision']:12.4f} {r['Recall']:10.4f}")
    mean_iou  = float(np.mean([r['IoU']       for r in rows]))
    mean_prec = float(np.mean([r['Precision'] for r in rows]))
    mean_rec  = float(np.mean([r['Recall']    for r in rows]))
    print(f"{'-'*54}")
    print(f"{'MEAN':<10} {mean_iou:8.4f} {mean_prec:12.4f} {mean_rec:10.4f}")
    print(f"{'='*54}")

    if mean_iou > 0.1:
        print("\n  Model is producing meaningful predictions.")
    elif mean_iou > 0.01:
        print("\n  Model is detecting something — IoU is low, may need more training.")
    else:
        print("\n  WARNING: IoU near zero — model may not be learning or threshold is wrong.")

    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        for r in rows:
            writer.writerow([r['RC'], 'Frame', 'IoU', 'Precision', 'Recall'])
            for fd in r['per_frame']:
                writer.writerow(['', fd['frame'], f"{fd['IoU']:.4f}",
                                 f"{fd['Precision']:.4f}", f"{fd['Recall']:.4f}"])
            writer.writerow(['', 'GLOBAL', f"{r['IoU']:.4f}",
                             f"{r['Precision']:.4f}", f"{r['Recall']:.4f}"])
            writer.writerow([])
        writer.writerow(['MEAN', '', f"{mean_iou:.4f}", f"{mean_prec:.4f}", f"{mean_rec:.4f}"])
    print(f"\n  Results saved -> {os.path.abspath(out_csv)}")

    # -- Thesis figures --------------------------------------------------------
    tp_cfg         = config.get('thesis_plots', {})
    raw_prediction = bool(tp_cfg.get('raw_prediction', True))
    thr_plot       = float(tp_cfg.get('threshold', threshold))
    n_plots        = int(tp_cfg.get('n_plots', 5))
    if tp_cfg.get('enable', True):
        print(f"\n{'='*54}")
        print(f"THESIS FIGURES  ({n_plots} frames per RC, "
              f"{'raw prediction' if raw_prediction else f'threshold={thr_plot}'})")
        print(f"{'='*54}")
        generate_thesis_plots(rc_names, base_dir, config, model, device,
                              out_dir, threshold=thr_plot, n_plots=n_plots,
                              raw_prediction=raw_prediction)


# -----------------------------------------------------------------------------
# WEATHER MODE -- metric helpers
# -----------------------------------------------------------------------------

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
        data = np.load(f) if f.endswith('.npy') else np.loadtxt(f, delimiter=',')
    except Exception:
        return None, None
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if len(data) == 0:
        return None, None
    # Apply Doppler filter: column 3 is velocity (m/s); keep approaching targets only.
    # Using it as a score with threshold=0.5 would discard all negative-velocity detections.
    if data.shape[1] > 3:
        data = data[data[:, 3] < -1.8]
    if len(data) == 0:
        return None, None
    pts    = data[:, :3].astype(np.float32)
    scores = np.ones(len(pts), dtype=np.float32)   # binary: passed Doppler gate
    return pts, scores


def compute_ap(frames_data):
    all_scores, all_in_box = [], []
    for fd in frames_data:
        corners = fd['box_corners']
        if corners is None:
            continue
        if 'in_box' in fd:
            scores = fd['scores']
            if scores is None or len(scores) == 0:
                continue
            in_box = fd['in_box']
        else:
            pts, scores = fd.get('pts'), fd.get('scores')
            if pts is None or len(pts) == 0:
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

    try:
        return float(np.trapezoid(prec, rec))
    except AttributeError:
        return float(np.trapz(prec, rec))


def compute_pd_pfa(frames_data, threshold):
    n_frames = n_detected = total = outside = 0
    for fd in frames_data:
        corners = fd['box_corners']
        if corners is None:
            continue
        n_frames += 1
        if 'in_box' in fd:
            scores = fd['scores']
            if scores is None or len(scores) == 0:
                continue
            mask     = scores >= threshold
            total   += int(mask.sum())
            in_box_t = fd['in_box'][mask]
            outside += int((~in_box_t).sum())
            if in_box_t.any():
                n_detected += 1
        else:
            pts, scores = fd.get('pts'), fd.get('scores')
            if pts is None:
                continue
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
            if (fd['box_corners'] is None
                    or fd['lidar_pts'] is None or len(fd['lidar_pts']) == 0):
                continue
            if 'pts_in_box_thresh' in fd:
                pts_in = fd['pts_in_box_thresh']
            else:
                pts, scores = fd.get('pts'), fd.get('scores')
                if pts is None or len(pts) == 0:
                    continue
                mask   = (scores >= threshold) if scores is not None else np.ones(len(pts), dtype=bool)
                pts_in = pts[mask]
                pts_in = pts_in[points_in_box(pts_in, fd['box_corners'])] if len(pts_in) else pts_in
            if len(pts_in) == 0:
                continue
            val = chamfer_distance(pts_in, fd['lidar_pts'])
            if not np.isnan(val):
                cd_vals.append(val)
        cd = float(np.mean(cd_vals)) if cd_vals else np.nan

    # Point density: group frames by box_range band, compute pts_in_box/box_volume per frame
    bands       = ((0, 5), (5, 10), (10, 15), (15, 20))
    density_acc = {f"{a}-{b}m": [] for a, b in bands}
    for fd in frames:
        if fd['box_corners'] is None or np.isnan(fd.get('box_range', np.nan)):
            continue
        if 'pts_in_box_thresh' in fd:
            pts_in = fd['pts_in_box_thresh']
        else:
            pts, scores = fd.get('pts'), fd.get('scores')
            if pts is None or len(pts) == 0:
                continue
            mask   = (scores >= threshold) if scores is not None else np.ones(len(pts), dtype=bool)
            pts_t  = pts[mask]
            pts_in = pts_t[points_in_box(pts_t, fd['box_corners'])] if len(pts_t) else pts_t
        if len(pts_in) == 0:
            continue
        # bin this frame into the band that matches its box range
        r = fd['box_range']
        for a, b in bands:
            if a <= r < b:
                density_acc[f"{a}-{b}m"].append(len(pts_in) / (fd['box_volume'] + 1e-9))
                break

    density = {k: float(np.mean(v)) if v else np.nan for k, v in density_acc.items()}

    # Per-range-band AP / P_d / P_fa
    range_metrics = {}
    for r_min, r_max in bands:
        band_frames = [fd for fd in frames
                       if not np.isnan(fd.get('box_range', np.nan))
                       and r_min <= fd['box_range'] < r_max]
        if not band_frames:
            continue
        b_ap       = compute_ap(band_frames)
        b_pd, b_fa = compute_pd_pfa(band_frames, threshold)
        range_metrics[f"{r_min}-{r_max}m"] = {'AP': b_ap, 'P_d': b_pd, 'P_fa': b_fa,
                                               'n': len(band_frames)}

    return {'AP': ap, 'P_d': p_d, 'P_fa': p_fa, 'CD': cd,
            'density': density, 'range_metrics': range_metrics}


def _box_stats(corners):
    """Return (box_volume, box_range) from corner array."""
    dims   = corners.max(axis=0) - corners.min(axis=0)
    center = (corners.max(axis=0) + corners.min(axis=0)) / 2
    return float(dims[0] * dims[1] * dims[2]), float(np.sqrt(center[0]**2 + center[1]**2))


def _collect_cfar_frames(rc_dir, sf, weather, calib_files):
    """Collect CFAR frames independently from all calib files (not tied to DL timestamps).

    Each calib file defines one evaluation frame. CFAR files are matched by
    nearest timestamp within 200 ms. Label files (for Chamfer) are matched
    within 100 ms when weather == 'clear'.
    """
    cfar_dir   = os.path.join(rc_dir, sf.get('cfar', 'cfar'))
    label_dir  = os.path.join(rc_dir, sf.get('labels', 'labels'))

    has_cfar = os.path.isdir(cfar_dir) and (
        glob.glob(os.path.join(cfar_dir, '*.npy')) +
        glob.glob(os.path.join(cfar_dir, '*.txt')))
    if not has_cfar:
        return []

    label_files = sorted(glob.glob(os.path.join(label_dir, '*.npy')))
    label_ts    = np.array([_extract_ts_ms(f) for f in label_files]) if label_files else np.array([])

    frames = []
    for cf in calib_files:
        ts_ms = _extract_ts_ms(cf)
        corners, r2l, R_r2l = _parse_calib_file(cf)
        if corners is None:
            continue
        box_volume, box_range = _box_stats(corners)

        # GT LiDAR pts for Chamfer distance (clear weather only)
        lidar_pts_in_box = None
        if weather == 'clear' and len(label_ts) > 0:
            diffs = np.abs(label_ts - ts_ms)
            best  = int(np.argmin(diffs))
            if diffs[best] <= 100:
                lbl_pts, _ = occupancy_to_points(np.load(label_files[best]), threshold=0.5)
                lbl_pts    = lbl_pts + r2l
                if len(lbl_pts) > 0:
                    lidar_pts_in_box = lbl_pts[points_in_box(lbl_pts, corners)]

        # Load CFAR: Doppler-filtered, binary scores, full coordinate transform
        pts_c, scores_c = _load_cfar(cfar_dir, ts_ms)
        if pts_c is not None and len(pts_c):
            pts_c = _cfar_to_lidar(pts_c, R_r2l, r2l)

        frames.append({
            'pts':         pts_c,
            'scores':      scores_c,
            'box_corners': corners,
            'lidar_pts':   lidar_pts_in_box,
            'box_volume':  box_volume,
            'box_range':   box_range,
        })
    return frames


def _range_band(r):
    """Return band label string for a box range in metres."""
    if np.isnan(r): return 'unknown'
    for lo, hi in [(0,5),(5,10),(10,15),(15,20),(20,30)]:
        if lo <= r < hi:
            return f'{lo}-{hi}m'
    return f'{r:.0f}m+'


def collect_frames(rc_folders, base_dir, config, model, device, threshold, weather):
    """Return (dl_frames, cfar_frames, match_log) for all RC folders.

    DL frames: one per prepared-dataset frame (synced radar+LiDAR).
    CFAR frames use a hybrid approach:
      - Matched: one CFAR per DL frame at the same timestamp — ensures fair
        per-band comparison for the 0-5m and 5-10m bands.
      - Extended: remaining calib frames not covered by any DL timestamp —
        adds the 10-15m and 15-20m bands where DL has no labels.

    match_log: list of dicts, one per CFAR frame, recording the matching
    details. Written to frame_match_log.csv by run_weather.
    """
    sf          = config.get('subfolders', {})
    dl_frames   = []
    cfar_frames = []
    match_log   = []

    for rc_name in rc_folders:
        rc_dir    = os.path.join(base_dir, rc_name) if base_dir else rc_name
        calib_dir = os.path.join(rc_dir, sf.get('calib', 'calib'))
        cfar_dir  = os.path.join(rc_dir, sf.get('cfar',  'cfar'))

        calib_files = sorted(glob.glob(os.path.join(calib_dir, '*.txt')))
        txt_ts      = np.array([_extract_ts_ms(f) for f in calib_files]) if calib_files else np.array([])

        # Pre-index CFAR timestamps for delta calculation in the log
        cfar_files = sorted(glob.glob(os.path.join(cfar_dir, '*.txt')) +
                            glob.glob(os.path.join(cfar_dir, '*.npy')))
        cfar_ts    = np.array([_extract_ts_ms(f) for f in cfar_files]) if cfar_files else np.array([])

        # ── DL frames: one per prepared-dataset frame ─────────────────────────
        try:
            ds = RadarDataset(rc_dir, augment=False, config=_build_ds_config(config, rc_dir))
        except Exception as e:
            import traceback
            print(f"  [ERROR] Skipping {rc_name} DL: {e}")
            traceback.print_exc()
            continue

        dl_ts_list = []

        for idx in tqdm(range(len(ds)), desc=rc_name):
            radar_tensor, label_tensor = ds[idx]
            sample = ds.matched_data[idx]
            ts_ms  = _extract_ts_ms(sample['power'])
            dl_ts_list.append(ts_ms)

            corners, r2l, R_r2l = _find_calib(calib_files, txt_ts, ts_ms)
            box_volume, box_range = _box_stats(corners) if corners is not None else (0.0, np.nan)

            with torch.no_grad():
                pred_np = torch.sigmoid(
                    model(radar_tensor.unsqueeze(0).to(device))
                )[0].cpu().numpy()

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
            })

            # ── Matched CFAR: same timestamp as this DL frame (1-to-1) ────────
            pts_c, scores_c = _load_cfar(cfar_dir, ts_ms)
            cfar_found = pts_c is not None and len(pts_c) > 0
            if cfar_found and corners is not None:
                pts_c = _cfar_to_lidar(pts_c, R_r2l, r2l)
            cfar_frames.append({
                'pts':         pts_c,
                'scores':      scores_c,
                'box_corners': corners,
                'lidar_pts':   lidar_pts_in_box,
                'box_volume':  box_volume,
                'box_range':   box_range,
            })

            # Log entry for this matched frame
            cfar_ts_matched = ''
            delta_ms        = ''
            if len(cfar_ts) > 0:
                ci = int(np.argmin(np.abs(cfar_ts - ts_ms)))
                if abs(cfar_ts[ci] - ts_ms) <= 200:
                    cfar_ts_matched = cfar_ts[ci]
                    delta_ms        = abs(cfar_ts[ci] - ts_ms)
            match_log.append({
                'rc':           rc_name,
                'type':         'matched',
                'dl_ts_ms':     ts_ms,
                'cfar_ts_ms':   cfar_ts_matched,
                'delta_ms':     delta_ms,
                'cfar_found':   cfar_found,
                'box_range_m':  f'{box_range:.2f}' if not np.isnan(box_range) else '',
                'band':         _range_band(box_range),
            })

        # ── Extended CFAR: calib timestamps NOT covered by any DL frame ───────
        # Provides metrics for 10-15m and 15-20m where DL has no labels.
        if dl_ts_list:
            dl_ts_arr   = np.array(dl_ts_list)
            extra_calib = [f for f in calib_files
                           if np.min(np.abs(dl_ts_arr - _extract_ts_ms(f))) > 200]
        else:
            extra_calib = calib_files
        if extra_calib:
            ext_frames = _collect_cfar_frames(rc_dir, sf, weather, extra_calib)
            cfar_frames.extend(ext_frames)

            # Log entries for extended frames
            for cf, ef in zip(extra_calib, ext_frames):
                calib_ts_ms = _extract_ts_ms(cf)
                cfar_found_ext = ef.get('pts') is not None and len(ef.get('pts', [])) > 0
                cfar_ts_ext = ''
                delta_ext   = ''
                if len(cfar_ts) > 0:
                    ci = int(np.argmin(np.abs(cfar_ts - calib_ts_ms)))
                    if abs(cfar_ts[ci] - calib_ts_ms) <= 200:
                        cfar_ts_ext = cfar_ts[ci]
                        delta_ext   = abs(cfar_ts[ci] - calib_ts_ms)
                br = ef.get('box_range', float('nan'))
                match_log.append({
                    'rc':           rc_name,
                    'type':         'extended',
                    'dl_ts_ms':     '',
                    'cfar_ts_ms':   cfar_ts_ext,
                    'delta_ms':     delta_ext,
                    'cfar_found':   cfar_found_ext,
                    'box_range_m':  f'{br:.2f}' if not np.isnan(br) else '',
                    'band':         _range_band(br),
                })

    return dl_frames, cfar_frames, match_log


# -----------------------------------------------------------------------------
# THESIS FIGURE GENERATION
# -----------------------------------------------------------------------------

def _compute_ae_re_maps(power_np, elev_np, config):
    """Elevation-indexed AE and RE maps (mirrors predict.py logic)."""
    num_e   = 64
    norm    = config.get('normalization', {})
    max_ang = norm.get('elevation_max_angle', 0.7854)
    is_norm = norm.get('normalize_elevation', False)

    e_norm = elev_np if is_norm else np.clip(elev_np / (max_ang + 1e-9), -1.0, 1.0)
    e_bins = ((e_norm + 1.0) / 2.0 * (num_e - 1)).clip(0, num_e - 1).astype(int)

    p_mov  = power_np[2:]   # skip static Doppler channels
    e_mov  = e_bins[2:]
    d, r, a = p_mov.shape

    def _proj(dim_idx, size):
        s = np.zeros((num_e, size), dtype=np.float64)
        c = np.zeros((num_e, size), dtype=np.float64)
        g = np.broadcast_to(np.arange(size).reshape([1 if i != dim_idx else size
                            for i in range(3)]), (d, r, a))
        np.add.at(s, (e_mov.ravel(), g.ravel()), p_mov.ravel())
        np.add.at(c, (e_mov.ravel(), g.ravel()), 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            m = np.where(c > 0, s / c, 0).astype(np.float32)
        if m.max() > 0:
            m /= m.max()
        return m

    # AE: elevation x azimuth
    ae = np.zeros((num_e, a), dtype=np.float64)
    ac = np.zeros((num_e, a), dtype=np.float64)
    ag = np.broadcast_to(np.arange(a)[np.newaxis, np.newaxis, :], (d, r, a))
    np.add.at(ae, (e_mov.ravel(), ag.ravel()), p_mov.ravel())
    np.add.at(ac, (e_mov.ravel(), ag.ravel()), 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        ae_map = np.where(ac > 0, ae / ac, 0).astype(np.float32)
    if ae_map.max() > 0: ae_map /= ae_map.max()

    # RE: elevation x range
    re = np.zeros((num_e, r), dtype=np.float64)
    rc2 = np.zeros((num_e, r), dtype=np.float64)
    rg = np.broadcast_to(np.arange(r)[np.newaxis, :, np.newaxis], (d, r, a))
    np.add.at(re, (e_mov.ravel(), rg.ravel()), p_mov.ravel())
    np.add.at(rc2, (e_mov.ravel(), rg.ravel()), 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        re_map = np.where(rc2 > 0, re / rc2, 0).astype(np.float32)
    if re_map.max() > 0: re_map /= re_map.max()

    return ae_map, re_map


def _save_pred_plot(rc_name, ts_str, frame_idx,
                    radar_bev, ae_map, re_map,
                    gt_bev, gt_fv, gt_sv,
                    pred_bev, pred_fv, pred_sv,
                    out_path, raw_prediction=True):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    xt  = np.linspace(0, 255, 7)
    xl  = np.round(np.degrees(np.arcsin(np.linspace(-1, 1, 7)))).astype(int)

    fig = plt.figure(figsize=(18, 12))
    gs  = fig.add_gridspec(3, 3, hspace=0.38, wspace=0.32)

    def _ishow(ax, data, title, cmap, xlabel, ylabel, vmin=0, vmax=1, az_ticks=False):
        im = ax.imshow(data, cmap=cmap, origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8); ax.set_ylabel(ylabel, fontsize=8)
        if az_ticks:
            ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=7)
        plt.colorbar(im, ax=ax)
        return im

    # Row 0 -- Input
    bev_arr = radar_bev.numpy()
    bev_max = float(bev_arr.max()) if bev_arr.max() > 0 else 1.0
    _ishow(fig.add_subplot(gs[0, 0]), bev_arr,
           'Input: Power BEV',  'turbo', 'Azimuth (deg)', 'Range (Bins)', vmin=0, vmax=bev_max, az_ticks=True)
    ae_max = float(ae_map.max()) if ae_map.max() > 0 else 1.0
    _ishow(fig.add_subplot(gs[0, 1]), ae_map,
           'Input: AE Map',     'turbo', 'Azimuth (deg)', 'Elevation (Bins)', vmin=0, vmax=ae_max, az_ticks=True)
    re_max = float(re_map.max()) if re_map.max() > 0 else 1.0
    _ishow(fig.add_subplot(gs[0, 2]), re_map,
           'Input: RE Map',     'turbo', 'Range (Bins)',  'Elevation (Bins)', vmin=0, vmax=re_max)

    # Row 1 -- GT
    no_gt = np.zeros((64, 256))
    _ishow(fig.add_subplot(gs[1, 0]),
           gt_bev.numpy() if gt_bev is not None else no_gt,
           'GT: BEV' + ('' if gt_bev is not None else ' (no GT)'),
           'gray', 'Azimuth (deg)', 'Range (Bins)', vmin=0, vmax=1, az_ticks=True)
    _ishow(fig.add_subplot(gs[1, 1]),
           gt_fv.numpy() if gt_fv is not None else no_gt,
           'GT: Front View', 'gray', 'Azimuth (deg)', 'Height (Bins)', vmin=0, vmax=1, az_ticks=True)
    _ishow(fig.add_subplot(gs[1, 2]),
           gt_sv.numpy() if gt_sv is not None else no_gt,
           'GT: Side View', 'gray', 'Range (Bins)', 'Height (Bins)', vmin=0, vmax=1)

    # Row 2 -- Prediction
    pred_label = 'Raw Prediction' if raw_prediction else 'Prediction (thresholded)'
    _ishow(fig.add_subplot(gs[2, 0]), pred_bev.numpy(),
           f'Pred: BEV ({pred_label})',        'magma', 'Azimuth (deg)', 'Range (Bins)', az_ticks=True)
    _ishow(fig.add_subplot(gs[2, 1]), pred_fv.numpy(),
           f'Pred: Front View ({pred_label})', 'magma', 'Azimuth (deg)', 'Height (Bins)', az_ticks=True)
    _ishow(fig.add_subplot(gs[2, 2]), pred_sv.numpy(),
           f'Pred: Side View ({pred_label})',  'magma', 'Range (Bins)',  'Height (Bins)')

    fig.suptitle(f"{rc_name}  |  Frame {frame_idx:03d}  |  ts={ts_str}", fontsize=13)
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def _save_camera_proj(calib_txt, pco_dir, pred_np, threshold,
                      rc_name, ts_str, frame_idx, out_path,
                      saveroad_dir='', project_points_mod=None):
    """project_points_mod is the Cython extension; if None, falls back to NumPy."""
    use_numpy = project_points_mod is None
    try:
        import cv2
    except ImportError:
        print("    [CAM] cv2 not installed — cannot save camera projection")
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    img_name, K, r_t, radar_to_lidar = parse_calibration(calib_txt)
    if not img_name:
        print(f"    [CAM] frame {frame_idx}: PCO_frame missing in {os.path.basename(calib_txt)}")
        return
    img_path = os.path.join(pco_dir, img_name)
    if not os.path.exists(img_path):
        print(f"    [CAM] frame {frame_idx}: camera image not found: {img_name}")
        return

    pts_3d, probs = occupancy_to_points(pred_np, threshold)
    if len(pts_3d) == 0:
        print(f"    [CAM] frame {frame_idx}: 0 points above threshold {threshold:.3f}")
        return
    pts_3d = pts_3d + radar_to_lidar
    if len(pts_3d) > 150_000:
        top    = np.argsort(probs)[-150_000:]
        pts_3d = pts_3d[top]; probs = probs[top]

    img = cv2.imread(img_path)
    if img is None:
        print(f"    [CAM] frame {frame_idx}: cv2 could not read {img_path}")
        return

    if use_numpy:
        px, front = _project_pts_cam(pts_3d.astype(np.float64), K, r_t)
        h2, w2 = img.shape[:2]
        valid = _in_image(px, h2, w2)
        cmap = plt.cm.turbo
        for (x, y), p in zip(px[valid], probs[front][valid]):
            bgr = tuple(int(c * 255) for c in reversed(cmap(float(p))[:3]))
            cv2.circle(img, (int(x), int(y)), 3, bgr, -1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        from utils.project_to_image import project_to_image
        img_rgb = cv2.cvtColor(
            project_to_image(pts_3d, probs, img, K, r_t, project_points_mod),
            cv2.COLOR_BGR2RGB
        )

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(img_rgb)
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap='turbo', norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04).set_label('Confidence')
    ax.set_title(f"{rc_name}  |  Frame {frame_idx:03d}  |  ts={ts_str}", fontsize=13)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── Camera projection helpers ─────────────────────────────────────────────────

def _parse_calib_full(txt_path):
    """Return (img_name, K, r_t, corners, t_r2l, R_r2l) from a calib .txt."""
    img_name, K, r_t, t_r2l = parse_calibration(txt_path)
    with open(txt_path) as f:
        content = f.read()
    m_bb  = re.search(r'"BoundingBox":([\d\s.-]+),', content)
    corners = np.array(m_bb.group(1).split(), dtype=float).reshape(-1, 3) if m_bb else None
    R_r2l = np.eye(3)
    m_rot = re.search(r'"Rotation_Radar_to_Lidar":\s*([-\d\s.e+]+),', content)
    if m_rot:
        vals = np.array(m_rot.group(1).strip().split(), dtype=float)
        if len(vals) == 9:
            R_r2l = vals.reshape(3, 3)
    return img_name, K, r_t, corners, t_r2l, R_r2l


def _axangle2mat(axis, angle):
    c, s = np.cos(angle), np.sin(angle); t = 1 - c
    x, y, z = axis
    return np.array([[t*x*x+c,   t*x*y-s*z, t*x*z+s*y],
                     [t*x*y+s*z, t*y*y+c,   t*y*z-s*x],
                     [t*x*z-s*y, t*y*z+s*x, t*z*z+c  ]])


def _project_pts_cam(pts, K, r_t):
    if len(pts) == 0:
        return np.empty((0, 2), dtype=np.int32), np.zeros(0, dtype=bool)
    rot  = (_axangle2mat([0,0,1], r_t[2])
            @ _axangle2mat([0,1,0], r_t[1])
            @ _axangle2mat([1,0,0], r_t[0]))
    cam   = (rot @ pts.T).T + r_t[3:]
    front = cam[:, 2] > 0.2
    proj  = (K @ cam[front].T).T
    proj  = proj / proj[:, 2:3]
    return proj[:, :2].astype(np.int32), front


def _in_image(px, h, w):
    return (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)


def _draw_bbox_cam(img, corners, K, r_t, color=(0, 0, 220), thickness=2):
    rot  = (_axangle2mat([0,0,1], r_t[2])
            @ _axangle2mat([0,1,0], r_t[1])
            @ _axangle2mat([1,0,0], r_t[0]))
    pts3d = corners[[0, 1, 2, 3, 5, 6, 7, 8]]
    cam   = (rot @ pts3d.T).T + r_t[3:]
    if np.all(cam[:, 2] <= 0):
        return img
    proj  = (K @ cam.T).T
    proj  = proj / proj[:, 2:3]
    px    = proj[:, :2].astype(int)
    def _line(i, j):
        if cam[i, 2] > 0 and cam[j, 2] > 0:
            cv2.line(img, tuple(px[i]), tuple(px[j]), color, thickness)
    for a, b in [(0,1),(1,2),(2,3),(3,0)]: _line(a, b)
    for a, b in [(4,5),(5,6),(6,7),(7,4)]: _line(a, b)
    for a, b in [(0,4),(1,5),(2,6),(3,7)]: _line(a, b)
    return img


def generate_camera_projection_plots(rc_folders, base_dir, config, model, device,
                                      threshold, out_dir, n_plots=5):
    """Generate n_plots 2-panel camera projection figures per RC folder.

    Each panel shows the camera image with GT bounding box (red):
      Left  — DL model occupancy (green)
      Right — CFAR Doppler-filtered detections (orange)

    Frames are chosen equally spaced across all calib files so the full
    range band is represented, not just the prepared-dataset timestamps.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if cv2 is None:
        print('  [SKIP] camera projection: cv2 not installed')
        return

    sf         = config.get('subfolders', {})
    DL_COLOR   = (20,  255,  57)   # neon green  (BGR)
    CFAR_COLOR = (0,   140, 255)   # orange      (BGR)
    BOX_COLOR  = (0,   0,   220)   # red         (BGR)

    for rc_name in rc_folders:
        rc_dir    = os.path.join(base_dir, rc_name) if base_dir else rc_name
        calib_dir = os.path.join(rc_dir, sf.get('calib', 'calib'))
        cfar_dir  = os.path.join(rc_dir, sf.get('cfar',  'cfar'))
        # pco_dir: top-level config key overrides subfolder default
        pco_dir_override = config.get('pco_dir', '').strip()
        pco_dir = pco_dir_override if pco_dir_override else os.path.join(rc_dir, sf.get('pco', 'pco'))

        calib_files = sorted(glob.glob(os.path.join(calib_dir, '*.txt')))
        if not calib_files:
            print(f'  [SKIP] {rc_name}: no calib files')
            continue

        try:
            ds = RadarDataset(rc_dir, augment=False, config=_build_ds_config(config, rc_dir))
            power_ts = np.array([_extract_ts_ms(ds.matched_data[i]['power'])
                                  for i in range(len(ds))])
        except Exception as e:
            print(f'  [SKIP] {rc_name} dataset: {e}')
            continue

        has_cfar = os.path.isdir(cfar_dir) and (
            glob.glob(os.path.join(cfar_dir, '*.npy')) +
            glob.glob(os.path.join(cfar_dir, '*.txt')))

        # Only use calib files that have a matching DL frame (within 100ms) so
        # camera projection shows frames where DL prediction is available.
        if len(power_ts) > 0:
            dl_calib = [cf for cf in calib_files
                        if np.min(np.abs(power_ts - _extract_ts_ms(cf))) <= 100]
        else:
            dl_calib = calib_files
        if not dl_calib:
            dl_calib = calib_files  # fallback: use all if nothing matched
        idxs     = np.linspace(0, len(dl_calib) - 1, n_plots, dtype=int)
        selected = [dl_calib[i] for i in idxs]

        rc_out = os.path.join(out_dir, 'camera_projection', rc_name)
        os.makedirs(rc_out, exist_ok=True)

        for fi, cf in enumerate(selected):
            ts_ms = _extract_ts_ms(cf)
            try:
                img_name, K, r_t, corners, t_r2l, R_r2l = _parse_calib_full(cf)
            except Exception:
                continue
            if corners is None:
                continue

            center    = (corners.max(0) + corners.min(0)) / 2
            box_range = float(np.sqrt(center[0]**2 + center[1]**2))

            img_path = os.path.join(pco_dir, img_name)
            if not os.path.exists(img_path):
                continue
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            h, w = img_bgr.shape[:2]

            # DL panel
            img_dl = img_bgr.copy()
            n_dl   = 0
            di = int(np.argmin(np.abs(power_ts - ts_ms)))
            if abs(power_ts[di] - ts_ms) <= 100:
                radar_tensor, _ = ds[di]
                try:
                    with torch.no_grad():
                        pred_np = torch.sigmoid(
                            model(radar_tensor.unsqueeze(0).to(device))
                        )[0].cpu().numpy()
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        torch.cuda.empty_cache()
                        with torch.no_grad():
                            pred_np = torch.sigmoid(
                                model(radar_tensor.unsqueeze(0).cpu())
                            )[0].numpy()
                    else:
                        raise
                finally:
                    del radar_tensor
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                pts_dl, _ = occupancy_to_points(pred_np, threshold)
                if len(pts_dl) > 0:
                    pts_dl_l = pts_dl.astype(np.float64)   # already in LiDAR frame
                    px_d, front_d = _project_pts_cam(pts_dl_l, K, r_t)
                    valid = _in_image(px_d, h, w)
                    for (x, y) in px_d[valid]:
                        cv2.circle(img_dl, (int(x), int(y)), 4, DL_COLOR, -1)
                    n_dl = int(valid.sum())
            _draw_bbox_cam(img_dl, corners, K, r_t, BOX_COLOR, thickness=2)

            # CFAR panel
            img_cfar = img_bgr.copy()
            n_cfar   = 0
            if has_cfar:
                pts_c, _ = _load_cfar(cfar_dir, ts_ms)
                if pts_c is not None and len(pts_c):
                    pts_c = _cfar_to_lidar(pts_c, R_r2l, t_r2l)
                    px_c, _ = _project_pts_cam(pts_c.astype(np.float64), K, r_t)
                    valid = _in_image(px_c, h, w)
                    for (x, y) in px_c[valid]:
                        cv2.circle(img_cfar, (int(x), int(y)), 5, CFAR_COLOR, -1)
                    n_cfar = int(valid.sum())
            _draw_bbox_cam(img_cfar, corners, K, r_t, BOX_COLOR, thickness=2)

            # 2-panel figure
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            for ax, im, title in zip(
                    axes,
                    [img_dl,   img_cfar],
                    [f'DL Model ({n_dl} pts)',
                     f'CFAR Doppler-filtered ({n_cfar} pts)']):
                ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                ax.set_title(title, fontsize=12)
                ax.axis('off')

            fig.legend(
                handles=[
                    mpatches.Patch(facecolor='#DC0000', edgecolor='white', label='GT Bounding Box'),
                    mpatches.Patch(facecolor='#39FF14', edgecolor='white', label='DL Occupancy'),
                    mpatches.Patch(facecolor='#FF8C00', edgecolor='white', label='CFAR Detection'),
                ],
                loc='lower center', ncol=3, fontsize=10,
                framealpha=0.85, bbox_to_anchor=(0.5, -0.01))
            fig.suptitle(f'{rc_name}  |  range = {box_range:.1f} m', fontsize=13, y=1.01)
            plt.tight_layout()

            out_path = os.path.join(rc_out, f'frame_{fi+1:02d}_range{box_range:.1f}m.png')
            plt.savefig(out_path, dpi=120, bbox_inches='tight')
            plt.close(fig)
            print(f'    frame {fi+1}/{n_plots}  range={box_range:.1f}m  '
                  f'DL={n_dl}  CFAR={n_cfar}  -> {os.path.basename(out_path)}')

        n_saved = len([f for f in os.listdir(rc_out) if f.endswith('.png')])
        print(f'  {rc_name}: {n_saved}/{n_plots} frames saved -> {rc_out}')


def generate_thesis_plots(rc_folders, base_dir, config, model, device,
                          out_dir, threshold=0.4, n_plots=5, raw_prediction=True):
    """Generate n_plots equally spaced prediction + camera frames per RC folder."""
    import matplotlib
    matplotlib.use('Agg')
    import traceback

    sf              = config.get('subfolders', {})
    saveroad_dir    = config.get('saveroad_dir', '').strip()
    tp_cfg          = config.get('thesis_plots', {})
    do_camera       = bool(tp_cfg.get('camera_projection', True)) and bool(saveroad_dir)

    # Validate saveroad_dir and load tools ONCE before the frame loop
    project_points_mod = None
    if do_camera:
        if not os.path.isdir(saveroad_dir):
            print(f"  [CAM ERROR] saveroad_dir not found: {saveroad_dir}")
            print(f"              Camera projection disabled for this run.")
            do_camera = False
        else:
            sys.path.insert(0, saveroad_dir)
            try:
                from tools import project_points_v2_withPC as _pm
                project_points_mod = _pm
                print(f"  Camera projection tools loaded OK from: {saveroad_dir}")
            except ImportError as e:
                print(f"  [CAM WARN] project_points_v2_withPC not found: {e}")
                print(f"             Falling back to NumPy projection (Cython not compiled for this platform).")
                print(f"             To build Cython: cd {saveroad_dir}/tools && python setup_v2_withPC.py build_ext --inplace")
                # project_points_mod stays None → _save_camera_proj uses NumPy fallback

    for rc_name in rc_folders:
        rc_dir = os.path.join(base_dir, rc_name) if base_dir else rc_name

        try:
            ds = RadarDataset(rc_dir, augment=False,
                              config=_build_ds_config(config, rc_dir))
        except Exception as e:
            print(f"  [SKIP] {rc_name}: {e}")
            continue

        n = len(ds)
        if n == 0:
            continue

        # Equally spaced indices: avoid very first/last frame
        start   = min(5, n - 1)
        end     = max(start, n - 6)
        indices = list(dict.fromkeys(
            np.linspace(start, end, n_plots).astype(int).tolist()
        ))

        # Calib + pco for camera projection
        calib_dir  = os.path.join(rc_dir, sf.get('calib', 'calib'))
        pco_dir    = os.path.join(rc_dir, sf.get('pco', 'pco'))
        txt_files  = sorted(glob.glob(os.path.join(calib_dir, '*.txt')))
        txt_ts     = np.array([_extract_ts_ms(f) for f in txt_files]) if txt_files else np.array([])
        has_camera = os.path.isdir(pco_dir) and len(txt_files) > 0

        plot_dir = os.path.join(out_dir, 'thesis_figures', rc_name, 'prediction_plots')
        cam_dir  = os.path.join(out_dir, 'thesis_figures', rc_name, 'camera_projection')
        os.makedirs(plot_dir, exist_ok=True)
        if has_camera:
            os.makedirs(cam_dir, exist_ok=True)

        print(f"\n  Generating thesis plots: {rc_name}  ({len(indices)} of {n} frames)")

        saved_plots = 0
        for idx in indices:
            try:
                sample     = ds.matched_data[idx]
                ts_ms      = _extract_ts_ms(sample['power'])
                ts_str     = os.path.basename(sample['power']).replace('.npy', '')
                radar_tensor, label_tensor = ds[idx]

                power_np = radar_tensor[0].numpy()
                elev_np  = radar_tensor[1].numpy()

                with torch.no_grad():
                    pred_prob = torch.sigmoid(
                        model(radar_tensor.unsqueeze(0).to(device))
                    )[0].cpu()

                pred_np  = pred_prob.numpy()
                ae_map, re_map = _compute_ae_re_maps(power_np, elev_np, config)

                radar_bev = torch.from_numpy(power_np).max(dim=0)[0]
                pred_bev  = pred_prob.max(dim=0)[0]
                pred_fv   = pred_prob.max(dim=1)[0]
                pred_sv   = pred_prob.max(dim=2)[0]

                if label_tensor is not None:
                    try:
                        gt      = label_tensor.float() if isinstance(label_tensor, torch.Tensor) \
                                  else torch.from_numpy(label_tensor.astype(np.float32))
                        gt_bev  = gt.max(dim=0)[0]
                        gt_fv   = gt.max(dim=1)[0]
                        gt_sv   = gt.max(dim=2)[0]
                    except Exception:
                        gt_bev = gt_fv = gt_sv = None
                else:
                    gt_bev = gt_fv = gt_sv = None

                cam_threshold = 0.05 if raw_prediction else threshold

                _save_pred_plot(
                    rc_name, ts_str, idx,
                    radar_bev, ae_map, re_map,
                    gt_bev, gt_fv, gt_sv,
                    pred_bev, pred_fv, pred_sv,
                    os.path.join(plot_dir, f'frame_{idx:03d}_{ts_str}.png'),
                    raw_prediction=raw_prediction
                )
                saved_plots += 1

                if do_camera and has_camera and len(txt_ts) > 0:
                    diff = np.abs(txt_ts - ts_ms)
                    best = int(np.argmin(diff))
                    if diff[best] < 200:
                        try:
                            _save_camera_proj(
                                txt_files[best], pco_dir, pred_np,
                                cam_threshold, rc_name, ts_str, idx,
                                os.path.join(cam_dir, f'frame_{idx:03d}_{ts_str}.png'),
                                saveroad_dir=saveroad_dir,
                                project_points_mod=project_points_mod,
                            )
                        except Exception as e:
                            print(f"    [CAM ERROR] frame {idx}: {e}")
                            traceback.print_exc()
                    else:
                        print(f"    [CAM] frame {idx}: no calib within 200ms (gap={diff[best]}ms)")

            except Exception as e:
                print(f"    [WARN] Frame {idx} skipped: {e}")
                traceback.print_exc()

        print(f"    Plots  -> {os.path.abspath(plot_dir)}  ({saved_plots}/{len(indices)} saved)")
        if do_camera and has_camera:
            print(f"    Camera -> {os.path.abspath(cam_dir)}")


# -----------------------------------------------------------------------------
# WEATHER MODE
# -----------------------------------------------------------------------------

def run_weather(config, ckpt, out_dir):
    w_cfg          = config.get('weather', {})
    threshold      = float(w_cfg.get('threshold', 0.4))
    # Support both new key (eval_splits) and old key (weather.weather_splits)
    weather_splits = config.get('eval_splits') or w_cfg.get('weather_splits', {})
    base_dir       = config.get('base_dir', '')

    if not weather_splits or not any(weather_splits.values()):
        print("ERROR: set eval_splits in eval_config.yaml")
        return

    eval_metrics = bool(config.get('eval_metrics', True))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = _load_model(config, ckpt, device)
    print(f"\nWeather eval -- threshold={threshold}  device={device}")
    if not eval_metrics:
        print("  [INFO] eval_metrics: false — skipping AP/P_d/P_fa/CD computation")

    all_results   = {}
    all_match_logs = []

    if eval_metrics:
        for weather, rc_folders in weather_splits.items():
            if not rc_folders:
                continue
            print(f"\n{'='*60}")
            print(f"  {weather.upper()} -- {rc_folders}")
            print('='*60)

            dl_frames, cfar_frames, log = collect_frames(
                rc_folders, base_dir, config, model, device,
                threshold, weather)
            all_match_logs.extend(log)

            print(f"  {len(dl_frames)} frames collected")
            dl_m = compute_weather_metrics(dl_frames, threshold, weather)
            all_results[weather] = {'DL': dl_m}

            if cfar_frames and any(len(f.get('scores', [])) > 0 or f.get('pts') is not None for f in cfar_frames):
                cfar_m = compute_weather_metrics(cfar_frames, 0.5, weather)
                all_results[weather]['CFAR'] = cfar_m

    if all_results:
        # -- Results table -----------------------------------------------------
        print(f"\n{'='*60}")
        print("RESULTS TABLE")
        print(f"{'='*60}")
        print(f"{'Weather':<8} {'Method':<6} {'AP':>6} {'P_d':>6} {'P_fa':>6} {'CD':>9}")
        print('-' * 48)
        for weather in ['clear', 'fog', 'rain']:
            if weather not in all_results:
                continue
            for method, m in all_results[weather].items():
                cd_s = f"{m['CD']:9.4f}" if not np.isnan(m['CD']) else '      N/A'
                print(f"{weather:<8} {method:<6} {m['AP']:6.3f} {m['P_d']:6.3f} {m['P_fa']:6.3f} {cd_s}")

        # -- Degradation -------------------------------------------------------
        if 'clear' in all_results:
            print(f"\n{'='*60}")
            print("DEGRADATION  (clear -> weather)  lower % = more robust")
            print(f"{'='*60}")
            print(f"{'Weather':<8} {'Method':<6} {'AP deg%':>8} {'P_d deg%':>9} {'P_fa deg%':>10}")
            print('-' * 45)
            def _deg(c, w):
                return f"{(c - w) / c * 100:8.1f}%" if c > 0 else '     N/A'
            for weather in ['fog', 'rain']:
                if weather not in all_results:
                    continue
                for method in ['DL', 'CFAR']:
                    if method not in all_results.get('clear', {}) or \
                       method not in all_results.get(weather, {}):
                        continue
                    c = all_results['clear'][method]
                    w = all_results[weather][method]
                    print(f"{weather:<8} {method:<6} "
                          f"{_deg(c['AP'], w['AP'])} "
                          f"{_deg(c['P_d'], w['P_d'])} "
                          f"{_deg(c['P_fa'], w['P_fa'])}")

        # -- Point density -----------------------------------------------------
        print(f"\n{'='*60}")
        print("POINT DENSITY  (pts/m^3 inside GT box)")
        print(f"{'='*60}")
        print(f"{'Weather':<8} {'Method':<6} {'0-5m':>8} {'5-10m':>8} {'10-15m':>8} {'15-20m':>8}")
        print('-' * 50)
        def _fd(v):
            return f"{v:8.3f}" if not np.isnan(v) else '     N/A'
        for weather in ['clear', 'fog', 'rain']:
            if weather not in all_results:
                continue
            for method, m in all_results[weather].items():
                d = m['density']
                print(f"{weather:<8} {method:<6} "
                      f"{_fd(d.get('0-5m',   np.nan))} "
                      f"{_fd(d.get('5-10m',  np.nan))} "
                      f"{_fd(d.get('10-15m', np.nan))} "
                      f"{_fd(d.get('15-20m', np.nan))}")

        # -- Range-band P_d / AP / P_fa ----------------------------------------
        range_bands = ['0-5m', '5-10m', '10-15m', '15-20m']
        has_range = any(
            bool(all_results[w][meth].get('range_metrics'))
            for w in all_results for meth in all_results[w]
        )
        if has_range:
            print(f"\n{'='*60}")
            print("RANGE-BAND DETECTION  (P_d / AP per distance band)")
            print(f"{'='*60}")
            print(f"{'Weather':<8} {'Method':<6} {'Band':<10} {'n':>5} {'AP':>6} {'P_d':>6} {'P_fa':>6}")
            print('-' * 52)
            for weather in ['clear', 'fog', 'rain']:
                if weather not in all_results:
                    continue
                for method, m in all_results[weather].items():
                    rm = m.get('range_metrics', {})
                    for band in range_bands:
                        if band not in rm:
                            continue
                        bm = rm[band]
                        print(f"{weather:<8} {method:<6} {band:<10} {bm['n']:>5} "
                              f"{bm['AP']:6.3f} {bm['P_d']:6.3f} {bm['P_fa']:6.3f}")

        # -- Save CSV ----------------------------------------------------------
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, 'weather_results.csv')
        with open(out_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Weather', 'Method', 'AP', 'P_d', 'P_fa', 'Chamfer_Distance',
                             'Density_0-5m', 'Density_5-10m', 'Density_10-15m', 'Density_15-20m'])
            for weather in ['clear', 'fog', 'rain']:
                if weather not in all_results:
                    continue
                for method, m in all_results[weather].items():
                    d = m['density']
                    writer.writerow([
                        weather, method,
                        f"{m['AP']:.4f}", f"{m['P_d']:.4f}", f"{m['P_fa']:.4f}",
                        f"{m['CD']:.4f}" if not np.isnan(m['CD']) else 'N/A',
                        f"{d.get('0-5m',   np.nan):.4f}" if not np.isnan(d.get('0-5m',   np.nan)) else 'N/A',
                        f"{d.get('5-10m',  np.nan):.4f}" if not np.isnan(d.get('5-10m',  np.nan)) else 'N/A',
                        f"{d.get('10-15m', np.nan):.4f}" if not np.isnan(d.get('10-15m', np.nan)) else 'N/A',
                        f"{d.get('15-20m', np.nan):.4f}" if not np.isnan(d.get('15-20m', np.nan)) else 'N/A',
                    ])
            writer.writerow([])
            writer.writerow(['Weather', 'Method', 'Band', 'n_frames', 'AP', 'P_d', 'P_fa'])
            for weather in ['clear', 'fog', 'rain']:
                if weather not in all_results:
                    continue
                for method, m in all_results[weather].items():
                    for band, bm in m.get('range_metrics', {}).items():
                        writer.writerow([weather, method, band, bm['n'],
                                         f"{bm['AP']:.4f}", f"{bm['P_d']:.4f}", f"{bm['P_fa']:.4f}"])
        print(f"\n  Results saved -> {os.path.abspath(out_csv)}")

        # -- Frame match log ---------------------------------------------------
        if all_match_logs:
            log_csv = os.path.join(out_dir, 'frame_match_log.csv')
            log_fields = ['rc', 'type', 'dl_ts_ms', 'cfar_ts_ms',
                          'delta_ms', 'cfar_found', 'box_range_m', 'band']
            with open(log_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=log_fields)
                writer.writeheader()
                writer.writerows(all_match_logs)
            print(f"  Match log saved -> {os.path.abspath(log_csv)}")
            matched_n  = sum(1 for r in all_match_logs if r['type'] == 'matched')
            extended_n = sum(1 for r in all_match_logs if r['type'] == 'extended')
            found_n    = sum(1 for r in all_match_logs if r['cfar_found'])
            print(f"  matched={matched_n}  extended={extended_n}  cfar_found={found_n}/{matched_n+extended_n}")

    # -- Thesis figures --------------------------------------------------------
    tp_cfg         = config.get('thesis_plots', {})
    n_plots        = int(tp_cfg.get('n_plots', 5))
    raw_prediction = bool(tp_cfg.get('raw_prediction', True))
    thr_plot       = float(tp_cfg.get('threshold', threshold))
    all_rc = list(dict.fromkeys(
        rc for folders in weather_splits.values() for rc in folders
    ))

    if tp_cfg.get('enable', True):
        device_p = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_p  = _load_model(config, ckpt, device_p)
        print(f"\n{'='*60}")
        print(f"THESIS FIGURES  ({n_plots} frames per RC folder, "
              f"{'raw prediction' if raw_prediction else f'threshold={thr_plot}'})")
        print(f"{'='*60}")
        generate_thesis_plots(all_rc, base_dir, config, model_p, device_p,
                              out_dir, threshold=thr_plot, n_plots=n_plots,
                              raw_prediction=raw_prediction)

    if tp_cfg.get('camera_projection', False):
        device_p = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_p  = _load_model(config, ckpt, device_p)
        print(f"\n{'='*60}")
        print(f"CAMERA PROJECTION  ({n_plots} frames per RC folder, DL vs CFAR)")
        print(f"{'='*60}")
        generate_camera_projection_plots(
            all_rc, base_dir, config, model_p, device_p,
            threshold, out_dir, n_plots=n_plots)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

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
        print(f"ERROR: unknown eval_mode '{mode}' -- use 'basic' or 'weather'")


if __name__ == '__main__':
    main()
