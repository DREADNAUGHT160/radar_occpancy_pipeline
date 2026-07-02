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

    try:
        return float(np.trapezoid(prec, rec))
    except AttributeError:
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

    # Point density: group frames by box_range band, compute pts_in_box/box_volume per frame
    bands       = ((0, 5), (5, 10), (10, 15), (15, 20))
    density_acc = {f"{a}-{b}m": [] for a, b in bands}
    for fd in frames:
        if fd['pts'] is None or fd['box_corners'] is None or np.isnan(fd.get('box_range', np.nan)):
            continue
        scores = fd['scores']
        mask   = (scores >= threshold) if scores is not None else np.ones(len(fd['pts']), dtype=bool)
        pts_t  = fd['pts'][mask]
        if len(pts_t) == 0:
            continue
        pts_in = pts_t[points_in_box(pts_t, fd['box_corners'])]
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


def collect_frames(rc_folders, base_dir, config, model, device, threshold, weather):
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

        # CFAR lives at base_dir/<rc_name>/<subfolders.cfar>/ (set in eval_config.yaml)
        cfar_rc = os.path.join(rc_dir, sf.get('cfar', 'cfar'))
        if not os.path.isdir(cfar_rc) or not (
                glob.glob(os.path.join(cfar_rc, '*.npy')) +
                glob.glob(os.path.join(cfar_rc, '*.txt'))):
            cfar_rc = ''

        for idx in tqdm(range(len(ds)), desc=rc_name):
            radar_tensor, label_tensor = ds[idx]
            sample = ds.matched_data[idx]
            ts_ms  = _extract_ts_ms(sample['power'])

            corners, r2l = _find_calib(txt_files, txt_ts, ts_ms)
            box_volume   = 0.0
            box_range    = np.nan
            if corners is not None:
                dims       = corners.max(axis=0) - corners.min(axis=0)
                box_volume = float(dims[0] * dims[1] * dims[2])
                center     = (corners.max(axis=0) + corners.min(axis=0)) / 2
                box_range  = float(np.sqrt(center[0]**2 + center[1]**2))

            with torch.no_grad():
                pred_np = torch.sigmoid(
                    model(radar_tensor.unsqueeze(0).to(device))
                )[0].cpu().numpy()

            # Use a small floor threshold to discard obvious background voxels;
            # this bounds memory use in compute_ap (trapz over all sorted points)
            # while preserving the full shape of the PR curve above 5% confidence.
            pts_dl, scores_dl = occupancy_to_points(pred_np, threshold=0.05)
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
                'box_volume': box_volume, 'box_range': box_range,
            })

            if cfar_rc:
                pts_c, scores_c = _load_cfar(cfar_rc, ts_ms)
                # Radar .txt files use opposite Y axis vs LiDAR frame (right-hand vs
                # left-hand convention in the SAVEROAD export). Negate Y to align with
                # the LiDAR-frame bounding boxes used for point-in-box evaluation.
                if pts_c is not None and len(pts_c):
                    pts_c = pts_c.copy()
                    pts_c[:, 1] = -pts_c[:, 1]
                cfar_frames.append({
                    'pts': pts_c, 'scores': scores_c,
                    'box_corners': corners, 'lidar_pts': lidar_pts_in_box,
                    'box_volume': box_volume, 'box_range': box_range,
                })

    return dl_frames, cfar_frames


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
                      rc_name, ts_str, frame_idx, out_path, saveroad_dir=''):
    if not saveroad_dir or not os.path.isdir(saveroad_dir):
        return
    try:
        import cv2
    except ImportError:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from utils.project_to_image import project_to_image

    sys.path.insert(0, saveroad_dir)
    try:
        from tools import project_points_v2_withPC as project_points_mod
    except ImportError:
        return

    img_name, K, r_t, radar_to_lidar = parse_calibration(calib_txt)
    if not img_name:
        return
    img_path = os.path.join(pco_dir, img_name)
    if not os.path.exists(img_path):
        return

    pts_3d, probs = occupancy_to_points(pred_np, threshold)
    if len(pts_3d) == 0:
        return
    pts_3d = pts_3d + radar_to_lidar
    if len(pts_3d) > 150_000:
        top    = np.argsort(probs)[-150_000:]
        pts_3d = pts_3d[top]; probs = probs[top]

    img = cv2.imread(img_path)
    if img is None:
        return

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


def generate_thesis_plots(rc_folders, base_dir, config, model, device,
                          out_dir, threshold=0.4, n_plots=5, raw_prediction=True):
    """Generate n_plots equally spaced prediction + camera frames per RC folder."""
    import matplotlib
    matplotlib.use('Agg')

    sf              = config.get('subfolders', {})
    saveroad_dir    = config.get('saveroad_dir', '').strip()
    tp_cfg          = config.get('thesis_plots', {})
    do_camera       = bool(tp_cfg.get('camera_projection', True)) and bool(saveroad_dir)

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
                                saveroad_dir=saveroad_dir
                            )
                        except Exception as e:
                            print(f"    [WARN] Camera frame {idx}: {e}")

            except Exception as e:
                print(f"    [WARN] Frame {idx} skipped: {e}")

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = _load_model(config, ckpt, device)
    print(f"\nWeather eval -- threshold={threshold}  device={device}")

    all_results = {}

    for weather, rc_folders in weather_splits.items():
        if not rc_folders:
            continue
        print(f"\n{'='*60}")
        print(f"  {weather.upper()} -- {rc_folders}")
        print('='*60)

        dl_frames, cfar_frames = collect_frames(
            rc_folders, base_dir, config, model, device,
            threshold, weather)

        print(f"  {len(dl_frames)} frames collected")
        dl_m = compute_weather_metrics(dl_frames, threshold, weather)
        all_results[weather] = {'DL': dl_m}

        if cfar_frames and any(f['pts'] is not None for f in cfar_frames):
            cfar_m = compute_weather_metrics(cfar_frames, 0.5, weather)
            all_results[weather]['CFAR'] = cfar_m

    # -- Results table ---------------------------------------------------------
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

    # -- Degradation -----------------------------------------------------------
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
                c = all_results['clear'][weather]    if False else all_results['clear'][method]
                w = all_results[weather][method]
                print(f"{weather:<8} {method:<6} "
                      f"{_deg(c['AP'], w['AP'])} "
                      f"{_deg(c['P_d'], w['P_d'])} "
                      f"{_deg(c['P_fa'], w['P_fa'])}")

    # -- Point density ---------------------------------------------------------
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

    # -- Range-band P_d / AP / P_fa -------------------------------------------
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

    # -- Save CSV --------------------------------------------------------------
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
        # Range-band rows
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

    # -- Thesis figures --------------------------------------------------------
    tp_cfg         = config.get('thesis_plots', {})
    n_plots        = int(tp_cfg.get('n_plots', 5))
    raw_prediction = bool(tp_cfg.get('raw_prediction', True))
    thr_plot       = float(tp_cfg.get('threshold', threshold))
    if tp_cfg.get('enable', True):
        device_p = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_p  = _load_model(config, ckpt, device_p)
        all_rc   = list(dict.fromkeys(
            rc for folders in weather_splits.values() for rc in folders
        ))
        print(f"\n{'='*60}")
        print(f"THESIS FIGURES  ({n_plots} frames per RC folder, "
              f"{'raw prediction' if raw_prediction else f'threshold={thr_plot}'})")
        print(f"{'='*60}")
        generate_thesis_plots(all_rc, base_dir, config, model_p, device_p,
                              out_dir, threshold=thr_plot, n_plots=n_plots,
                              raw_prediction=raw_prediction)


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
