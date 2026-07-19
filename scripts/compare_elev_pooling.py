"""
Compare stride vs max-pool elevation downsampling.

Layout per frame (2 rows x 3 cols):
  Row 0 (top):    Power BEV  |  AE Map (STRIDE)   |  RE Map (STRIDE)
  Row 1 (bottom): [blank]    |  AE Map (MAX-POOL)  |  RE Map (MAX-POOL)

Data sources (auto-detected per RC folder):
  - If rad_elev/ (512-depth) exists: compute both from raw
  - Else uses rad_elev_pooled/ (stride, 128) + rad_elev_maxpool/ (max-pool, 128)

Usage:
  python scripts/compare_elev_pooling.py --config configs/eval_datamasters_fog.yaml
  python scripts/compare_elev_pooling.py --config configs/eval_rc011_pool.yaml
"""

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Elevation pooling ────────────────────────────────────────────────────────

def stride_elev(data):
    if data.shape[0] == 512:
        return data[::4]
    return data

def maxpool_elev(data):
    if data.shape[0] == 512:
        blocks = data.reshape(128, 4, data.shape[1], data.shape[2])
        return blocks.max(axis=1)
    return data


# ── Power normalization ──────────────────────────────────────────────────────

def normalize_power(power, mn=-100.0, mx=-31.7):
    db = 10.0 * np.log10(power + 1e-10)
    db = np.clip(db, mn, mx)
    return (db - mn) / (mx - mn)


# ── AE / RE maps ─────────────────────────────────────────────────────────────

def compute_ae_re(power_norm, elev_128, max_ang=0.7854, num_e=64):
    e_norm = np.clip(elev_128 / (max_ang + 1e-9), -1.0, 1.0)
    e_bins = ((e_norm + 1.0) / 2.0 * (num_e - 1)).clip(0, num_e - 1).astype(int)

    p_mov = power_norm[2:]
    e_mov = e_bins[2:]
    d, r, a = p_mov.shape

    ae = np.zeros((num_e, a)); ac = np.zeros((num_e, a))
    ag = np.broadcast_to(np.arange(a)[np.newaxis, np.newaxis, :], (d, r, a))
    np.add.at(ae, (e_mov.ravel(), ag.ravel()), p_mov.ravel())
    np.add.at(ac, (e_mov.ravel(), ag.ravel()), 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        ae_map = np.where(ac > 0, ae / ac, 0).astype(np.float32)
    if ae_map.max() > 0: ae_map /= ae_map.max()

    re = np.zeros((num_e, r)); rc2 = np.zeros((num_e, r))
    rg = np.broadcast_to(np.arange(r)[np.newaxis, :, np.newaxis], (d, r, a))
    np.add.at(re, (e_mov.ravel(), rg.ravel()), p_mov.ravel())
    np.add.at(rc2, (e_mov.ravel(), rg.ravel()), 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        re_map = np.where(rc2 > 0, re / rc2, 0).astype(np.float32)
    if re_map.max() > 0: re_map /= re_map.max()

    return ae_map, re_map


# ── Per-frame plot ────────────────────────────────────────────────────────────

def save_comparison(rc_name, ts_str, frame_idx, power_norm,
                    elev_stride, elev_max, out_path, max_ang=0.7854):
    ae_s, re_s = compute_ae_re(power_norm, elev_stride, max_ang)
    ae_m, re_m = compute_ae_re(power_norm, elev_max,    max_ang)

    bev = power_norm.max(axis=0)
    xt  = np.linspace(0, power_norm.shape[2] - 1, 7)
    xl  = np.round(np.degrees(np.arcsin(np.linspace(-1, 1, 7)))).astype(int)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"{rc_name}  |  Frame {frame_idx:04d}  |  ts={ts_str}", fontsize=12)

    def show(ax, data, title, xlabel, ylabel, az_ticks=False):
        vmax = float(data.max()) if data.max() > 0 else 1.0
        im = ax.imshow(data, cmap='turbo', origin='lower', aspect='auto',
                       vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        if az_ticks:
            ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 0 — STRIDE
    show(axes[0, 0], bev,  'Power BEV (max Doppler)',
         'Azimuth (deg)', 'Range (Bins)', az_ticks=True)
    show(axes[0, 1], ae_s, 'AE Map — STRIDE elev [::4]',
         'Azimuth (deg)', 'Elevation (Bins)', az_ticks=True)
    show(axes[0, 2], re_s, 'RE Map — STRIDE elev [::4]',
         'Range (Bins)', 'Elevation (Bins)')

    # Row 1 — MAX-POOL
    axes[1, 0].axis('off')
    show(axes[1, 1], ae_m, 'AE Map — MAX-POOL elev (blocks of 4)',
         'Azimuth (deg)', 'Elevation (Bins)', az_ticks=True)
    show(axes[1, 2], re_m, 'RE Map — MAX-POOL elev (blocks of 4)',
         'Range (Bins)', 'Elevation (Bins)')

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=120)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',  default='configs/eval_datamasters_fog.yaml')
    parser.add_argument('--out_dir', default='verification_output/elev_pool_compare')
    args = parser.parse_args()

    import yaml
    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    base_dir = cfg.get('base_dir', '').strip()
    max_ang  = cfg.get('normalization', {}).get('elevation_max_angle', 0.7854)
    pwr_min  = cfg.get('normalization', {}).get('power_min_val', -100.0)
    pwr_max  = cfg.get('normalization', {}).get('power_max_val', -31.7)
    splits   = cfg.get('eval_splits', {})
    rc_names = [rc for rcs in splits.values() for rc in rcs]

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\nConfig    : {args.config}")
    print(f"Base dir  : {base_dir}")
    print(f"RC folders: {rc_names}")
    print(f"Out dir   : {args.out_dir}\n")

    for rc_name in rc_names:
        rc_dir = os.path.join(base_dir, rc_name) if base_dir else rc_name

        # Power: prefer pooled (128) else raw
        power_dir = os.path.join(rc_dir, 'rad_power_pooled')
        if not os.path.isdir(power_dir):
            power_dir = os.path.join(rc_dir, 'rad_power')

        # Elevation sources
        raw_elev_dir    = os.path.join(rc_dir, 'rad_elev')          # 512-depth
        stride_elev_dir = os.path.join(rc_dir, 'rad_elev_pooled')   # 128 stride
        max_elev_dir    = os.path.join(rc_dir, 'rad_elev_maxpool')  # 128 max-pool

        has_raw    = os.path.isdir(raw_elev_dir)    and len(glob.glob(raw_elev_dir    + '/*.npy')) > 0
        has_stride = os.path.isdir(stride_elev_dir) and len(glob.glob(stride_elev_dir + '/*.npy')) > 0
        has_max    = os.path.isdir(max_elev_dir)    and len(glob.glob(max_elev_dir    + '/*.npy')) > 0

        if has_raw:
            elev_s_dir = raw_elev_dir
            elev_m_dir = raw_elev_dir
            use_raw = True
            print(f"{rc_name}: using raw rad_elev/ (512) for both stride + max-pool")
        elif has_stride and has_max:
            elev_s_dir = stride_elev_dir
            elev_m_dir = max_elev_dir
            use_raw = False
            print(f"{rc_name}: using rad_elev_pooled/ (stride) + rad_elev_maxpool/")
        else:
            print(f"[SKIP] {rc_name}: no elevation source found")
            continue

        power_files = sorted(glob.glob(os.path.join(power_dir, '*.npy')))
        elev_s_map  = {os.path.basename(f): f for f in glob.glob(elev_s_dir + '/*.npy')}
        elev_m_map  = {os.path.basename(f): f for f in glob.glob(elev_m_dir + '/*.npy')}

        matched = [(pf, elev_s_map[os.path.basename(pf)], elev_m_map[os.path.basename(pf)])
                   for pf in power_files
                   if os.path.basename(pf) in elev_s_map
                   and os.path.basename(pf) in elev_m_map]

        if not matched:
            print(f"[SKIP] {rc_name}: no matching files")
            continue

        out_rc = os.path.join(args.out_dir, rc_name.replace('/', '_'))
        os.makedirs(out_rc, exist_ok=True)
        print(f"  {len(matched)} frames -> {out_rc}")

        skipped_corrupt = 0
        for idx, (pf, ef_s, ef_m) in enumerate(tqdm(matched, desc=rc_name, unit='frame')):
            ts_str   = os.path.splitext(os.path.basename(pf))[0]
            out_path = os.path.join(out_rc, f'frame_{idx:04d}_{ts_str}.png')

            if os.path.exists(out_path):
                continue  # already rendered

            try:
                power_raw = np.load(pf).astype(np.float32)
                es_raw    = np.load(ef_s).astype(np.float32)
                em_raw    = np.load(ef_m).astype(np.float32)
            except (ValueError, OSError) as e:
                tqdm.write(f'  [SKIP] frame {idx:04d} {ts_str}: {e}')
                skipped_corrupt += 1
                continue

            # Ensure (D, R, A)
            for arr_name, arr in [('power', power_raw), ('elev_s', es_raw), ('elev_m', em_raw)]:
                pass  # done below per-array
            if power_raw.ndim == 3 and power_raw.shape[2] in (512, 128):
                power_raw = power_raw.transpose(2, 0, 1)
            if es_raw.ndim == 3 and es_raw.shape[2] in (512, 128):
                es_raw = es_raw.transpose(2, 0, 1)
            if em_raw.ndim == 3 and em_raw.shape[2] in (512, 128):
                em_raw = em_raw.transpose(2, 0, 1)

            # Pool power if 512
            if power_raw.shape[0] == 512:
                power_raw = power_raw.reshape(128, 4, power_raw.shape[1],
                                              power_raw.shape[2]).max(axis=1)

            # Apply pooling to raw elevation if using 512-depth source
            if use_raw:
                elev_s = stride_elev(es_raw)
                elev_m = maxpool_elev(em_raw)
            else:
                elev_s = es_raw   # already 128 stride
                elev_m = em_raw   # already 128 max-pool

            power_norm = normalize_power(power_raw, pwr_min, pwr_max)
            save_comparison(rc_name, ts_str, idx, power_norm,
                            elev_s, elev_m, out_path, max_ang)

    print(f"\nDone. Plots saved to {args.out_dir}")


if __name__ == '__main__':
    main()
