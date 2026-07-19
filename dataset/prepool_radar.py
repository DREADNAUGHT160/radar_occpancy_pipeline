"""
Pre-pool radar cubes at extraction time and copy matched CFAR files.

For each RC folder listed in the config, reads .mat files from Radar/ and:
  - Power:     512 → 128 Doppler bins via max pooling (blocks of 4)
  - Elevation: 512 → 128 Doppler bins via stride (every 4th bin, matches original model_pipeline)
  - CFAR:      matched via calib Radar_frame field (mat ts → nearest calib →
               Radar_frame → data/radar/<ms>.txt), saved as cfar/<ts_sec>.txt

CFAR matching chain:
  .mat TimeStamp  →  nearest calib file (labels_new2/)
                  →  "Radar_frame" field in calib  →  data/radar/<ms>.txt
                  →  copied as cfar/<ts_sec>.txt

This mirrors how thesis_eval.py resolves CFAR: the calib file is the reliable
bridge between the radar cube timestamp and the CFAR point cloud file.

Outputs per RC folder:
  <base_dir>/<RC>/rad_power_pooled/<ts>.npy   — (128, H, W) max-pooled power
  <base_dir>/<RC>/rad_elev_pooled/<ts>.npy    — (128, H, W) max-pooled elevation
  <base_dir>/<RC>/cfar/<ts>.txt               — CFAR matched via calib
  <base_dir>/<RC>/calib/<orig_name>.txt       — calibration files (as-is)

Usage:
  python dataset/prepool_radar.py --config configs/prepare_config.yaml
  python dataset/prepool_radar.py --config configs/prepare_config.yaml --cfar_only
"""

import os
import glob
import shutil
import argparse
import yaml
import numpy as np
import h5py
from tqdm import tqdm
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_source_folder(raw_data_dir, rc_name):
    """Return the first subfolder of raw_data_dir matching *_<RCNAME>."""
    for pattern in [f"*_{rc_name}", f"*_{rc_name.upper()}"]:
        matches = glob.glob(os.path.join(raw_data_dir, pattern))
        if matches:
            return matches[0]
    return None


def pool_power(data):
    """Max-pool 512-bin power cube to 128 bins (blocks of 4, numpy)."""
    if data.ndim == 3 and data.shape[2] == 512:
        data = data.transpose(2, 0, 1)      # (H, W, 512) → (512, H, W)
    if data.shape[0] != 512:
        return data                          # already pooled / wrong shape
    blocks = data.reshape(128, 4, data.shape[1], data.shape[2])
    return blocks.max(axis=1)               # (128, H, W)


def stride_elev(data):
    """Stride 512-bin elevation cube to 128 bins (every 4th bin).

    Matches the original model_pipeline behaviour (see prepool_elev.py comment).
    Max-pool on elevation angle data would discard downward-pointing bins.
    """
    if data.ndim == 3 and data.shape[2] == 512:
        data = data.transpose(2, 0, 1)      # (H, W, 512) → (512, H, W)
    if data.shape[0] != 512:
        return data                          # already strided / wrong shape
    return data[::4, :, :]                  # (128, H, W)


def argmax_gather_elev(power, elev):
    """Pick elevation at the Doppler bin with maximum power within each block of 4.

    For each of the 128 output bins, finds which of the 4 input Doppler bins
    has the highest power, then reads elevation from that exact bin.
    This guarantees the elevation angle always corresponds to the actual detection.

    power: (512, H, W) — raw power cube
    elev:  (512, H, W) — raw elevation cube
    returns: (128, H, W)
    """
    H, W = power.shape[1], power.shape[2]
    p_blocks = power.reshape(128, 4, H, W)
    e_blocks = elev.reshape(128, 4, H, W)
    idx = p_blocks.argmax(axis=1)                              # (128, H, W)
    return np.take_along_axis(e_blocks,
                              idx[:, np.newaxis, :, :],
                              axis=1).squeeze(1)               # (128, H, W)


def build_calib_radar_index(calib_src_dir):
    """Build index: calib_ts_ms → Radar_frame filename, from data/labels_new2/.

    Each calib file contains a "Radar_frame" field naming the exact CFAR file
    that was synchronised with that camera frame. This is the reliable bridge
    between .mat timestamps and CFAR point clouds.
    """
    files = sorted(glob.glob(os.path.join(calib_src_dir, '*.txt')))
    ts_list    = []
    radar_refs = {}   # ts_ms → radar_frame filename (e.g. "1649837844448.txt")
    for f in files:
        m = re.search(r'(\d+\.\d+|\d+)', os.path.basename(f))
        if not m:
            continue
        val   = float(m.group(0))
        ts_ms = int(val * 1000) if val < 1e11 else int(val)
        try:
            content = open(f, encoding='utf-8', errors='ignore').read()
            rf = re.search(r'"Radar_frame"\s*:\s*"([^"]+)"', content)
            if rf:
                radar_refs[ts_ms] = rf.group(1)
                ts_list.append(ts_ms)
        except Exception:
            pass
    return radar_refs, np.array(ts_list, dtype=np.int64)


def find_cfar_via_calib(radar_refs, calib_ts, mat_ts_ms, cfar_src_dir,
                        threshold_ms=100):
    """mat timestamp → nearest calib → Radar_frame → CFAR file path."""
    if len(calib_ts) == 0:
        return None
    diffs = np.abs(calib_ts - mat_ts_ms)
    best  = int(np.argmin(diffs))
    if diffs[best] > threshold_ms:
        return None
    radar_frame = radar_refs[calib_ts[best]]          # e.g. "1649837844448.txt"
    cfar_path   = os.path.join(cfar_src_dir, radar_frame)
    return cfar_path if os.path.exists(cfar_path) else None


# ── Calib copy ────────────────────────────────────────────────────────────────

def copy_calib(src_folder, dst_folder, calib_sub='calib'):
    """Copy data/labels_new2/*.txt → calib/ keeping original filenames."""
    calib_src = os.path.join(src_folder, 'data', 'labels_new2')
    calib_dst = os.path.join(dst_folder, calib_sub)
    if not os.path.isdir(calib_src):
        print(f"    [WARN] No labels_new2 folder at {calib_src} — calib/ skipped")
        return 0, 0
    os.makedirs(calib_dst, exist_ok=True)
    files = sorted(glob.glob(os.path.join(calib_src, '*.txt')))
    copied = skipped = 0
    for src in files:
        dst = os.path.join(calib_dst, os.path.basename(src))
        if os.path.exists(dst):
            skipped += 1
        else:
            shutil.copy2(src, dst)
            copied += 1
    return copied, skipped


# ── Per-RC processing ─────────────────────────────────────────────────────────

def process_rc(src_folder, dst_folder, cfar_sub='cfar', calib_sub='calib',
               cfar_only=False):
    radar_src  = os.path.join(src_folder, 'Radar')
    cfar_src   = os.path.join(src_folder, 'data', 'radar')
    calib_src  = os.path.join(src_folder, 'data', 'labels_new2')

    mat_files = sorted(glob.glob(os.path.join(radar_src, '*.mat')))
    if not mat_files:
        print(f"    [WARN] No .mat files in {radar_src}")
        return

    out_power        = os.path.join(dst_folder, 'rad_power_pooled')
    out_elev         = os.path.join(dst_folder, 'rad_elev_pooled')    # stride
    out_elev_max     = os.path.join(dst_folder, 'rad_elev_maxpool')   # max-pool
    out_elev_argmax  = os.path.join(dst_folder, 'rad_elev_argmax')    # argmax-gather
    out_cfar         = os.path.join(dst_folder, cfar_sub)
    os.makedirs(out_power,       exist_ok=True)
    os.makedirs(out_elev,        exist_ok=True)
    os.makedirs(out_elev_max,    exist_ok=True)
    os.makedirs(out_elev_argmax, exist_ok=True)
    os.makedirs(out_cfar,        exist_ok=True)

    # Build calib→Radar_frame index for CFAR matching
    radar_refs, calib_ts = build_calib_radar_index(calib_src)
    has_calib = len(calib_ts) > 0
    if not has_calib:
        print(f"    [WARN] No calib files with Radar_frame in {calib_src} — cfar/ skipped")

    print(f"    .mat files   : {len(mat_files)}")
    print(f"    calib frames : {len(calib_ts)}  (used for CFAR matching)")
    print(f"    CFAR src     : {cfar_src}")

    saved = skipped = errors = 0
    cfar_copied = cfar_skipped = cfar_miss = 0

    for mat_path in tqdm(mat_files, unit='frame', leave=False):
        try:
            with h5py.File(mat_path, 'r') as f:
                ts_ms = int(f['TimeStamp'][0, 0])   # ms from .mat
                ts    = ts_ms / 1000.0              # seconds — output filename

                p_out  = os.path.join(out_power,       f'{ts}.npy')
                e_out  = os.path.join(out_elev,        f'{ts}.npy')
                em_out = os.path.join(out_elev_max,    f'{ts}.npy')
                ea_out = os.path.join(out_elev_argmax, f'{ts}.npy')
                c_out  = os.path.join(out_cfar,        f'{ts}.txt')

                # Radar cubes — skip if cfar_only mode
                if not cfar_only:
                    if os.path.exists(p_out) and os.path.exists(e_out) \
                            and os.path.exists(em_out) and os.path.exists(ea_out):
                        skipped += 1
                    else:
                        elev_raw  = np.array(f['RAD_elev'],  dtype=np.float32)
                        power_raw = np.array(f['RAD_power'], dtype=np.float32)
                        # ensure (D, H, W)
                        if elev_raw.ndim == 3 and elev_raw.shape[2] == 512:
                            elev_raw = elev_raw.transpose(2, 0, 1)
                        if power_raw.ndim == 3 and power_raw.shape[2] == 512:
                            power_raw = power_raw.transpose(2, 0, 1)
                        power  = pool_power(power_raw)
                        elev_s = stride_elev(elev_raw)
                        elev_m = elev_raw.reshape(128, 4,
                                                   elev_raw.shape[1],
                                                   elev_raw.shape[2]).max(axis=1)
                        elev_a = argmax_gather_elev(power_raw, elev_raw)
                        np.save(p_out,  power)
                        np.save(e_out,  elev_s)
                        np.save(em_out, elev_m)
                        np.save(ea_out, elev_a)
                        saved += 1

                # CFAR: mat ts → nearest calib → Radar_frame → data/radar/
                if os.path.exists(c_out):
                    cfar_skipped += 1
                elif has_calib:
                    cfar_path = find_cfar_via_calib(radar_refs, calib_ts,
                                                    ts_ms, cfar_src)
                    if cfar_path:
                        shutil.copy2(cfar_path, c_out)
                        cfar_copied += 1
                    else:
                        cfar_miss += 1

        except Exception as e:
            print(f"    [ERROR] {os.path.basename(mat_path)}: {e}")
            errors += 1

    if not cfar_only:
        print(f"    rad_power_pooled/  : saved={saved}  skipped={skipped}  errors={errors}")
        print(f"    rad_elev_pooled/   : saved={saved}  skipped={skipped}  errors={errors}")
        print(f"    rad_elev_maxpool/  : saved={saved}  skipped={skipped}  errors={errors}")
        print(f"    rad_elev_argmax/   : saved={saved}  skipped={skipped}  errors={errors}")
    print(f"    cfar/             : copied={cfar_copied}  skipped={cfar_skipped}  "
          f"no_match={cfar_miss}")
    if cfar_miss > 0:
        print(f"    [NOTE] {cfar_miss} mat frames had no calib match within 100 ms")

    # Calib files (needed for eval bounding-box metrics)
    c, s = copy_calib(src_folder, dst_folder, calib_sub)
    print(f"    calib/            : copied={c}  skipped={s}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Pre-pool radar cubes and copy matched CFAR files.'
    )
    parser.add_argument('--config', default='configs/prepare_config.yaml',
                        help='Path to prepare_config.yaml')
    parser.add_argument('--cfar_only', action='store_true',
                        help='Skip radar cube extraction; only redo CFAR matching')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    ds_cfg       = config.get('dataset', {})
    raw_data_dir = ds_cfg.get('raw_data_dir', '').strip()
    base_dir     = ds_cfg.get('base_dir',     '').strip()
    sf           = ds_cfg.get('subfolders',   {})
    cfar_sub     = sf.get('cfar',  'cfar')
    calib_sub    = sf.get('calib', 'calib')

    if not raw_data_dir or not os.path.isdir(raw_data_dir):
        print(f"ERROR: dataset.raw_data_dir not set or missing: '{raw_data_dir}'")
        return
    if not base_dir:
        print("ERROR: dataset.base_dir is not set in config.")
        return

    rc_names = list(dict.fromkeys(
        ds_cfg.get('train', []) + ds_cfg.get('val', []) + ds_cfg.get('test', [])
    ))
    if not rc_names:
        print("No RC folders listed under dataset.train/val/test.")
        return

    print(f"raw_data_dir : {raw_data_dir}")
    print(f"base_dir     : {base_dir}")
    print(f"RC folders   : {rc_names}")
    print(f"Pooling      : power=max(blocks of 4)  elevation=stride(4)")
    print(f"CFAR         : mat ts -> nearest calib -> Radar_frame -> data/radar/")
    if args.cfar_only:
        print(f"Mode         : --cfar_only (radar cubes skipped)\n")
    else:
        print()

    for rc_name in rc_names:
        print(f"{'='*60}")
        print(f"  {rc_name}")
        print(f"{'='*60}")

        src = find_source_folder(raw_data_dir, rc_name)
        if not src:
            print(f"  [SKIP] No source folder for {rc_name} in {raw_data_dir}")
            continue
        print(f"  Source : {src}")

        dst = os.path.join(base_dir, rc_name)
        process_rc(src, dst, cfar_sub=cfar_sub, calib_sub=calib_sub,
                   cfar_only=args.cfar_only)
        print()

    print("Done.")


if __name__ == '__main__':
    main()
