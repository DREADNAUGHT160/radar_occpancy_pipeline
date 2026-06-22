"""
Step 0 - Prepare all datasets for training.

Reads the config file and, for every RC folder listed under dataset.train/val/test,
runs three preparation steps:

  Step 1  Radar extraction    .mat -> rad_power/ + rad_elev/
  Step 2  LiDAR labels        .h5  -> labels/
  Step 3  Camera assets       .png -> pco/     .txt -> calib/

Skips any file that already exists. After all datasets are processed, the
inference.camera section of the config YAML is auto-updated with pco_dir and
label_txt_dir entries for every prepared dataset.

Usage:
  python dataset/prepare_dataset.py --config configs/prepare_config.yaml
"""

import os
import re
import glob
import shutil
import argparse
import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

from dataset.extract_mat_to_npy import extract as extract_radar
from dataset.generate_lidar_labels import sync_timestamps, process_lidar_h5


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_source_folder(raw_data_dir, rc_name):
    """Return the first subfolder of raw_data_dir matching *_<RCNAME>."""
    pattern = os.path.join(raw_data_dir, f"*_{rc_name}")
    matches = glob.glob(pattern)
    if not matches:
        pattern_ci = os.path.join(raw_data_dir, f"*_{rc_name.upper()}")
        matches = glob.glob(pattern_ci)
    return matches[0] if matches else None


def copy_files(src_dir, dst_dir, pattern="*"):
    """Copy files matching pattern from src_dir to dst_dir; skip existing."""
    os.makedirs(dst_dir, exist_ok=True)
    files = glob.glob(os.path.join(src_dir, pattern))
    copied, skipped = 0, 0
    for src in files:
        dst = os.path.join(dst_dir, os.path.basename(src))
        if os.path.exists(dst):
            skipped += 1
        else:
            shutil.copy2(src, dst)
            copied += 1
    return copied, skipped


def generate_labels(radar_npy_dir, lidar_pcd_dir, out_dir, sync_threshold):
    """Run LiDAR label generation using shared functions."""
    os.makedirs(out_dir, exist_ok=True)

    radar_files = sorted(glob.glob(os.path.join(radar_npy_dir, 'rad_power', '*.npy')))
    lidar_files = sorted(glob.glob(os.path.join(lidar_pcd_dir, '*.h5')))

    if not radar_files:
        print(f"    [WARN] No radar npy files found in {radar_npy_dir}/rad_power/")
        return 0, 0, 0, 0
    if not lidar_files:
        print(f"    [WARN] No LiDAR .h5 files found in {lidar_pcd_dir}")
        return 0, 0, 0, 0

    matched = sync_timestamps(radar_files, lidar_files, sync_threshold)
    print(f"    Synced {len(matched)}/{len(radar_files)} radar-LiDAR pairs "
          f"(threshold {sync_threshold} ms)")
    if not matched:
        return 0, 0, 0, len(radar_files)

    saved, skipped = 0, 0
    for m in tqdm(matched, desc="    Labels", unit="frame", leave=False):
        out_path = os.path.join(out_dir, f"{m['r_ts']}.npy")
        if os.path.exists(out_path):
            skipped += 1
            continue
        grid = process_lidar_h5(m['lidar'])
        if grid is not None:
            np.save(out_path, grid)
            saved += 1

    return saved, skipped, len(matched), len(radar_files)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/prepare_config.yaml',
                        help='Path to prepare_config.yaml (or train_config.yaml)')
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    ds_cfg       = config.get('dataset', {})
    raw_data_dir = ds_cfg.get('raw_data_dir', '').strip()
    base_dir     = ds_cfg.get('base_dir', '').strip()
    threshold    = ds_cfg.get('sync_threshold_ms', 100)
    sf           = ds_cfg.get('subfolders', {})

    if not raw_data_dir or not os.path.isdir(raw_data_dir):
        print(f"ERROR: dataset.raw_data_dir is not set or does not exist: '{raw_data_dir}'")
        print("Set it in your config file and re-run.")
        return

    if not base_dir:
        print("ERROR: dataset.base_dir is not set in config.")
        return

    # Collect unique RC names across all splits
    rc_names = list(dict.fromkeys(
        ds_cfg.get('train', []) + ds_cfg.get('val', []) + ds_cfg.get('test', [])
    ))
    if not rc_names:
        print("No datasets listed under dataset.train/val/test in config.")
        return

    print(f"raw_data_dir : {raw_data_dir}")
    print(f"base_dir     : {base_dir}")
    print(f"Datasets     : {rc_names}\n")

    prepared = []

    for rc_name in rc_names:
        print(f"{'='*60}")
        print(f"  {rc_name}")
        print(f"{'='*60}")

        src_folder = find_source_folder(raw_data_dir, rc_name)
        if not src_folder:
            print(f"  [SKIP] No source folder found for {rc_name} in {raw_data_dir}")
            continue
        print(f"  Source : {src_folder}")

        dst_folder = os.path.join(base_dir, rc_name)
        radar_src  = os.path.join(src_folder, 'Radar')
        lidar_src  = os.path.join(src_folder, 'data', 'pcd')
        pco_src    = os.path.join(src_folder, 'data', 'pco')
        calib_src  = os.path.join(src_folder, 'data', 'labels_new2')

        # Resolve destination subfolder names from config (or use defaults)
        rp_sub    = sf.get('rad_power', 'rad_power')
        lbl_sub   = sf.get('labels',    'labels')
        pco_sub   = sf.get('pco',       'pco')
        calib_sub = sf.get('calib',     'calib')
        cfar_sub  = sf.get('cfar',      'cfar')

        # -- Step 1: Radar extraction ------------------------------------------
        print(f"  Step 1: Radar extraction")
        if os.path.isdir(radar_src) and glob.glob(os.path.join(radar_src, '*.mat')):
            extract_radar(radar_src, dst_folder)
        else:
            existing = len(glob.glob(os.path.join(dst_folder, rp_sub, '*.npy')))
            if existing:
                print(f"    Already extracted ({existing} frames) — skipping.")
            else:
                print(f"    [WARN] No .mat files in {radar_src}")

        # -- Step 2: LiDAR labels ----------------------------------------------
        n_matched, n_radar = 0, 0
        print(f"  Step 2: LiDAR label generation")
        if os.path.isdir(lidar_src):
            saved, skipped, n_matched, n_radar = generate_labels(
                dst_folder, lidar_src, os.path.join(dst_folder, lbl_sub), threshold)
            print(f"    Saved={saved}  Skipped={skipped}")
        else:
            n_radar   = len(glob.glob(os.path.join(dst_folder, rp_sub, '*.npy')))
            n_matched = len(glob.glob(os.path.join(dst_folder, lbl_sub, '*.npy')))
            if n_radar:
                print(f"    Already generated ({n_matched} labels) — skipping.")
            else:
                print(f"    [WARN] No LiDAR pcd folder at {lidar_src}")

        # -- Step 3: Camera images and calibration files -----------------------
        print(f"  Step 3: Camera assets")
        pco_dst   = os.path.join(dst_folder, pco_sub)
        calib_dst = os.path.join(dst_folder, calib_sub)

        if os.path.isdir(pco_src):
            c, s = copy_files(pco_src, pco_dst, '*.png')
            print(f"    pco/   : copied={c}  skipped={s}")
        else:
            print(f"    [WARN] No pco folder at {pco_src}")

        if os.path.isdir(calib_src):
            c, s = copy_files(calib_src, calib_dst, '*.txt')
            print(f"    calib/ : copied={c}  skipped={s}")
        else:
            print(f"    [WARN] No labels_new2 folder at {calib_src}")

        # -- Step 4: CFAR point clouds -----------------------------------------
        # Source: data/radar/*.txt  (SAVEROAD export: X,Y,Z,Doppler,Power per row)
        # Dest:   <dst>/cfar/*.txt
        # The evaluator applies Y-axis negation automatically to align with
        # the LiDAR coordinate frame used by bounding boxes and DL predictions.
        print(f"  Step 4: CFAR point clouds")
        cfar_src = os.path.join(src_folder, 'data', 'radar')
        cfar_dst = os.path.join(dst_folder, cfar_sub)
        if os.path.isdir(cfar_src):
            c, s = copy_files(cfar_src, cfar_dst, '*.txt')
            print(f"    cfar/  : copied={c}  skipped={s}")
        else:
            print(f"    [WARN] No radar folder at {cfar_src} — CFAR will be skipped in eval")

        if n_radar:
            print(f"  >> {n_matched}/{n_radar} radar frames have a matching LiDAR frame")

        prepared.append(rc_name)
        print()

    print(f"\nDone. Prepared datasets: {prepared}")


if __name__ == '__main__':
    main()
