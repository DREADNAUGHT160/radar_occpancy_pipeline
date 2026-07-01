"""
Copy updated calibration files and CFAR point clouds from raw SAVEROAD source
folders into an existing prepared dataset.

Run this when:
  - Calibration files have been updated in the raw data (overwrites existing calib/)
  - CFAR point clouds are missing from the prepared dataset

Source layout (SAVEROAD raw collection):
  <raw_data_dir>/<NN>_<RC>/
    data/labels_new2/   <- calibration .txt files
    data/radar/         <- CFAR point cloud .txt files

Destination layout (prepared dataset):
  <base_dir>/<RC>/
    calib/              <- calibration files (overwritten if updated)
    cfar/               <- CFAR files (skipped if already present)

Usage:
  python dataset/update_calib_cfar.py --config configs/eval_weather.yaml
  python dataset/update_calib_cfar.py --config configs/eval_weather.yaml --rc RC019
"""

import os
import re
import glob
import shutil
import argparse
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))


def find_source_folder(raw_data_dir, rc_name):
    """Find the raw SAVEROAD collection folder matching *_<RC> pattern."""
    pattern = os.path.join(raw_data_dir, f'*_{rc_name}')
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def copy_files(src_dir, dst_dir, pattern, overwrite=False):
    os.makedirs(dst_dir, exist_ok=True)
    copied = skipped = 0
    for src in glob.glob(os.path.join(src_dir, pattern)):
        dst = os.path.join(dst_dir, os.path.basename(src))
        if os.path.exists(dst) and not overwrite:
            skipped += 1
        else:
            shutil.copy2(src, dst)
            copied += 1
    return copied, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/eval_weather.yaml')
    parser.add_argument('--rc',     default=None,
                        help='Single RC folder to update (e.g. RC019). '
                             'Default: all RC folders in eval_splits.')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    raw_data_dir = config.get('raw_data_dir', '').strip()
    base_dir     = config.get('base_dir', '').strip()
    sf           = config.get('subfolders', {})
    calib_sub    = sf.get('calib', 'calib')
    cfar_sub     = sf.get('cfar',  'cfar')

    # Source subfolder paths inside the raw SAVEROAD collection
    calib_src_sub = 'data/labels_new2'
    cfar_src_sub  = config.get('cfar_src_subfolder', 'data/radar')

    if not raw_data_dir:
        print("ERROR: raw_data_dir is not set in the config.")
        return
    if not base_dir:
        print("ERROR: base_dir is not set in the config.")
        return

    # Collect RC names to process
    if args.rc:
        rc_names = [args.rc]
    else:
        splits = config.get('eval_splits') or config.get('weather', {}).get('weather_splits', {})
        rc_names = [rc for folders in splits.values() for rc in folders]

    if not rc_names:
        print("No RC folders found. Set eval_splits in config or pass --rc.")
        return

    print(f"Config       : {args.config}")
    print(f"raw_data_dir : {raw_data_dir}")
    print(f"base_dir     : {base_dir}")
    print(f"RC folders   : {rc_names}\n")

    for rc_name in rc_names:
        print(f"{'='*50}")
        print(f"  {rc_name}")
        print(f"{'='*50}")

        src_folder = find_source_folder(raw_data_dir, rc_name)
        if not src_folder:
            print(f"  [SKIP] No source folder found for {rc_name} in {raw_data_dir}")
            continue
        print(f"  Source : {src_folder}")

        dst_folder = os.path.join(base_dir, rc_name)
        if not os.path.isdir(dst_folder):
            print(f"  [SKIP] Destination not found: {dst_folder}")
            continue

        # -- Calibration (overwrite — professor updated these) ----------------
        calib_src = os.path.join(src_folder, *calib_src_sub.split('/'))
        calib_dst = os.path.join(dst_folder, calib_sub)
        if os.path.isdir(calib_src):
            c, s = copy_files(calib_src, calib_dst, '*.txt', overwrite=True)
            print(f"  calib/ : updated={c}  unchanged={s}  (source: {calib_src_sub})")
        else:
            print(f"  calib/ : [WARN] source not found at {calib_src}")

        # -- CFAR (skip existing — only copy what is missing) -----------------
        cfar_src = os.path.join(src_folder, *cfar_src_sub.split('/'))
        cfar_dst = os.path.join(dst_folder, cfar_sub)
        if os.path.isdir(cfar_src):
            c, s = copy_files(cfar_src, cfar_dst, '*.txt', overwrite=False)
            print(f"  cfar/  : copied={c}   skipped={s}   (source: {cfar_src_sub})")
        else:
            print(f"  cfar/  : [WARN] source not found at {cfar_src}")

    print("\nDone.")


if __name__ == '__main__':
    main()
