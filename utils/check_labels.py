"""
Scan all label .npy files in the dataset and report corrupt / empty ones.

Usage:
  python utils/check_labels.py --config configs/train_config.yaml
  python utils/check_labels.py --config configs/train_config.yaml --fix   # delete corrupt files so prepare_dataset can regenerate them
"""
import os
import sys
import argparse
import yaml
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def scan_rc(rc_dir, rc_name):
    label_dir = os.path.join(rc_dir, 'labels')
    if not os.path.isdir(label_dir):
        return [], f"  [{rc_name}] labels/ folder not found at {label_dir}"

    files = sorted(f for f in os.listdir(label_dir) if f.endswith('.npy'))
    if not files:
        return [], f"  [{rc_name}] no .npy files in labels/"

    corrupt = []
    for fname in files:
        fpath = os.path.join(label_dir, fname)
        size  = os.path.getsize(fpath)
        if size == 0:
            corrupt.append((fpath, 'empty file (0 bytes)'))
            continue
        try:
            arr = np.load(fpath, allow_pickle=False)
            if arr.size == 0:
                corrupt.append((fpath, f'zero-element array shape={arr.shape}'))
        except Exception as e:
            corrupt.append((fpath, str(e)))

    return corrupt, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train_config.yaml')
    parser.add_argument('--fix',    action='store_true',
                        help='Delete corrupt files so prepare_dataset.py can regenerate them')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    base_dir = cfg.get('dataset', {}).get('base_dir', '')
    ds       = cfg.get('dataset', {})
    all_rcs  = ds.get('train', []) + ds.get('val', []) + ds.get('test', [])

    if not all_rcs:
        print("No RC folders found in config.")
        return

    print(f"\nScanning {len(all_rcs)} RC folders in {base_dir}\n")

    total_corrupt = 0
    total_files   = 0

    for rc_name in all_rcs:
        rc_dir  = os.path.join(base_dir, rc_name)
        label_dir = os.path.join(rc_dir, 'labels')

        if not os.path.isdir(label_dir):
            print(f"  [SKIP] {rc_name}: labels/ not found")
            continue

        all_files = [f for f in os.listdir(label_dir) if f.endswith('.npy')]
        total_files += len(all_files)

        corrupt, warn = scan_rc(rc_dir, rc_name)
        if warn:
            print(warn)
            continue

        if corrupt:
            total_corrupt += len(corrupt)
            print(f"  [FAIL] {rc_name}: {len(corrupt)}/{len(all_files)} corrupt file(s)")
            for fpath, reason in corrupt:
                print(f"         {os.path.basename(fpath)}  ({reason})")
                if args.fix:
                    os.remove(fpath)
                    print(f"         -> deleted")
        else:
            print(f"  [ OK ] {rc_name}: {len(all_files)} files, all clean")

    print(f"\n{'='*55}")
    print(f"  Total label files scanned : {total_files}")
    print(f"  Corrupt / empty           : {total_corrupt}")
    if total_corrupt > 0 and not args.fix:
        print(f"\n  Re-run with --fix to delete them so prepare_dataset.py")
        print(f"  can regenerate the missing files.")
    elif total_corrupt > 0 and args.fix:
        print(f"\n  Corrupt files deleted. Run prepare_dataset.py to regenerate.")
    else:
        print(f"  Dataset is clean.")
    print(f"{'='*55}\n")


if __name__ == '__main__':
    main()
