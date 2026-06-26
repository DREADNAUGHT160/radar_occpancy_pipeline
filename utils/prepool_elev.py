"""
Pre-pool elevation cube Doppler axis (512 -> 128) for all RC folders.
Saves results to rad_elev_pooled/ alongside the original rad_elev/.

Run this AFTER prepool_doppler.py has finished.
The dataloader auto-detects rad_elev_pooled/ and uses it directly.

Usage:
  python utils/prepool_elev.py --config configs/train_config.yaml
  python utils/prepool_elev.py --config configs/train_config.yaml --rc RC019
  python utils/prepool_elev.py --config configs/train_config.yaml --splits train val
"""
import os
import sys
import argparse
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pool_file(src_path, dst_path):
    """Load a (512, H, W) elevation file, max-pool to (128, H, W), save.

    GPU: F.max_pool3d kernel=(4,1,1)
    CPU: numpy reshape + max
    """
    arr = np.load(src_path)
    if arr.shape[0] == 128:
        np.save(dst_path, arr)
        return 'skip'
    if arr.shape[0] != 512:
        return f'unexpected shape {arr.shape}'

    if DEVICE.type == 'cuda':
        t      = torch.from_numpy(arr).float().to(DEVICE)
        t      = t.unsqueeze(0).unsqueeze(0)                        # (1, 1, 512, H, W)
        t      = F.max_pool3d(t, kernel_size=(4, 1, 1), stride=(4, 1, 1))
        pooled = t.squeeze(0).squeeze(0).cpu().numpy()              # (128, H, W)
    else:
        blocks = arr.reshape(128, 4, arr.shape[1], arr.shape[2])
        pooled = blocks.max(axis=1)

    np.save(dst_path, pooled)
    return 'ok'


def prepool_rc(rc_dir, force=False):
    src_dir = os.path.join(rc_dir, 'rad_elev')
    dst_dir = os.path.join(rc_dir, 'rad_elev_pooled')

    if not os.path.isdir(src_dir):
        return 0, 0, f"rad_elev/ not found in {rc_dir}"

    files = sorted(f for f in os.listdir(src_dir) if f.endswith('.npy'))
    if not files:
        return 0, 0, "no .npy files in rad_elev/"

    os.makedirs(dst_dir, exist_ok=True)
    done = skipped = 0
    for fname in tqdm(files, desc=os.path.basename(rc_dir), leave=False):
        dst = os.path.join(dst_dir, fname)
        if os.path.exists(dst) and not force:
            skipped += 1
            continue
        result = pool_file(os.path.join(src_dir, fname), dst)
        if result == 'ok':
            done += 1
        elif result == 'skip':
            skipped += 1

    return done, skipped, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',  default='configs/train_config.yaml')
    parser.add_argument('--rc',      default=None,
                        help='Process only this RC folder')
    parser.add_argument('--splits',  nargs='+', default=['train', 'val', 'test'],
                        help='Which splits to process (default: all)')
    parser.add_argument('--force',   action='store_true',
                        help='Re-pool even if rad_elev_pooled/ file already exists')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    base_dir = cfg.get('dataset', {}).get('base_dir', '')
    ds       = cfg.get('dataset', {})

    if args.rc:
        rc_list = [args.rc]
    else:
        rc_list = []
        for split in args.splits:
            rc_list += ds.get(split, [])
        rc_list = list(dict.fromkeys(rc_list))

    device_name = torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'
    print(f"\nPre-pooling elevation Doppler 512->128 for {len(rc_list)} RC folder(s)")
    print(f"Base dir  : {base_dir}")
    print(f"Device    : {DEVICE} ({device_name})")
    print(f"Method    : F.max_pool3d kernel=(4,1,1)  [CPU fallback: numpy max]")
    print(f"Output    : <RC>/rad_elev_pooled/\n")

    total_done = total_skip = 0
    for rc_name in rc_list:
        rc_dir = os.path.join(base_dir, rc_name) if base_dir else rc_name
        done, skipped, err = prepool_rc(rc_dir, force=args.force)
        if err:
            print(f"  [SKIP] {rc_name}: {err}")
        else:
            total_done += done
            total_skip += skipped
            status = f"pooled {done}" if done else f"all {skipped} already pooled"
            print(f"  [ OK ] {rc_name}: {status}")

    print(f"\n{'='*50}")
    print(f"  Newly pooled : {total_done} files")
    print(f"  Already done : {total_skip} files")
    print(f"{'='*50}\n")
    print("Done. The dataloader will now use rad_elev_pooled/ automatically.")


if __name__ == '__main__':
    main()
