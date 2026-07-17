"""
Precompute argmax-gathered elevation pooling (512 -> 128) for all RC folders.

For each 4-bin Doppler group, selects the elevation value with the largest
|magnitude| — not power's peak bin. RAD_elev is sparse enough (~99.6% exactly
zero) that elevation's own magnitude reliably identifies the real detection
without needing power at all; power files are only used to find which frames
have a paired elev file, never loaded into the pooling itself. This replaces
the default max-pool, which always keeps the numerically largest raw value in
each group — since zero background always beats a negative detection, max-pool
silently erases every negative elevation reading (targets below sensor height).

Usage:
    python elev_pool.py --base_dir /path/to/radar_dataset

Output:
    <RC>/rad_elev_pooled/<timestamp>.npy  (128, H, W) per frame

The dataloader reads rad_elev_pooled/ automatically if it exists, so no
other changes are needed — just run this script once before training.
"""

import os
import glob
import argparse
import numpy as np


def pool_elev(elev):
    """512->128: elevation gathered from the Doppler bin with largest |elevation| magnitude.
    RAD_elev is sparse — only the detected target's Doppler bin is non-zero.
    abs().argmax() finds that bin directly, no power dependency needed."""
    if elev.ndim == 3 and elev.shape[2] == 512:
        elev = elev.transpose(2, 0, 1)
    H, W    = elev.shape[1], elev.shape[2]
    e_blocks = elev.reshape(128, 4, H, W)
    idx      = np.abs(e_blocks).argmax(axis=1)                               # (128, H, W)
    elev_p   = np.take_along_axis(e_blocks, idx[:, np.newaxis], axis=1).squeeze(1)
    return elev_p                                                             # (128, H, W)


def process_rc(rc_dir):
    power_dir  = os.path.join(rc_dir, 'rad_power')
    elev_dir   = os.path.join(rc_dir, 'rad_elev')
    out_dir    = os.path.join(rc_dir, 'rad_elev_pooled')

    power_files = sorted(glob.glob(os.path.join(power_dir, '*.npy')))
    elev_files  = sorted(glob.glob(os.path.join(elev_dir,  '*.npy')))

    if not power_files:
        print(f"  [SKIP] No rad_power files in {rc_dir}")
        return
    if not elev_files:
        print(f"  [SKIP] No rad_elev files in {rc_dir}")
        return

    # Remove old pooled files (max-pool or stride) before writing argmax
    import shutil
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
        print(f"  Removed old rad_elev_pooled/")
    os.makedirs(out_dir)

    # Match by filename (timestamp) — power only used to find paired elev files
    elev_map = {os.path.basename(f): f for f in elev_files}
    matched  = [(p, elev_map[os.path.basename(p)])
                for p in power_files if os.path.basename(p) in elev_map]
    del power_files  # power not loaded during pooling

    print(f"  {os.path.basename(rc_dir)}: {len(matched)} frames -> {out_dir}")
    for i, (p_path, e_path) in enumerate(matched):
        out_path = os.path.join(out_dir, os.path.basename(p_path))
        elev   = np.load(e_path)
        elev_p = pool_elev(elev)
        np.save(out_path, elev_p)
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(matched)}")

    # Sanity check: verify negative values present (stride/argmax preserve them, max-pool removes them)
    sample = np.load(os.path.join(out_dir, os.path.basename(matched[0][0])))
    status = 'OK (has negative elevations)' if sample.min() < 0 else 'WARN: no negative values found'
    print(f"    Done. min={sample.min():.4f}  [{status}]")


def main():
    parser = argparse.ArgumentParser(description='Precompute argmax elevation pooling.')
    parser.add_argument('--base_dir', required=True,
                        help='Root radar dataset directory containing RC* folders')
    args = parser.parse_args()

    base = args.base_dir

    # Single RC folder passed directly (has rad_power/ inside)
    if os.path.isdir(os.path.join(base, 'rad_power')):
        rc_dirs = [base]
    else:
        rc_dirs = sorted(set(
            [d for d in glob.glob(os.path.join(base, '*', 'RC*')) if os.path.isdir(d)] +
            [d for d in glob.glob(os.path.join(base, 'RC*'))      if os.path.isdir(d)]
        ))

    if not rc_dirs:
        print(f"No RC folders found under {base}")
        return

    print(f"Found {len(rc_dirs)} RC folder(s)")
    for rc_dir in rc_dirs:
        process_rc(rc_dir)

    print("\nDone. rad_elev_pooled/ folders are ready for training.")


if __name__ == '__main__':
    main()
