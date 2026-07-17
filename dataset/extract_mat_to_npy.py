"""
Step 1 — Extract radar data from MATLAB (.mat) files to NumPy arrays.

Reads RAD_power and RAD_elev from HDF5-format .mat files (MATLAB v7.3)
and saves them as <timestamp>.npy inside rad_power/ and rad_elev/ subfolders.

Usage:
  python dataset/extract_mat_to_npy.py \
    --input  "E:/raw/RC019/Radar" \
    --output "E:/dataset/RC019"
"""
import os
import argparse
import numpy as np
import h5py
from tqdm import tqdm
import glob


def extract(input_dir, output_dir):
    os.makedirs(os.path.join(output_dir, 'rad_power'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'rad_elev'),  exist_ok=True)

    mat_files = sorted(glob.glob(os.path.join(input_dir, '*.mat')))
    print(f"Found {len(mat_files)} .mat files in {input_dir}")

    skipped = 0
    for mat_path in tqdm(mat_files):
        try:
            with h5py.File(mat_path, 'r') as f:
                ts = int(f['TimeStamp'][0, 0])

                power_path = os.path.join(output_dir, 'rad_power', f'{ts}.npy')
                elev_path  = os.path.join(output_dir, 'rad_elev',  f'{ts}.npy')

                if os.path.exists(power_path) and os.path.exists(elev_path):
                    skipped += 1
                    continue

                power = np.array(f['RAD_power'], dtype=np.float32)
                elev  = np.array(f['RAD_elev'],  dtype=np.float32)

                np.save(power_path, power)
                np.save(elev_path,  elev)

        except Exception as e:
            print(f"  [ERROR] {os.path.basename(mat_path)}: {e}")

    print(f"\nDone. Extracted to: {output_dir}")
    print(f"  rad_power/ : {len(os.listdir(os.path.join(output_dir, 'rad_power')))} files")
    print(f"  rad_elev/  : {len(os.listdir(os.path.join(output_dir, 'rad_elev')))} files")
    if skipped:
        print(f"  Skipped (already exist): {skipped}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  required=True, help='Folder containing .mat radar cube files')
    parser.add_argument('--output', required=True, help='Output dataset folder (rad_power/ and rad_elev/ created inside)')
    args = parser.parse_args()
    extract(args.input, args.output)
