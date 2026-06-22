"""
Copy CFAR radar point cloud files into the prepared dataset folder.

Run this once before thesis_eval.py if CFAR comparison is needed.
Reads from eval_config.yaml and copies:

  <raw_data_dir>/<NN>_<RC>/<cfar_src_subfolder>/*.txt
      ->  <base_dir>/<RC>/<subfolders.cfar>/

Skips files that already exist. Safe to re-run.

Usage:
  python dataset/copy_cfar.py --config configs/eval_config.yaml
"""

import os
import glob
import shutil
import argparse
import yaml


def find_source_folder(raw_data_dir, rc_name):
    for pattern in [f"*_{rc_name}", f"*_{rc_name.upper()}"]:
        matches = glob.glob(os.path.join(raw_data_dir, pattern))
        if matches:
            return matches[0]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/eval_config.yaml')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    raw_data_dir     = config.get('raw_data_dir', '').strip()
    base_dir         = config.get('base_dir', '').strip()
    cfar_src_sub     = config.get('cfar_src_subfolder', 'data/radar').strip()
    cfar_dst_sub     = config.get('subfolders', {}).get('cfar', 'cfar')

    # RC list comes from eval_splits (all conditions combined)
    eval_splits = config.get('eval_splits', {})
    rc_names = list(dict.fromkeys(
        rc for folders in eval_splits.values() for rc in (folders or [])
    ))

    if not raw_data_dir or not os.path.isdir(raw_data_dir):
        print(f"ERROR: raw_data_dir not set or does not exist: '{raw_data_dir}'")
        print("Set raw_data_dir in eval_config.yaml and re-run.")
        return
    if not base_dir:
        print("ERROR: base_dir not set in eval_config.yaml.")
        return
    if not rc_names:
        print("No RC folders listed under eval_splits in eval_config.yaml.")
        return

    print(f"raw_data_dir     : {raw_data_dir}")
    print(f"base_dir         : {base_dir}")
    print(f"cfar source path : <RC_source>/{cfar_src_sub}/")
    print(f"cfar dest folder : <base_dir>/<RC>/{cfar_dst_sub}/")
    print(f"RC folders       : {rc_names}\n")

    for rc_name in rc_names:
        src_folder = find_source_folder(raw_data_dir, rc_name)
        if not src_folder:
            print(f"[SKIP] {rc_name} -- no source folder matching *_{rc_name} in {raw_data_dir}")
            continue

        cfar_src = os.path.join(src_folder, *cfar_src_sub.split('/'))
        cfar_dst = os.path.join(base_dir, rc_name, cfar_dst_sub)

        if not os.path.isdir(cfar_src):
            print(f"[SKIP] {rc_name} -- CFAR source not found: {cfar_src}")
            continue

        files = glob.glob(os.path.join(cfar_src, '*.txt'))
        if not files:
            print(f"[SKIP] {rc_name} -- no .txt files in {cfar_src}")
            continue

        os.makedirs(cfar_dst, exist_ok=True)
        copied, skipped = 0, 0
        for src in files:
            dst = os.path.join(cfar_dst, os.path.basename(src))
            if os.path.exists(dst):
                skipped += 1
            else:
                shutil.copy2(src, dst)
                copied += 1

        print(f"{rc_name:10s}  copied={copied}  skipped={skipped}  -> {cfar_dst}")

    print("\nDone.")


if __name__ == '__main__':
    main()
