"""
Standalone camera projection figure generator.

Generates 2-panel DL-vs-CFAR camera overlay figures for all RC folders
listed in eval_splits, without running the full evaluation pipeline.

Use this when thesis_eval.py fails mid-run or you only want the figures.

Usage:
  python utils/gen_camera_proj.py --config configs/eval_config.yaml
  python utils/gen_camera_proj.py --config configs/eval_config.yaml --checkpoint checkpoints/<run_id>/best_model.pth
  python utils/gen_camera_proj.py --config configs/eval_config.yaml --n_plots 10
  python utils/gen_camera_proj.py --config configs/eval_config.yaml --rc RC019 RC031

Output: <out_dir>/camera_projection/<rc_name>/frame_NN_rangeX.Xm.png
        where out_dir comes from the config file.

Requirements:
  - pco/ subfolder must exist inside each RC folder in the prepared dataset
  - calib/ and cfar/ subfolders must also exist (standard dataset layout)
"""
import os
import sys
import argparse
import yaml
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.thesis_eval import _load_model, generate_camera_projection_plots


def main():
    parser = argparse.ArgumentParser(
        description='Generate DL-vs-CFAR camera projection figures.')
    parser.add_argument('--config',     default='configs/eval_config.yaml',
                        help='Path to eval_config.yaml')
    parser.add_argument('--checkpoint', default=None,
                        help='Override checkpoint path in config')
    parser.add_argument('--n_plots',    type=int, default=None,
                        help='Number of frames per RC folder (overrides config)')
    parser.add_argument('--rc',         nargs='+', default=None,
                        help='Process only these RC folders, e.g. --rc RC019 RC031')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    ckpt = args.checkpoint or config.get('checkpoint', '')
    if not ckpt or not os.path.exists(ckpt):
        print(f"ERROR: checkpoint not found: {ckpt}")
        sys.exit(1)

    base_dir = config.get('base_dir', '')
    out_dir  = config.get('out_dir',  'verification_output/camera_proj')

    tp_cfg  = config.get('thesis_plots', {})
    n_plots = args.n_plots or int(tp_cfg.get('n_plots', 5))
    threshold = float(config.get('weather', {}).get('threshold',
                      tp_cfg.get('threshold', 0.4)))

    # Collect RC folders — use --rc override, or everything from eval_splits
    if args.rc:
        all_rc = args.rc
    else:
        splits = config.get('eval_splits', {})
        all_rc = list(dict.fromkeys(
            rc for folders in splits.values() for rc in (folders or [])))

    if not all_rc:
        print("ERROR: no RC folders found. Set eval_splits in config or use --rc.")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading checkpoint: {ckpt}")
    print(f"Device: {device}")
    model  = _load_model(config, ckpt, device)

    print(f"\nRC folders : {all_rc}")
    print(f"Frames/RC  : {n_plots}")
    print(f"Threshold  : {threshold}")
    print(f"Output dir : {os.path.abspath(out_dir)}/camera_projection/")
    print("=" * 60)

    generate_camera_projection_plots(
        all_rc, base_dir, config, model, device,
        threshold, out_dir, n_plots=n_plots)

    print("\nDone.")


if __name__ == '__main__':
    main()
