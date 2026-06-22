"""
Generate equally spaced prediction plots for visual model verification.

Produces the same 3-row mosaic as evaluate.py (radar input / GT / prediction)
but only for N equally spaced frames instead of every frame -- fast to run.

Usage:
  python utils/eval_plots.py --config configs/eval_config.yaml --checkpoint checkpoints/<run_id>/best_model.pth
  python utils/eval_plots.py --config configs/eval_config.yaml --n_plots 12
"""
import os
import sys
import argparse
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.factory import ModelFactory
from dataset.dataloader import RadarDataset


def _build_ds_config(base_cfg, rc_dir):
    sf = base_cfg.get('subfolders', {})
    return {
        'model':   base_cfg.get('model', {}),
        'dataset': {
            'radar_dir':         rc_dir,
            'lidar_path':        os.path.join(rc_dir, sf.get('labels', 'labels')),
            'sync_threshold_ms': 100,
            'subfolders':        sf,
            'normalization':     base_cfg.get('normalization', {}),
            'filter_bboxes':     False,
            'label_text_dir':    '',
        },
    }


def _build_ae_re(radar_tensor, norm_cfg):
    p_np   = radar_tensor[0].numpy()
    e_np   = radar_tensor[1].numpy() if radar_tensor.shape[0] > 1 else np.zeros_like(p_np)
    max_a  = norm_cfg.get('elevation_max_angle', 0.7854)
    e_norm = np.clip(e_np / max_a, -1.0, 1.0)
    e_bins = ((e_norm + 1.0) / 2.0 * 63).clip(0, 63).astype(int)

    n_e, n_r, n_a = 64, p_np.shape[1], p_np.shape[2]
    p_m = p_np[2:]       # skip static Doppler bins
    e_b = e_bins[2:]
    d, r, a = p_m.shape

    a_grid = np.tile(np.arange(a)[np.newaxis, np.newaxis, :], (d, r, 1))
    r_grid = np.tile(np.arange(r)[np.newaxis, :, np.newaxis], (d, 1, a))

    ae_sum = np.zeros((n_e, n_a)); ae_cnt = np.zeros((n_e, n_a))
    np.add.at(ae_sum, (e_b.flatten(), a_grid.flatten()), p_m.flatten())
    np.add.at(ae_cnt, (e_b.flatten(), a_grid.flatten()), 1)
    ae_map = np.where(ae_cnt > 0, ae_sum / ae_cnt, 0)
    if ae_map.max() > 0: ae_map /= ae_map.max()

    re_sum = np.zeros((n_e, n_r)); re_cnt = np.zeros((n_e, n_r))
    np.add.at(re_sum, (e_b.flatten(), r_grid.flatten()), p_m.flatten())
    np.add.at(re_cnt, (e_b.flatten(), r_grid.flatten()), 1)
    re_map = np.where(re_cnt > 0, re_sum / re_cnt, 0)
    if re_map.max() > 0: re_map /= re_map.max()

    bev = torch.max(radar_tensor[0], dim=0)[0].numpy()
    return bev, ae_map, re_map


def plot_frame(idx, radar_tensor, label_tensor, pred_prob, norm_cfg,
               timestamp, pco_dir, out_dir):
    bev, ae_map, re_map = _build_ae_re(radar_tensor, norm_cfg)

    label    = label_tensor
    pred_bev = torch.max(pred_prob, dim=0)[0];  label_bev = torch.max(label, dim=0)[0]
    pred_fv  = torch.max(pred_prob, dim=1)[0];  label_fv  = torch.max(label, dim=1)[0]
    pred_re  = torch.max(pred_prob, dim=2)[0];  label_re  = torch.max(label, dim=2)[0]

    # Try to find a matching camera image
    img_path = None
    if pco_dir:
        for ext in ('.jpg', '.jpeg', '.png'):
            candidate = os.path.join(pco_dir, timestamp + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

    ncols = 4 if img_path else 3
    xticks     = np.linspace(0, 255, 7)
    xlabels_az = np.round(np.degrees(np.arcsin(np.linspace(-1, 1, 7)))).astype(int)

    fig = plt.figure(figsize=(6 * ncols, 10))
    gs  = fig.add_gridspec(3, ncols, hspace=0.35, wspace=0.25)

    # -- Row 0: Radar input ----------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(bev, cmap='turbo', origin='lower', aspect='auto', vmin=0, vmax=1)
    ax.set_title('Input: Power BEV')
    ax.set_xticks(xticks); ax.set_xticklabels(xlabels_az)

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(ae_map, cmap='turbo', origin='lower', aspect='auto')
    ax.set_title('Input: AE Map')
    ax.set_xticks(xticks); ax.set_xticklabels(xlabels_az)

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(re_map, cmap='turbo', origin='lower', aspect='auto')
    ax.set_title('Input: RE Map')

    if img_path:
        ax = fig.add_subplot(gs[0, 3])
        ax.imshow(Image.open(img_path))
        ax.set_title('Camera Image'); ax.axis('off')

    # -- Row 1: Ground truth ---------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(label_bev.numpy(), cmap='gray', origin='lower', aspect='auto')
    ax.set_title('GT: BEV')
    ax.set_xticks(xticks); ax.set_xticklabels(xlabels_az)

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(label_fv.numpy(), cmap='gray', origin='lower', aspect='auto')
    ax.set_title('GT: Front View')
    ax.set_xticks(xticks); ax.set_xticklabels(xlabels_az)

    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(label_re.numpy(), cmap='gray', origin='lower', aspect='auto')
    ax.set_title('GT: Side View')

    # -- Row 2: Prediction (raw probability) -----------------------------------
    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(pred_bev.numpy(), cmap='magma', origin='lower', aspect='auto', vmin=0, vmax=1)
    ax.set_title('Pred: BEV (raw prob)')
    ax.set_xticks(xticks); ax.set_xticklabels(xlabels_az)

    ax = fig.add_subplot(gs[2, 1])
    ax.imshow(pred_fv.numpy(), cmap='magma', origin='lower', aspect='auto', vmin=0, vmax=1)
    ax.set_title('Pred: Front View (raw prob)')
    ax.set_xticks(xticks); ax.set_xticklabels(xlabels_az)

    ax = fig.add_subplot(gs[2, 2])
    ax.imshow(pred_re.numpy(), cmap='magma', origin='lower', aspect='auto', vmin=0, vmax=1)
    ax.set_title('Pred: Side View (raw prob)')

    plt.suptitle(f"Frame {idx:03d}  |  ts={timestamp}", fontsize=13, y=1.01)
    plt.savefig(os.path.join(out_dir, f'plot_{idx:03d}_{timestamp}.png'),
                bbox_inches='tight', dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='configs/eval_config.yaml')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--dataset',    default=None,
                        help='RC folder name (overrides eval_config basic.dataset)')
    parser.add_argument('--n_plots',    type=int, default=None,
                        help='Number of equally spaced frames to plot (overrides config)')
    parser.add_argument('--out_dir',    default=None)
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    basic_cfg  = config.get('basic', {})
    plots_cfg  = config.get('eval_plots', {})

    ckpt       = args.checkpoint or config.get('checkpoint', '')
    rc_name    = args.dataset    or plots_cfg.get('dataset') or basic_cfg.get('dataset', '')
    n_plots    = args.n_plots    or plots_cfg.get('n_plots', 10)
    base_dir   = config.get('base_dir', '')
    pco_dir    = plots_cfg.get('pco_dir', '')
    out_dir    = args.out_dir or os.path.join(config.get('out_dir', 'verification_output/eval'), 'plots')

    if not ckpt:
        print("ERROR: set checkpoint in eval_config.yaml or pass --checkpoint")
        return
    if not rc_name:
        print("ERROR: set eval_plots.dataset or basic.dataset in eval_config.yaml")
        return

    rc_dir = os.path.join(base_dir, rc_name) if base_dir else rc_name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ModelFactory.get_model(config).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    ds = RadarDataset(rc_dir, augment=False, config=_build_ds_config(config, rc_dir))
    total = len(ds)
    print(f"\nDataset: {rc_name}  ({total} frames)")
    print(f"Generating {n_plots} equally spaced plots  ->  {out_dir}")

    # Pick equally spaced indices
    indices = np.linspace(0, total - 1, min(n_plots, total)).astype(int)

    os.makedirs(out_dir, exist_ok=True)
    norm_cfg = config.get('normalization', {})

    for idx in tqdm(indices, desc="Plotting"):
        radar_tensor, label_tensor = ds[int(idx)]
        sample   = ds.matched_data[int(idx)]
        ts       = os.path.basename(sample['power']).replace('.npy', '')

        with torch.no_grad():
            pred_prob = torch.sigmoid(
                model(radar_tensor.unsqueeze(0).to(device))
            )[0].cpu()

        plot_frame(int(idx), radar_tensor, label_tensor, pred_prob,
                   norm_cfg, ts, pco_dir, out_dir)

    print(f"\nSaved {len(indices)} plots to: {os.path.abspath(out_dir)}")


if __name__ == '__main__':
    main()
