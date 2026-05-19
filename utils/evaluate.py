"""
Evaluation script — generates 3-row visualisation mosaic for every frame in the dataset.

Each output image shows:
  Row 1: radar input (BEV power, azimuth-elevation map, range-elevation map, reference image)
  Row 2: ground-truth occupancy label (BEV, front view, side view)
  Row 3: model prediction (BEV, front view, side view) — raw probability, no threshold

Usage:
  python utils/evaluate.py \
    --config    configs/train_config.yaml \
    --checkpoint E:/checkpoints/20260417_090255/best_model.pth \
    --out_dir   verification_output/my_run/eval
"""
import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import glob

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.factory import ModelFactory
from dataset.dataloader import RadarDataset


def _get_latest_checkpoint(output_dir):
    subdirs = [d for d in glob.glob(os.path.join(output_dir, '*'))
               if os.path.isdir(d)]
    if not subdirs:
        raise FileNotFoundError(f"No run directories in {output_dir}")
    latest = max(subdirs, key=os.path.getctime)
    ckpt   = os.path.join(latest, 'best_model.pth')
    if not os.path.exists(ckpt):
        ckpt = os.path.join(latest, 'final_model.pth')
    print(f"Using checkpoint: {ckpt}")
    return ckpt


def _compute_metrics(pred_binary, label_binary):
    p   = pred_binary.bool()
    l   = label_binary.bool()
    tp  = (p & l).sum().item()
    fp  = (p & ~l).sum().item()
    fn  = (~p & l).sum().item()
    iou = tp / (tp + fp + fn + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    return iou, prec, rec


def evaluate_and_plot():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='configs/train_config.yaml')
    parser.add_argument('--dataset',    default=None,
                        help='Dataset folder name to evaluate (e.g. RC019). '
                             'Uses config.dataset.base_dir/<dataset>. '
                             'Defaults to the first entry in config.dataset.val.')
    parser.add_argument('--checkpoint', default=None,
                        help='Path to .pth checkpoint. Defaults to config.inference.checkpoint or latest run.')
    parser.add_argument('--out_dir',    default=None,
                        help='Output directory. Defaults to config.inference.out_dir/eval.')
    args, _ = parser.parse_known_args()

    with open(args.config, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    inf      = config.get('inference', {})
    base_dir = config['dataset'].get('base_dir', '')

    # Resolve dataset folder
    ds_name = args.dataset or (config['dataset'].get('val', [None])[0])
    if ds_name and base_dir:
        radar_dir = os.path.join(base_dir, ds_name)
    elif ds_name:
        radar_dir = ds_name
    else:
        radar_dir = config['dataset'].get('radar_dir', '')

    ckpt    = args.checkpoint or inf.get('checkpoint') or _get_latest_checkpoint(config['logging']['output_dir'])
    out_dir = args.out_dir    or os.path.join(inf.get('out_dir', 'verification_output'), ds_name or 'eval', 'eval')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds_config = {**config, 'dataset': {**config['dataset'], 'radar_dir': radar_dir,
                                        'lidar_path': os.path.join(radar_dir, 'labels')}}
    full_ds = RadarDataset(radar_dir, augment=False, config=ds_config)
    print(f"Evaluating {len(full_ds)} frames in {radar_dir}")

    model = ModelFactory.get_model(config).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    os.makedirs(out_dir, exist_ok=True)

    label_text_dir = config['dataset'].get('label_text_dir', '')
    img_dir        = label_text_dir.replace('labels_new2', 'image').replace('label', 'image')

    xticks     = np.linspace(0, 255, 7)
    xlabels_az = np.round(np.degrees(np.arcsin(np.linspace(-1, 1, 7)))).astype(int)

    for idx in tqdm(range(len(full_ds))):
        radar_tensor, label_tensor = full_ds[idx]
        sample_info = full_ds.matched_data[idx]

        # Optional reference image
        img_path = None
        if img_dir and sample_info.get('power'):
            candidate = os.path.join(img_dir,
                                     os.path.basename(sample_info['power']).replace('.npy', '.jpg'))
            if os.path.exists(candidate):
                img_path = candidate

        with torch.no_grad():
            pred_prob = torch.sigmoid(model(radar_tensor.unsqueeze(0).to(device)))[0].cpu()

        # ── Build radar feature maps ───────────────────────────────────────────
        p_np  = radar_tensor[0].numpy()
        e_np  = radar_tensor[1].numpy() if radar_tensor.shape[0] > 1 else np.zeros_like(p_np)
        max_a = config['dataset']['normalization'].get('elevation_max_angle', 0.7854)
        e_norm = np.clip(e_np / max_a, -1.0, 1.0)
        e_bins = ((e_norm + 1.0) / 2.0 * 63).clip(0, 63).astype(int)

        n_e, n_r, n_a = 64, p_np.shape[1], p_np.shape[2]
        radar_bev = torch.max(radar_tensor[0], dim=0)[0]

        p_m = p_np[2:]      # skip static near-zero Doppler bins
        e_b = e_bins[2:]
        d, r, a = p_m.shape
        a_grid = np.tile(np.arange(a)[np.newaxis, np.newaxis, :], (d, r, 1))
        r_grid = np.tile(np.arange(r)[np.newaxis, :, np.newaxis], (d, 1, a))

        ae_sum   = np.zeros((n_e, n_a)); ae_cnt = np.zeros((n_e, n_a))
        np.add.at(ae_sum, (e_b.flatten(), a_grid.flatten()), p_m.flatten())
        np.add.at(ae_cnt, (e_b.flatten(), a_grid.flatten()), 1)
        radar_fv = np.where(ae_cnt > 0, ae_sum / ae_cnt, 0)
        if radar_fv.max() > 0: radar_fv /= radar_fv.max()

        re_sum   = np.zeros((n_e, n_r)); re_cnt = np.zeros((n_e, n_r))
        np.add.at(re_sum, (e_b.flatten(), r_grid.flatten()), p_m.flatten())
        np.add.at(re_cnt, (e_b.flatten(), r_grid.flatten()), 1)
        radar_re = np.where(re_cnt > 0, re_sum / re_cnt, 0)
        if radar_re.max() > 0: radar_re /= radar_re.max()

        # ── Projections ────────────────────────────────────────────────────────
        label     = label_tensor
        pred_bev  = torch.max(pred_prob, dim=0)[0];  label_bev = torch.max(label, dim=0)[0]
        pred_fv   = torch.max(pred_prob, dim=1)[0];  label_fv  = torch.max(label, dim=1)[0]
        pred_re   = torch.max(pred_prob, dim=2)[0];  label_re  = torch.max(label, dim=2)[0]

        ncols = 4 if img_path else 3
        fig   = plt.figure(figsize=(6 * ncols, 12))
        gs    = fig.add_gridspec(3, ncols)

        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(radar_bev.numpy(), cmap='turbo', origin='lower', aspect='auto', vmin=0, vmax=1)
        ax.set_title('Input: Power (BEV)'); ax.set_xticks(xticks); ax.set_xticklabels(xlabels_az)

        ax = fig.add_subplot(gs[0, 1])
        ax.imshow(radar_fv, cmap='turbo', origin='lower', aspect='auto')
        ax.set_title('Input: AE Map'); ax.set_xticks(xticks); ax.set_xticklabels(xlabels_az)

        ax = fig.add_subplot(gs[0, 2])
        ax.imshow(radar_re, cmap='turbo', origin='lower', aspect='auto')
        ax.set_title('Input: RE Map')

        if img_path:
            ax = fig.add_subplot(gs[0, 3])
            ax.imshow(Image.open(img_path)); ax.axis('off'); ax.set_title('Reference Image')

        ax = fig.add_subplot(gs[1, 0])
        ax.imshow(label_bev.numpy(), cmap='gray', origin='lower', aspect='auto')
        ax.set_title('GT: BEV'); ax.set_xticks(xticks); ax.set_xticklabels(xlabels_az)

        ax = fig.add_subplot(gs[1, 1])
        ax.imshow(label_fv.numpy(), cmap='gray', origin='lower', aspect='auto')
        ax.set_title('GT: Front View'); ax.set_xticks(xticks); ax.set_xticklabels(xlabels_az)

        ax = fig.add_subplot(gs[1, 2])
        ax.imshow(label_re.numpy(), cmap='gray', origin='lower', aspect='auto')
        ax.set_title('GT: Side View')

        ax = fig.add_subplot(gs[2, 0])
        ax.imshow(pred_bev.numpy(), cmap='magma', origin='lower', aspect='auto', vmin=0, vmax=1)
        ax.set_title('Pred: BEV'); ax.set_xticks(xticks); ax.set_xticklabels(xlabels_az)

        ax = fig.add_subplot(gs[2, 1])
        ax.imshow(pred_fv.numpy(), cmap='magma', origin='lower', aspect='auto', vmin=0, vmax=1)
        ax.set_title('Pred: Front View'); ax.set_xticks(xticks); ax.set_xticklabels(xlabels_az)

        ax = fig.add_subplot(gs[2, 2])
        ax.imshow(pred_re.numpy(), cmap='magma', origin='lower', aspect='auto', vmin=0, vmax=1)
        ax.set_title('Pred: Side View')

        ts = os.path.basename(sample_info['power']).replace('.npy', '')
        plt.suptitle(f"Sample {idx:03d} | Timestamp: {ts} | Raw Probability (no threshold)", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'eval_{ts}.png'), bbox_inches='tight', dpi=120)
        plt.close(fig)

    print(f"\nSaved {len(full_ds)} evaluation plots to: {os.path.abspath(out_dir)}")


if __name__ == '__main__':
    evaluate_and_plot()
