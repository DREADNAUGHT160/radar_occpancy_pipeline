"""
Main training script for the radar 3D occupancy model.

Usage:
  python training/train.py --config configs/train_config.yaml

  # To set CUDA memory allocator for large batches:
  $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"  # PowerShell
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     # bash
"""
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import traceback
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.factory import ModelFactory
from dataset.dataloader import RadarDataset
from utils.config_utils import resolve_splits
from utils.logger import setup_logger
from utils.report import ReportGenerator
from utils.tb_logger import log_train_images, log_eval_images, log_epoch_summary
from training.losses import get_loss_function, FocalLoss, FocalDiceLoss, BCEDiceLoss, TverskyLoss

BINARY_LOSSES = (nn.BCEWithLogitsLoss, FocalLoss, FocalDiceLoss, BCEDiceLoss, TverskyLoss)


class Trainer:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config['logging']['output_dir'] = os.path.join(
            self.config['logging']['output_dir'], self.run_id)
        os.makedirs(self.config['logging']['output_dir'], exist_ok=True)

        self.logger = setup_logger(
            self.config['experiment_name'], self.config['logging']['output_dir'])
        self.logger.info(f"Run ID: {self.run_id}")
        self.logger.info(f"Config:\n{json.dumps(self.config, indent=2)}")

        self.report = ReportGenerator(self.config['logging']['output_dir'], self.run_id)

        if self.config['logging'].get('tensorboard', False):
            tb_dir = os.path.join(self.config['logging']['output_dir'], 'tensorboard')
            self.writer = SummaryWriter(log_dir=tb_dir)
        else:
            self.writer = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        self.scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
        self.logger.info(f"Device: {self.device}")

    def _load_config(self, config_path):
        with open(config_path, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        config['_config_path'] = config_path
        return config

    # ── Data ──────────────────────────────────────────────────────────────────

    def setup_data(self):
        self.config = resolve_splits(self.config)
        self.logger.info("Setting up data loaders...")
        bs         = self.config['dataset']['batch_size']
        nw         = self.config['dataset']['num_workers']
        val_split  = self.config['dataset'].get('val_split',  0.0)
        test_split = self.config['dataset'].get('test_split', 0.0)

        train_sets, val_sets, test_sets = [], [], []

        base_dir    = self.config['dataset'].get('base_dir', '')
        train_names = self.config['dataset'].get('train', [])
        val_names   = self.config['dataset'].get('val',   [])
        test_names  = self.config['dataset'].get('test',  [])
        threshold   = self.config['dataset'].get('sync_threshold_ms', 100)

        sf      = self.config['dataset'].get('subfolders', {})
        lbl_sub = sf.get('labels', 'labels')

        def _load(name, augment):
            folder = os.path.join(base_dir, name) if base_dir else name
            ds_cfg = {**self.config, 'dataset': {
                **self.config['dataset'],
                'radar_dir':         folder,
                'lidar_path':        os.path.join(folder, lbl_sub),
                'sync_threshold_ms': threshold,
            }}
            return RadarDataset(folder, augment=augment, config=ds_cfg)

        if train_names or val_names or test_names:
            # ── Named-folder format ────────────────────────────────────────────
            # dataset:
            #   base_dir: E:/dataset
            #   train: [RC019, RC013]
            #   val:   [RC002]
            #   test:  [RC014]
            for name in train_names:
                ds = _load(name, augment=True)
                train_sets.append(ds)
                self.logger.info(f"{name}: train → {len(ds)} frames")
            for name in val_names:
                ds = _load(name, augment=False)
                val_sets.append(ds)
                self.logger.info(f"{name}: val → {len(ds)} frames")
            for name in test_names:
                ds = _load(name, augment=False)
                test_sets.append(ds)
                self.logger.info(f"{name}: test → {len(ds)} frames")

        else:
            # ── Legacy format: single radar_dir + optional extra_datasets ──────
            full_ds = RadarDataset(
                self.config['dataset']['radar_dir'],
                augment=self.config['dataset'].get('augmentations', True),
                config=self.config,
            )
            total   = len(full_ds)
            v_size  = int(total * val_split)
            t_size  = int(total * test_split)
            tr_size = total - v_size - t_size
            if tr_size <= 0:
                raise ValueError(f"Train size {tr_size} <= 0. Check val/test splits.")
            gen = torch.Generator().manual_seed(42)
            tr_ds, v_ds, te_ds = random_split(full_ds, [tr_size, v_size, t_size], generator=gen)
            full_ds._train_indices = set(tr_ds.indices)
            full_ds.augment = True
            train_sets.append(tr_ds)
            if v_size > 0: val_sets.append(v_ds)
            if t_size > 0: test_sets.append(te_ds)
            self.logger.info(f"Split — Train: {tr_size}  Val: {v_size}  Test: {t_size}")

            for extra in self.config['dataset'].get('extra_datasets', []):
                extra_cfg = {**self.config, 'dataset': {
                    **self.config['dataset'],
                    'radar_dir':         extra['radar_dir'],
                    'lidar_path':        extra['lidar_path'],
                    'sync_threshold_ms': extra.get('sync_threshold_ms', threshold),
                }}
                extra_ds = RadarDataset(extra['radar_dir'], augment=True, config=extra_cfg)
                train_sets.append(extra_ds)
                self.logger.info(f"Extra: {extra['radar_dir']} — {len(extra_ds)} frames")

        def _concat(sets):
            if not sets:   return None
            return ConcatDataset(sets) if len(sets) > 1 else sets[0]

        combined_train = _concat(train_sets)
        combined_val   = _concat(val_sets)
        combined_test  = _concat(test_sets)

        if combined_train is None:
            raise ValueError("No training data. Check your datasets config.")

        self.logger.info(
            f"Total — train={len(combined_train)}  "
            f"val={len(combined_val) if combined_val else 0}  "
            f"test={len(combined_test) if combined_test else 0}"
        )

        pin = torch.cuda.is_available()
        self.train_loader = DataLoader(combined_train, bs, shuffle=True,  num_workers=nw, pin_memory=pin)
        self.val_loader   = DataLoader(combined_val,   bs, shuffle=False, num_workers=nw) if combined_val  else None
        self.test_loader  = DataLoader(combined_test,  bs, shuffle=False, num_workers=nw) if combined_test else None

    # ── Model ─────────────────────────────────────────────────────────────────

    def setup_model(self):
        self.logger.info("Setting up model...")
        self.model = ModelFactory.get_model(self.config).to(self.device)

        pretrained = self.config.get('training', {}).get('pretrained_checkpoint')
        if pretrained and os.path.exists(pretrained):
            self.model.load_state_dict(torch.load(pretrained, map_location=self.device))
            self.logger.info(f"Loaded pretrained weights: {pretrained}")
        elif pretrained:
            self.logger.warning(f"pretrained_checkpoint not found: {pretrained}")

    # ── Optimiser / Loss ──────────────────────────────────────────────────────

    def setup_training(self):
        self.criterion = get_loss_function(self.config, self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['lr'],
            weight_decay=float(self.config['training']['weight_decay']),
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config['training']['epochs'])

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self):
        self.setup_data()
        self.setup_model()
        self.setup_training()

        num_epochs    = self.config['training']['epochs']
        best_val_loss = float('inf')
        self.logger.info(f"Starting training — {num_epochs} epochs  loss={self.criterion}")

        try:
            for epoch in range(num_epochs):
                self.current_epoch = epoch + 1
                train_loss = self._train_one_epoch(epoch, num_epochs)
                self.scheduler.step()

                val_loss, val_acc, val_iou, val_prec, val_rec = self.validate()

                self.logger.info(
                    f"Epoch {self.current_epoch}/{num_epochs} — "
                    f"Train: {train_loss:.4f}  Val: {val_loss:.4f}  "
                    f"IoU: {val_iou:.4f}  Prec: {val_prec:.4f}  Rec: {val_rec:.4f}")

                if self.writer:
                    self.writer.add_scalar('Loss/train', train_loss, self.current_epoch)
                    self.writer.add_scalar('Loss/val',   val_loss,   self.current_epoch)
                    self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], self.current_epoch)
                    self.writer.add_scalar('Metrics/IoU',       val_iou,  self.current_epoch)
                    self.writer.add_scalar('Metrics/Precision', val_prec, self.current_epoch)
                    self.writer.add_scalar('Metrics/Recall',    val_rec,  self.current_epoch)
                    log_epoch_summary(self.writer, self.current_epoch,
                                      train_loss, val_loss, val_acc, val_iou,
                                      val_prec, val_rec,
                                      self.optimizer.param_groups[0]['lr'], num_epochs)

                self.report.log_metric(self.current_epoch, train_loss, val_loss, val_acc)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_path = os.path.join(self.config['logging']['output_dir'], 'best_model.pth')
                    torch.save(self.model.state_dict(), save_path)
                    self.logger.info(f"Best model saved (val_loss={val_loss:.4f})")

            final_path = os.path.join(self.config['logging']['output_dir'], 'final_model.pth')
            torch.save(self.model.state_dict(), final_path)
            self.logger.info(f"Final model saved: {final_path}")

            self.report.save_report()
            if self.writer:
                self.writer.close()

            self._run_post_training()

            test_loss, test_acc, test_iou, test_prec, test_rec = self.test()
            self.logger.info(
                f"Test — Loss: {test_loss:.4f}  Acc: {test_acc:.4f}  "
                f"IoU: {test_iou:.4f}  Prec: {test_prec:.4f}  Rec: {test_rec:.4f}")

        except Exception as e:
            self.logger.error(f"Training crashed: {e}\n{traceback.format_exc()}")
            raise

    def _train_one_epoch(self, epoch, num_epochs):
        self.model.train()
        running_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")

        for i, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            if isinstance(self.criterion, BINARY_LOSSES):
                labels = labels.float()

            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = self.model(images)
                loss    = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (i + 1)})

            if self.writer:
                self.writer.add_scalar('Loss/train_batch', loss.item(),
                                       epoch * len(self.train_loader) + i)
                if i == 0:
                    log_train_images(self.writer, images, labels, outputs, epoch + 1)

        return running_loss / len(self.train_loader)

    # ── Evaluation ────────────────────────────────────────────────────────────

    def validate(self):
        if not self.val_loader:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        return self._evaluate(self.val_loader, "Validation")

    def test(self):
        if not self.test_loader:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        return self._evaluate(self.test_loader, "Test")

    def _evaluate(self, loader, mode="Validation"):
        self.model.eval()
        val_loss, correct, total = 0.0, 0, 0
        tp = fp = fn = 0
        images_logged = False

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                if isinstance(self.criterion, BINARY_LOSSES):
                    labels = labels.float()

                outputs  = self.model(images)
                loss     = self.criterion(outputs, labels)
                val_loss += loss.item()

                if isinstance(self.criterion, BINARY_LOSSES):
                    preds_prob = torch.sigmoid(outputs)
                    predicted  = (preds_prob > 0.5).float()
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    preds_prob   = torch.softmax(outputs, dim=1)

                total   += labels.numel()
                correct += (predicted == labels).sum().item()
                pred_b   = predicted.bool()
                label_b  = labels.bool()
                tp += (pred_b &  label_b).sum().item()
                fp += (pred_b & ~label_b).sum().item()
                fn += (~pred_b & label_b).sum().item()

                if not images_logged:
                    log_eval_images(self.writer, images, labels, preds_prob, mode, self.current_epoch)
                    images_logged = True

        avg_loss  = val_loss / max(len(loader), 1)
        accuracy  = correct  / max(total, 1)
        iou       = tp / (tp + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)

        if self.writer:
            self.writer.add_scalar(f'Metrics/{mode}_IoU',       iou,       self.current_epoch)
            self.writer.add_scalar(f'Metrics/{mode}_Precision', precision, self.current_epoch)
            self.writer.add_scalar(f'Metrics/{mode}_Recall',    recall,    self.current_epoch)
            self.writer.add_scalar(f'Metrics/{mode}_Accuracy',  accuracy,  self.current_epoch)

        return avg_loss, accuracy, iou, precision, recall

    # ── Post-training evaluation ──────────────────────────────────────────────

    def _run_post_training(self):
        if not self.config['logging'].get('save_inference_images', False):
            return
        import subprocess
        cfg_path        = self.config.get('_config_path', 'configs/train_config.yaml')
        run_id          = os.path.basename(self.config['logging']['output_dir'])
        base_dir        = f"verification_output/{run_id}"
        best_checkpoint = os.path.join(self.config['logging']['output_dir'], 'best_model.pth')

        scripts = [
            ("Full evaluation mosaic",
             ["python", str(ROOT / "utils" / "evaluate.py"),
              "--config", cfg_path,
              "--checkpoint", best_checkpoint,
              "--out_dir", f"{base_dir}/eval_all_data_views"]),
            ("Predict on test split",
             ["python", str(ROOT / "utils" / "predict.py"),
              "--config", cfg_path,
              "--checkpoint", best_checkpoint,
              "--out_dir", f"{base_dir}/predict_test"]),
        ]
        for desc, cmd in scripts:
            try:
                self.logger.info(f"Running: {desc}")
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                if result.stdout:
                    self.logger.info(result.stdout)
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Post-training step failed [{desc}]:\n{e.stderr}")
                print(f"\n[ERROR] {desc} failed:\n{e.stderr}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_config.yaml')
    args = parser.parse_args()
    trainer = Trainer(args.config)
    trainer.train()
