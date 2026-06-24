import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import glob
import re


class RadarDataset(Dataset):
    """
    PyTorch Dataset for radar 3D occupancy prediction.

    Expects the following folder layout:
        <radar_dir>/
            rad_power/   — .npy power cubes,     filenames = <timestamp_ms>.npy
            rad_elev/    — .npy elevation cubes,  same filenames as power

    Each power cube has shape (512, H, W) or (H, W, 512); the dataloader
    transposes and downsamples to (128, 256, 256) by default.

    Labels are .npy occupancy grids with shape (64, 256, 256), generated
    by dataset/generate_lidar_labels.py. The dataloader flips the azimuth
    axis (dim=-1) on load to align with the radar convention.
    """

    def __init__(self, radar_dir, augment=False, config=None):
        self.root_dir = radar_dir
        self.augment  = augment
        self.config   = config

        # Resolve power / elevation directories
        sf = (config or {}).get('dataset', {}).get('subfolders', {})
        raw_power_dir    = os.path.join(radar_dir, sf.get('rad_power', 'rad_power'))
        pooled_power_dir = os.path.join(radar_dir, 'rad_power_pooled')

        # Use pre-pooled folder if it exists — skips runtime Doppler downsampling
        if os.path.isdir(pooled_power_dir) and \
                len(glob.glob(os.path.join(pooled_power_dir, '*.npy'))) > 0:
            self.power_dir      = pooled_power_dir
            self.doppler_pooled = True
        else:
            self.power_dir      = raw_power_dir
            self.doppler_pooled = False

        self.elev_dir  = os.path.join(radar_dir, sf.get('rad_elev',  'rad_elev'))

        self.power_paths = sorted(glob.glob(os.path.join(self.power_dir, '*.npy')))
        if not self.power_paths:
            raise FileNotFoundError(f"No .npy power files found in {self.power_dir}")
        print(f"Dataset: {len(self.power_paths)} frames in {self.power_dir}")

        # Normalization statistics (defaults; overridden by auto_normalization)
        self.stats = {
            'power_min': -100.0,
            'power_max': -40.0,
            'elev_max':   0.7854,
        }

        if config and config.get('dataset', {}).get('normalization', {}).get('auto_normalization', False):
            self._compute_stats()

        # Resolve LiDAR label directory
        lidar_path = config['dataset'].get('lidar_path') if config else None
        if lidar_path and os.path.isdir(lidar_path):
            self.lidar_paths = sorted(glob.glob(os.path.join(lidar_path, '*.npy')))
            print(f"Dataset: {len(self.lidar_paths)} label files in {lidar_path}")
        else:
            print("Dataset: no lidar_path configured — dummy zero labels will be used.")
            self.lidar_paths = []

        if self.lidar_paths:
            threshold = config['dataset'].get('sync_threshold_ms', 100) if config else 100
            print(f"Dataset: syncing with threshold {threshold} ms")
            self.matched_data = self._sync_timestamps(self.power_paths, self.lidar_paths, threshold)
        else:
            self.matched_data = [{'power': p, 'label': None} for p in self.power_paths]

    # ── Timestamp sync ────────────────────────────────────────────────────────

    def _extract_timestamp(self, filepath):
        match = re.search(r'(\d+\.\d+|\d+)', os.path.basename(filepath))
        if match:
            val = float(match.group(0))
            return int(val * 1000) if val < 1e11 else int(val)
        return 0

    def _sync_timestamps(self, radar_files, lidar_files, threshold_ms=100):
        matched    = []
        lidar_ts   = np.array([self._extract_timestamp(f) for f in lidar_files])
        for r_file in radar_files:
            r_ts  = self._extract_timestamp(r_file)
            diffs = np.abs(lidar_ts - r_ts)
            best  = np.argmin(diffs)
            if diffs[best] < threshold_ms:
                matched.append({'power': r_file, 'label': lidar_files[best]})
        print(f"Dataset: synced {len(matched)}/{len(radar_files)} radar frames.")
        return matched

    # ── Auto-normalization ────────────────────────────────────────────────────

    def _compute_stats(self):
        print("Dataset: computing normalization statistics from 20 random frames...")
        sample_paths = self.power_paths[:20]
        pow_mins, pow_maxs, elev_maxs = [], [], []
        for p_path in sample_paths:
            try:
                data = np.load(p_path)
                if self.config['dataset']['normalization'].get('power_log_transform', True):
                    data = 10 * np.log10(data + 1e-10)
                pow_mins.append(data.min())
                pow_maxs.append(data.max())
                e_path = os.path.join(self.elev_dir, os.path.basename(p_path))
                if os.path.exists(e_path):
                    elev_maxs.append(np.abs(np.load(e_path)).max())
            except Exception:
                continue
        if pow_mins:
            self.stats['power_min'] = float(np.percentile(pow_mins, 1))
            self.stats['power_max'] = float(np.percentile(pow_maxs, 98))
            print(f"  power range: {self.stats['power_min']:.1f} to {self.stats['power_max']:.1f} dB")
        if elev_maxs:
            self.stats['elev_max'] = float(np.max(elev_maxs))

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self):
        return len(self.matched_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample     = self.matched_data[idx]
        power_path = sample['power']
        label_path = sample['label']

        # Per-sample augment flag (train.py marks training indices via _train_indices)
        should_augment = self.augment
        if hasattr(self, '_train_indices'):
            should_augment = idx in self._train_indices

        cfg     = self.config or {}
        model   = cfg.get('model', {})
        n_cls   = model.get('num_classes', 64)
        doppler = model.get('doppler_depth', 128)
        in_ch   = model.get('in_channels', 2)

        # Load power cube
        try:
            power = np.load(power_path).astype(np.float32)
        except Exception as e:
            print(f"Error loading {power_path}: {e}")
            return torch.zeros(in_ch, doppler, 256, 256), torch.zeros(n_cls, 256, 256)

        # Load elevation cube (same filename, different folder)
        fname     = os.path.basename(power_path)
        elev_path = os.path.join(self.elev_dir, fname)
        try:
            elev = np.load(elev_path).astype(np.float32) if os.path.exists(elev_path) else np.zeros_like(power)
        except Exception:
            elev = np.zeros_like(power)

        power = self._preprocess(power, is_power=True)
        elev  = self._preprocess(elev,  is_power=False)

        # [D, A, R] → [D, R, A] to match label axes (elevation, range, azimuth)
        power = np.transpose(power, (0, 2, 1))
        elev  = np.transpose(elev,  (0, 2, 1))

        if in_ch == 1:
            radar_tensor = torch.from_numpy(power[np.newaxis]).float()
        else:
            radar_tensor = torch.from_numpy(np.stack([power, elev], axis=0)).float()

        # Radar-to-LiDAR range shift using per-frame calibration file
        if label_path:
            radar_tensor = self._apply_range_shift(radar_tensor, label_path)

        # Elevation masking (suppress low-power elevation bins)
        radar_tensor = self._apply_elevation_mask(radar_tensor)

        # Load label
        label_valid = False
        if label_path and os.path.exists(label_path):
            try:
                label = np.load(label_path)
                if label.size == 0:
                    raise ValueError("empty array")
                label_tensor = torch.from_numpy(label.astype(np.float32))
                if label_tensor.ndim == 2:
                    label_tensor = label_tensor.unsqueeze(0)
                label_valid = True
            except Exception as e:
                import warnings
                warnings.warn(f"Skipping corrupt label file ({e}): {label_path}")

        if label_valid:
            # Apply bounding-box crop if enabled
            label_tensor = self._apply_bbox_filter(label_tensor, label_path, n_cls)
            # Align azimuth: LiDAR and radar have opposite azimuth conventions
            label_tensor = torch.flip(label_tensor, [-1])
        else:
            _, _, h, w   = radar_tensor.shape
            label_tensor = torch.zeros((n_cls, h, w), dtype=torch.float)

        return radar_tensor, label_tensor

    # ── Preprocessing helpers ─────────────────────────────────────────────────

    def _preprocess(self, data, is_power=True):
        # Transpose to (D, H, W) if stored as (H, W, D)
        if data.ndim == 3 and data.shape[2] > data.shape[0]:
            data = data.transpose(2, 0, 1)

        model_cfg        = self.config.get('model', {}) if self.config else {}
        use_full_doppler = model_cfg.get('use_full_doppler', False)
        doppler_pool     = model_cfg.get('doppler_pool', 'max')   # 'max' | 'mean' | 'stride'

        # 4× Doppler downsampling: 512 → 128
        # Skip if data already came from rad_power_pooled/ (pre-pooled on disk)
        if is_power and getattr(self, 'doppler_pooled', False):
            return data  # already (128, H, W)

        if not use_full_doppler and data.shape[0] == 512:
            if is_power:
                blocks = data.reshape(128, 4, data.shape[1], data.shape[2])
                if doppler_pool == 'mean':
                    data = blocks.mean(axis=1)
                elif doppler_pool == 'stride':
                    data = data[::4, :, :]
                elif doppler_pool == 'torch_max':
                    # PyTorch max_pool3d along the Doppler axis — matches original model_pipeline
                    t    = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)  # (1,1,512,H,W)
                    t    = F.max_pool3d(t, kernel_size=(4, 1, 1), stride=(4, 1, 1))  # (1,1,128,H,W)
                    data = t.squeeze(0).squeeze(0).numpy()                            # (128,H,W)
                else:                          # 'max' (default, numpy)
                    data = blocks.max(axis=1)
            else:
                data = data[::4, :, :]

        norm = (self.config or {}).get('dataset', {}).get('normalization', {})
        if not norm.get('enable', False):
            return data

        auto = norm.get('auto_normalization', False)
        if is_power:
            if norm.get('power_log_transform', False):
                p_min = self.stats['power_min'] if auto else norm.get('power_min_val', -100.0)
                p_max = self.stats['power_max'] if auto else norm.get('power_max_val', -40.0)
                data  = 10 * np.log10(data + 1e-10)
                data  = np.clip(data, p_min, p_max)
                data  = (data - p_min) / (p_max - p_min + 1e-6)
            else:
                val97 = float(np.percentile(data, 97))
                p_max = val97 if val97 > 1e-6 else 1e-6
                data  = np.clip(data, 0.0, p_max)
                data  = np.sqrt(data / p_max)
                data  = np.clip(data, 0.0, 1.0)
        else:
            if norm.get('normalize_elevation', False):
                max_angle = self.stats['elev_max'] if auto else norm.get('elevation_max_angle', 0.7854)
                if max_angle > 0:
                    data = np.clip(data / max_angle, -1.0, 1.0)

        return data

    def _apply_range_shift(self, radar_tensor, label_path):
        """Shift radar range bins to compensate for sensor offset from calibration file."""
        label_text_dir = (self.config or {}).get('dataset', {}).get('label_text_dir', '')
        if not label_text_dir:
            return radar_tensor
        txt_path = os.path.join(label_text_dir, os.path.basename(label_path).replace('.npy', '.txt'))
        if not os.path.exists(txt_path):
            return radar_tensor
        try:
            with open(txt_path) as f:
                content = f.read()
            match = re.search(r'"Translation_Radar_to_Lidar":\s*([-\d\s.]+)\s*,', content)
            if match:
                vals = np.array(match.group(1).strip().split(), dtype=float)
                if len(vals) >= 1:
                    shift_bins = -int(round(vals[0] / (25.6 / 256)))
                    if shift_bins != 0:
                        radar_tensor = torch.roll(radar_tensor, shifts=shift_bins, dims=-2)
                        if shift_bins > 0:
                            radar_tensor[..., :shift_bins, :] = 0
                        else:
                            radar_tensor[..., shift_bins:, :] = 0
        except Exception:
            pass
        return radar_tensor

    def _apply_elevation_mask(self, radar_tensor):
        """Zero out elevation values where radar power is below threshold."""
        norm = (self.config or {}).get('dataset', {}).get('normalization', {})
        if not (norm.get('enable', False) and norm.get('mask_elevation', False)):
            return radar_tensor
        if radar_tensor.shape[0] < 2:
            return radar_tensor
        p_min      = norm.get('power_min_val', -100.0)
        p_max      = norm.get('power_max_val', -40.0)
        thresh     = norm.get('power_threshold', -60.0)
        thresh_n   = max(0.0, min(1.0, (thresh - p_min) / (p_max - p_min)))
        mask       = (radar_tensor[0] > thresh_n).float()
        radar_tensor[1] = radar_tensor[1] * mask
        return radar_tensor

    def _apply_bbox_filter(self, label_tensor, label_path, n_cls):
        """Restrict labels to within a calibrated bounding box region."""
        if not (self.config or {}).get('dataset', {}).get('filter_bboxes', False):
            return label_tensor
        label_text_dir = (self.config or {}).get('dataset', {}).get('label_text_dir', '')
        if not label_text_dir:
            return label_tensor
        txt_path = os.path.join(label_text_dir, os.path.basename(label_path).replace('.npy', '.txt'))
        if not os.path.exists(txt_path):
            return label_tensor
        try:
            with open(txt_path) as f:
                content = f.read()
            match = re.search(r'"BoundingBox":([\d\s.-]+),', content)
            if not match:
                return label_tensor
            corners = np.array(match.group(1).split(), dtype=float).reshape(-1, 3)
            x, y, z = corners[:, 0], corners[:, 1], corners[:, 2]
            R_BINS, A_BINS = 256, 256
            MAX_RANGE, FOV_RAD, PHI_MAX = 25.6, np.deg2rad(180.0), 0.5236
            r = np.sqrt(x**2 + y**2)
            r_idx = np.clip(np.floor(r / MAX_RANGE * R_BINS).astype(int), 0, R_BINS - 1)
            sin_theta = np.sin(np.arctan2(y, x))
            sin_half  = np.sin(FOV_RAD / 2.0)
            a_idx = np.clip(np.floor(((sin_theta + sin_half) / (2 * sin_half)) * A_BINS).astype(int), 0, A_BINS - 1)
            phi   = np.arctan2(z, r) - np.deg2rad(5.0)
            z_idx = np.clip(np.floor(((np.sin(phi) + np.sin(PHI_MAX)) / (2 * np.sin(PHI_MAX))) * n_cls).astype(int), 0, n_cls - 1)
            mask = torch.zeros_like(label_tensor)
            mask[z_idx.min():z_idx.max()+1, r_idx.min():r_idx.max()+1, a_idx.min():a_idx.max()+1] = 1
            label_tensor = label_tensor * mask
        except Exception as e:
            print(f"BBox filter error {txt_path}: {e}")
        return label_tensor
