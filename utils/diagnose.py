"""
Pipeline Diagnostic Tool
========================
Run this when something goes wrong -- it checks everything and saves
a full report folder you can send for diagnosis.

Usage:
  python utils/diagnose.py --config configs/train_config.yaml
  python utils/diagnose.py --config configs/train_config.yaml --checkpoint checkpoints/<run_id>/best_model.pth

Output:
  diagnostic_output/<timestamp>/
    report.txt          human-readable summary of every check
    report.json         machine-readable (same data)
    plots/              sample data plots (input + GT) for each split
    errors.txt          only the failures -- quick triage file
"""
import os
import sys
import time
import json
import platform
import datetime
import argparse
import traceback
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Colours for terminal output ───────────────────────────────────────────────
GREEN  = '\033[92m'
YELLOW = '\033[93m'
RED    = '\033[91m'
BLUE   = '\033[94m'
RESET  = '\033[0m'
BOLD   = '\033[1m'

def _ok(msg):   return f"  {GREEN}[OK]{RESET}     {msg}"
def _warn(msg): return f"  {YELLOW}[WARN]{RESET}   {msg}"
def _fail(msg): return f"  {RED}[FAIL]{RESET}   {msg}"
def _info(msg): return f"  {BLUE}[INFO]{RESET}   {msg}"


# ── Report accumulator ────────────────────────────────────────────────────────

class Report:
    def __init__(self):
        self.sections = []          # list of (title, lines)
        self.errors   = []          # only failures
        self.warnings = []
        self._cur_title = ''
        self._cur_lines = []
        self._data = {}             # structured JSON data

    def section(self, title):
        if self._cur_lines:
            self.sections.append((self._cur_title, list(self._cur_lines)))
        self._cur_title = title
        self._cur_lines = []
        print(f"\n{BOLD}{'─'*60}{RESET}")
        print(f"{BOLD}  {title}{RESET}")
        print(f"{BOLD}{'─'*60}{RESET}")

    def ok(self, msg, key=None, value=None):
        line = _ok(msg)
        self._cur_lines.append(('OK', msg))
        print(line)
        if key: self._data[key] = value if value is not None else True

    def warn(self, msg, key=None, value=None):
        line = _warn(msg)
        self._cur_lines.append(('WARN', msg))
        self.warnings.append(f"[{self._cur_title}] {msg}")
        print(line)
        if key: self._data[key] = value if value is not None else 'WARNING'

    def fail(self, msg, key=None, value=None):
        line = _fail(msg)
        self._cur_lines.append(('FAIL', msg))
        self.errors.append(f"[{self._cur_title}] {msg}")
        print(line)
        if key: self._data[key] = value if value is not None else 'FAILED'

    def info(self, msg, key=None, value=None):
        line = _info(msg)
        self._cur_lines.append(('INFO', msg))
        print(line)
        if key: self._data[key] = value if value is not None else msg

    def set(self, key, value):
        self._data[key] = value

    def finalise(self):
        if self._cur_lines:
            self.sections.append((self._cur_title, list(self._cur_lines)))

    def save(self, out_dir):
        self.finalise()

        # ── report.txt ────────────────────────────────────────────────────────
        txt_path = os.path.join(out_dir, 'report.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Pipeline Diagnostic Report\n")
            f.write(f"Generated: {datetime.datetime.now()}\n")
            f.write('=' * 60 + '\n\n')

            for title, lines in self.sections:
                f.write(f"\n{'─'*60}\n{title}\n{'─'*60}\n")
                for status, msg in lines:
                    f.write(f"  [{status:4s}]  {msg}\n")

            f.write(f"\n{'='*60}\n")
            f.write(f"SUMMARY\n{'='*60}\n")
            f.write(f"  Errors   : {len(self.errors)}\n")
            f.write(f"  Warnings : {len(self.warnings)}\n")

            if self.errors:
                f.write(f"\n{'─'*60}\nFAILURES\n{'─'*60}\n")
                for e in self.errors:
                    f.write(f"  {e}\n")

            if self.warnings:
                f.write(f"\n{'─'*60}\nWARNINGS\n{'─'*60}\n")
                for w in self.warnings:
                    f.write(f"  {w}\n")

        # ── errors.txt ────────────────────────────────────────────────────────
        err_path = os.path.join(out_dir, 'errors.txt')
        with open(err_path, 'w', encoding='utf-8') as f:
            f.write(f"Pipeline Diagnostic — Quick Triage\n")
            f.write(f"Generated: {datetime.datetime.now()}\n\n")
            if not self.errors and not self.warnings:
                f.write("No errors or warnings found.\n")
            if self.errors:
                f.write(f"FAILURES ({len(self.errors)})\n{'─'*40}\n")
                for e in self.errors:
                    f.write(f"  {e}\n")
            if self.warnings:
                f.write(f"\nWARNINGS ({len(self.warnings)})\n{'─'*40}\n")
                for w in self.warnings:
                    f.write(f"  {w}\n")

        # ── report.json ───────────────────────────────────────────────────────
        self._data['errors']   = self.errors
        self._data['warnings'] = self.warnings
        self._data['n_errors'] = len(self.errors)
        self._data['n_warnings'] = len(self.warnings)
        json_path = os.path.join(out_dir, 'report.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, indent=2, default=str)

        return txt_path, err_path, json_path


# =============================================================================
# 1. ENVIRONMENT
# =============================================================================

def check_environment(R):
    R.section("1. Environment")

    R.info(f"Platform : {platform.platform()}", 'platform', platform.platform())
    R.info(f"Python   : {sys.version.split()[0]}", 'python', sys.version.split()[0])

    try:
        import torch
        R.ok(f"PyTorch  : {torch.__version__}", 'pytorch', torch.__version__)
    except ImportError:
        R.fail("PyTorch not installed -- cannot continue", 'pytorch', None)
        return None

    # CUDA
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        for i in range(n_gpus):
            props  = torch.cuda.get_device_properties(i)
            vram   = props.total_memory / 1024**3
            R.ok(f"GPU {i}    : {props.name}  |  VRAM: {vram:.1f} GB",
                 f'gpu_{i}', {'name': props.name, 'vram_gb': round(vram, 2)})
        try:
            # allocate a small tensor to confirm GPU is actually usable
            t = torch.zeros(1).cuda()
            del t
            R.ok("GPU usable: test allocation succeeded", 'gpu_usable', True)
        except Exception as e:
            R.fail(f"GPU allocation failed: {e}", 'gpu_usable', False)
    else:
        R.warn("CUDA not available -- training will run on CPU (very slow)", 'cuda', False)

    # Key packages
    pkg_checks = [
        ('numpy',      lambda: __import__('numpy').__version__),
        ('yaml',       lambda: __import__('yaml').__version__),
        ('matplotlib', lambda: __import__('matplotlib').__version__),
        ('tqdm',       lambda: __import__('tqdm').__version__),
        ('PIL',        lambda: __import__('PIL').__version__),
        ('scipy',      lambda: __import__('scipy').__version__),
        ('cv2',        lambda: __import__('cv2').__version__),
    ]
    for pkg, get_ver in pkg_checks:
        try:
            ver = get_ver()
            R.ok(f"{pkg:<12}: {ver}", f'pkg_{pkg}', ver)
        except ImportError:
            if pkg == 'cv2':
                R.warn("opencv-python not installed -- camera projection will be skipped",
                       f'pkg_{pkg}', None)
            else:
                R.fail(f"{pkg} not installed", f'pkg_{pkg}', None)

    return torch


# =============================================================================
# 2. CONFIG
# =============================================================================

def check_config(R, config_path):
    R.section("2. Config File")

    if not os.path.exists(config_path):
        R.fail(f"Config not found: {config_path}", 'config_path', None)
        return None

    R.ok(f"Config file found: {config_path}", 'config_path', config_path)

    try:
        import yaml
        with open(config_path, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        R.ok("Config parsed successfully (valid YAML)", 'config_valid', True)
    except Exception as e:
        R.fail(f"Config parse error: {e}", 'config_valid', False)
        return None

    # Required keys
    required = [
        ('dataset.base_dir',    lambda c: c.get('dataset', {}).get('base_dir')),
        ('training.epochs',     lambda c: c.get('training', {}).get('epochs')),
        ('training.lr',         lambda c: c.get('training', {}).get('lr')),
        ('training.loss',       lambda c: c.get('training', {}).get('loss')),
        ('model.name',          lambda c: c.get('model', {}).get('name')),
        ('logging.output_dir',  lambda c: c.get('logging', {}).get('output_dir')),
    ]
    for key, getter in required:
        val = getter(config)
        if val is not None:
            R.ok(f"{key} = {val}", f'cfg_{key}', val)
        else:
            R.fail(f"Missing required key: {key}", f'cfg_{key}', None)

    # Dataset splits
    ds = config.get('dataset', {})
    for split in ('train', 'val', 'test'):
        folders = ds.get(split, [])
        if folders:
            R.ok(f"dataset.{split}: {len(folders)} folders", f'split_{split}', len(folders))
        elif split == 'train':
            R.fail("dataset.train is empty -- no training data configured", f'split_{split}', 0)
        else:
            R.warn(f"dataset.{split} is empty", f'split_{split}', 0)

    # Numeric sanity
    epochs = config.get('training', {}).get('epochs', 0)
    lr     = config.get('training', {}).get('lr', 0)
    bs     = config.get('dataset',  {}).get('batch_size', 0)

    if epochs <= 0:   R.fail(f"epochs={epochs} must be > 0")
    else:             R.ok(f"epochs = {epochs}")

    if not (1e-6 < lr < 1.0): R.warn(f"lr={lr} looks unusual (expected 1e-5 to 1e-2)")
    else:                      R.ok(f"lr = {lr}")

    if bs <= 0: R.warn(f"batch_size={bs} -- will default to 1")
    else:       R.ok(f"batch_size = {bs}")

    return config


# =============================================================================
# 3. PATHS & DATA
# =============================================================================

def check_paths(R, config):
    R.section("3. Paths & Data Files")

    import numpy as np

    ds       = config.get('dataset', {})
    base_dir = ds.get('base_dir', '')
    sf       = ds.get('subfolders', {})
    pw_sub   = sf.get('rad_power', 'rad_power')
    el_sub   = sf.get('rad_elev',  'rad_elev')
    lb_sub   = sf.get('labels',    'labels')
    ca_sub   = sf.get('calib',     'calib')
    cf_sub   = sf.get('cfar',      'cfar')
    pc_sub   = sf.get('pco',       'pco')

    # base_dir
    if not base_dir:
        R.fail("dataset.base_dir is empty -- set the path to prepared .npy data")
    elif not os.path.isdir(base_dir):
        R.fail(f"base_dir does not exist: {base_dir}")
    else:
        R.ok(f"base_dir exists: {base_dir}", 'base_dir', base_dir)

    summary = {}

    for split in ('train', 'val', 'test'):
        folders = ds.get(split, [])
        if not folders:
            continue

        R.info(f"\n  -- {split.upper()} split ({len(folders)} folders) --")

        for rc in folders:
            rc_dir = os.path.join(base_dir, rc) if base_dir else rc
            rc_ok  = os.path.isdir(rc_dir)
            info   = {'rc': rc, 'split': split}

            if not rc_ok:
                R.fail(f"{rc}: folder not found at {rc_dir}")
                info['exists'] = False
                summary[rc] = info
                continue

            # subfolder checks
            for sub_name, sub_key in [(pw_sub, 'rad_power'), (el_sub, 'rad_elev'),
                                       (lb_sub, 'labels')]:
                sub_path = os.path.join(rc_dir, sub_name)
                if not os.path.isdir(sub_path):
                    R.fail(f"{rc}/{sub_name}: folder missing")
                    info[sub_key] = 0
                else:
                    n = len([f for f in os.listdir(sub_path) if f.endswith('.npy')])
                    info[sub_key] = n
                    if n == 0:
                        R.fail(f"{rc}/{sub_name}: 0 .npy files found")
                    else:
                        R.ok(f"{rc}/{sub_name}: {n} files")

            # optional folders
            for sub_name, sub_key in [(ca_sub, 'calib'), (cf_sub, 'cfar'), (pc_sub, 'pco')]:
                sub_path = os.path.join(rc_dir, sub_name)
                if os.path.isdir(sub_path):
                    n = len(os.listdir(sub_path))
                    R.ok(f"{rc}/{sub_name}: {n} files (optional)")
                    info[sub_key] = n
                else:
                    R.info(f"{rc}/{sub_name}: not present (optional)")
                    info[sub_key] = 0

            # timestamp sync
            pw_dir = os.path.join(rc_dir, pw_sub)
            lb_dir = os.path.join(rc_dir, lb_sub)
            if os.path.isdir(pw_dir) and os.path.isdir(lb_dir):
                pw_files = sorted([f for f in os.listdir(pw_dir) if f.endswith('.npy')])
                lb_files = sorted([f for f in os.listdir(lb_dir) if f.endswith('.npy')])
                if pw_files and lb_files:
                    # quick check: first file shape
                    try:
                        arr = np.load(os.path.join(pw_dir, pw_files[0]))
                        R.ok(f"{rc}/rad_power shape: {arr.shape}  dtype: {arr.dtype}")
                        info['power_shape'] = list(arr.shape)
                        if arr.shape[0] not in (128, 512):
                            R.warn(f"{rc}: unexpected Doppler depth {arr.shape[0]} (expected 128 or 512)")
                    except Exception as e:
                        R.fail(f"{rc}: cannot load power file: {e}")

                    try:
                        arr = np.load(os.path.join(lb_dir, lb_files[0]))
                        R.ok(f"{rc}/labels shape   : {arr.shape}  dtype: {arr.dtype}")
                        info['label_shape'] = list(arr.shape)
                        if arr.shape != (64, 256, 256):
                            R.warn(f"{rc}: unexpected label shape {arr.shape} (expected (64,256,256))")
                    except Exception as e:
                        R.fail(f"{rc}: cannot load label file: {e}")

                    # count sync matches
                    import re
                    def _ts(fname):
                        m = re.search(r'(\d+\.\d+|\d+)', fname)
                        if m:
                            v = float(m.group(0))
                            return int(v * 1000) if v < 1e11 else int(v)
                        return 0

                    pw_ts = np.array([_ts(f) for f in pw_files])
                    lb_ts = np.array([_ts(f) for f in lb_files])
                    matched = 0
                    for t in pw_ts:
                        diff = np.abs(lb_ts - t)
                        if diff.min() < 100:
                            matched += 1
                    pct = matched / len(pw_ts) * 100
                    info['sync_matched'] = matched
                    info['sync_total']   = len(pw_files)
                    if pct < 50:
                        R.fail(f"{rc}: only {matched}/{len(pw_files)} frames synced ({pct:.0f}%) -- "
                               f"check timestamps or sync_threshold_ms")
                    elif pct < 90:
                        R.warn(f"{rc}: {matched}/{len(pw_files)} frames synced ({pct:.0f}%)")
                    else:
                        R.ok(f"{rc}: {matched}/{len(pw_files)} frames synced ({pct:.0f}%)")

            info['exists'] = True
            summary[rc] = info

    R.set('data_summary', summary)
    return summary


# =============================================================================
# 4. GPU MEMORY ESTIMATE
# =============================================================================

def check_gpu_memory(R, config, torch):
    R.section("4. GPU Memory Estimate")

    if not torch.cuda.is_available():
        R.warn("No CUDA GPU -- skipping memory estimate")
        return

    props     = torch.cuda.get_device_properties(0)
    vram_gb   = props.total_memory / 1024**3
    bs        = config.get('dataset', {}).get('batch_size', 6)
    doppler   = config.get('model', {}).get('doppler_depth', 128)
    in_ch     = config.get('model', {}).get('in_channels', 2)

    # Rough estimate: input tensor + 3D conv intermediates
    # Input: B × 2 × 128 × 256 × 256 × 4 bytes
    input_bytes = bs * in_ch * doppler * 256 * 256 * 4
    # 3D conv intermediates roughly 6× the input during backward
    est_gb = (input_bytes * 6) / 1024**3

    R.info(f"GPU VRAM      : {vram_gb:.1f} GB", 'vram_gb', round(vram_gb, 2))
    R.info(f"batch_size    : {bs}", 'batch_size', bs)
    R.info(f"Est. memory   : ~{est_gb:.1f} GB (rough estimate for forward + backward)",
           'est_memory_gb', round(est_gb, 2))

    if est_gb > vram_gb * 0.9:
        R.fail(f"Estimated memory ({est_gb:.1f} GB) exceeds GPU VRAM ({vram_gb:.1f} GB) -- "
               f"CUDA OOM likely. Reduce batch_size to {max(1, bs//2)} or less",
               'memory_ok', False)
    elif est_gb > vram_gb * 0.7:
        R.warn(f"Memory usage will be tight ({est_gb:.1f} GB / {vram_gb:.1f} GB). "
               f"OOM auto-recovery is enabled but consider reducing batch_size",
               'memory_ok', 'tight')
    else:
        R.ok(f"Memory looks fine ({est_gb:.1f} GB / {vram_gb:.1f} GB)",
             'memory_ok', True)

    # Free VRAM check
    try:
        free_gb = torch.cuda.mem_get_info()[0] / 1024**3
        R.info(f"Free VRAM now : {free_gb:.1f} GB", 'free_vram_gb', round(free_gb, 2))
        if free_gb < est_gb:
            R.warn(f"Free VRAM ({free_gb:.1f} GB) < estimated need ({est_gb:.1f} GB) -- "
                   f"other processes may be using the GPU")
    except Exception:
        pass


# =============================================================================
# 5. DATALOADER SPEED
# =============================================================================

def check_dataloader_speed(R, config, torch):
    R.section("5. DataLoader Speed")

    try:
        from dataset.dataloader import RadarDataset
        from torch.utils.data import DataLoader
    except ImportError as e:
        R.fail(f"Cannot import DataLoader: {e}")
        return

    ds_cfg = config.get('dataset', {})
    base_dir   = ds_cfg.get('base_dir', '')
    train_flds = ds_cfg.get('train', [])
    sf         = ds_cfg.get('subfolders', {})

    if not train_flds:
        R.warn("No train folders configured -- skipping speed test")
        return

    # Use first available train folder
    rc_name = None
    rc_dir  = None
    for rc in train_flds:
        d = os.path.join(base_dir, rc) if base_dir else rc
        if os.path.isdir(d):
            rc_name = rc
            rc_dir  = d
            break

    if not rc_dir:
        R.fail("No train folder found on disk -- cannot run speed test")
        return

    ds_sub_cfg = {
        **config,
        'dataset': {
            **ds_cfg,
            'radar_dir':  rc_dir,
            'lidar_path': os.path.join(rc_dir, sf.get('labels', 'labels')),
        }
    }

    try:
        ds     = RadarDataset(rc_dir, augment=False, config=ds_sub_cfg)
        n_test = min(10, len(ds))
        if n_test == 0:
            R.fail("Dataset has 0 matched frames -- cannot test speed")
            return

        bs  = max(1, min(2, ds_cfg.get('batch_size', 1)))
        nw  = ds_cfg.get('num_workers', 0)

        loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw)
        R.info(f"Testing {n_test} batches  (bs={bs}, num_workers={nw})  from {rc_name}")

        t0      = time.time()
        batches = 0
        for i, (imgs, lbls) in enumerate(loader):
            if i >= n_test // bs:
                break
            batches += 1

        elapsed    = time.time() - t0
        frames_sec = (batches * bs) / max(elapsed, 1e-6)
        secs_epoch = len(ds) / frames_sec

        R.ok(f"Loaded {batches * bs} frames in {elapsed:.1f}s  ({frames_sec:.1f} frames/sec)",
             'frames_per_sec', round(frames_sec, 1))
        R.info(f"Est. time per epoch ({len(ds)} frames): {secs_epoch/60:.1f} min",
               'est_epoch_min', round(secs_epoch / 60, 1))

        if frames_sec < 1.0:
            R.warn("Very slow loading (<1 frame/sec) -- increase num_workers or check disk speed")
        elif frames_sec < 5.0:
            R.warn(f"Slow loading ({frames_sec:.1f} frames/sec) -- consider increasing num_workers")
        else:
            R.ok(f"Loading speed looks fine ({frames_sec:.1f} frames/sec)")

    except Exception as e:
        R.fail(f"DataLoader speed test failed: {e}\n    {traceback.format_exc().splitlines()[-1]}")


# =============================================================================
# 6. MODEL
# =============================================================================

def check_model(R, config, torch, checkpoint_path=None):
    R.section("6. Model")

    try:
        from models.factory import ModelFactory
    except ImportError as e:
        R.fail(f"Cannot import ModelFactory: {e}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model
    try:
        model = ModelFactory.get_model(config).to(device)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        R.ok(f"Model created: {config.get('model', {}).get('name', '?')}  "
             f"({n_params:.1f}M params)", 'model_params_M', round(n_params, 2))
    except Exception as e:
        R.fail(f"Model creation failed: {e}\n    {traceback.format_exc().splitlines()[-1]}")
        return

    # Load checkpoint
    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            R.fail(f"Checkpoint not found: {checkpoint_path}", 'checkpoint', None)
        else:
            try:
                state = torch.load(checkpoint_path, map_location=device, weights_only=True)
                model.load_state_dict(state)
                ckpt_mb = os.path.getsize(checkpoint_path) / 1024**2
                R.ok(f"Checkpoint loaded: {checkpoint_path}  ({ckpt_mb:.1f} MB)",
                     'checkpoint', checkpoint_path)
            except RuntimeError as e:
                R.fail(f"Checkpoint incompatible with model: {e}", 'checkpoint', 'incompatible')
                return
            except Exception as e:
                R.fail(f"Checkpoint load failed: {e}", 'checkpoint', 'failed')
                return
    else:
        R.info("No checkpoint provided -- testing forward pass with random weights",
               'checkpoint', None)

    # Forward pass
    try:
        in_ch  = config.get('model', {}).get('in_channels', 2)
        dop    = config.get('model', {}).get('doppler_depth', 128)
        dummy  = torch.zeros(1, in_ch, dop, 256, 256).to(device)
        model.eval()
        with torch.no_grad():
            t0  = time.time()
            out = model(dummy)
            dt  = (time.time() - t0) * 1000
        R.ok(f"Forward pass OK -- input {tuple(dummy.shape)} -> output {tuple(out.shape)}  "
             f"({dt:.1f} ms/frame)", 'forward_pass', True)

        n_cls = config.get('model', {}).get('num_classes', 64)
        if out.shape[1] != n_cls:
            R.warn(f"Output channels ({out.shape[1]}) != num_classes ({n_cls}) in config")
    except torch.cuda.OutOfMemoryError:
        R.fail("CUDA OOM during forward pass with batch_size=1 -- GPU has too little VRAM even for inference",
               'forward_pass', False)
    except Exception as e:
        R.fail(f"Forward pass failed: {e}\n    {traceback.format_exc().splitlines()[-1]}",
               'forward_pass', False)


# =============================================================================
# 7. SAMPLE PLOTS
# =============================================================================

def check_plots(R, config, out_dir):
    R.section("7. Sample Data Plots")

    try:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from dataset.dataloader import RadarDataset
    except ImportError as e:
        R.fail(f"Cannot import plotting dependencies: {e}")
        return

    ds_cfg   = config.get('dataset', {})
    base_dir = ds_cfg.get('base_dir', '')
    sf       = ds_cfg.get('subfolders', {})
    norm_cfg = ds_cfg.get('normalization', {})
    plots_saved = 0

    for split in ('train', 'val', 'test'):
        folders = ds_cfg.get(split, [])
        if not folders:
            continue

        # Take first available folder per split
        for rc in folders:
            rc_dir = os.path.join(base_dir, rc) if base_dir else rc
            if not os.path.isdir(rc_dir):
                continue

            ds_sub = {
                **config,
                'dataset': {
                    **ds_cfg,
                    'radar_dir':  rc_dir,
                    'lidar_path': os.path.join(rc_dir, sf.get('labels', 'labels')),
                }
            }

            try:
                import torch
                ds    = RadarDataset(rc_dir, augment=False, config=ds_sub)
                total = len(ds)
                if total == 0:
                    R.warn(f"{rc}: 0 matched frames -- no plot generated")
                    continue

                n_frames = min(3, total)
                step     = max(1, total // n_frames)
                indices  = [i * step for i in range(n_frames)]

                rc_plot_dir = os.path.join(out_dir, 'plots', split, rc)
                os.makedirs(rc_plot_dir, exist_ok=True)

                for idx in indices:
                    radar_t, label_t = ds[idx]
                    ts = os.path.basename(ds.matched_data[idx]['power']).replace('.npy', '')

                    p_np = radar_t[0].numpy()
                    bev  = p_np.max(axis=0)

                    # AE map
                    max_a  = norm_cfg.get('elevation_max_angle', 0.7854)
                    e_np   = radar_t[1].numpy() if radar_t.shape[0] > 1 else np.zeros_like(p_np)
                    e_bins = ((np.clip(e_np / max_a, -1, 1) + 1) / 2 * 63).clip(0, 63).astype(int)
                    p_m    = p_np[2:]
                    e_b    = e_bins[2:]
                    d, r, a = p_m.shape
                    ag = np.tile(np.arange(a)[np.newaxis, np.newaxis, :], (d, r, 1))
                    ae_s = np.zeros((64, a)); ae_c = np.zeros((64, a))
                    with np.errstate(invalid='ignore'):
                        np.add.at(ae_s, (e_b.flatten(), ag.flatten()), p_m.flatten())
                        np.add.at(ae_c, (e_b.flatten(), ag.flatten()), 1)
                        ae_map = np.where(ae_c > 0, ae_s / ae_c, 0)
                    if ae_map.max() > 0: ae_map /= ae_map.max()

                    lbl_bev = label_t.max(dim=0)[0].numpy()
                    lbl_fv  = label_t.max(dim=1)[0].numpy()

                    xt = np.linspace(0, 255, 7)
                    xl = np.round(np.degrees(np.arcsin(np.linspace(-1, 1, 7)))).astype(int)

                    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
                    fig.suptitle(f"{rc}  [{split}]  |  frame {idx:03d}  |  ts={ts}", fontsize=12)

                    axes[0,0].imshow(bev, cmap='turbo', origin='lower', aspect='auto', vmin=0, vmax=1)
                    axes[0,0].set_title('Input: Power BEV')
                    axes[0,0].set_xticks(xt); axes[0,0].set_xticklabels(xl)

                    axes[0,1].imshow(ae_map, cmap='turbo', origin='lower', aspect='auto')
                    axes[0,1].set_title('Input: AE Map')
                    axes[0,1].set_xticks(xt); axes[0,1].set_xticklabels(xl)

                    axes[0,2].imshow(p_np.max(axis=2), cmap='turbo', origin='lower', aspect='auto')
                    axes[0,2].set_title('Input: RE Map (max over azimuth)')

                    axes[1,0].imshow(lbl_bev, cmap='gray', origin='lower', aspect='auto')
                    axes[1,0].set_title('GT: BEV')
                    axes[1,0].set_xticks(xt); axes[1,0].set_xticklabels(xl)

                    axes[1,1].imshow(lbl_fv, cmap='gray', origin='lower', aspect='auto')
                    axes[1,1].set_title('GT: Front View')
                    axes[1,1].set_xticks(xt); axes[1,1].set_xticklabels(xl)

                    axes[1,2].imshow(label_t.max(dim=2)[0].numpy(), cmap='gray', origin='lower', aspect='auto')
                    axes[1,2].set_title('GT: Side View')

                    plt.tight_layout()
                    fname = os.path.join(rc_plot_dir, f'frame_{idx:03d}_{ts}.png')
                    plt.savefig(fname, bbox_inches='tight', dpi=110)
                    plt.close(fig)
                    plots_saved += 1

                R.ok(f"{split}/{rc}: {len(indices)} plots saved -> {rc_plot_dir}")
                break  # one folder per split is enough for diagnostics

            except Exception as e:
                R.fail(f"{split}/{rc}: plot failed: {e}\n    {traceback.format_exc().splitlines()[-1]}")
                break

    if plots_saved > 0:
        R.info(f"Total plots saved: {plots_saved}", 'plots_saved', plots_saved)
    else:
        R.warn("No plots generated -- check data paths above")


# =============================================================================
# 8. CHECKPOINT FOLDER
# =============================================================================

def check_checkpoints(R, config):
    R.section("8. Checkpoint Folder")

    out_dir = config.get('logging', {}).get('output_dir', 'checkpoints')

    if not os.path.exists(out_dir):
        R.warn(f"Checkpoint dir does not exist yet: {out_dir}  (normal before first run)")
        return

    R.ok(f"Checkpoint dir exists: {out_dir}")

    runs = sorted([d for d in os.listdir(out_dir)
                   if os.path.isdir(os.path.join(out_dir, d))], reverse=True)

    if not runs:
        R.warn("No run folders found inside checkpoint dir -- no training runs recorded yet")
        return

    R.info(f"Found {len(runs)} run folder(s). Latest: {runs[0]}")

    latest = os.path.join(out_dir, runs[0])
    for fname in ('best_model.pth', 'final_model.pth', 'training.log'):
        fpath = os.path.join(latest, fname)
        if os.path.exists(fpath):
            size_mb = os.path.getsize(fpath) / 1024**2
            R.ok(f"{fname}: found ({size_mb:.1f} MB)", f'ckpt_{fname}', fpath)
        else:
            if fname.endswith('.pth'):
                R.fail(f"{fname}: NOT found in {latest} -- training may have crashed before saving",
                       f'ckpt_{fname}', None)
            else:
                R.warn(f"{fname}: not found in {latest}", f'ckpt_{fname}', None)

    R.set('latest_run', latest)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pipeline diagnostic tool")
    parser.add_argument('--config',     default='configs/train_config.yaml')
    parser.add_argument('--checkpoint', default=None,
                        help='Path to a .pth checkpoint to validate')
    parser.add_argument('--out_dir',    default='diagnostic_output',
                        help='Root folder for diagnostic outputs')
    args = parser.parse_args()

    run_id  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Pipeline Diagnostic Tool{RESET}")
    print(f"{BOLD}  Config     : {args.config}{RESET}")
    print(f"{BOLD}  Output     : {out_dir}{RESET}")
    print(f"{BOLD}  Checkpoint : {args.checkpoint or 'none'}{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    R = Report()
    R.set('run_id',     run_id)
    R.set('config',     args.config)
    R.set('checkpoint', args.checkpoint)
    R.set('generated',  str(datetime.datetime.now()))

    # Run all checks
    torch  = check_environment(R)
    config = check_config(R, args.config)

    if config:
        check_paths(R, config)
        if torch:
            check_gpu_memory(R, config, torch)
            check_dataloader_speed(R, config, torch)
            check_model(R, config, torch, args.checkpoint)
        check_plots(R, config, out_dir)
        check_checkpoints(R, config)

    # Final summary
    R.section("Summary")
    total_err  = len(R.errors)
    total_warn = len(R.warnings)

    if total_err == 0 and total_warn == 0:
        R.ok("All checks passed -- pipeline looks healthy")
    elif total_err == 0:
        R.warn(f"{total_warn} warning(s) -- review before running training")
    else:
        R.fail(f"{total_err} error(s) and {total_warn} warning(s) -- fix errors before training")

    txt_path, err_path, json_path = R.save(out_dir)

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Diagnostic complete{RESET}")
    print(f"  Errors   : {RED}{total_err}{RESET}")
    print(f"  Warnings : {YELLOW}{total_warn}{RESET}")
    print(f"\n  Saved to : {os.path.abspath(out_dir)}")
    print(f"  report.txt  -> full details")
    print(f"  errors.txt  -> quick triage (send this first)")
    print(f"  report.json -> machine-readable")
    print(f"  plots/      -> sample data visualisations")
    print(f"{BOLD}{'='*60}{RESET}\n")

    if total_err > 0:
        print(f"{RED}FAILURES:{RESET}")
        for e in R.errors:
            print(f"  {RED}x{RESET} {e}")
    if total_warn > 0:
        print(f"{YELLOW}WARNINGS:{RESET}")
        for w in R.warnings:
            print(f"  {YELLOW}!{RESET} {w}")


if __name__ == '__main__':
    main()
