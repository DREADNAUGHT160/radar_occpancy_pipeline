"""
Shared config helpers.

resolve_splits(config)
    When dataset.auto_split.enabled is true, scans dataset.base_dir for RC
    folders and fills in config.dataset.train / val / test using the configured
    ratios and seed.  Returns the (possibly mutated) config dict.
    Called once at startup by both train.py and predict.py.
"""
import os
import copy
import numpy as np


def resolve_splits(config):
    """Populate train/val/test lists from auto_split if enabled.

    Returns a deep-copied config with dataset.train/val/test set.
    No-ops if auto_split.enabled is false or lists are already populated.
    """
    ds   = config['dataset']
    auto = ds.get('auto_split', {})

    if not auto.get('enabled', False):
        return config

    # If explicit lists are already set, respect them.
    if ds.get('train') or ds.get('val') or ds.get('test'):
        return config

    base_dir = ds.get('base_dir', '')
    if not base_dir or not os.path.isdir(base_dir):
        print(f"auto_split: base_dir '{base_dir}' not found — skipping auto split.")
        return config

    rad_power_sub = ds.get('subfolders', {}).get('rad_power', 'rad_power')

    # Discover RC folders: any direct child that contains a rad_power subfolder.
    rc_folders = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
        and os.path.isdir(os.path.join(base_dir, d, rad_power_sub))
    ])

    if not rc_folders:
        print(f"auto_split: no RC folders with '{rad_power_sub}/' found in {base_dir}.")
        return config

    rng = np.random.default_rng(auto.get('seed', 42))
    shuffled = rc_folders.copy()
    rng.shuffle(shuffled)

    n           = len(shuffled)
    train_ratio = float(auto.get('train_ratio', 0.70))
    val_ratio   = float(auto.get('val_ratio',   0.15))
    n_train     = max(1, round(n * train_ratio))
    n_val       = max(0, round(n * val_ratio))
    n_test      = n - n_train - n_val

    train_split = shuffled[:n_train]
    val_split   = shuffled[n_train:n_train + n_val]
    test_split  = shuffled[n_train + n_val:]

    print(f"auto_split: {n} RC folders found in {base_dir}")
    print(f"  train ({len(train_split)}): {train_split}")
    print(f"  val   ({len(val_split)}):   {val_split}")
    print(f"  test  ({len(test_split)}):  {test_split}")

    config = copy.deepcopy(config)
    config['dataset']['train'] = train_split
    config['dataset']['val']   = val_split
    config['dataset']['test']  = test_split
    return config
