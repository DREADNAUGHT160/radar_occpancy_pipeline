#!/usr/bin/env python3
"""
eval_preflight.py — mandatory pre-flight checks before thesis_eval.

Checks (all must PASS):
  1. Calib -> rad_power_pooled sync per RC, with per-range-band breakdown.
  2. Label-guard: fog/rain RCs with label dirs must have the
     'has_labels and weather == clear' fix in thesis_eval.py.
     If labels exist but fix is absent -> FAIL, do not proceed.
  3. Radar_frame coverage: calib files' internal Radar_frame timestamp
     must match a pooled power frame within 200 ms. This is what
     _find_calib_by_radar_frame uses; low coverage -> few bounding boxes.
  4. Elevation pooled negative values: sampled rad_elev_pooled files
     must contain negative values (argmax_gather preserves sign).

Exit 0 on all-PASS, exit 1 on any FAIL.

Usage:
    conda run -n thesis_model python utils/eval_preflight.py \
        --config configs/eval_xxx.yaml
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import yaml

SEP  = "=" * 70
SEP2 = "-" * 70

RANGE_BANDS     = [(0, 5), (5, 15), (15, 25), (25, None)]
RF_THRESHOLD_MS = 200   # _find_calib_by_radar_frame default
LABEL_FIX_RE    = re.compile(r"has_labels\s+and\s+weather\s*==\s*['\"]clear['\"]")


# ── helpers ───────────────────────────────────────────────────────────────────

def _ts_ms(path):
    stem = re.sub(r'\.[a-zA-Z]+$', '', os.path.basename(path))
    m    = re.search(r'(\d+\.\d+|\d+)', stem)
    val  = float(m.group(1))
    return val * 1000 if val < 1e11 else val


def _parse_range(txt_path):
    try:
        with open(txt_path) as f:
            text = f.read()
        m = re.search(r'"Center_front"\s*:\s*([\d.\-e+]+)\s+([\d.\-e+]+)', text)
        if not m:
            return None
        return float(np.sqrt(float(m.group(1))**2 + float(m.group(2))**2))
    except Exception:
        return None


def _radar_frame_ts(txt_path):
    try:
        with open(txt_path) as f:
            text = f.read()
        m = re.search(r'"Radar_frame"\s*:\s*"?([\d.]+)', text)
        if not m:
            return None
        return _ts_ms(m.group(1))
    except Exception:
        return None


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Check 1: calib -> rad_power_pooled sync ───────────────────────────────────

def check_calib_pooled_sync(rc_dir, calib_sf, pooled_sf, threshold_ms, rc_name):
    calib_dir  = os.path.join(rc_dir, calib_sf)
    pooled_dir = os.path.join(rc_dir, pooled_sf)

    calib_files  = sorted(glob.glob(os.path.join(calib_dir,  '*.txt')))
    pooled_files = sorted(glob.glob(os.path.join(pooled_dir, '*.npy')))

    r = {'rc': rc_name, 'status': 'PASS', 'issues': []}

    if not calib_files:
        r['status'] = 'FAIL'
        r['issues'].append(f"No calib files found in {calib_dir}")
        return r
    if not pooled_files:
        r['status'] = 'FAIL'
        r['issues'].append(f"No rad_power_pooled files found in {pooled_dir}")
        return r

    pooled_ts = np.array([_ts_ms(f) for f in pooled_files])

    rows = []
    for cf in calib_files:
        rng = _parse_range(cf)
        ct  = _ts_ms(cf)
        idx = int(np.argmin(np.abs(pooled_ts - ct)))
        dlt = float(np.abs(pooled_ts[idx] - ct))
        rows.append({'range': rng, 'delta': dlt, 'ok': dlt <= threshold_ms})

    deltas = np.array([x['delta'] for x in rows])
    r['total']      = len(rows)
    r['synced']     = sum(1 for x in rows if x['ok'])
    r['delta_mean'] = float(deltas.mean())
    r['delta_max']  = float(deltas.max())

    bands = {}
    for lo, hi in RANGE_BANDS:
        label     = f"{lo}-{hi if hi else 'inf'}m"
        band_rows = [x for x in rows if x['range'] is not None
                     and x['range'] >= lo and (hi is None or x['range'] < hi)]
        bands[label] = {'total': len(band_rows),
                        'synced': sum(1 for x in band_rows if x['ok'])}
    r['bands'] = bands

    bad = [x for x in rows if not x['ok']]
    if bad:
        r['status'] = 'FAIL'
        r['issues'].append(
            f"{len(bad)}/{len(rows)} calib files exceed {threshold_ms} ms threshold")

    return r


# ── Check 2: label guard for fog/rain ─────────────────────────────────────────

def check_label_guard(base_dir, eval_splits, label_sf):
    weather_rcs = []
    for weather, rcs in eval_splits.items():
        if weather in ('fog', 'rain'):
            for rc in (rcs or []):
                weather_rcs.append((weather, rc))

    r = {'status': 'PASS', 'issues': [], 'warnings': []}
    rcs_with_labels = []
    for weather, rc in weather_rcs:
        ldir = os.path.join(base_dir, rc, label_sf)
        if os.path.isdir(ldir) and glob.glob(os.path.join(ldir, '*.npy')):
            rcs_with_labels.append((weather, rc))

    r['rcs_with_labels'] = rcs_with_labels

    if not rcs_with_labels:
        r['note'] = 'No label dirs found for fog/rain RCs — calib-driven path active by default'
        return r

    # Labels exist for weather RCs -> verify the fix is in thesis_eval.py
    candidates = [
        os.path.join(os.getcwd(), 'utils', 'thesis_eval.py'),
        os.path.join(os.getcwd(), 'thesis_eval.py'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'thesis_eval.py'),
    ]
    thesis_eval_path = next((p for p in candidates if os.path.exists(p)), None)

    fix_present = False
    if thesis_eval_path:
        with open(thesis_eval_path) as f:
            fix_present = bool(LABEL_FIX_RE.search(f.read()))

    r['thesis_eval_path'] = thesis_eval_path or '(not found)'
    r['fix_present']      = fix_present

    if not fix_present:
        r['status'] = 'FAIL'
        r['issues'].append(
            f"{len(rcs_with_labels)} fog/rain RC(s) have label dirs but "
            f"thesis_eval.py is MISSING the `has_labels and weather == 'clear'` "
            f"guard — label-sync would skip calib-aligned frames. Apply the fix.")
    else:
        r['warnings'].append(
            f"{len(rcs_with_labels)} fog/rain RC(s) have label dirs; "
            f"fix confirmed present -> calib-driven path will activate correctly")

    return r


# ── Check 3: Radar_frame -> pooled coverage ───────────────────────────────────

def check_radar_frame_coverage(rc_dir, calib_sf, pooled_sf, rc_name):
    calib_dir  = os.path.join(rc_dir, calib_sf)
    pooled_dir = os.path.join(rc_dir, pooled_sf)

    calib_files  = sorted(glob.glob(os.path.join(calib_dir,  '*.txt')))
    pooled_files = sorted(glob.glob(os.path.join(pooled_dir, '*.npy')))

    r = {'rc': rc_name, 'status': 'PASS', 'issues': []}

    if not calib_files or not pooled_files:
        r['status'] = 'SKIP'
        return r

    pooled_ts = np.array([_ts_ms(f) for f in pooled_files])
    matched = no_rf = bad = 0

    for cf in calib_files:
        rf_ts = _radar_frame_ts(cf)
        if rf_ts is None:
            no_rf += 1
            continue
        idx = int(np.argmin(np.abs(pooled_ts - rf_ts)))
        dlt = float(np.abs(pooled_ts[idx] - rf_ts))
        if dlt <= RF_THRESHOLD_MS:
            matched += 1
        else:
            bad += 1

    total = len(calib_files)
    denom = total - no_rf
    r['total']   = total
    r['no_rf']   = no_rf
    r['matched'] = matched
    r['bad']     = bad
    r['pct']     = 100.0 * matched / denom if denom > 0 else 0.0

    if no_rf == total:
        r['status'] = 'WARN'
        r['issues'].append("No Radar_frame field found in any calib file")
    elif r['pct'] < 50.0:
        r['status'] = 'WARN'
        r['issues'].append(
            f"Only {r['pct']:.0f}% matched via Radar_frame (within {RF_THRESHOLD_MS}ms) "
            f"— expect low bounding-box coverage in eval")

    return r


# ── Check 4: elevation pooled has negative values ─────────────────────────────

def check_elev_negatives(rc_dir, elev_sf, rc_name, n_sample=10):
    elev_dir   = os.path.join(rc_dir, elev_sf)
    elev_files = sorted(glob.glob(os.path.join(elev_dir, '*.npy')))

    r = {'rc': rc_name, 'status': 'PASS', 'issues': []}

    if not elev_files:
        r['status'] = 'FAIL'
        r['issues'].append(f"No elev_pooled files found in {elev_dir}")
        return r

    step   = max(1, len(elev_files) // n_sample)
    sample = elev_files[::step][:n_sample]
    neg_count = sum(1 for f in sample if np.load(f).min() < 0)

    r['total_files'] = len(elev_files)
    r['sampled']     = len(sample)
    r['with_neg']    = neg_count

    if neg_count == 0:
        r['status'] = 'FAIL'
        r['issues'].append(
            f"None of {len(sample)} sampled files have negative values — "
            f"argmax_gather should preserve sign; check elev_pool.py")

    return r


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Pre-flight checks for thesis_eval')
    ap.add_argument('--config', required=True, help='eval YAML config (same as thesis_eval.py)')
    args = ap.parse_args()

    cfg         = _load_yaml(args.config)
    base_dir    = cfg.get('base_dir', '')
    subs        = cfg.get('subfolders', {})
    calib_sf    = subs.get('calib',  'calib')
    label_sf    = subs.get('labels', 'labels')
    elev_sf     = 'rad_elev_pooled'
    pooled_sf   = 'rad_power_pooled'
    thresh_ms   = cfg.get('sync_threshold_ms', 100)
    eval_splits = cfg.get('eval_splits', {})

    all_rcs = []
    for weather, rcs in eval_splits.items():
        for rc in (rcs or []):
            all_rcs.append((weather, rc))

    print(SEP)
    print("  THESIS-EVAL PRE-FLIGHT CHECK")
    print(f"  Config  : {args.config}")
    print(f"  Base    : {base_dir}")
    print(f"  RCs     : {[rc for _, rc in all_rcs]}")
    print(f"  Sync thr: {thresh_ms} ms")
    print(SEP)

    overall_pass = True

    # ── 1. Calib → pooled sync ────────────────────────────────────────────────
    print("\nCHECK 1 — Calib -> rad_power_pooled sync (per-range-band)")
    print(SEP2)
    for weather, rc in all_rcs:
        rc_d = os.path.join(base_dir, rc)
        res  = check_calib_pooled_sync(rc_d, calib_sf, pooled_sf, thresh_ms, rc)
        tag  = res['status']
        if tag == 'FAIL':
            overall_pass = False
        print(f"  {rc} [{weather}] -> [{tag}]")
        if 'total' in res:
            print(f"    frames  : {res['synced']}/{res['total']} synced  "
                  f"(mean {res['delta_mean']:.1f} ms  max {res['delta_max']:.1f} ms)")
            for band, bst in res['bands'].items():
                if bst['total'] > 0:
                    ok = 'ok' if bst['synced'] == bst['total'] else 'MISS'
                    print(f"    {band:12s}: {bst['synced']}/{bst['total']} [{ok}]")
        for iss in res.get('issues', []):
            print(f"    [FAIL] {iss}")

    # ── 2. Label guard ─────────────────────────────────────────────────────────
    print(f"\nCHECK 2 — Label-sync guard for fog/rain RCs")
    print(SEP2)
    lg  = check_label_guard(base_dir, eval_splits, label_sf)
    tag = lg['status']
    if tag == 'FAIL':
        overall_pass = False
    print(f"  [{tag}]")
    if lg.get('rcs_with_labels'):
        print(f"  Fog/rain RCs with label dirs : {[rc for _, rc in lg['rcs_with_labels']]}")
        fix = lg.get('fix_present', False)
        print(f"  weather=='clear' fix present : {'YES' if fix else 'NO  <-- FAIL'}")
        print(f"  thesis_eval.py               : {lg.get('thesis_eval_path')}")
    else:
        print(f"  {lg.get('note', 'OK')}")
    for iss in lg.get('issues', []):
        print(f"  [FAIL] {iss}")
    for w in lg.get('warnings', []):
        print(f"  [~] {w}")

    # ── 3. Radar_frame coverage ───────────────────────────────────────────────
    print(f"\nCHECK 3 — Radar_frame -> pooled frame coverage ({RF_THRESHOLD_MS} ms window)")
    print(SEP2)
    for weather, rc in all_rcs:
        rc_d = os.path.join(base_dir, rc)
        res  = check_radar_frame_coverage(rc_d, calib_sf, pooled_sf, rc)
        tag  = res['status']
        if tag == 'FAIL':
            overall_pass = False
        print(f"  {rc} [{tag}]")
        if 'total' in res:
            print(f"    matched : {res['matched']}/{res['total'] - res['no_rf']} "
                  f"({res['pct']:.0f}%)   no_rf_field={res['no_rf']}   bad={res['bad']}")
        for iss in res.get('issues', []):
            lvl = 'WARN' if res['status'] == 'WARN' else 'FAIL'
            print(f"    [{lvl}] {iss}")

    # ── 4. Elevation negative values ──────────────────────────────────────────
    print(f"\nCHECK 4 — Elevation pooled files have negative values")
    print(SEP2)
    for weather, rc in all_rcs:
        rc_d = os.path.join(base_dir, rc)
        res  = check_elev_negatives(rc_d, elev_sf, rc)
        tag  = res['status']
        if tag == 'FAIL':
            overall_pass = False
        print(f"  {rc} [{tag}]")
        if 'sampled' in res:
            print(f"    total={res['total_files']}  sampled={res['sampled']}  "
                  f"with_negatives={res['with_neg']}/{res['sampled']}")
        for iss in res.get('issues', []):
            print(f"    [FAIL] {iss}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print(SEP)
    if overall_pass:
        print("  RESULT : ALL CHECKS PASSED -- safe to run thesis_eval")
    else:
        print("  RESULT : PREFLIGHT FAILED -- fix the issues above before running thesis_eval")
    print(SEP)

    sys.exit(0 if overall_pass else 1)


if __name__ == '__main__':
    main()
