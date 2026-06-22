"""
Timestamp sync diagnostic tool.

Give it one or more raw recording folders and it checks how many radar (.mat),
LiDAR (.h5), and camera (.png) frames match by timestamp.

Outputs two CSV files:
  sync_report.csv        — one row per folder (summary)
  sync_report_frames.csv — one row per radar frame (full detail)

Usage:
  python dataset/check_sync.py --folder "D:/data/18_RC019"
  python dataset/check_sync.py --folder "D:/data/18_RC019" "D:/data/12_RC013"
  python dataset/check_sync.py --folder "D:/data/18_RC019" --threshold 100 --out sync_report.csv
"""

import os
import re
import glob
import argparse
import csv
import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# ── Timestamp helpers ─────────────────────────────────────────────────────────

def detect_format(value):
    return 'decimal_s' if value < 1e11 else 'int_ms'


def extract_ts_ms(filepath):
    """Extract timestamp from filename as float ms — LiDAR decimal-s kept full precision, radar int-ms divided by 1000 then back."""
    stem = os.path.basename(filepath)
    stem = re.sub(r'\.(npy|h5|png|txt|mat)$', '', stem)
    m = re.search(r'[\d.]+', stem)
    if not m:
        return None, None, None
    raw_str = m.group(0)
    val     = float(raw_str)
    fmt     = detect_format(val)
    if fmt == 'decimal_s':
        ts_ms      = val * 1000          # float ms — no truncation
        conversion = f'{raw_str} * 1000 = {ts_ms}'
    else:
        ts_ms      = float(val)
        conversion = f'float({raw_str}) = {ts_ms}'
    return ts_ms, raw_str, conversion


def ts_from_mat(mat_path):
    """Read ms timestamp directly from HDF5 .mat TimeStamp field."""
    if not HAS_H5PY:
        return None
    try:
        with h5py.File(mat_path, 'r') as f:
            return int(f['TimeStamp'][0, 0])
    except Exception:
        return None


def gap_stats(ts_array):
    if len(ts_array) < 2:
        return 0, 0
    gaps = np.diff(ts_array)
    return int(gaps.max()), int((gaps > 200).sum())


def find_source_folder(raw_data_dir, rc_name):
    for pattern in [f'*_{rc_name}', f'*_{rc_name.upper()}', rc_name]:
        matches = glob.glob(os.path.join(raw_data_dir, pattern))
        if matches:
            return matches[0]
    return None


# ── Per-RC analysis ───────────────────────────────────────────────────────────

def analyse_rc(rc_name, src_folder, threshold_ms):
    """
    Returns (summary_row dict, frame_rows list of dicts).
    """
    notes      = []
    frame_rows = []

    summary = {
        'RC_Name':               rc_name,
        'Source_Folder':         os.path.basename(src_folder),
        'Radar_Mat_Files':       0,
        'Radar_Start_ms':        '',
        'Radar_End_ms':          '',
        'Radar_Span_s':          '',
        'Radar_Avg_Gap_ms':      '',
        'Radar_Ts_Source':       'HDF5 TimeStamp field (int ms)',
        'LiDAR_H5_Files':        0,
        'LiDAR_Start_ms':        '',
        'LiDAR_End_ms':          '',
        'LiDAR_Span_s':          '',
        'LiDAR_Avg_Gap_ms':      '',
        'LiDAR_Max_Gap_ms':      '',
        'LiDAR_Gaps_Over_200ms': '',
        'LiDAR_Ts_Format':       '',
        'LiDAR_Ts_Conversion':   '',
        'PCO_PNG_Files':         0,
        'PCO_Start_ms':          '',
        'PCO_End_ms':            '',
        'PCO_Span_s':            '',
        'PCO_Avg_Gap_ms':        '',
        'PCO_Max_Gap_ms':        '',
        'PCO_Gaps_Over_200ms':   '',
        'PCO_Ts_Format':         '',
        'PCO_Ts_Conversion':     '',
        'Threshold_ms':          threshold_ms,
        'Radar_LiDAR_Matched':   0,
        'Radar_LiDAR_Match_Pct': '',
        'Radar_PCO_Matched':     0,
        'Radar_PCO_Match_Pct':   '',
        'Prof_LiDAR_Total':      0,
        'Prof_Within_Threshold': 0,
        'Prof_Bad_Matches':      0,
        'Prof_Match_Pct':        '',
        'Notes':                 '',
    }

    # ── Radar .mat ────────────────────────────────────────────────────────────
    radar_src = os.path.join(src_folder, 'Radar')
    mat_files = sorted(glob.glob(os.path.join(radar_src, '*.mat')))

    if not mat_files:
        notes.append('No .mat files in Radar/')
        summary['Notes'] = ' | '.join(notes)
        return summary, frame_rows

    if not HAS_H5PY:
        notes.append('h5py not installed')
        summary['Notes'] = ' | '.join(notes)
        return summary, frame_rows

    radar_data = []
    for f in mat_files:
        ts = ts_from_mat(f)
        if ts is not None:
            radar_data.append((ts, os.path.basename(f)))
    radar_data.sort(key=lambda x: x[0])

    if not radar_data:
        notes.append('Could not read TimeStamp from .mat files')
        summary['Notes'] = ' | '.join(notes)
        return summary, frame_rows

    radar_ts  = np.array([r[0] for r in radar_data])
    radar_avg = round(float(np.diff(radar_ts).mean()), 1) if len(radar_ts) > 1 else 0

    summary['Radar_Mat_Files']  = len(radar_ts)
    summary['Radar_Start_ms']   = int(radar_ts[0])
    summary['Radar_End_ms']     = int(radar_ts[-1])
    summary['Radar_Span_s']     = round((radar_ts[-1] - radar_ts[0]) / 1000, 3)
    summary['Radar_Avg_Gap_ms'] = radar_avg

    # ── LiDAR .h5 ─────────────────────────────────────────────────────────────
    lidar_src   = os.path.join(src_folder, 'data', 'pcd')
    lidar_files = sorted(glob.glob(os.path.join(lidar_src, '*.h5')))
    lidar_data  = []

    if lidar_files:
        for f in lidar_files:
            ts_ms, raw_str, conv = extract_ts_ms(f)
            if ts_ms is not None:
                lidar_data.append((ts_ms, raw_str, conv, os.path.basename(f)))
        lidar_data.sort(key=lambda x: x[0])

        lidar_ts = np.array([d[0] for d in lidar_data])
        max_gap, n_big = gap_stats(lidar_ts)
        sample_fmt = detect_format(float(lidar_data[0][1]))
        sample_conv = lidar_data[0][2].split(' = ')[0]

        summary['LiDAR_H5_Files']        = len(lidar_ts)
        summary['LiDAR_Start_ms']        = int(lidar_ts[0])
        summary['LiDAR_End_ms']          = int(lidar_ts[-1])
        summary['LiDAR_Span_s']          = round((lidar_ts[-1] - lidar_ts[0]) / 1000, 3)
        summary['LiDAR_Avg_Gap_ms']      = round(float(np.diff(lidar_ts).mean()), 1) if len(lidar_ts) > 1 else 0
        summary['LiDAR_Max_Gap_ms']      = max_gap
        summary['LiDAR_Gaps_Over_200ms'] = n_big
        summary['LiDAR_Ts_Format']       = sample_fmt
        summary['LiDAR_Ts_Conversion']   = ('int(val*1000) — decimal s to ms'
                                             if sample_fmt == 'decimal_s'
                                             else 'int(val) — already ms')
        if max_gap > 500:
            notes.append(f'LiDAR dropout {max_gap}ms')
    else:
        notes.append('No LiDAR .h5 files in data/pcd/')

    # ── PCO .png ──────────────────────────────────────────────────────────────
    pco_src   = os.path.join(src_folder, 'data', 'pco')
    pco_files = sorted(glob.glob(os.path.join(pco_src, '*.png')))
    pco_data  = []

    if pco_files:
        for f in pco_files:
            ts_ms, raw_str, conv = extract_ts_ms(f)
            if ts_ms is not None:
                pco_data.append((ts_ms, raw_str, conv, os.path.basename(f)))
        pco_data.sort(key=lambda x: x[0])

        pco_ts = np.array([d[0] for d in pco_data])
        max_gap, n_big = gap_stats(pco_ts)
        sample_fmt = detect_format(float(pco_data[0][1]))

        summary['PCO_PNG_Files']        = len(pco_ts)
        summary['PCO_Start_ms']         = int(pco_ts[0])
        summary['PCO_End_ms']           = int(pco_ts[-1])
        summary['PCO_Span_s']           = round((pco_ts[-1] - pco_ts[0]) / 1000, 3)
        summary['PCO_Avg_Gap_ms']       = round(float(np.diff(pco_ts).mean()), 1) if len(pco_ts) > 1 else 0
        summary['PCO_Max_Gap_ms']       = max_gap
        summary['PCO_Gaps_Over_200ms']  = n_big
        summary['PCO_Ts_Format']        = sample_fmt
        summary['PCO_Ts_Conversion']    = ('int(val*1000) — decimal s to ms'
                                           if sample_fmt == 'decimal_s'
                                           else 'int(val) — already ms')
        if max_gap > 500:
            notes.append(f'PCO dropout {max_gap}ms')
    else:
        notes.append('No PCO .png files in data/pco/')

    summary['Notes'] = ' | '.join(notes)

    # ── Per-frame rows ────────────────────────────────────────────────────────
    lidar_ts_arr = np.array([d[0] for d in lidar_data]) if lidar_data else np.array([])
    pco_ts_arr   = np.array([d[0] for d in pco_data])   if pco_data   else np.array([])

    lidar_matched_count = 0
    pco_matched_count   = 0

    for idx, (r_ts, r_file) in enumerate(radar_data):
        row = {
            'RC_Name':            rc_name,
            'Frame_Index':        idx + 1,
            'Radar_File':         r_file,
            'Radar_Ts_ms':        r_ts,
            'Radar_Ts_Source':    'HDF5 TimeStamp field (int ms)',

            'LiDAR_Matched':      'NO',
            'LiDAR_File':         '',
            'LiDAR_Ts_Original':  '',
            'LiDAR_Ts_ms':        '',
            'LiDAR_Delta_ms':     '',
            'LiDAR_Ts_Conversion':'',

            'PCO_Matched':        'NO',
            'PCO_File':           '',
            'PCO_Ts_Original':    '',
            'PCO_Ts_ms':          '',
            'PCO_Delta_ms':       '',
            'PCO_Ts_Conversion':  '',
            'Prof_Accept':        '',
        }

        if len(lidar_ts_arr) > 0:
            diffs = np.abs(lidar_ts_arr - r_ts)
            best  = int(np.argmin(diffs))
            delta = round(float(diffs[best]), 4)
            d     = lidar_data[best]
            row['LiDAR_File']         = d[3]
            row['LiDAR_Ts_Original']  = d[1]
            row['LiDAR_Ts_ms']        = round(float(d[0]), 4)
            row['LiDAR_Delta_ms']     = delta
            row['LiDAR_Ts_Conversion']= d[2]
            if delta <= threshold_ms:
                row['LiDAR_Matched'] = 'YES'
                row['Prof_Accept']   = 'YES'
                lidar_matched_count += 1
            else:
                row['LiDAR_Matched'] = f'NO (nearest {delta}ms away)'
                row['Prof_Accept']   = f'YES (no threshold — delta={delta}ms)'

        if len(pco_ts_arr) > 0:
            diffs = np.abs(pco_ts_arr - r_ts)
            best  = int(np.argmin(diffs))
            delta = round(float(diffs[best]), 4)
            d     = pco_data[best]
            row['PCO_File']         = d[3]
            row['PCO_Ts_Original']  = d[1]
            row['PCO_Ts_ms']        = round(float(d[0]), 4)
            row['PCO_Delta_ms']     = delta
            row['PCO_Ts_Conversion']= d[2]
            if delta <= threshold_ms:
                row['PCO_Matched'] = 'YES'
                pco_matched_count += 1
            else:
                row['PCO_Matched'] = f'NO (nearest {delta}ms away)'

        frame_rows.append(row)

    summary['Radar_LiDAR_Matched']   = lidar_matched_count
    summary['Radar_LiDAR_Match_Pct'] = f"{100*lidar_matched_count/len(radar_ts):.1f}%"
    summary['Radar_PCO_Matched']      = pco_matched_count
    summary['Radar_PCO_Match_Pct']    = f"{100*pco_matched_count/len(radar_ts):.1f}%"

    # ── Professor's method: lidar-centric, no threshold ───────────────────
    prof_within = 0
    prof_bad    = 0
    if len(lidar_ts_arr) > 0:
        for l_ts in lidar_ts_arr:
            diffs = np.abs(radar_ts - l_ts)
            if diffs.min() <= threshold_ms:
                prof_within += 1
            else:
                prof_bad += 1
    summary['Prof_LiDAR_Total']       = len(lidar_ts_arr)
    summary['Prof_Within_Threshold']  = prof_within
    summary['Prof_Bad_Matches']       = prof_bad
    summary['Prof_Match_Pct']         = f"{100*prof_within/len(lidar_ts_arr):.1f}%" if len(lidar_ts_arr) > 0 else ''

    return summary, frame_rows


# ── Column definitions ────────────────────────────────────────────────────────

SUMMARY_COLS = [
    'RC_Name', 'Source_Folder',
    'Radar_Mat_Files', 'Radar_Start_ms', 'Radar_End_ms', 'Radar_Span_s',
    'Radar_Avg_Gap_ms', 'Radar_Ts_Source',
    'LiDAR_H5_Files', 'LiDAR_Start_ms', 'LiDAR_End_ms', 'LiDAR_Span_s',
    'LiDAR_Avg_Gap_ms', 'LiDAR_Max_Gap_ms', 'LiDAR_Gaps_Over_200ms',
    'LiDAR_Ts_Format', 'LiDAR_Ts_Conversion',
    'PCO_PNG_Files', 'PCO_Start_ms', 'PCO_End_ms', 'PCO_Span_s',
    'PCO_Avg_Gap_ms', 'PCO_Max_Gap_ms', 'PCO_Gaps_Over_200ms',
    'PCO_Ts_Format', 'PCO_Ts_Conversion',
    'Threshold_ms',
    'Radar_LiDAR_Matched', 'Radar_LiDAR_Match_Pct',
    'Radar_PCO_Matched',   'Radar_PCO_Match_Pct',
    'Prof_LiDAR_Total', 'Prof_Within_Threshold', 'Prof_Bad_Matches', 'Prof_Match_Pct',
    'Notes',
]

FRAME_COLS = [
    'RC_Name', 'Frame_Index',
    'Radar_File', 'Radar_Ts_ms', 'Radar_Ts_Source',
    'LiDAR_Matched', 'LiDAR_File', 'LiDAR_Ts_Original', 'LiDAR_Ts_ms',
    'LiDAR_Delta_ms', 'LiDAR_Ts_Conversion',
    'PCO_Matched',   'PCO_File',   'PCO_Ts_Original',   'PCO_Ts_ms',
    'PCO_Delta_ms',  'PCO_Ts_Conversion',
    'Prof_Accept',
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder',    nargs='+', required=True,
                        help='One or more raw recording folders to check '
                             '(e.g. "D:/data/18_RC019" "D:/data/12_RC013")')
    parser.add_argument('--threshold', type=int, default=100,
                        help='Timestamp match threshold in ms (default 100)')
    parser.add_argument('--out',       default='sync_report.csv',
                        help='Summary CSV path. Frames CSV will be <name>_frames.csv')
    args = parser.parse_args()

    threshold = args.threshold

    # Validate folders
    folders = []
    for f in args.folder:
        f = os.path.abspath(f)
        if not os.path.isdir(f):
            print(f"[WARN] Folder not found, skipping: {f}")
        else:
            folders.append(f)

    if not folders:
        print("ERROR: No valid folders provided.")
        return

    print(f"Threshold : {threshold} ms")
    print(f"Folders   : {len(folders)}\n")

    all_summaries  = []
    all_frame_rows = []

    for src in folders:
        rc_name = os.path.basename(src)
        print(f"  Analysing {rc_name} ...")
        summary, frame_rows = analyse_rc(rc_name, src, threshold)
        all_summaries.append(summary)
        all_frame_rows.extend(frame_rows)

        print(f"    Radar frames         : {summary['Radar_Mat_Files']}")
        print(f"    Our method  (LiDAR)  : {summary['Radar_LiDAR_Matched']}/{summary['Radar_Mat_Files']}  ({summary['Radar_LiDAR_Match_Pct']})")
        print(f"    Our method  (PCO)    : {summary['Radar_PCO_Matched']}/{summary['Radar_Mat_Files']}  ({summary['Radar_PCO_Match_Pct']})")
        print(f"    Prof method (LiDAR)  : {summary['Prof_Within_Threshold']}/{summary['Prof_LiDAR_Total']} within {threshold}ms  |  {summary['Prof_Bad_Matches']} bad matches  ({summary['Prof_Match_Pct']})")
        if summary['Notes']:
            print(f"    Notes         : {summary['Notes']}")
        print()

    # Write summary CSV
    out_summary = os.path.abspath(args.out)
    with open(out_summary, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_summaries)

    # Write frames CSV
    base, ext     = os.path.splitext(out_summary)
    out_frames    = base + '_frames' + ext
    with open(out_frames, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FRAME_COLS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_frame_rows)

    print(f"Summary CSV : {out_summary}")
    print(f"Frames  CSV : {out_frames}")


if __name__ == '__main__':
    main()
