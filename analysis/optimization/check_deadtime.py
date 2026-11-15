'''
check_deadtime.py

Scan all .Spe files in one or more folders, compute live time, real time,
dead time (%) and rate (counts / live time) for each spectrum, and print
a table with flag codes:

    OK        dead < 5%          safe for resolution measurement
    WARNING   5% < dead < 20%    suspicious, inspect
    CRITICAL  dead > 20%         distorted peak, EXCLUDE

Also saves the full table to results/tables/optimization/deadtime_summary.csv
and prints the list of spectra that should be excluded or re-measured.

Workflow
--------
    $ uv run check_deadtime.py

Customization
-------------
- FOLDERS: dict {label: path}. Add or remove entries as needed.
- DEAD_OK / DEAD_WARN: thresholds on dead time (in %).
- RATE_WARN: threshold on rate (in Hz), only for informational purposes.
'''

from pathlib import Path
import numpy as np
import pandas as pd

from lab_compton.maestro_reader import read_dir
from lab_compton.config import parse_filename


# =========================================================================
# CONFIG
# =========================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT    = PROJECT_ROOT / "data" / "Optimization"
OUT_CSV      = PROJECT_ROOT / "results" / "tables" / "optimization" / "deadtime_summary.csv"

FOLDERS = {
    "Scint1": DATA_ROOT / "Scintillator1" / "Spe",
    "Scint0": DATA_ROOT / "Scintillator0" / "Spe",
}

# Thresholds on dead time (%)
DEAD_OK   = 5.0
DEAD_WARN = 20.0

# Thresholds on rate (Hz) - informational only, not used for flag
RATE_OK   = 10_000
RATE_WARN = 50_000


# =========================================================================
# Flag logic
# =========================================================================
def flag(dead_pct, rate_hz):
    '''Return a colored flag based on dead time (primary) and rate (info).'''
    if not (np.isfinite(dead_pct) and np.isfinite(rate_hz)):
        return "  ?  "
    if dead_pct > DEAD_WARN:
        return " RED "     # critical
    if dead_pct > DEAD_OK:
        return "WARN "     # warning
    return " OK  "


# =========================================================================
# Analyse one folder
# =========================================================================
def analyse_folder(folder, label):
    print(f"\n{'='*92}")
    print(f"{label}: {folder}")
    print('='*92)

    if not folder.is_dir():
        print(f"  [skip] folder not found")
        return []

    spectra = read_dir(folder, pattern="*.Spe")
    print(f"  Found {len(spectra)} spectra")

    rows = []
    for name, s in sorted(spectra.items()):
        total = int(s.counts.sum())
        live  = float(s.live)
        real  = float(s.real)

        # Dead time (%)
        dead_pct = 100.0 * (1.0 - live / real) if real > 0 else np.nan

        # Rate (counts / live time)
        rate_hz = total / live if live > 0 else np.nan

        # V, G, S from filename
        key = parse_filename(name) if name else None
        V, G, S = (key if key is not None else (None, None, None))

        rows.append({
            "file":         name,
            "V":            V,
            "G":            G,
            "S":            S,
            "total":        total,
            "live_s":       live,
            "real_s":       real,
            "dead_%":       dead_pct,
            "rate_Hz":      rate_hz,
            "flag":         flag(dead_pct, rate_hz),
        })

    # Print table
    print()
    print(f"{'file':32s} {'total':>10s} {'live(s)':>9s} {'real(s)':>9s} "
          f"{'dead%':>7s} {'rate(Hz)':>10s}  flag")
    print('-' * 92)
    for r in rows:
        dead_s = f"{r['dead_%']:6.1f}%" if np.isfinite(r['dead_%']) else "   n/a"
        rate_s = f"{r['rate_Hz']:10.0f}" if np.isfinite(r['rate_Hz']) else "       n/a"
        print(f"{r['file']:32s} {r['total']:>10d} {r['live_s']:>9.1f} "
              f"{r['real_s']:>9.1f} {dead_s:>7s} {rate_s:>10s}  {r['flag']}")

    # Count
    n_ok   = sum(1 for r in rows if r['flag'] == ' OK  ')
    n_warn = sum(1 for r in rows if r['flag'] == 'WARN ')
    n_red  = sum(1 for r in rows if r['flag'] == ' RED ')
    print('-' * 92)
    print(f"  OK: {n_ok}   WARN: {n_warn}   RED: {n_red}   (total {len(rows)})")

    return rows


# =========================================================================
# Main
# =========================================================================
def main():
    all_rows = []
    for label, folder in FOLDERS.items():
        rows = analyse_folder(folder, label)
        for r in rows:
            r["scintillator"] = label
        all_rows.extend(rows)

    if not all_rows:
        print("\nNo spectra found.")
        return

    df = pd.DataFrame(all_rows)

    # Save CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nFull table saved to {OUT_CSV}")

    # Print list of problem spectra (to be excluded or re-measured)
    print("\n" + "="*92)
    print("SPECTRA TO EXCLUDE OR RE-MEASURE (dead time > 5%)")
    print("="*92)

    problems = df[df["flag"].isin(["WARN ", " RED "])].copy()
    problems = problems.sort_values("dead_%", ascending=False)

    if problems.empty:
        print("  None. All spectra are within safe limits (dead < 5%).")
        return

    for _, r in problems.iterrows():
        if r["flag"] == " RED ":
            action = "EXCLUDE  (peak distorted by pile-up)"
        else:
            action = "inspect  (moderate dead time)"
        print(f"  {r['flag']}  {r['scintillator']:7s}  {r['file']:32s}  "
              f"dead={r['dead_%']:5.1f}%  rate={r['rate_Hz']:>8.0f} Hz  →  {action}")

    # Summary by scintillator
    print("\n" + "="*92)
    print("SUMMARY")
    print("="*92)
    for label in df["scintillator"].unique():
        sub = df[df["scintillator"] == label]
        n_ok   = int((sub["flag"] == " OK  ").sum())
        n_warn = int((sub["flag"] == "WARN ").sum())
        n_red  = int((sub["flag"] == " RED ").sum())
        med_dead = sub["dead_%"].median()
        med_rate = sub["rate_Hz"].median()
        print(f"  {label:7s}  n={len(sub):3d}  OK={n_ok:3d}  WARN={n_warn:3d}  "
              f"RED={n_red:3d}   median dead = {med_dead:5.1f}%   "
              f"median rate = {med_rate:8.0f} Hz")


if __name__ == "__main__":
    main()
