'''
inspect_scintillator1.py

Plot the raw spectra of Scintillator0 to see where the peaks are.
'''

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from lab_compton.maestro_reader import read_dir

DATA_DIR = Path("/home/ale_bomei/Uni/lab_analysis/data/Optimization/Scintillator1/Spe")

spectra = read_dir(DATA_DIR, pattern="*.Spe")
print(f"Found {len(spectra)} spectra")

fig, axes = plt.subplots(5, 4, figsize=(20, 16))
axes = axes.flatten()

for i, (name, s) in enumerate(sorted(spectra.items())):
    ax = axes[i]
    # Log scale: see both high peaks and low tails
    ax.step(s.channels, s.counts + 0.5, where="mid", color="steelblue")
    ax.set_yscale("log")
    ax.set_xlabel("channel")
    ax.set_ylabel("counts (log)")
    ax.set_title(f"{name}\nmax={s.counts.max()} counts, "
                 f"total={s.counts.sum():.0f}")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, len(s.channels) - 1)

    # Zoom hints: where is 95% of the counts?
    cumsum = np.cumsum(s.counts)
    idx_50 = np.searchsorted(cumsum, 0.5 * cumsum[-1])
    idx_95 = np.searchsorted(cumsum, 0.95 * cumsum[-1])
    ax.axvspan(0, idx_95, alpha=0.08, color="orange",
               label=f"95% counts < ch {idx_95}")
    ax.axvline(s.counts.argmax(), color="red", ls="--", lw=1,
               label=f"peak at ch {s.counts.argmax()}")
    #ax.legend(fontsize=8)

plt.tight_layout()
plt.show()
