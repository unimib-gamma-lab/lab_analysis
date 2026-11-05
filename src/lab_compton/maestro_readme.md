# Maestro.py - README

```markdown
# maestro.py

Read and vectorise MCA spectra saved by Maestro (`.Spe` ASCII format).

The module does three things:

1. **Parse** a single `.Spe` file into a `Spectrum` object.
2. **Batch read** a folder of files into a `dict {name: Spectrum}`.
3. **Vectorise** the batch into a `SpectrumBatch` with `(n_file, n_ch)` arrays.

## Import

```python
from maestro import (
    Spectrum,          # type of a single spectrum
    SpectrumBatch,     # type of a vectorised batch
    read_spe,          # read one .Spe file
    read_data,         # read only (channels, counts) from a file
    read_dir,          # read a whole folder
    to_arrays,         # vectorise a dict of Spectrum
)
```

## The data model

### `Spectrum`

`Spectrum` is an immutable `NamedTuple` with the following fields:

| Field | Type | Meaning |
|---|---|---|
| `channels` | `np.ndarray` | channel indices (e.g. 0..1023) |
| `counts` | `np.ndarray` | counts per channel |
| `live` | `float` | live time (s) |
| `real` | `float` | real time (s) = live + dead |
| `date` | `datetime \| None` | measurement date |
| `date_str` | `str` | raw date string |
| `description` | `str` | `SPEC_ID` lines joined with `" \| "` |
| `path` | `Path` | source file path |
| `raw_sections` | `dict` | every section of the file |

Access by attribute (`s.channels`, `s.live`, ...) or, if you prefer,
unpack as a tuple:

```python
channels, counts, live, real, date, date_str, desc, path, raw = s
```

### `SpectrumBatch`

`SpectrumBatch` is the vectorised view of N spectra:

| Field | Shape | Meaning |
|---|---|---|
| `names` | `list[str]` | file stems (no extension) |
| `channels` | `(n_ch,)` | shared channel axis |
| `counts` | `(n_file, n_ch)` | counts |
| `live` | `(n_file,)` | live time per file |
| `real` | `(n_file,)` | real time per file |
| `rates` | `(n_file, n_ch)` | `counts / live[:, None]` |
| `dates` | `list[datetime \| None]` | dates aligned with the rows |

## Usage: a single file

```python
from maestro import read_spe, read_data

# Full metadata
s = read_spe("spettro.Spe")
print(s.live, s.real, s.date)
print(s.channels.shape, s.counts.sum())

# Only channels and counts (shortcut)
channels, counts = read_data("spettro.Spe")
```

## Usage: a whole folder

```python
from maestro import read_dir

spectra = read_dir("dati/")
```

`read_dir` accepts five parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `folder` | — | folder to scan |
| `pattern` | `"*.Spe"` | filename filter, **case-insensitive** |
| `sort_by` | `"name"` | `"name"` (natural: `run2 < run10`) or `"date"` |
| `skip_errors` | `True` | skip corrupt files with a warning |
| `recursive` | `False` | if `True`, scan subfolders too |

Examples:

```python
# Only files starting with "compton_", sorted by measurement date
spectra = read_dir("dati/", pattern="compton_*.Spe", sort_by="date")

# Include subfolders
spectra = read_dir("dati/", recursive=True)

# Blow up on the first error instead of skipping corrupt files
spectra = read_dir("dati/", skip_errors=False)
```

The result is an **ordered** `dict` according to `sort_by`:

```python
for name, s in spectra.items():
    print(name, s.live, s.counts.sum(), s.date)
```

## Usage: vectorisation

If you need to run fits or statistical analyses, vectorise the batch:

```python
from maestro import read_dir, to_arrays

spectra = read_dir("dati/", sort_by="date")
b = to_arrays(spectra)

b.counts.shape          # (n_file, n_ch)
b.live.shape            # (n_file,)
b.rates.shape           # (n_file, n_ch)

# Total rate per file (c/s)
total_rates = b.rates.sum(axis=1)

# Mean spectrum (c/s per channel)
mean_rates = b.rates.mean(axis=0)

# Dead time per file, in percent
dead_time_pct = 100 * (1 - b.live / b.real)
```

`to_arrays` checks that:

- the dict is not empty;
- all spectra have the **same number of channels** (otherwise `ValueError` listing the distinct sizes);
- all `live` values are **> 0** (otherwise `ValueError` listing the offending file names).

## Full example: rate vs time

```python
import numpy as np
import matplotlib.pyplot as plt
from maestro import read_dir, to_arrays

spectra = read_dir("dati/", sort_by="date")
b = to_arrays(spectra)

# Total rate (c/s) against measurement date
dates = [d for d in b.dates if d is not None]  # mind the None entries
totals = b.rates.sum(axis=1)

plt.plot(dates, totals[:len(dates)], "o-")
plt.xlabel("date")
plt.ylabel("total rate (c/s)")
plt.show()
```


