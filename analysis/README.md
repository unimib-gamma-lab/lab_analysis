## Library reference (`src/lab_compton/`)

The package `lab_compton` contains the code shared by every notebook and script in `analysis/`. After `uv sync` it can be imported from anywhere (see [Quick start](#quick-start)).

| Module | What it does |
| --- | --- |
| [`config`](#config) | Extracts the HV / gain / shaping values `(V, G, S)` from a spectrum file name |
| [`parameters`](#parameters) | Physical constants (electron mass, the two <sup>22</sup>Na lines, ...) and selection cuts for the 511 keV fits |
| [`maestro_reader`](#maestro_reader) | Reads Maestro `.Spe` files into `Spectrum` objects |
| [`spectrum_fit`](#spectrum_fit) | Fits a single peak (signal + background models), computes FWHM and resolution |
| [`resolution_scan`](#resolution_scan) | Fits the peak of every spectrum of a scan, then R(V) for each (G, S); finds the optimum setting |
| [`scan_plotting`](#scan_plotting) | Plots and text summaries for the scan |
| [`systematics`](#systematics) | Systematic uncertainty on R from the fit window and from the peak model |

**Conventions used everywhere**

- Everything is done in **channels**, not in energy, so no calibration error has to be propagated. The resolution is `R = FWHM / (mu - zero_channel)` (dimensionless); `zero_channel = 0` is fine to compare settings with each other, pass the calibrated value for absolute numbers.
- File names carry the acquisition settings: `V` = high voltage, `G` = amplifier gain, `S` = shaping time. See [`config`](#config) for the accepted patterns.

The `**Imports**` line at the top of each block below shows which modules depend on which.

### config

Parses `(V, G, S)` out of a file name, e.g. `hv1000_g050_sh1.Spe`, `HV1000-g50-sh1.Spe` and `run_V1000_G050_S1.Spe` all give `(1000, 50, 1)`. A `p` is a decimal point (`s0p5` gives `0.5`, returned as a float). It returns `None` if the name does not match, and the scan functions then skip that file.

```python
from lab_compton.config import parse_filename

V, G, S = parse_filename("hv1000_g050_sh1.Spe")
```

<!-- API:config:START -->
**Imports** — standard library only

**Functions**

- `parse_filename(name)` — Extract (V, G, S) from a filename.

<!-- API:config:END -->

### parameters

Constants and thresholds in one place, imported as `import lab_compton.parameters as par`. `optimization.py` uses `R_MIN_511`, `R_MAX_511` and `D_NDOF_MAX` to discard fits that are not a good 511 keV peak (resolution outside the window, or reduced deviance too high).

<!-- API:parameters:START -->
**Imports** — third-party: `numpy`

**Constants**

- `ELECTRON_RADIUS` = `2.8179403262e-13` — classical electron radius [cm]
- `ELECTRON_MASS` = `510.99895` — electron rest energy [keV]
- `FWHM_FACTOR` = `2.0 * np.sqrt(2.0 * np.log(2.0))`
- `E1_KEV` = `ELECTRON_MASS` — Na22 511 KeV peak
- `E2_KEV` = `1274.53` — Na22 1274 KeV peak
- `R_MIN_511` = `0.055` — lower bound of R for the 511 keV peak
- `R_MAX_511` = `0.2` — upper bound
- `D_NDOF_MAX` = `3.0` — max reduced deviance for a usable fit

<!-- API:parameters:END -->

### maestro_reader

Reads the ASCII `.Spe` files written by Maestro. `read_spe` returns a `Spectrum` (channels, counts, live and real time, date, description); `read_dir` reads a whole folder and returns `{file stem: Spectrum}` in natural order (`run2` before `run10`) or ordered by measurement date. Unreadable files are skipped with a warning (`skip_errors=False` makes them raise instead). `read_spe` checks that the number of counts matches the header and that live time does not exceed real time. `to_arrays` stacks many spectra with the same channel axis into 2-D arrays (`counts`, `rates = counts / live`).

```python
from lab_compton.maestro_reader import read_dir, to_arrays

spectra = read_dir("data/Optimization/Scintillator1/Spe", sort_by="date")
for name, s in spectra.items():
    dead_pct = 100 * (1 - s.live / s.real)
    print(name, s.counts.sum(), f"{dead_pct:.1f}% dead")

batch = to_arrays(spectra)      # batch.counts has shape (n_file, n_channels)
```

<!-- API:maestro_reader:START -->
**Imports** — third-party: `numpy`

**Functions**

- `read_spe(path)` — Read a single .Spe file (Maestro's ASCII format).
- `read_data(path)` — Shortcut: read a .Spe file and return only channels and counts.
- `read_dir(folder, pattern='*.Spe', sort_by='name', skip_errors=True, recursive=False)` — Read every .Spe file matching `pattern` in a folder.
- `to_arrays(spectra)` — Stack a {name: Spectrum} dict into a SpectrumBatch.

**Classes**

- `Spectrum` — Container for one spectrum (one .Spe file).
- `SpectrumBatch` — Vectorised view of many spectra, all with the same channel axis.

<!-- API:maestro_reader:END -->

### spectrum_fit

The fitting engine: one peak in one spectrum, in a window of channels. The fit is an extended binned negative log-likelihood minimised with `iminuit` (Migrad, with Minos for asymmetric errors on `n_sig`, `mu` and `sigma`); all normalisations are bounded to be positive.

A model is a **signal + background (+ optional Compton step)**. Models are registered in `MODELS` under the key `"<signal>_<background>"` or `"<signal>_<background>_step"` (for example `"gauss_exp"`, `"cb_lin"`, `"gauss_lin_step"`): 5 signals x 6 backgrounds x 2 = 60 models.

| Signal key | Shape |
| --- | --- |
| `gauss` | Gaussian |
| `cb` | One-sided Crystal Ball (tail at low energy) |
| `agauss` | Two-piece Gaussian; FWHM is exactly 2.3548 sigma for any asymmetry |
| `emg` | Exponentially modified Gaussian, tail at low energy; `mu` is the centre of the Gaussian component, not the mode |
| `jsu` | Johnson SU, parametrised by mean and standard deviation |

| Background key | Shape |
| --- | --- |
| `flat` | Constant |
| `lin` | Straight line, parametrised by its values at the two window edges |
| `exp` | Exponential, same parametrisation as `lin` |
| `poly2`, `poly3` | Quadratic / cubic polynomial in the Bernstein basis (non-negative everywhere) |
| `exppoly2` | Exponential of a quadratic |

`fit_peak` returns a `FitResult`. The resolution comes from `peak_resolution`; the FWHM is computed numerically from the fitted shape, so it is correct for the non-Gaussian signals as well. `fit_scan` repeats the same fit on many spectra. The helper `_find_peak_window` (used by `load_scan` and by some scripts) locates the most prominent peak and returns a window of `fit_half_fwhm` (default 2.5) times the FWHM measured on the data, on each side of the peak.

```python
from lab_compton.maestro_reader import read_spe
from lab_compton.spectrum_fit import fit_peak

s = read_spe("data/Optimization/Scintillator1/Spe/<file>.Spe")
fit = fit_peak(s.channels, s.counts, window=(lo, hi), model="gauss_exp")   # lo, hi: channels

mu, mu_err = fit.centroid()
fwhm, fwhm_err = fit.fwhm()
R, R_err = fit.peak_resolution(zero_channel=0.0)
print(fit.reduced_deviance(), fit.warnings())
```

<!-- API:spectrum_fit:START -->
**Imports** — third-party: `iminuit`, `numpy`, `scipy` · internal: `parameters`

**Constants**

- `INV_SQRT_2PI` = `1.0 / math.sqrt(2.0 * math.pi)`

**Functions**

- `fwhm_from_logpdf(logpdf, mode=None, step=1.0, *, method='brentq', xtol=1e-13, rtol=1e-13, max_doublings=60, mode_search=…)` — FWHM of a unimodal density, in the units of the variable of `logpdf`.
- `fit_peak(channels, counts, window, *, model='gauss_lin', p0=None, strategy=2, minos=…)` — Fit a single peak with the chosen composite model.
- `fit_scan(spectra, window, *, model='gauss_lin', p0=None, zero_channel=0.0, zero_channel_err=0.0, **fit_kw)` — Fit the same peak on many spectra, one after another.

**Classes**

- `GaussianSignal` — gaussian peak
- `CrystalBallSignal` — One sided crystall ball (tail at low energy).
- `AsymGaussianSignal` — Two-piece (split) Gaussian, mode at mu
- `EMGSignal` — Exponentially modified Gaussian with the tail at LOW energy (scipy's exponnorm has the tail on the right, so x is mirrored: z -> -z).
- `JohnsonSUSignal` — Johnson SU, parametrised by MEAN (mu) and STANDARD DEVIATION (sigma) plus two shape parameters.
- `LinearBackground` — Linear bkg outside the window (instead of y = mx + q, I take the line passing through (x1, y1) and (x2, y2) given by lo_edge and high_edge.
- `FlatBackground` — Constant bkg, b(x) = b_flat.
- `BernsteinBackground` — Polynomial bkg of a given degree in the Bernstein basis on t = (x - lo_edge)/(hi_edge - lo_edge)
- `Poly2Background` — Quadratic (Bernstein).
- `Poly3Background` — Cubic (Bernstein).
- `ExpPoly2Background` — exp(quadratic): b(t) = exp(p(t)), t = (x - lo_edge)/(hi_edge - lo_edge), with p the parabola through (ln b_left, ln b_mid, ln b_right) at t = 0, 1/2, 1.
- `ExpBackground` — Exponential bkg parametrised, like the linear one, by its values at the window edges
- `ComptonStep` — Step function.
- `CompositeModel` — CDF sum of more components.
- `FitResult` — Container for the result of a single peak fit. Methods: `param`, `centroid`, `fwhm`, `fwhm_asym`, `n_events`, `n_events_asym`, `peak_resolution`, `residuals`, `reduced_deviance`, `warnings`, `summary`.

<!-- API:spectrum_fit:END -->

### resolution_scan

Turns a folder of spectra taken at different (V, G, S) into a resolution study. `load_scan` reads every `.Spe`, takes `(V, G, S)` from the file name, fits the main peak and returns a `DataFrame` sorted by `(G, S, V)` with columns `mu`, `fwhm`, `R` and their errors, window, deviance, warnings, ... By default it fits the most prominent peak of each spectrum; pass `mu_ref` (a channel, or a function `(V, G, S) -> channel`) to look near a given position, and `gain` / `shaping` to keep only some settings.

`fit_groups` then fits R as a function of V separately for every `(G, S)`, with two models:

- a **parabola** `R(V) = R0 + k (V - V0)^2` (robust, needs at least 3 valid points);
- the **PMT model** `R(V)^2 = R_intr^2 + (A/V)^2 + (B V)^2` (needs at least 4 points): `R_intr` is the intrinsic resolution, `A/V` the electronic noise (decreases with HV), `B V` the multiplication noise / saturation (increases with HV). The optimum is `V_opt = sqrt(A/B)`, fitted directly with its Minos interval.

It returns `(results, summary)`: `results[(G, S)]` holds the data and both fits, `summary` has one row per `(G, S)`. `best_setting` picks the `(G, S)` with the lowest resolution using `method="pmt"` (needs a well-defined minimum), `"parabola"` or `"data"` (lowest measured R, works also on a flat curve), and lists the settings compatible with it within `n_sigma`.

The plotting functions of [`scan_plotting`](#scan_plotting) can also be imported from here (backward compatibility).

```python
from lab_compton.resolution_scan import load_scan, fit_groups, best_setting

df = load_scan("data/Optimization/Scintillator1/Spe", min_channel=30)
results, summary = fit_groups(df)
best = best_setting(summary, method="data")
print(best["best"])          # the best (G, S)
print(best["compatible"])    # settings compatible with it within 1 sigma
```

<!-- API:resolution_scan:START -->
**Imports** — third-party: `iminuit`, `numpy`, `pandas` · internal: `config`, `maestro_reader`, `scan_plotting`, `spectrum_fit`

**Functions**

- `load_scan(folder, mu_ref=None, *, gain=None, shaping=None, model='gauss_exp', zero_channel=0.0, zero_channel_err=0.0, search_half=None, fit_half_fwhm=2.5, window_kw=None, min_channel=0, parse=…, recursive=False, verbose=True)` — Load every .Spe in `folder`, fit the main peak, return a DataFrame sorted by (G, S, V).
- `fit_parabola_R_V(df, *, V_guess=None, minos=True)` — Fit R(V) = R0 + k * (V - V0)^2.
- `fit_pmt_R_V(df, *, p0=None, minos=True)` — Fit R(V) with the PMT model R(V)^2 = R_intr^2 + (A/V)^2 + (B*V)^2.
- `fit_groups(df, *, by=('G', 'S'), minos=True)` — Fit R(V) separately for every (G, S) combination.
- `best_setting(summary, *, method='pmt', n_sigma=1.0)` — Pick the (G, S) with the lowest resolution, and list the ones that are statistically compatible with it.

<!-- API:resolution_scan:END -->

### scan_plotting

Presentation only: nothing here changes a fit or a result. `matplotlib` is imported inside each function, so the module can be imported on a machine without a display. Plot functions return the matplotlib `Axes` / `Figure` so that you can customise and save them.

- R(V) with its fits: `plot_scan` (one group), `plot_groups` (one panel per `(G, S)`), `plot_optimum_map` (optimum R or V over the `(G, S)` grid, best setting starred).
- R as a function of the gain: `plot_R_vs_G`, `plot_R_vs_G_grid`.
- Text summaries: `print_summary`, `print_group_summary`, `print_summary_R_vs_G`.

<!-- API:scan_plotting:START -->
**Imports** — third-party: `matplotlib`, `numpy` · internal: `resolution_scan`

**Functions**

- `plot_scan(df, res_par=None, res_pmt=None, *, title=None, ax=None, show_bands=True, ylabel=…)` — Plot R(V) with both fits overlaid.
- `print_summary(df, res_par=None, res_pmt=None)` — Print a compact summary of the scan and of both fits.
- `plot_groups(df, results, *, ncols=3, **plot_kw)` — One R(V) panel per (G, S).
- `print_group_summary(summary)` — Print one line per (G, S): optimum HV and resolution.
- `plot_optimum_map(summary, *, method='pmt', value='R', best=None, ax=None)` — Map over the (G, S) grid of the optimum resolution (value="R", in %) or of the optimum HV (value="V").
- `plot_R_vs_G(df, *, title=None, ax=None, log_x=True, marker_size=7)` — Plot R vs G (gain) at fixed HV.
- `plot_R_vs_G_grid(df, *, ncols=3, log_x=True, marker_size=6)` — Grid of R vs G panels, one per V (HV) that has at least 2 G values.
- `print_summary_R_vs_G(df)` — Text summary: for each V that has >= 2 G values, print

<!-- API:scan_plotting:END -->

### systematics

Estimates the systematic uncertainty on R by repeating the whole scan with different analysis choices. For each file the spread is `(max R - min R) / 2` over the variants with a valid fit, a conservative estimate that covers the whole range of plausible values.

- `window_systematics`: changes the fit window (`fit_half_fwhm` in 1.5, 2.0, 2.5, 3.0, 4.0 by default).
- `model_systematics`: changes the peak model. By default it uses **every** model in `MODELS` (minus those in `exclude`), which means one full scan per model and can take a long time: pass `models=[...]` to restrict it.
- `combine_systematics`: the two effects are treated as independent and added in quadrature (`syst_tot`).

Extra keyword arguments (`min_channel`, `gain`, `shaping`, `zero_channel`, ...) are passed to `load_scan`.

```python
from lab_compton.systematics import (
    window_systematics, model_systematics, combine_systematics,
)

df_w, sp_w = window_systematics("data/Optimization/Scintillator1/Spe", min_channel=30)
df_m, sp_m = model_systematics("data/Optimization/Scintillator1/Spe", min_channel=30,
                               models=["gauss_exp", "cb_lin"])
combined = combine_systematics(sp_w, sp_m)    # syst_window, syst_model, syst_tot per file
```

<!-- API:systematics:START -->
**Imports** — third-party: `numpy`, `pandas` · internal: `resolution_scan`, `spectrum_fit`

**Functions**

- `window_systematics(folder, *, variants=…, verbose=True, **load_kw)` — Re-fit the scan with several fit_half_fwhm values.
- `model_systematics(folder, *, models=None, exclude=(), verbose=True, **load_kw)` — Re-fit the scan with several peak models.
- `print_systematics(spread, label='systematic')` — Print a compact summary of the per-file spread.
- `combine_systematics(spread_window, spread_model)` — Combine window and model systematics in quadrature, per file.

<!-- API:systematics:END -->

### Keeping this section up to date

The `<!-- API:... -->` blocks above are **generated from the code** by `tools/update_readme.py` (it only reads the files, it does not import or run them). Never edit what is between the markers: change the docstring in the source instead, the first sentence is what ends up here. Everything outside the markers is written by hand.

- Once per clone: `git config core.hooksPath .githooks`. From then on, every commit that touches a file in `src/lab_compton/` regenerates the blocks and adds `README.md` to the same commit.
- By hand: `uv run python tools/update_readme.py`. To only check: `uv run python tools/update_readme.py --check` (exit code 1 if the README is out of date). GitHub runs the same check on every pull request.
- New library: add `src/lab_compton/<name>.py`, then in this section a `### <name>` heading, a short description, and an empty `<!-- API:<name>:START -->` / `<!-- API:<name>:END -->` pair. Until you do, the script warns and the pull request check fails.
