'''
utils.py

We insert here useful functions we need, in this way we define them once and the main files are cleaner

'''

from pathlib import Path #I need it to act on the path
import numpy as np
from datetime import datetime
from typing import NamedTuple, Optional #I need it to create the class
import re #I just need it in _name_sort
from fnmatch import fnmatch
import warnings

class Spectrum(NamedTuple):
    '''
    Immutable container for one spectrum (one .Spe file).

    Fields
    ------
    channels : np.ndarray
        Channel indices (e.g. 0..1023).
    counts : np.ndarray
        Raw counts per channel (same length as `channels`).
    live : float
        Live time, in seconds.
    real : float
        Real time, in seconds (live + dead time).
    date : datetime | None
        Measurement date parsed from DATE_MEA, or None if missing/invalid.
    date_str : str
        Original date string, kept for reference.
    description : str
        SPEC_ID lines joined with " | ".
    path : Path
        Full path of the source file.
    raw_sections : dict
        All sections of the file, for advanced use.

    Notes
    -----
    Attribute access is preferred (s.channels, s.live, ...),
    but tuple unpacking also works:
        channels, counts, live, real, date, date_str, desc, path, raw = s
    '''
    channels:     np.ndarray
    counts:       np.ndarray
    live:         float
    real:         float
    date:         Optional[datetime]
    date_str:     str
    description:  str
    path:         Path
    raw_sections: dict

def _parse_date(date_str):
    '''
    Convert a Maestro date string into a datetime object.

    The expected format is "MM/DD/YYYY HH:MM:SS" (US order: month first).
    Returns None if the string is empty or the format does not match.
    '''
    
    try:
        return datetime.strptime(date_str.strip(), "%m/%d/%Y %H:%M:%S")
    except (ValueError, AttributeError):
        return None
        
def _name_sort(name):
    '''
    Return a sort key for natural ordering of file names.

    "run2" < "run10" (numeric compare), instead of the lexicographic
    "run10" < "run2".  re.split(r"(\\d+)", name) splits the name into
    alternating text/number tokens, then numbers are converted to int
    and text is lowercased (case-insensitive sort).
    '''
    return [
        int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)
        ] 

def read_spe(path):
    '''
    Read a single .Spe file (Maestro's ASCII format).

    The file is split into sections delimited by lines starting with "$".
    Only the sections needed to build the Spectrum are parsed; the rest
    is kept in `raw_sections` for advanced use.

    Parameters
    ----------
    path : str | Path
        Path to the .Spe file.

    Returns
    -------
    Spectrum
        NamedTuple with channels, counts, live/real times, date,
        description, path and raw sections.

    Raises
    ------
    ValueError
        If the number of counts does not match the header, or if
        live time exceeds real time.
    KeyError
        If mandatory sections (DATA, MEAS_TIM) are missing.

    Examples
    --------
    >>> s = read_spe("spettro.Spe")
    >>> s.channels, s.counts, s.live
    '''
    sections, key = {}, None
    for line in Path(path).read_text(errors="replace").splitlines():
        
        #In a spe file each "information" starts with "$"
        if line.startswith("$"):
            key = line.strip().lstrip("$").rstrip(":") #remove $, blanks and :
            sections[key] = [] #I create a section if there's a $
        elif key is not None and line.strip():
            sections[key].append(line.strip())
    
    #I take the first and last channel from the first row of data
    #I use map to apply int to each  element
    #The [:2 is truncate to the first two tokens, just to avoid errors
    first, last = map(int, sections["DATA"][0].split()[:2])
    
    #I take all the lines from the second raw and create numpy array
    counts = np.array([int(x) for x in sections["DATA"][1:]])
    if len(counts) != last - first + 1:
        raise ValueError(
            f"{path}: {len(counts)} read channles, "
            f"header declares {last - first + 1}"
        )

    live, real = map(float, sections["MEAS_TIM"][0].split()[:2])
    if live > real:
        raise ValueError(f"{path}: live time ({live}) > real time ({real})")
    
    date_str = sections.get("DATE_MEA", [""])[0]
    
    return Spectrum(
        channels     = np.arange(first, last + 1), #channels
        counts       = counts,                     
        live         = live,                       #live time
        real         = real,                       #real time
        date         = _parse_date(date_str),      #datetime
        date_str     = date_str,
        description  = " | ".join(sections.get("SPEC_ID", [])),
        path         = Path(path),
        raw_sections = sections,                   #all the rest
    )
    
def read_data(path):
    '''
    Shortcut: read a .Spe file and return only channels and counts.

    Parameters
    ----------
    path : str | Path
        Path to the .Spe file.

    Returns
    -------
    (np.ndarray, np.ndarray)
        (channels, counts)

    Examples
    --------
    >>> channels, counts = read_data("spettro.Spe")
    '''
    s = read_spe(path)
    return s.channels, s.counts


def read_dir(
    folder, 
    pattern="*.Spe", 
    sort_by="name", 
    skip_errors=True, 
    recursive=False
):

    '''
    Read every .Spe file matching `pattern` in a folder.

    Parameters
    ----------
    folder : str | Path
        Folder to scan.
    pattern : str
        Filename filter, case-insensitive (e.g. "*.spe", "compton_*.spe").
        The default "*.Spe" matches any case (.Spe, .SPE, .SpE, ...).
    sort_by : {"name", "date"}
        "name" -> natural order (run2 < run10).
        "date" -> order by measurement date from the header
                  (files with missing date go last).
    skip_errors : bool
        If True, unreadable files are skipped with a warning;
        if False, the first error propagates.
    recursive : bool
        If True, scan subfolders too.

    Returns
    -------
    dict {stem: Spectrum}
        Ordered according to `sort_by` (Python dicts preserve insertion
        order since 3.7).

    Raises
    ------
    FileNotFoundError
        If the folder does not exist, or no file matches `pattern`.
    ValueError
        If `sort_by` is not one of the allowed values, or two files
        share the same stem (same name, different extension).

    Examples
    --------
    >>> spectra = read_dir("dati/", pattern="compton_*.spe", sort_by="date")
    >>> for name, s in spectra.items():
    ...     print(name, s.live, s.counts.sum())
    '''
    
    #I convert folder in Path object for security
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"folder not found: {folder}")
    if sort_by not in ("name", "date"):
        raise ValueError("sort_by must be 'name' or 'date'")
    
    #I choose iterator to match the file name patterns: 
    # glob only folder, rglob also subfolder
    if recursive:
        it = folder.rglob("*")
    else:
        it = folder.iterdir()

    # I take files from the dir; .lower() + fnmatch = case-insensitive match
    paths = [
        p for p in it if p.is_file() and 
        fnmatch(p.name.lower(), pattern.lower())
    ]
    
    if not paths:
        raise FileNotFoundError(f"no files matching '{pattern}' in {folder}")
    
    #I sort the files by name (number)
    paths.sort(key=lambda p: _name_sort(p.name))
    
    spectra = {}
    for path in paths:
        if path.stem in spectra:
            raise ValueError(f"duplicate stem (same name, different extension): {path.stem}")
        try: 
            spectra[path.stem] = read_spe(path)
        except Exception as e:
            if not skip_errors:
                raise
            warnings.warn(f"skipped {path.name}: {e}", stacklevel=2)
            
    #Sort by date if active
    if sort_by == "date":
        #for each pair kv (key-value), I take kv[1], i.e. Spectrum, and .date
        spectra = dict(sorted(
            spectra.items(),
            key=lambda kv: kv[1].date or datetime.min,
        ))
    
    return spectra

    
class SpectrumBatch(NamedTuple):
    '''
    Vectorised view of many spectra, all with the same channel axis.

    Fields
    ------
    names : list[str]
        File stems, in the same order as the rows of the arrays.
    channels : np.ndarray, shape (n_ch,)
        Shared channel axis.
    counts : np.ndarray, shape (n_file, n_ch)
        Raw counts per file per channel.
    live : np.ndarray, shape (n_file,)
        Live time per file (seconds).
    real : np.ndarray, shape (n_file,)
        Real time per file (seconds).
    rates : np.ndarray, shape (n_file, n_ch)
        counts / live[:, None], i.e. counts per second.
    dates : list[datetime | None]
        Measurement date per file, aligned with the rows.

    Notes
    -----
    For Poisson fits, prefer `counts` and use `live` as exposure.
    `rates` is a convenience for plots and quick checks.
    '''
    
    names:    list                  
    channels: np.ndarray            
    counts:   np.ndarray            
    live:     np.ndarray            
    real:     np.ndarray            
    rates:    np.ndarray            
    dates:    list                  
    
def to_arrays(spectra):
    '''
    Stack a {name: Spectrum} dict into a SpectrumBatch.

    All spectra must share the same number of channels; otherwise a
    ValueError is raised listing the distinct sizes.

    Parameters
    ----------
    spectra : dict {str: Spectrum}
        Output of `read_dir` (or any dict of Spectrum objects).

    Returns
    -------
    SpectrumBatch
        Vectorised arrays: counts (n_file, n_ch), live (n_file,),
        real (n_file,), rates (n_file, n_ch), plus names, channels,
        dates aligned with the rows.

    Raises
    ------
    ValueError
        If `spectra` is empty, if channel counts differ, or if any
        live time is non-positive.

    Examples
    --------
    >>> b = to_arrays(read_dir("dati/"))
    >>> b.counts.shape          # (n_file, n_ch)
    >>> b.rates.sum(axis=1)     # total rate per file (c/s)

    >>> names, channels, counts, live, real, rates, dates = b
    '''
    
    #I check the spectra are not empy
    names = list(spectra)
    if not names:
        raise ValueError("empty spectra dict")
    
    sizes = {len(spectra[n].counts) for n in names}
    if len(sizes) != 1:
        raise ValueError(f"inconsistent channel counts: {sorted(sizes)}")
    
    channels = spectra[names[0]].channels
    counts   = np.stack([spectra[n].counts for n in names])
    live     = np.array([spectra[n].live  for n in names])
    real     = np.array([spectra[n].real  for n in names])
    dates    = [spectra[n].date          for n in names]

    #I check live time is positive
    if np.any(live <= 0):
        bad = [names[i] for i in np.where(live <= 0)[0]]
        raise ValueError(f"non-positive live time in: {bad}")

    rates = counts / live[:, None]
    return SpectrumBatch(names, channels, counts, live, real, rates, dates)
    
    
    
