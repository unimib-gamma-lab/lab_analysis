import re

# Accepts filenames like:
#   hv1000_g050_sh1.Spe
#   HV1000-g50-sh1.Spe
#   run_V1000_G050_S1.Spe

# in config.py
_NUM = r"\d+(?:p\d+)?"
_FILENAME_RE = re.compile(
    rf"v(?P<V>{_NUM})[_\-]?g(?P<G>{_NUM})[_\-]?sh?(?P<S>{_NUM})",
    re.IGNORECASE,
)

def _to_number(token):
    '''"050" -> 50, "0p5" -> 0.5'''
    x = float(token.lower().replace("p", "."))
    return int(x) if x.is_integer() else x

def parse_filename(name):
    '''
    Extract (V, G, S) from a filename.

    Returns
    -------
    (V, G, S) : tuple of int, or None if the name does not match.
    '''
    m = _FILENAME_RE.search(name)
    if not m:
        return None
    return _to_number(m.group("V")), _to_number(m.group("G")), _to_number(m.group("S"))
