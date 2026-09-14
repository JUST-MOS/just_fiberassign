import io
import numpy as np
import requests

from functools import lru_cache
from pathlib import Path
from scipy.interpolate import interp1d

DATA_DIR = Path(__file__).resolve().parent / 'data'

def _load_npz(filename):
    if str(filename).startswith(('http://', 'https://')):
        response = requests.get(filename, timeout = 60)
        response.raise_for_status()
        return np.load(io.BytesIO(response.content))
    return np.load(filename)

@lru_cache(maxsize = None)
def load_target(filename):
    with _load_npz(filename) as data:
        return data['target_coord'], data['target_pri'], data['target_sub'], data['target_z'], data['target_mr']

@lru_cache(maxsize = None)
def load_tile(filename):
    with _load_npz(filename) as data:
        return data['tile_pass'], data['tile_coord'], data['tile_rotation']

@lru_cache(maxsize = None)
def load_platescale(telescope, spec_dir = DATA_DIR):
    columns = [('radius', 'f8'), ('theta', 'f8'), ('radial_platescale', 'f8'), ('az_platescale', 'f8'), ('arclength', 'f8')]
    filename = Path(spec_dir) / f'{telescope}_platescale.txt'

    try:
        return np.loadtxt(filename, usecols = [0, 1, 6, 7, 8], dtype = columns)
    except (IndexError, ValueError):
        ps = np.loadtxt(filename, usecols = [0, 1, 6, 7, 7], dtype = columns)
        rzs = np.genfromtxt(Path(spec_dir) / f'{telescope}_rzsn.txt', names = True, skip_header = 7)
        ps['arclength'] = interp1d(rzs['R'], rzs['S'], kind = 'quadratic')(ps['radius'])
        return ps

@lru_cache(maxsize = None)
def _platescale_interp(telescope, spec_dir = DATA_DIR):
    ps = load_platescale(telescope, spec_dir)
    kw = dict(kind = 'quadratic', bounds_error = False, fill_value = 'extrapolate')
    return interp1d(ps['theta'], ps['radius'], **kw), interp1d(ps['radius'], ps['theta'], **kw), \
           interp1d(ps['radius'], ps['radial_platescale'], **kw)

def rotate_xy(xy, angle_deg):
    xy = np.asarray(xy, dtype = float)
    q = np.radians(angle_deg)
    c, s = np.cos(q), np.sin(q)
    return np.column_stack((c * xy[..., 0] - s * xy[..., 1], s * xy[..., 0] + c * xy[..., 1]))

def load_focalplane(telescope, spec_dir = DATA_DIR):
    return np.loadtxt(Path(spec_dir) / f'{telescope}_focalplane.csv', delimiter = ',', usecols = (0, 1))

def _get_radius_mm(theta, telescope, spec_dir = DATA_DIR):
    radius = _platescale_interp(telescope, spec_dir)[0](theta)
    return float(radius) if np.isscalar(theta) else radius

def _get_radius_deg(x, y, telescope, spec_dir = DATA_DIR):
    return np.asarray(_platescale_interp(telescope, spec_dir)[1](np.hypot(x, y)), dtype = float)

def _patrol_radius_deg(x, y, telescope, patrol_radius_mm = 6.0, spec_dir = DATA_DIR):
    return _platescale_interp(telescope, spec_dir)[2](np.hypot(x, y)) * patrol_radius_mm / 3600.0

def xy2radec(telra, teldec, x, y, telescope, rotation = 0.0, spec_dir = DATA_DIR):
    xy = rotate_xy(np.column_stack((np.ravel(x), np.ravel(y))), rotation)
    r = np.radians(_get_radius_deg(xy[:, 0], xy[:, 1], telescope, spec_dir))
    q = np.arctan2(xy[:, 1], xy[:, 0])
    sr, cr = np.sin(r), np.cos(r)

    tra, tdec = np.radians([telra, teldec])
    cra, sra, cdec, sdec = np.cos(tra), np.sin(tra), np.cos(tdec), np.sin(tdec)
    xyz = np.einsum('ij,j...->i...', [[cra * cdec, -sra, -cra * sdec], [sra * cdec, cra, -sra * sdec], [sdec, 0.0, cdec]],
                    np.stack((cr, -sr * np.cos(q), sr * np.sin(q))))

    return np.column_stack((np.degrees(np.mod(np.arctan2(xyz[1], xyz[0]), 2 * np.pi)),
                            np.degrees(np.arcsin(np.clip(xyz[2], -1, 1)))))

def radec2xy(telra, teldec, ra, dec, telescope, rotation = 0.0, spec_dir = DATA_DIR):
    ra, dec = np.radians(ra), np.radians(dec)
    tra, tdec = np.radians([telra, teldec])
    cra, sra, cdec, sdec = np.cos(tra), np.sin(tra), np.cos(tdec), np.sin(tdec)

    xyz = np.einsum('ij,j...->i...', [[cra * cdec, sra * cdec, sdec], [-sra, cra, 0.0], [-cra * sdec, -sra * sdec, cdec]],
                    np.stack((np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec))))
    rr, dd = np.arctan2(xyz[1], xyz[0]), np.arcsin(np.clip(xyz[2], -1, 1))
    theta = np.degrees(2 * np.arcsin(np.sqrt(np.sin(dd / 2) ** 2 + np.cos(dd) * np.sin(rr / 2) ** 2)))
    radius = _get_radius_mm(theta, telescope, spec_dir)
    q = np.arctan2(xyz[2], -xyz[1])

    return rotate_xy(np.column_stack((radius * np.cos(q), radius * np.sin(q))), -rotation)