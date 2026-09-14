import io
import numpy as np
import matplotlib.pyplot as plt
import requests

from functools import lru_cache
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
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

def show_focalplane(text = None, text_scale = 0.9, text_dx = 0.0, text_dy = 0.0, text_buffer = 3.0, font = 'STIXGeneral', weight = 1000,
                    text_ec = '0.75', text_fc = '0.75', logo = True, logo_scale = 0.7, logo_dx = 0.0, logo_dy = 180.0, logo_buffer = 1.0,
                    logo_ec = '0.75', logo_fc = '#A71E2D', ec = '0.75', fc = 'none', dpi = 250, transparent = False, save_path = None,
                    logo_path = DATA_DIR / 'sjtu_astro_logo.png', focalplane_path = DATA_DIR / 'just_focalplane.csv'):
    xy = np.loadtxt(focalplane_path, delimiter = ',', usecols = (0, 1))
    mn, mx = xy.min(0), xy.max(0)
    c, d = 0.5 * (mn + mx), 0.515 * max(mx - mn)

    def text_mask():
        tp = TextPath((0, 0), text, size = 1,
                      prop = FontProperties(family = font, weight = weight))
        bb = tp.get_extents()
        tp = Affine2D().translate(-0.5 * (bb.x0 + bb.x1), -0.5 * (bb.y0 + bb.y1)).scale(
            text_scale * 2 * np.hypot(xy[:, 0], xy[:, 1]).max() / max(bb.width, bb.height)
        ).translate(text_dx, text_dy).transform_path(tp)
        return np.any([tp.contains_points(xy + dxy) for dxy in
                       [[0, 0], [text_buffer, 0], [-text_buffer, 0], [0, text_buffer], [0, -text_buffer]]], axis = 0)

    def logo_mask():
        alpha = plt.imread(logo_path)
        alpha = alpha[:, :, 3] if alpha.shape[-1] == 4 else np.any(alpha[:, :, :3] < 0.99, axis = 2)
        h, w = alpha.shape
        mask = np.zeros(len(xy), dtype = bool)
        for dxy in [[0, 0], [logo_buffer, 0], [-logo_buffer, 0], [0, logo_buffer], [0, -logo_buffer]]:
            u = ((xy[:, 0] + dxy[0] - c[0] - logo_dx) / (logo_scale * 2 * d) + 0.5) * (w - 1)
            v = (0.5 - (xy[:, 1] + dxy[1] - c[1] - logo_dy) /
                 (logo_scale * 2 * d * h / w)) * (h - 1)
            inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            mask[inside] |= alpha[v[inside].astype(int), u[inside].astype(int)] > 0.5
        return mask

    edgecolors = np.full(len(xy), ec, dtype = object)
    facecolors = np.full(len(xy), fc, dtype = object)
    if logo:
        mask = logo_mask()
        edgecolors[mask], facecolors[mask] = logo_ec, logo_fc
    if text is not None:
        mask = text_mask()
        edgecolors[mask], facecolors[mask] = text_ec, text_fc

    plt.figure(figsize = (5, 5))
    plt.axes([0.05, 0.05, 0.9, 0.9])
    plt.scatter(xy[:, 0], xy[:, 1], s = 25, facecolors = facecolors,
                edgecolors = edgecolors, linewidths = 0.25)
    plt.xlim(c[0] - d, c[0] + d)
    plt.ylim(c[1] - d, c[1] + d)
    plt.gca().set_aspect('equal')
    plt.axis('off')
    if save_path is not None:
        plt.savefig(Path(save_path) / 'just_focalplane.png', dpi = dpi, transparent = transparent)
    plt.show()