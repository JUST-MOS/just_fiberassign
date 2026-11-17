import healpy as hp
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np

from .utils import DATA_DIR, _get_radius_deg, _get_radius_mm, load_focalplane

from matplotlib import cm, colors
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
from mpl_toolkits.basemap import Basemap
from numba import njit
from scipy.spatial import KDTree

def plot_focalplane(text = None, text_scale = 0.9, text_dx = 0.0, text_dy = 0.0, text_buffer = 3.0, font = 'STIXGeneral', weight = 1000,
                    text_ec = '0.75', text_fc = '0.75', logo = True, logo_scale = 0.7, logo_dx = 0.0, logo_dy = 180.0, logo_buffer = 1.0,
                    logo_ec = '0.75', logo_fc = '#A71E2D', ec = '0.75', fc = 'none',
                    logo_path = DATA_DIR / 'sjtu_astro_logo.png', focalplane_path = DATA_DIR / 'just_focalplane.csv', ax = None, size = 25):
    xy = np.loadtxt(focalplane_path, delimiter = ',', usecols = (0, 1))
    mn, mx = xy.min(0), xy.max(0)
    c, d = 0.5 * (mn + mx), 0.515 * max(mx - mn)

    edgecolors = np.full(len(xy), ec, dtype = object)
    facecolors = np.full(len(xy), fc, dtype = object)
    if logo:
        alpha = plt.imread(logo_path)
        alpha = alpha[:, :, 3] if alpha.shape[-1] == 4 else np.any(alpha[:, :, :3] < 0.99, axis = 2)
        h, w = alpha.shape
        span = logo_scale * 2 * d
        mask = np.zeros(len(xy), dtype = bool)
        for dx, dy in ((0, 0), (logo_buffer, 0), (-logo_buffer, 0), (0, logo_buffer), (0, -logo_buffer)):
            u = ((xy[:, 0] + dx - c[0] - logo_dx) / span + 0.5) * (w - 1)
            v = (0.5 - (xy[:, 1] + dy - c[1] - logo_dy) / (span * h / w)) * (h - 1)
            inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            mask[inside] |= alpha[v[inside].astype(int), u[inside].astype(int)] > 0.5
        edgecolors[mask], facecolors[mask] = logo_ec, logo_fc

    if text:
        tp = TextPath((0, 0), text, size = 1, prop = FontProperties(family = font, weight = weight))
        bb = tp.get_extents()
        scale = text_scale * 2 * np.hypot(xy[:, 0], xy[:, 1]).max() / max(bb.width, bb.height)
        tp = Affine2D().translate(-0.5 * (bb.x0 + bb.x1), -0.5 * (bb.y0 + bb.y1)).scale(scale).translate(text_dx, text_dy).transform_path(tp)
        mask = np.any([tp.contains_points(xy + dxy) for dxy in
                       ((0, 0), (text_buffer, 0), (-text_buffer, 0), (0, text_buffer), (0, -text_buffer))], axis = 0)
        edgecolors[mask], facecolors[mask] = text_ec, text_fc

    own_figure = ax is None
    if ax is None:
        fig = plt.gcf() if plt.get_fignums() else plt.figure(figsize = (5, 5))
        ax = fig.gca() if fig.axes else fig.add_axes([0.05, 0.05, 0.9, 0.9])

    ax.scatter(xy[:, 0], xy[:, 1], s = size, facecolors = facecolors, edgecolors = edgecolors, linewidths = 0.25)
    ax.set_xlim(c[0] - d, c[0] + d)
    ax.set_ylim(c[1] - d, c[1] + d)
    ax.set_aspect('equal')
    ax.axis('off')
    return ax

def _radec_unit(coord, dec = None):
    ra, dec = np.radians(np.asarray(coord)).T if dec is None else (np.radians(coord), np.radians(dec))
    cdec = np.cos(dec)
    return np.column_stack((cdec * np.cos(ra), cdec * np.sin(ra), np.sin(dec)))

def _radec_to_healpix(nside, ra, dec):
    return hp.ang2pix(nside, np.radians(90.0 - dec), np.radians(ra % 360.0))

def _project_focalplane(xyz, basis, tile_ids, telescope):
    xyz = np.ascontiguousarray(xyz.T)
    local = np.empty_like(xyz)
    boundaries = np.r_[0, np.flatnonzero(tile_ids[1:] != tile_ids[:-1]) + 1, len(tile_ids)]

    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        local[:, start:stop] = np.einsum('ij,j...->i...', basis[tile_ids[start]], xyz[:, start:stop])
    rr, dd = np.arctan2(local[1], local[0]), np.arcsin(np.clip(local[2], -1, 1))
    theta = np.degrees(2 * np.arcsin(np.sqrt(np.sin(dd / 2) ** 2 + np.cos(dd) * np.sin(rr / 2) ** 2)))
    radius, q = _get_radius_mm(theta, telescope), np.arctan2(local[2], -local[1])
    return np.column_stack((radius * np.cos(q), radius * np.sin(q)))

def _build_patrol_lookup(tree, r_outer):
    step = max(1.0, r_outer / 512)
    k = int(np.ceil(r_outer / step))
    axis = np.arange(-k, k + 1) * step
    xx, yy = np.meshgrid(axis, axis)
    distance = tree.query(np.column_stack((xx.ravel(), yy.ravel())))[0].reshape(xx.shape)
    return axis, distance

def _test_patrol(xy, tree, r_patrol, lookup):
    if lookup is None:
        return tree.query(xy, k = 1, distance_upper_bound = r_patrol)[0] <= r_patrol

    axis, distance = lookup
    step = axis[1] - axis[0]
    ix = np.clip(np.rint((xy[:, 0] - axis[0]) / step).astype(int), 0, len(axis) - 1)
    iy = np.clip(np.rint((xy[:, 1] - axis[0]) / step).astype(int), 0, len(axis) - 1)
    d = distance[iy, ix]
    error = np.hypot(xy[:, 0] - axis[ix], xy[:, 1] - axis[iy]) + 64 * np.finfo(float).eps * (abs(axis[-1]) + r_patrol + 1)
    inside = d < r_patrol
    uncertain = np.abs(d - r_patrol) <= error
    inside[uncertain] = tree.query(xy[uncertain], k = 1, distance_upper_bound = r_patrol)[0] <= r_patrol
    return inside

@njit
def _accumulate_pass(counts, indices, inside):
    for i in range(len(indices)):
        if inside[i]:
            counts[indices[i]] += 1

def _count_pass_pixels(fxy, tile_coord, nside, r_patrol, telescope, pixels = None):
    counts = np.zeros(hp.nside2npix(nside) if pixels is None else len(pixels), dtype = np.int32)
    if not len(tile_coord) or not len(counts) or r_patrol <= 0:
        return counts

    tree = KDTree(fxy)
    r_outer = np.hypot(fxy[:, 0], fxy[:, 1]).max() + r_patrol
    angle = np.radians(float(_get_radius_deg(r_outer, 0.0, telescope)))
    chord = 2 * np.sin(angle / 2)

    ra, dec = np.radians(np.asarray(tile_coord, dtype = float)).T
    cra, sra, cdec, sdec = np.cos(ra), np.sin(ra), np.cos(dec), np.sin(dec)
    tile_xyz = np.column_stack((cdec * cra, cdec * sra, sdec))
    basis = np.stack((tile_xyz,
                      np.column_stack((-sra, cra, np.zeros_like(ra))),
                      np.column_stack((-cra * sdec, -sra * sdec, cdec))), axis = 1)
    radius = 2 * np.arcsin(np.minimum(1, (chord + np.abs(np.linalg.norm(tile_xyz, axis = 1) - 1) + 64 * np.finfo(float).eps) / 2))

    pix_buffer = np.empty(131072, dtype = np.int64)
    index_buffer, tile_buffer = np.empty_like(pix_buffer), np.empty_like(pix_buffer)
    lookup = None
    corners = hp.ang2pix(nside, np.pi / 2, [0.0, np.pi / 2]) if nside % 2 else None

    def accumulate(size):
        nonlocal lookup
        pix, indices, tile_ids = pix_buffer[:size], index_buffer[:size], tile_buffer[:size]
        theta, phi = hp.pix2ang(nside, pix)
        ra, dec = np.degrees(phi), 90.0 - np.degrees(theta)
        valid = (ra >= 0) & (ra <= 90) & (dec >= 0) & (dec <= 90)
        if corners is not None:
            valid &= (pix != corners[0]) & (pix != corners[1])
        ra, dec, indices, tile_ids = ra[valid], dec[valid], indices[valid], tile_ids[valid]

        xyz = _radec_unit(ra, dec)
        delta = xyz - tile_xyz[tile_ids]
        valid = np.einsum('ij,ij->i', delta, delta) <= chord * chord
        xyz, indices, tile_ids = xyz[valid], indices[valid], tile_ids[valid]
        if not len(indices):
            return

        xy = _project_focalplane(xyz, basis, tile_ids, telescope)
        if lookup is None and len(xy) >= 65536:
            lookup = _build_patrol_lookup(tree, r_outer)
        _accumulate_pass(counts, indices, _test_patrol(xy, tree, r_patrol, lookup))

    size = 0
    for tile_id, (vec, rad) in enumerate(zip(tile_xyz, radius)):
        near = hp.query_disc(nside, vec, rad, inclusive = True)
        if pixels is None:
            indices = near
        else:
            indices = np.minimum(np.searchsorted(pixels, near), len(pixels) - 1)
            valid = pixels[indices] == near
            near, indices = near[valid], indices[valid]

        start = 0
        while start < len(near):
            take = min(len(near) - start, len(pix_buffer) - size)
            pix_buffer[size:size + take] = near[start:start + take]
            index_buffer[size:size + take] = indices[start:start + take]
            tile_buffer[size:size + take] = tile_id
            size += take
            start += take
            if size == len(pix_buffer):
                accumulate(size)
                size = 0
    if size:
        accumulate(size)
    return counts

def _calc_pass_healpix(fxy, tile_coord, nside = 4096, r_patrol = 6.0, telescope = 'just'):
    return _count_pass_pixels(fxy, tile_coord, nside, r_patrol, telescope)

def _project_healpix(m, nside, nxy):
    x, y = np.linspace(m.xmin, m.xmax, nxy), np.linspace(m.ymin, m.ymax, nxy)
    rows = max(1, 262144 // max(1, nxy))
    for start in range(0, nxy, rows):
        xx, yy = np.meshgrid(x, y[start:start + rows])
        ra, dec = m(xx, yy, inverse = True)
        valid = np.isfinite(ra) & np.isfinite(dec) & (np.abs(ra) < 1.e20) & (np.abs(dec) <= 90.0)
        pix = np.full(ra.shape, -1, dtype = np.int64)
        pix[valid] = _radec_to_healpix(nside, ra[valid], dec[valid])
        yield start, pix

def _sample_healpix(m, sky_map, nside, nxy):
    values = np.full((nxy, nxy), np.nan)
    for start, pix in _project_healpix(m, nside, nxy):
        valid = pix >= 0
        values[start:start + len(pix)][valid] = sky_map[pix[valid]]
    return values

@njit
def _accumulate_targets(pix, assigned, n_total, n_assigned):
    for i in range(len(pix)):
        p = pix[i]
        n_total[p] += 1
        if assigned[i]:
            n_assigned[p] += 1

def _count_targets(gal_ra, gal_dec, assigned, nside):
    ra, dec = np.atleast_1d(gal_ra), np.atleast_1d(gal_dec)
    dtype = np.result_type(ra, dec)
    ra, dec = np.asarray(ra, dtype = dtype), np.asarray(dec, dtype = dtype)
    if ra.ndim != 1 or dec.ndim != 1:
        coord = np.column_stack((ra, dec))
        ra, dec = coord[:, 0], coord[:, 1]

    selection = np.asarray(assigned)
    npix = hp.nside2npix(nside)
    if selection.dtype == bool and selection.ndim == 1:
        if len(selection) != len(ra):
            raise IndexError('Boolean selection must match the target array length')
        n_total, n_assigned = np.zeros(npix, dtype = np.int64), np.zeros(npix, dtype = np.int64)
        for start in range(0, len(ra), 262144):
            stop = start + 262144
            pix = _radec_to_healpix(nside, ra[start:stop], dec[start:stop])
            _accumulate_targets(pix, selection[start:stop], n_total, n_assigned)
        return n_total, n_assigned

    pix = np.empty(len(ra), dtype = np.int64)
    for start in range(0, len(ra), 262144):
        stop = start + 262144
        pix[start:stop] = _radec_to_healpix(nside, ra[start:stop], dec[start:stop])
    return np.bincount(pix, minlength = npix), np.bincount(pix[assigned], minlength = npix)

def _init_sky_map(ax = None):
    if ax is None:
        fig = plt.gcf() if plt.get_fignums() else plt.figure(figsize = (5, 5))
        ax = fig.gca() if fig.axes else fig.add_axes([0.14, 0.14, 0.83, 0.83])
    lim_angle = np.degrees(4 * np.arcsin(np.sin(np.radians(45)) * np.sqrt(2 / (1 + np.cos(np.radians(45)) ** 2)) / 2))
    width = 4 * 6_370_997.0 * np.sin(np.radians(lim_angle) / 4)
    m = Basemap(width = width, height = width, resolution = None, projection = 'laea', lat_0 = 45, lon_0 = 45, ax = ax)
    return ax, m

def _draw_sky_map(ax, m, data, cmap, vmin, vmax, tile_name, cbar_cmap, norm, ticks, ticklabels,
                  cbar_label, tick_length, boundaries = None):
    cmap.set_bad(alpha = 0)
    m.imshow(data, origin = 'lower', interpolation = 'nearest', cmap = cmap, vmin = vmin, vmax = vmax, zorder = 1)
    m.drawparallels(np.arange(-90.0, 91.0, 15.0), labels = [1, 0, 0, 0],
                    fmt = lambda x: rf'${x:.0f}^\circ$', color = 'gray', textcolor = 'k',
                    linewidth = 0.25, dashes = [4, 2], fontsize = 15, latmax = 90.0, zorder = 3)
    m.drawmeridians(np.arange(-180.0, 181.0, 15.0), labels = [0, 0, 0, 1],
                    fmt = lambda x: rf'${x % 360:.0f}^\circ$', color = 'gray', textcolor = 'k',
                    linewidth = 0.25, dashes = [4, 2], fontsize = 15, latmax = 90.0, zorder = 3)

    ax.set_xlabel(r'$\mathrm{R.A.}$', fontsize = 15, labelpad = 25.0)
    ax.set_ylabel(r'$\mathrm{Dec.}$', fontsize = 15, labelpad = 27.0)
    ax.text(0.025, 0.975, tile_name, fontsize = 15, color = 'k', transform = ax.transAxes, ha = 'left', va = 'top',
            path_effects = [path_effects.withStroke(linewidth = 2.0, foreground = 'w')])

    h = 0.02 / ax.get_position().height
    cax = ax.inset_axes([2 / 3, 1.0 - h, 1 / 3, h])
    cbar = ax.figure.colorbar(cm.ScalarMappable(norm = norm, cmap = cbar_cmap), cax = cax,
                              orientation = 'horizontal', boundaries = boundaries)
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.ax.xaxis.set_label_position('bottom')
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticklabels)
    cbar.set_label(cbar_label, fontsize = 15, labelpad = 1.0)
    cbar.ax.tick_params(axis = 'x', labelsize = 10, length = tick_length, pad = 3.0)
    return ax

def plot_coverage(tile_coord, tile_name, nside = 4096, nxy = 2001, ax = None):
    ax, m = _init_sky_map(ax)
    tile_coord = np.asarray(tile_coord)
    mask = (tile_coord[:, 0] >= 0) & (tile_coord[:, 0] <= 90) & (tile_coord[:, 1] >= 0) & (tile_coord[:, 1] <= 90)
    tile_coord = tile_coord[mask]

    image_pix = np.empty((nxy, nxy), dtype = np.int64)
    for start, pix in _project_healpix(m, nside, nxy):
        image_pix[start:start + len(pix)] = pix
    pixels = np.unique(image_pix)
    pixels = pixels[pixels >= 0]
    counts = _count_pass_pixels(load_focalplane('just'), tile_coord, nside, 6.0, 'just', pixels)

    values = np.full((nxy, nxy), np.nan)
    rows = max(1, 262144 // max(1, nxy))
    for start in range(0, nxy, rows):
        pix = image_pix[start:start + rows]
        valid = pix >= 0
        values[start:start + len(pix)][valid] = counts[np.searchsorted(pixels, pix[valid])]

    vmax = 4
    his_cmap = colors.LinearSegmentedColormap.from_list('tjet', plt.get_cmap('jet')(np.linspace(0.2, 0.8, 101)))
    bar_cmap = colors.ListedColormap(np.vstack(([1, 1, 1, 1], his_cmap(np.linspace(0, 1, vmax)))))
    boundaries = np.arange(-0.5, vmax + 1.5, 1)
    return _draw_sky_map(ax, m, np.ma.masked_invalid(np.ma.masked_less(values, 1)), his_cmap, 1, vmax, tile_name,
                         bar_cmap, colors.BoundaryNorm(boundaries, bar_cmap.N), [0, 1, 2, 3, 4],
                         ['$0$', '$1$', '$2$', '$3$', r'$\geq 4$'], r'$N_{\mathrm{pass}}$', 0, boundaries)

def plot_complete(target_coord, assigned, tile_name, nside = 1024, nxy = 10_001, ax = None):
    ax, m = _init_sky_map(ax)
    target_coord = np.asarray(target_coord)
    n_total, n_assigned = _count_targets(target_coord[:, 0], target_coord[:, 1], assigned, nside)
    complete_map = np.divide(n_assigned, n_total, out = np.full(len(n_total), np.nan), where = n_total > 0)
    values = _sample_healpix(m, complete_map, nside, nxy)
    his_cmap = plt.get_cmap('inferno').copy()
    return _draw_sky_map(ax, m, np.ma.masked_invalid(values), his_cmap, 0, 1, tile_name,
                         his_cmap, colors.Normalize(vmin = 0, vmax = 1), [0.1, 0.3, 0.5, 0.7, 0.9],
                         ['$10\\%$', '$30\\%$', '$50\\%$', '$70\\%$', '$90\\%$'], r'$f_{\mathrm{complete}}$', 2)
