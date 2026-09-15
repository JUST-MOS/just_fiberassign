import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np

from astropy.cosmology import Planck18
from pathlib import Path
from scipy.cluster.vq import kmeans2
from scipy.spatial import KDTree
from scipy.stats import hypergeom
from tqdm import tqdm

EXAMPLES_DIR = Path(__file__).resolve().parent
CATALOG_FILE = EXAMPLES_DIR / 'output/cat_gal_mock_rlim20p5_ra0_4_dec0_4_tile_ipack_1pass_ra0_4_dec0_4.npz'

_POPCOUNT = np.array([bin(i).count('1') for i in range(256)], dtype = np.uint8)

def _popcount(x):
    x = np.ascontiguousarray(x, dtype = np.uint64)
    return _POPCOUNT[x.view(np.uint8).reshape(x.shape + (8,))].sum(axis = -1)

def _pip_joint(nreal):
    joint = np.zeros((nreal + 1, nreal + 1))
    for c1 in range(1, nreal + 1):
        for c2 in range(1, c1 + 1):
            c12 = np.arange(1, c2 + 1)
            value = np.sum(nreal / c12 * hypergeom.pmf(c12 - 1, nreal - 1, c1 - 1, c2 - 1))
            joint[c1, c2] = joint[c2, c1] = value
    return joint

def _pip_norm(bitweights, joint, nreal):
    recurrence = 1 + _popcount(bitweights).sum(axis = 1).astype(int)
    hist = np.bincount(recurrence, minlength = nreal + 1)
    return 0.5 * (hist @ joint @ hist - np.sum(joint[recurrence, recurrence]))

def _pairnorm(w1, w2, autocorr, ijack1, ijack2, njack):
    sum1, sum2 = np.sum(w1), np.sum(w2)
    norm = 0.5 * (sum1 ** 2 - np.sum(w1 ** 2)) if autocorr else sum1 * sum2

    if not njack:
        return norm, None, None

    sum1_jack = np.bincount(ijack1, weights = w1, minlength = njack)

    if autocorr:
        sumsq_jack = np.bincount(ijack1, weights = w1 ** 2, minlength = njack)
        auto_norm = 0.5 * (sum1_jack ** 2 - sumsq_jack)
        cross_norm = sum1_jack * (sum1 - sum1_jack)
    else:
        sum2_jack = np.bincount(ijack2, weights = w2, minlength = njack)
        auto_norm = sum1_jack * sum2_jack
        cross_norm = sum1_jack * (sum2 - sum2_jack) + (sum1 - sum1_jack) * sum2_jack

    return norm, auto_norm, cross_norm

def _paircount(x1, x2, rp_edges, pi_edges, w1 = None, w2 = None, autocorr = False,
               ijack1 = None, ijack2 = None, njack = 0, thread = 16, chunk = 2048, tree = None,
               pip_bitweights = None):
    weighted = w1 is not None
    w1 = np.ones(len(x1)) if w1 is None else np.asarray(w1)
    w2 = w1 if autocorr else (np.ones(len(x2)) if w2 is None else np.asarray(w2))
    ijack2 = ijack1 if autocorr else ijack2

    if pip_bitweights is not None:
        pip_bitweights = np.asarray(pip_bitweights, dtype = np.uint64)
        nreal = 1 + 64 * pip_bitweights.shape[1]
        pip_joint = _pip_joint(nreal)

    nweight = 1 + int(weighted) + int(pip_bitweights is not None)
    tree = KDTree(x2) if tree is None else tree
    nrp, npi = len(rp_edges) - 1, len(pi_edges) - 1
    nbin, rmax = nrp * npi, np.hypot(rp_edges[-1], pi_edges[-1])
    count = np.zeros((nweight, nbin))

    if njack:
        auto_count = np.zeros((nweight, njack * nbin))
        cross_count = np.zeros((nweight, njack * nbin))

    for start in range(0, len(x1), chunk):
        stop = min(start + chunk, len(x1))
        pairs = tree.query_ball_point(x1[start:stop], r = rmax, workers = thread, return_sorted = False)
        lens = np.fromiter((len(p) for p in pairs), dtype = int, count = len(pairs))
        if not lens.sum():
            continue

        i = np.repeat(np.arange(start, stop), lens)
        j = np.concatenate(pairs).astype(int, copy = False)

        if autocorr:
            keep = j > i
            i, j = i[keep], j[keep]

        sep = x1[i] - x2[j]
        los = 0.5 * (x1[i] + x2[j])
        pi = np.abs(np.einsum('ij,ij->i', sep, los) / np.linalg.norm(los, axis = 1))
        rp = np.sqrt(np.maximum(np.einsum('ij,ij->i', sep, sep) - pi ** 2, 0.0))

        keep = (rp >= rp_edges[0]) & (rp < rp_edges[-1]) & (pi >= pi_edges[0]) & (pi < pi_edges[-1])
        i, j, rp, pi = i[keep], j[keep], rp[keep], pi[keep]
        ibin = (np.searchsorted(rp_edges, rp, side = 'right') - 1) * npi + np.searchsorted(pi_edges, pi, side = 'right') - 1

        pair_weights = [np.ones(len(i))]
        if weighted:
            pair_weights.append(w1[i] * w2[j])
        if pip_bitweights is not None:
            nij = 1 + _popcount(pip_bitweights[i] & pip_bitweights[j]).sum(axis = 1)
            pair_weights.append(nreal / nij)

        for k, pair_weight in enumerate(pair_weights):
            count[k] += np.bincount(ibin, weights = pair_weight, minlength = nbin)

        if njack:
            j1, j2 = ijack1[i], ijack2[j]
            same = j1 == j2
            diff = ~same
            index = np.r_[j1[diff] * nbin + ibin[diff], j2[diff] * nbin + ibin[diff]]

            for k, pair_weight in enumerate(pair_weights):
                auto_count[k] += np.bincount(j1[same] * nbin + ibin[same], weights = pair_weight[same], minlength = njack * nbin)
                cross_count[k] += np.bincount(index, weights = np.r_[pair_weight[diff], pair_weight[diff]], minlength = njack * nbin)

    norm, auto_norm, cross_norm = [], [], []

    n, a, c = _pairnorm(np.ones(len(x1)), np.ones(len(x2)), autocorr, ijack1, ijack2, njack)
    norm.append(n)
    if njack:
        auto_norm.append(a)
        cross_norm.append(c)

    if weighted:
        n, a, c = _pairnorm(w1, w2, autocorr, ijack1, ijack2, njack)
        norm.append(n)
        if njack:
            auto_norm.append(a)
            cross_norm.append(c)

    if pip_bitweights is not None:
        norm.append(_pip_norm(pip_bitweights, pip_joint, nreal))

        if njack:
            a = np.zeros(njack)
            c = np.zeros(njack)
            for k in range(njack):
                mask = ijack1 == k
                a[k] = _pip_norm(pip_bitweights[mask], pip_joint, nreal)
                outside_norm = _pip_norm(pip_bitweights[~mask], pip_joint, nreal)
                c[k] = norm[-1] - a[k] - outside_norm
            auto_norm.append(a)
            cross_norm.append(c)

    norm = np.asarray(norm)
    count = count.reshape(nweight, nrp, npi) / norm[:, None, None]

    if not njack:
        return count

    auto_norm = np.asarray(auto_norm)
    cross_norm = np.asarray(cross_norm)
    alpha = njack / (2.0 + np.sqrt(2.0) * (njack - 1))
    jack_norm = norm[:, None] - auto_norm - alpha * cross_norm
    jack_count = count[:, None] * norm[:, None, None, None] \
               - auto_count.reshape(nweight, njack, nrp, npi) \
               - alpha * cross_count.reshape(nweight, njack, nrp, npi)
    jack_count /= jack_norm[:, :, None, None]

    return count, jack_count

def measure_wp(random_coord, random_z, data_coord, data_z, weight = None, pip_bitweights = None,
               rp_edges = np.logspace(np.log10(0.1), np.log10(10.0), 11), pi_edges = np.arange(0.0, 41.0, 1.0),
               njack = 32, thread = 16, label = ''):
    ra, dec = np.radians(random_coord).T
    chi = Planck18.comoving_distance(random_z).value * Planck18.h
    random_xyz = chi[:, None] * np.column_stack((np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)))

    ra, dec = np.radians(data_coord).T
    chi = Planck18.comoving_distance(data_z).value * Planck18.h
    data_xyz = chi[:, None] * np.column_stack((np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)))

    random_tree = KDTree(random_xyz)
    random_unit = random_xyz / np.linalg.norm(random_xyz, axis = 1)[:, None]
    data_unit = data_xyz / np.linalg.norm(data_xyz, axis = 1)[:, None]
    centers, _ = kmeans2(random_unit, njack, minit = '++', seed = 0)

    jack_tree = KDTree(centers)
    random_jack = jack_tree.query(random_unit, workers = thread)[1]
    data_jack = jack_tree.query(data_unit, workers = thread)[1]

    with tqdm(total = 3, desc = label) as progress:
        dd, dd_jack = _paircount(data_xyz, data_xyz, rp_edges, pi_edges, weight, weight, True,
                                 data_jack, data_jack, njack, thread, pip_bitweights = pip_bitweights)
        progress.update()

        dr, dr_jack = _paircount(data_xyz, random_xyz, rp_edges, pi_edges, weight, None, False,
                                 data_jack, random_jack, njack, thread, tree = random_tree)
        progress.update()

        rr, rr_jack = _paircount(random_xyz, random_xyz, rp_edges, pi_edges, autocorr = True,
                                 ijack1 = random_jack, ijack2 = random_jack, njack = njack, thread = thread, tree = random_tree)
        progress.update()

    if len(dd) == 3:
        dr = dr[[0, 1, 1]]
        dr_jack = dr_jack[[0, 1, 1]]

    rr, rr_jack = rr[0], rr_jack[0]

    xi = np.full_like(dd, np.nan)
    np.divide(dd - 2.0 * dr + rr[None], rr[None], out = xi, where = rr[None] > 0)

    xi_jack = np.full_like(dd_jack, np.nan)
    np.divide(dd_jack - 2.0 * dr_jack + rr_jack[None], rr_jack[None],
              out = xi_jack, where = rr_jack[None] > 0)

    dpi = np.diff(pi_edges)
    wp = 2.0 * np.sum(xi * dpi[None, None, :], axis = 2)
    wp_jack = 2.0 * np.sum(xi_jack * dpi[None, None, None, :], axis = 3)

    wp_err = np.zeros_like(wp)
    for i in range(len(wp)):
        cov = (njack - 1) * np.cov(wp_jack[i], rowvar = False, ddof = 0)
        wp_err[i] = np.sqrt(np.diag(np.atleast_2d(cov)))

    rp_cen = np.sqrt(rp_edges[:-1] * rp_edges[1:])

    if len(wp) == 1:
        return rp_cen, wp[0], wp_err[0]
    return rp_cen, wp, wp_err

def run_wp_validation(catalog_file = CATALOG_FILE, thread = 16, dpi = 250):
    output_dir = EXAMPLES_DIR / 'output'
    output_dir.mkdir(exist_ok = True)

    input_file = Path(catalog_file)
    if not input_file.is_file():
        input_file = output_dir / catalog_file

    stem = input_file.stem
    if stem.startswith('cat_'):
        stem = stem[len('cat_'):]

    auto_file = output_dir / f'wp_auto_{stem}.npz'
    plot_file = output_dir / f'wp_auto_{stem}.png'

    if auto_file.is_file():
        print(f'Loading cached data: {auto_file.name}')
        with np.load(auto_file) as data:
            rp = data['rp']
            wp_tru, wp_tru_err = data['wp_tru'], data['wp_tru_err']
            wp_fba, wp_fba_err = data['wp_fba'], data['wp_fba_err']
            wp_fba_iip, wp_fba_iip_err = data['wp_fba_iip'], data['wp_fba_iip_err']
            wp_fba_pip, wp_fba_pip_err = data['wp_fba_pip'], data['wp_fba_pip_err']
    else:
        print('Warning: For larger catalogs, use an optimized package, e.g., pycorr.')

        with np.load(input_file) as data:
            random_coord, random_z = data['random_coord'], data['random_z']
            patrolled_coord, patrolled_z = data['patrolled_coord'], data['patrolled_z']
            assigned_coord, assigned_z = data['assigned_coord'], data['assigned_z']
            assigned_iip_weight = data['assigned_iip_weight']
            assigned_pip_bitweights = data['assigned_pip_bitweights']

        rp, wp_tru, wp_tru_err = measure_wp(
            random_coord, random_z, patrolled_coord, patrolled_z,
            njack = 32, thread = thread, label = 'Truth'
        )
        rp, wp_fba_all, wp_fba_err_all = measure_wp(
            random_coord, random_z, assigned_coord, assigned_z,
            weight = assigned_iip_weight, pip_bitweights = assigned_pip_bitweights,
            njack = 32, thread = thread, label = 'Assigned'
        )
        wp_fba, wp_fba_iip, wp_fba_pip = wp_fba_all
        wp_fba_err, wp_fba_iip_err, wp_fba_pip_err = wp_fba_err_all

        np.savez_compressed(
            auto_file,
            rp = rp,
            wp_tru = wp_tru, wp_tru_err = wp_tru_err,
            wp_fba = wp_fba, wp_fba_err = wp_fba_err,
            wp_fba_iip = wp_fba_iip, wp_fba_iip_err = wp_fba_iip_err,
            wp_fba_pip = wp_fba_pip, wp_fba_pip_err = wp_fba_pip_err
        )

    plot_params = {
        'axes.linewidth': 1.0, 'font.family': 'STIXGeneral', 'font.size': 15, 'mathtext.fontset': 'stix',
        'ytick.major.width': 1.0, 'ytick.minor.width': 1.0, 'ytick.major.size': 4, 'ytick.minor.size': 2,
        'ytick.labelsize': 15, 'ytick.direction': 'in',
        'xtick.major.width': 1.0, 'xtick.minor.width': 1.0, 'xtick.major.size': 4, 'xtick.minor.size': 2,
        'xtick.labelsize': 15, 'xtick.direction': 'in', 'xtick.major.pad': 6, 'xtick.minor.pad': 6,
    }
    plt.rcParams.update(plot_params)

    r_, y_, g_, b_ = '#EA4335', '#FBBC05', '#34A853', '#4285F4'

    plt.figure(figsize = (5, 5))
    plt.axes([0.14, 0.14 + 0.83 / 4, 0.83, 0.83 * 3 / 4])

    plt.plot(rp, wp_tru, c = 'gray', lw = 1.0, ls = '--', alpha = 0.5, label = 'Truth')
    plt.fill_between(rp, wp_tru - wp_tru_err, wp_tru + wp_tru_err, color = 'gray', alpha = 0.2, lw = 0, zorder = 0)

    plt.errorbar(rp, wp_fba, yerr = wp_fba_err, c = b_, lw = 0.8, fmt = '^', ms = 8, mfc = 'None', mew = 0.8, zorder = 0,
                 path_effects = [path_effects.withStroke(linewidth = 1.6, foreground = 'w')], label = 'Assigned')
    plt.errorbar(rp, wp_fba_iip, yerr = wp_fba_iip_err, c = g_, lw = 0.8, fmt = 's', ms = 8, mfc = 'None', mew = 0.8, zorder = 0,
                 path_effects = [path_effects.withStroke(linewidth = 1.6, foreground = 'w')], label = 'IIP weighted')
    plt.errorbar(rp, wp_fba_pip, yerr = wp_fba_pip_err, c = r_, lw = 0.8, fmt = 'o', ms = 8, mfc = 'None', mew = 0.8, zorder = 0,
                 path_effects = [path_effects.withStroke(linewidth = 1.6, foreground = 'w')], label = 'PIP weighted')

    plt.xlim(0.1, 10.0)
    plt.xscale('log')
    plt.yscale('log')
    plt.gca().tick_params(labelbottom = False)
    plt.legend(loc = 3, labelspacing = 0.55, fontsize = 15, handletextpad = .5, handlelength = 1.0, frameon = False)
    plt.ylabel('$w_p$', fontsize = 15, labelpad = 6.0)

    plt.axes([0.14, 0.14, 0.83, 0.83 / 4])

    ratio_fba = wp_fba / wp_tru
    ratio_iip = wp_fba_iip / wp_tru
    ratio_pip = wp_fba_pip / wp_tru
    ratio_err = wp_tru_err / wp_tru
    ratio_fba_err = ratio_fba * np.sqrt((wp_fba_err / wp_fba) ** 2 + ratio_err ** 2)
    ratio_iip_err = ratio_iip * np.sqrt((wp_fba_iip_err / wp_fba_iip) ** 2 + ratio_err ** 2)
    ratio_pip_err = ratio_pip * np.sqrt((wp_fba_pip_err / wp_fba_pip) ** 2 + ratio_err ** 2)

    plt.fill_between(rp, 1.0 - ratio_err, 1.0 + ratio_err, color = 'gray', alpha = 0.2, lw = 0)
    plt.axhline(1.0, c = 'gray', lw = 1.0, ls = '--', alpha = 0.5, zorder = 0)

    plt.errorbar(rp, ratio_fba, yerr = ratio_fba_err, c = b_, lw = 0.8, fmt = '^', ms = 8, mfc = 'None', mew = 0.8, zorder = 0,
                 path_effects = [path_effects.withStroke(linewidth = 1.6, foreground = 'w')])
    plt.errorbar(rp, ratio_iip, yerr = ratio_iip_err, c = g_, lw = 0.8, fmt = 's', ms = 8, mfc = 'None', mew = 0.8, zorder = 0,
                 path_effects = [path_effects.withStroke(linewidth = 1.6, foreground = 'w')])
    plt.errorbar(rp, ratio_pip, yerr = ratio_pip_err, c = r_, lw = 0.8, fmt = 'o', ms = 8, mfc = 'None', mew = 0.8, zorder = 0,
                 path_effects = [path_effects.withStroke(linewidth = 1.6, foreground = 'w')])

    plt.xlim(0.1, 10.0)
    plt.xscale('log')
    plt.xlabel('$r_p$ $[h^{-1} \\mathrm{Mpc}]$', fontsize = 15, labelpad = 6.0)
    plt.ylabel('Ratio', fontsize = 15, labelpad = 6.0)
    plt.ylim(1.0 - 1.25 * np.nanmax(np.abs(np.r_[ratio_err, ratio_fba - 1.0, ratio_iip - 1.0, ratio_pip - 1.0])),
             1.0 + 1.25 * np.nanmax(np.abs(np.r_[ratio_err, ratio_fba - 1.0, ratio_iip - 1.0, ratio_pip - 1.0])))

    plt.savefig(plot_file, dpi = dpi, transparent = False, facecolor = 'w')
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-catalog', default = CATALOG_FILE)
    parser.add_argument('-thread', type = int, default = 16)
    parser.add_argument('-dpi', type = int, default = 250)
    args = parser.parse_args()
    run_wp_validation(args.catalog, thread = args.thread, dpi = args.dpi)