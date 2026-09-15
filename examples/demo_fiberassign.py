import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np

from just_fiberassign.catalog import gen_catalog, gen_random
from just_fiberassign.utils import load_focalplane, load_target, load_tile

from matplotlib.ticker import MaxNLocator
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
TARGET_FILE = EXAMPLES_DIR / 'input/target/gal_mock_rlim20p5_ra0_4_dec0_4.npz'
TILE_FILE = EXAMPLES_DIR / 'input/tiling/tile_ipack_1pass_ra0_4_dec0_4.npz'

def _sky_circle(ra, dec, radius = 0.575, n = 128):
    ra0, dec0, radius = np.radians([ra, dec, radius])
    phi = np.linspace(0.0, 2 * np.pi, n)

    center = np.array([np.cos(dec0) * np.cos(ra0), np.cos(dec0) * np.sin(ra0), np.sin(dec0)])
    east = np.array([-np.sin(ra0), np.cos(ra0), 0.0])
    north = np.cross(center, east)

    xyz = np.cos(radius) * center[:, None] + np.sin(radius) * (
        np.cos(phi) * east[:, None] + np.sin(phi) * north[:, None]
    )
    circle_ra = np.degrees(np.arctan2(xyz[1], xyz[0]))
    circle_dec = np.degrees(np.arcsin(np.clip(xyz[2], -1, 1)))
    circle_ra = ra + (circle_ra - ra + 180.0) % 360.0 - 180.0

    return circle_ra, circle_dec

def _refresh(fig, notebook = False):
    fig.canvas.draw()
    fig.canvas.flush_events()
    if not notebook:
        plt.pause(0.001)

def run_fiberassign(target_file = TARGET_FILE, tile_file = TILE_FILE, algorithm = 'greedy-classic',
                    n_random = 5000, thread = 16, dpi = 250, fancy = False, notebook = False):
    target_coord, target_pri, target_sub, target_z, _ = load_target(target_file)
    tile_pass, tile_coord, tile_rotation = load_tile(tile_file)
    fiber_coord = load_focalplane('just')

    output_dir = EXAMPLES_DIR / 'output'
    output_dir.mkdir(exist_ok = True)
    stem = f'{Path(target_file).stem}_{Path(tile_file).stem}'
    catalog_file = output_dir / f'cat_{stem}.npz'
    plot_file = output_dir / f'fba_{stem}.png'

    r_, y_, g_, b_ = '#EA4335', '#FBBC05', '#34A853', '#4285F4'

    plot_params = {
        'axes.linewidth': 1.0, 'font.family': 'STIXGeneral', 'font.size': 15, 'mathtext.fontset': 'stix',
        'ytick.major.width': 1.0, 'ytick.minor.width': 1.0, 'ytick.major.size': 4, 'ytick.minor.size': 2,
        'ytick.labelsize': 15, 'ytick.direction': 'in',
        'xtick.major.width': 1.0, 'xtick.minor.width': 1.0, 'xtick.major.size': 4, 'xtick.minor.size': 2,
        'xtick.labelsize': 15, 'xtick.direction': 'in', 'xtick.major.pad': 6, 'xtick.minor.pad': 6,
    }
    plt.rcParams.update(plot_params)

    single_edge = (10.58 - 0.12 * 5 - 0.03 * 5) / 4
    full_height = single_edge + 0.12 * 5 + 0.03 * 5

    tile_circles = [_sky_circle(ra, dec) for ra, dec in tile_coord]
    all_ra = np.concatenate([target_coord[:, 0]] + [circle[0] for circle in tile_circles])
    all_dec = np.concatenate([target_coord[:, 1]] + [circle[1] for circle in tile_circles])
    dx, dy = 0.05 * np.ptp(all_ra), 0.05 * np.ptp(all_dec)
    xlim = (all_ra.min() - dx, all_ra.max() + dx)
    ylim = (all_dec.min() - dy, all_dec.max() + dy)

    fig = plt.figure(figsize = (10.58, full_height))
    axes = []

    for i in range(4):
        ax = plt.axes([(0.12 * 5 + single_edge * i) / 10.58, 0.12 * 5 / full_height,
                       single_edge / 10.58, single_edge / full_height])
        axes.append(ax)

        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xlabel(rf'$\mathrm{{R.A.}}$ $\mathrm{{[deg]}}$', fontsize = 15, labelpad = 6.0)
        ax.tick_params(which = 'both', top = True, right = True, labelsize = 15)
        ax.minorticks_on()

        if i == 0:
            ax.set_ylabel(rf'$\mathrm{{Dec.}}$ $\mathrm{{[deg]}}$', fontsize = 15, labelpad = 6.0)
        else:
            ax.tick_params(axis = 'y', labelleft = False)

        ax.xaxis.set_major_locator(MaxNLocator(integer = True))
        ax.yaxis.set_major_locator(MaxNLocator(integer = True))

    if fancy:
        if not notebook:
            plt.ion()
            plt.show(block = False)
        else:
            plt.show()
        _refresh(fig, notebook = notebook)

    pass_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    axes[0].scatter(target_coord[:, 0], target_coord[:, 1], s = 0.25, marker = '.', lw = 0, color = 'gray')
    for j, p in enumerate(np.unique(tile_pass)):
        select = tile_pass == p
        color = pass_colors[j % len(pass_colors)]
        axes[0].scatter(tile_coord[select, 0], tile_coord[select, 1], s = 25, marker = '+', lw = 1.0, color = color)
        for k in np.flatnonzero(select):
            axes[0].plot(*tile_circles[k], lw = 0.5, color = color,
                         path_effects = [path_effects.withStroke(linewidth = 1, foreground = 'w')])

    axes[0].text(0.97, 0.97, rf'$N_g = {len(target_coord)}$', transform = axes[0].transAxes,
                 ha = 'right', va = 'top', fontsize = 12,
                 path_effects = [path_effects.withStroke(linewidth = 1, foreground = 'w')])

    if fancy:
        _refresh(fig, notebook = notebook)

    assigned_mask, iip_weight, pip_bitweights, patrolled_mask = gen_catalog(
        fiber_coord, tile_coord, tile_rotation, target_coord, target_pri, target_sub,
        algorithm = algorithm, thread = thread
    )

    axes[1].scatter(target_coord[:, 0], target_coord[:, 1], s = 0.25, marker = '.', lw = 0, color = 'gray')
    axes[1].scatter(target_coord[patrolled_mask, 0], target_coord[patrolled_mask, 1],
                    s = 1.0, marker = '.', lw = 0, color = b_)
    axes[1].text(0.97, 0.97, rf'$N_{{g, \, \mathrm{{patrolled}}}} = {np.sum(patrolled_mask)}$',
                 transform = axes[1].transAxes, ha = 'right', va = 'top', fontsize = 12,
                 path_effects = [path_effects.withStroke(linewidth = 1, foreground = 'w')])

    axes[2].scatter(target_coord[:, 0], target_coord[:, 1], s = 0.25, marker = '.', lw = 0, color = 'gray')
    axes[2].scatter(target_coord[assigned_mask, 0], target_coord[assigned_mask, 1],
                    s = 1.0, marker = '.', lw = 0, color = r_)
    axes[2].text(0.97, 0.97, rf'$N_{{g, \, \mathrm{{assigned}}}} = {np.sum(assigned_mask)}$',
                 transform = axes[2].transAxes, ha = 'right', va = 'top', fontsize = 12,
                 path_effects = [path_effects.withStroke(linewidth = 1, foreground = 'w')])
    axes[2].text(0.98, 0.02,
                 rf'$f_\mathrm{{observed}} = {np.sum(assigned_mask) / np.sum(patrolled_mask) * 100:.1f}\%, \,\,\,$'
                 rf'$f_\mathrm{{assigned}} = {np.sum(assigned_mask) / len(tile_coord) / len(fiber_coord) * 100:.1f}\%$',
                 transform = axes[2].transAxes, ha = 'right', va = 'bottom', fontsize = 12, linespacing = 1.5,
                 path_effects = [path_effects.withStroke(linewidth = 1, foreground = 'w')])

    if fancy:
        _refresh(fig, notebook = notebook)

    random_coord, random_z, random_patrol = gen_random(
        fiber_coord, tile_coord, tile_rotation, target_z, density = n_random
    )

    np.savez_compressed(
        catalog_file,
        patrolled_coord = target_coord[patrolled_mask], patrolled_z = target_z[patrolled_mask],
        assigned_coord = target_coord[assigned_mask], assigned_z = target_z[assigned_mask],
        assigned_iip_weight = iip_weight[assigned_mask], assigned_pip_bitweights = pip_bitweights[assigned_mask],
        random_coord = random_coord[random_patrol], random_z = random_z[random_patrol]
    )

    axes[3].scatter(random_coord[random_patrol, 0], random_coord[random_patrol, 1],
                    s = 1.0, marker = '.', lw = 0, color = g_)
    axes[3].text(0.97, 0.97, rf'$N_{{g, \, \mathrm{{random}}}} = {np.sum(random_patrol)}$',
                 transform = axes[3].transAxes, ha = 'right', va = 'top', fontsize = 12,
                 path_effects = [path_effects.withStroke(linewidth = 1, foreground = 'w')])

    if fancy:
        _refresh(fig, notebook = notebook)

    fig.savefig(plot_file, dpi = dpi, transparent = False, facecolor = 'w')

    if not fancy:
        plt.show()
    elif not notebook:
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-target', default = TARGET_FILE)
    parser.add_argument('-tile', default = TILE_FILE)
    parser.add_argument('-algorithm', default = 'greedy-classic')
    parser.add_argument('-n_random', type = int, default = 5000)
    parser.add_argument('-thread', type = int, default = 16)
    parser.add_argument('-dpi', type = int, default = 250)
    parser.add_argument('-fancy', action = 'store_true')
    parser.add_argument('-notebook', action = 'store_true')
    args = parser.parse_args()
    run_fiberassign(args.target, args.tile, algorithm = args.algorithm,
                    n_random = args.n_random, thread = args.thread, dpi = args.dpi,
                    fancy = args.fancy, notebook = args.notebook)