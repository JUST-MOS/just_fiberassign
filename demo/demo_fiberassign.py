import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np

from just_fiberassign.fiberassign import fiberassign
from just_fiberassign.utils import load_focalplane, load_target, load_tile

from matplotlib.ticker import MaxNLocator
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
TARGET_FILE = DEMO_DIR / 'input/target/gal_mock_rlim20p5_ra0_4_dec0_4.npz'
TILE_FILE = DEMO_DIR / 'input/tiling/tile_ipack_1pass_ra0_4_dec0_4.npz'

def run_fiberassign(target_file = TARGET_FILE, tile_file = TILE_FILE, algorithm = 'greedy-classic', dpi = 250):
    target_coord, target_pri, target_sub, _, _ = load_target(target_file)
    _, tile_coord, tile_rotation = load_tile(tile_file)
    fiber_coord = load_focalplane('just')

    mask, patrol = fiberassign(fiber_coord, tile_coord, tile_rotation, target_coord, target_pri, target_sub,
                               algorithm = algorithm)

    r_, y_, g_, b_ = '#EA4335', '#FBBC05', '#34A853', '#4285F4'

    plot_params = {
        'axes.linewidth': 1.0, 'font.family': 'STIXGeneral', 'font.size': 15, 'mathtext.fontset': 'stix',
        'ytick.major.width': 1.0, 'ytick.minor.width': 1.0, 'ytick.major.size': 4, 'ytick.minor.size': 2,
        'ytick.labelsize': 15, 'ytick.direction': 'in',
        'xtick.major.width': 1.0, 'xtick.minor.width': 1.0, 'xtick.major.size': 4, 'xtick.minor.size': 2,
        'xtick.labelsize': 15, 'xtick.direction': 'in', 'xtick.major.pad': 6, 'xtick.minor.pad': 6,
    }
    plt.rcParams.update(plot_params)

    single_edge = (10.58 - 0.12 * 5 - 0.03 * 5) / 3
    full_height = single_edge + 0.12 * 5 + 0.03 * 5

    masks = [None, patrol, mask]
    colors = [None, b_, r_]
    texts = [rf'$N_g = {len(target_coord)}$',
             rf'$N_{{g, \, \mathrm{{patrolled}}}} = {np.sum(patrol)}$',
             rf'$N_{{g, \, \mathrm{{assigned}}}} = {np.sum(mask)}$']

    plt.figure(figsize = (10.58, full_height))

    for i in range(3):
        plt.axes([(0.12 * 5 + single_edge * i) / 10.58, 0.12 * 5 / full_height, single_edge / 10.58, single_edge / full_height])

        plt.scatter(target_coord[:, 0], target_coord[:, 1], s = 0.25, marker = '.', lw = 0, color = 'gray')

        if masks[i] is not None:
            plt.scatter(target_coord[masks[i], 0], target_coord[masks[i], 1], s = 1.0, marker = '.', lw = 0, color = colors[i])
        else:
            plt.scatter(tile_coord[:, 0], tile_coord[:, 1], s = 25, marker = '+', lw = 1.0, color = 'k')

        plt.xlabel(rf'$\mathrm{{R.A.}}$ $\mathrm{{[deg]}}$', fontsize = 15, labelpad = 6.0)
        plt.tick_params(which = 'both', top = True, right = True, labelsize = 15)
        plt.minorticks_on()

        if i == 0:
            xlim, ylim = plt.xlim(), plt.ylim()
            plt.ylabel(rf'$\mathrm{{Dec.}}$ $\mathrm{{[deg]}}$', fontsize = 15, labelpad = 6.0)
        else:
            plt.xlim(xlim); plt.ylim(ylim)
            plt.tick_params(axis = 'y', labelleft = False)
            if i == 2:
                plt.text(0.98, 0.02,
                         rf'$f_\mathrm{{observed}} = {np.sum(mask) / np.sum(patrol) * 100:.1f}\%, \,\,\,$'
                         rf'$f_\mathrm{{assigned}} = {np.sum(mask) / len(tile_coord) / len(fiber_coord) * 100:.1f}\%$',
                         transform = plt.gca().transAxes, ha = 'right', va = 'bottom', fontsize = 15, linespacing = 1.5,
                         path_effects = [path_effects.withStroke(linewidth = 1, foreground = 'w')])

        plt.text(0.97, 0.97, texts[i], transform = plt.gca().transAxes, ha = 'right', va = 'top', fontsize = 15,
                 path_effects = [path_effects.withStroke(linewidth = 1, foreground = 'w')])
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer = True))
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer = True))

    output_dir = DEMO_DIR / 'output'
    output_dir.mkdir(exist_ok = True)
    output_file = output_dir / f'fba_{Path(target_file).stem}__{Path(tile_file).stem}.png'

    plt.savefig(output_file, dpi = dpi, transparent = False, facecolor = 'w')
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-target', default = TARGET_FILE)
    parser.add_argument('-tile', default = TILE_FILE)
    parser.add_argument('-algorithm', default = 'greedy-classic')
    parser.add_argument('-dpi', type = int, default = 250)
    args = parser.parse_args()
    run_fiberassign(args.target, args.tile, algorithm = args.algorithm, dpi = args.dpi)