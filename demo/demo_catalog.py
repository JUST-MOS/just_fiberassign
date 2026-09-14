import matplotlib.pyplot as plt
import numpy as np

from just_fiberassign.catalog import gen_catalog, gen_random
from just_fiberassign.utils import load_focalplane, load_target, load_tile

from matplotlib.ticker import MaxNLocator
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
TARGET_FILE = DEMO_DIR / 'input/target/gal_mock_rlim20p5_ra0_4_dec0_4.npz'
TILE_FILE = DEMO_DIR / 'input/tiling/tile_ipack_1pass_ra0_4_dec0_4.npz'

def run_catalog(target_file = TARGET_FILE, tile_file = TILE_FILE, thread = 16, dpi = 250):
    target_coord, target_pri, target_sub, target_z, target_mr = load_target(target_file)
    _, tile_coord, tile_rotation = load_tile(tile_file)
    fiber_coord = load_focalplane('just')

    assigned_mask, iip_weight, patrolled_mask = gen_catalog(
        fiber_coord, tile_coord, tile_rotation, target_coord, target_pri, target_sub,
        algorithm = 'greedy-classic', thread = thread
    )
    random_coord, random_z, random_patrol = gen_random(fiber_coord, tile_coord, tile_rotation, target_z)

    output_dir = DEMO_DIR / 'output'
    output_dir.mkdir(exist_ok = True)
    stem = f'{Path(target_file).stem}_{Path(tile_file).stem}'
    output_file = output_dir / f'cat_{stem}.npz'

    np.savez_compressed(
        output_file,
        patrolled_coord = target_coord[patrolled_mask], patrolled_z = target_z[patrolled_mask],
        assigned_coord = target_coord[assigned_mask], assigned_z = target_z[assigned_mask],
        assigned_iip_weight = iip_weight[assigned_mask],
        random_coord = random_coord[random_patrol], random_z = random_z[random_patrol]
    )

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

    plt.figure(figsize = (10.58, full_height))

    for i in range(3):
        plt.axes([(0.12 * 5 + single_edge * i) / 10.58, 0.12 * 5 / full_height, single_edge / 10.58, single_edge / full_height])

        if i == 0:
            plt.scatter(target_coord[:, 0], target_coord[:, 1], s = 0.25, marker = '.', lw = 0, color = 'gray')
            plt.scatter(target_coord[patrolled_mask, 0], target_coord[patrolled_mask, 1], s = 1.0, marker = '.', lw = 0, color = b_)
            xlim, ylim = plt.xlim(), plt.ylim()
        elif i == 1:
            plt.scatter(target_coord[:, 0], target_coord[:, 1], s = 0.25, marker = '.', lw = 0, color = 'gray')
            plt.scatter(target_coord[assigned_mask, 0], target_coord[assigned_mask, 1], s = 1.0, marker = '.', lw = 0, color = r_)
            plt.xlim(xlim); plt.ylim(ylim)
        else:
            plt.scatter(random_coord[random_patrol, 0], random_coord[random_patrol, 1], s = 1.0, marker = '.', lw = 0, color = g_)
            plt.xlim(xlim); plt.ylim(ylim)

        plt.xlabel(rf'$\mathrm{{R.A.}}$ $\mathrm{{[deg]}}$', fontsize = 15, labelpad = 6.0)
        plt.tick_params(which = 'both', top = True, right = True, labelsize = 15)
        plt.minorticks_on()

        if i == 0:
            plt.ylabel(rf'$\mathrm{{Dec.}}$ $\mathrm{{[deg]}}$', fontsize = 15, labelpad = 6.0)
        else:
            plt.tick_params(axis = 'y', labelleft = False)

        plt.gca().xaxis.set_major_locator(MaxNLocator(integer = True))
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer = True))

    plot_file = output_dir / f'cat_{stem}.png'
    plt.savefig(plot_file, dpi = dpi, transparent = False, facecolor = 'w')
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-target', default = TARGET_FILE)
    parser.add_argument('-tile', default = TILE_FILE)
    parser.add_argument('-thread', type = int, default = 16)
    parser.add_argument('-dpi', type = int, default = 250)
    args = parser.parse_args()
    run_catalog(args.target, args.tile, thread = args.thread, dpi = args.dpi)