import numpy as np

from .fiberassign import _fiberassign_prepared, _prepare_fiberassign, _prepare_solver, fiberpatrol

from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import KDTree
from tqdm import tqdm

def _radec2xyz(coord):
    ra, dec = np.radians(np.asarray(coord, dtype = float)).T
    return np.column_stack((np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)))

def _popcount_uint64(array):
    x = np.asarray(array, dtype = np.uint64).copy()
    x -= (x >> np.uint64(1)) & np.uint64(0x5555555555555555)
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.uint8)

def _angular_counts_fast(xyz1, tree1, theta_edges, xyz2 = None, tree2 = None, weight1 = None):
    autocorr = xyz2 is None
    xyz2 = xyz1 if autocorr else xyz2
    tree2 = tree1 if autocorr else tree2
    chord_edges = 2 * np.sin(np.radians(theta_edges) / 2)

    weight1 = None if weight1 is None else np.asarray(weight1, dtype = float)
    if autocorr:
        norm = len(xyz1) * (len(xyz1) - 1) / 2
        weights = None
    else:
        norm = (len(xyz1) if weight1 is None else weight1.sum()) * len(xyz2)
        weights = None if weight1 is None else (weight1, np.ones(len(xyz2)))

    if not len(xyz1) or not len(xyz2) or norm <= 0:
        return np.zeros(len(theta_edges) - 1), float(norm)

    counts = np.asarray(tree1.count_neighbors(tree2, chord_edges, weights = weights, cumulative = False), dtype = float)[1:]
    if autocorr:
        counts /= 2.0

    return counts, float(norm)

def _angular_counts(coord1, theta_edges, coord2 = None, weight1 = None, bitweights = None,
                    norm_weight = None, nrealizations = 129, thread = 16, chunk_size = 4096,
                    xyz1 = None, xyz2 = None, tree = None, progress_bar = None):
    coord1 = np.asarray(coord1, dtype = float)
    autocorr = coord2 is None
    coord2 = coord1 if autocorr else np.asarray(coord2, dtype = float)
    theta_edges = np.asarray(theta_edges, dtype = float)
    counts = np.zeros(len(theta_edges) - 1, dtype = float)

    weight1 = None if weight1 is None else np.asarray(weight1, dtype = float)
    bitweights = None if bitweights is None else np.atleast_2d(np.asarray(bitweights, dtype = np.uint64).T).T
    norm_weight = None if norm_weight is None else np.asarray(norm_weight, dtype = float)

    if weight1 is not None and bitweights is not None:
        raise ValueError('weight1 and bitweights cannot both be provided')

    if autocorr:
        norm = len(coord1) * (len(coord1) - 1) / 2 if norm_weight is None else (
            norm_weight.sum() ** 2 - np.sum(norm_weight ** 2)) / 2
    else:
        norm = (len(coord1) if weight1 is None else weight1.sum()) * len(coord2)

    if not len(coord1) or not len(coord2) or norm <= 0:
        return counts, float(norm)

    xyz1 = _radec2xyz(coord1) if xyz1 is None else xyz1
    xyz2 = xyz1 if autocorr else (_radec2xyz(coord2) if xyz2 is None else xyz2)
    tree = KDTree(xyz2) if tree is None else tree
    chord_edges = 2 * np.sin(np.radians(theta_edges) / 2)
    nbin = len(theta_edges) - 1

    for start in range(0, len(coord1), chunk_size):
        stop = min(start + chunk_size, len(coord1))
        neighbors = tree.query_ball_point(xyz1[start:stop], chord_edges[-1], workers = max(1, int(thread)))
        lens = np.fromiter((len(x) for x in neighbors), dtype = int, count = stop - start)
        if not lens.sum():
            if progress_bar is not None:
                progress_bar.update()
            continue

        i = np.repeat(np.arange(start, stop), lens)
        j = np.concatenate(neighbors).astype(int, copy = False)

        if autocorr:
            keep = j > i
            i, j = i[keep], j[keep]
            if not len(i):
                if progress_bar is not None:
                    progress_bar.update()
                continue

        chord = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * np.einsum('ij,ij->i', xyz1[i], xyz2[j])))
        ibin = np.searchsorted(chord_edges, chord, side = 'right') - 1
        keep = (ibin >= 0) & (ibin < nbin)
        i, j, ibin = i[keep], j[keep], ibin[keep]
        if not len(ibin):
            if progress_bar is not None:
                progress_bar.update()
            continue

        if bitweights is not None:
            nobs = 1 + _popcount_uint64(bitweights[i] & bitweights[j]).sum(axis = 1)
            pair_weight = nrealizations / nobs
        elif weight1 is not None:
            pair_weight = weight1[i]
        else:
            pair_weight = None

        counts += np.bincount(ibin, weights = pair_weight, minlength = nbin)[:nbin]

        if progress_bar is not None:
            progress_bar.update()

    return counts, float(norm)

def _iip_batch(args):
    prepared, target_pri, nsub, algorithm, seeds = args
    solver_state = _prepare_solver(prepared, algorithm)
    nassigned = np.zeros(prepared[-1], dtype = np.uint16)
    pip_bitweights = np.zeros((prepared[-1], 2), dtype = np.uint64)

    for seed in seeds:
        target_sub = np.random.default_rng(seed).random(nsub)
        assigned = _fiberassign_prepared(prepared, target_pri, target_sub, algorithm, solver_state)
        nassigned += assigned
        pip_bitweights[assigned, (seed - 1) // 64] |= np.uint64(1 << ((seed - 1) % 64))

    return nassigned, pip_bitweights, len(seeds)

def _sample_cap(ra, dec, radius, size, rng):
    ra, dec, radius = np.radians([ra, dec, radius])
    center = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    east = np.array([-np.sin(ra), np.cos(ra), 0.0])
    north = np.cross(center, east)

    cos_theta = rng.uniform(np.cos(radius), 1.0, size)
    theta = np.arccos(cos_theta)
    phi = rng.uniform(0.0, 2 * np.pi, size)
    xyz = np.cos(theta)[:, None] * center + np.sin(theta)[:, None] * (np.cos(phi)[:, None] * east + np.sin(phi)[:, None] * north)

    return np.column_stack((np.degrees(np.mod(np.arctan2(xyz[:, 1], xyz[:, 0]), 2 * np.pi)),
                            np.degrees(np.arcsin(np.clip(xyz[:, 2], -1, 1)))))

def gen_catalog(fiber_coord, tile_coord, tile_rotation, target_coord, target_pri, target_sub,
                algorithm = 'greedy-classic', r_patrol = 6.0, r_exclude = 1.6,
                telescope = 'just', thread = 16, progress = None):
    target_pri, target_sub = np.asarray(target_pri), np.asarray(target_sub)
    prepared = _prepare_fiberassign(fiber_coord, tile_coord, tile_rotation, target_coord,
                                    r_patrol = r_patrol, r_exclude = r_exclude, telescope = telescope)
    patrolled_mask = prepared[2]
    if progress is not None:
        progress('patrolled', patrolled_mask)

    solver_state = _prepare_solver(prepared, algorithm)
    assigned_mask = _fiberassign_prepared(prepared, target_pri, target_sub, algorithm, solver_state)
    if progress is not None:
        progress('assigned', assigned_mask)

    nassigned = assigned_mask.astype(np.uint16)
    pip_bitweights = np.zeros((len(target_coord), 2), dtype = np.uint64)

    nworker = min(max(int(thread), 1), 128)
    if nworker == 1:
        for seed in tqdm(range(1, 129), total = 128, desc = 'Alt Subpriority Reassignment'):
            target_sub_i = np.random.default_rng(seed).random(len(target_sub))
            assigned = _fiberassign_prepared(prepared, target_pri, target_sub_i, algorithm, solver_state)
            nassigned += assigned
            pip_bitweights[assigned, (seed - 1) // 64] |= np.uint64(1 << ((seed - 1) % 64))
    else:
        batches = [seeds for seeds in np.array_split(np.arange(1, 129), nworker) if len(seeds)]
        jobs = [(prepared, target_pri, len(target_sub), algorithm, seeds) for seeds in batches]
        with ProcessPoolExecutor(max_workers = len(batches)) as executor, tqdm(total = 128, desc = 'Alt Subpriority Reassignment') as progress_bar:
            for count, bitweights, nseed in executor.map(_iip_batch, jobs):
                nassigned += count
                pip_bitweights |= bitweights
                progress_bar.update(nseed)

    iip_weight = np.zeros(len(target_coord), dtype = float)
    iip_weight[assigned_mask] = 129.0 / nassigned[assigned_mask]

    if progress is not None:
        progress('weights', (iip_weight, pip_bitweights))

    return assigned_mask, iip_weight, pip_bitweights, patrolled_mask

def gen_random(fiber_coord, tile_coord, tile_rotation, target_z, density = 5000, tile_radius = 0.6,
               r_patrol = 6.0, telescope = 'just', seed = 0):
    tile_coord = np.atleast_2d(tile_coord)
    target_z = np.asarray(target_z)
    rng = np.random.default_rng(seed)

    radius = np.radians(tile_radius)
    area = 2 * np.pi * (1.0 - np.cos(radius)) * (180.0 / np.pi) ** 2
    n_tile = int(np.ceil(density * area))

    ra, dec = np.radians(tile_coord).T
    tile_xyz = np.column_stack((np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)))
    tree = KDTree(tile_xyz)
    chord = 2 * np.sin(radius / 2)

    random = []
    for i, tile in enumerate(tile_coord):
        coord = _sample_cap(tile[0], tile[1], tile_radius, n_tile, rng)

        ra, dec = np.radians(coord).T
        xyz = np.column_stack((np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)))
        covered = tree.query_ball_point(xyz, r = chord)
        keep = np.fromiter((min(c) == i for c in covered), dtype = bool, count = len(coord))
        random.append(coord[keep])

    random_coord = np.concatenate(random)
    random_z = rng.choice(target_z, size = len(random_coord), replace = True)
    patrolled_mask = fiberpatrol(fiber_coord, tile_coord, tile_rotation, random_coord,
                                 r_patrol = r_patrol, telescope = telescope)

    return random_coord, random_z, patrolled_mask

def gen_wang(patrolled_coord, assigned_coord, assigned_iip_weight, assigned_pip_bitweights,
             random_coord, theta_edges = None, thread = 16):
    theta_edges = np.logspace(-4, 0.5, 41) if theta_edges is None else np.asarray(theta_edges, dtype = float)
    assigned_iip_weight = np.asarray(assigned_iip_weight, dtype = float)
    assigned_pip_bitweights = np.asarray(assigned_pip_bitweights, dtype = np.uint64)
    nrealizations = 1 + 8 * assigned_pip_bitweights.dtype.itemsize * assigned_pip_bitweights.shape[1]

    patrolled_xyz = _radec2xyz(patrolled_coord)
    assigned_xyz = _radec2xyz(assigned_coord)
    random_xyz = _radec2xyz(random_coord)
    patrolled_tree = KDTree(patrolled_xyz)
    assigned_tree = KDTree(assigned_xyz)
    random_tree = KDTree(random_xyz)

    chunk_size = 4096
    n_chunks = (len(assigned_coord) + chunk_size - 1) // chunk_size

    with tqdm(total = n_chunks + 3, desc = 'Angular Upweight Calculation') as progress_bar:
        dd_parent, n_dd_parent = _angular_counts_fast(
            patrolled_xyz, patrolled_tree, theta_edges
        )
        progress_bar.update()

        dd_pip, n_dd_pip = _angular_counts(
            assigned_coord, theta_edges, bitweights = assigned_pip_bitweights,
            norm_weight = assigned_iip_weight, nrealizations = nrealizations, thread = thread,
            chunk_size = chunk_size, xyz1 = assigned_xyz, tree = assigned_tree, progress_bar = progress_bar
        )

        dr_parent, n_dr_parent = _angular_counts_fast(
            patrolled_xyz, patrolled_tree, theta_edges,
            xyz2 = random_xyz, tree2 = random_tree
        )
        progress_bar.update()

        dr_iip, n_dr_iip = _angular_counts_fast(
            assigned_xyz, assigned_tree, theta_edges,
            xyz2 = random_xyz, tree2 = random_tree, weight1 = assigned_iip_weight
        )
        progress_bar.update()

    dd_parent = dd_parent / n_dd_parent if n_dd_parent > 0 else dd_parent
    dd_pip = dd_pip / n_dd_pip if n_dd_pip > 0 else dd_pip
    dr_parent = dr_parent / n_dr_parent if n_dr_parent > 0 else dr_parent
    dr_iip = dr_iip / n_dr_iip if n_dr_iip > 0 else dr_iip

    wang_dd = np.ones(len(theta_edges) - 1)
    wang_dr = np.ones(len(theta_edges) - 1)
    np.divide(dd_parent, dd_pip, out = wang_dd, where = dd_pip > 0)
    np.divide(dr_parent, dr_iip, out = wang_dr, where = dr_iip > 0)

    theta = np.sqrt(theta_edges[:-1] * theta_edges[1:])
    return theta, wang_dd, wang_dr