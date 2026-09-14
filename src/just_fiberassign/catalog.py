import numpy as np

from .fiberassign import _fiberassign_prepared, _prepare_fiberassign, _prepare_solver, fiberpatrol

from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import KDTree
from tqdm import tqdm

def _iip_batch(args):
    prepared, target_pri, nsub, algorithm, seeds = args
    solver_state = _prepare_solver(prepared, algorithm)
    nassigned = np.zeros(prepared[-1], dtype = np.uint16)
    for seed in seeds:
        target_sub = np.random.default_rng(seed).random(nsub)
        nassigned += _fiberassign_prepared(prepared, target_pri, target_sub, algorithm, solver_state, thread = 1)
    return nassigned, len(seeds)

def gen_catalog(fiber_coord, tile_coord, tile_rotation, target_coord, target_pri, target_sub,
                algorithm = 'greedy-classic', r_patrol = 6.0, r_exclude = 1.6, telescope = 'just', thread = 16):
    target_pri, target_sub = np.asarray(target_pri), np.asarray(target_sub)
    prepared = _prepare_fiberassign(fiber_coord, tile_coord, tile_rotation, target_coord,
                                    r_patrol = r_patrol, r_exclude = r_exclude, telescope = telescope)
    solver_state = _prepare_solver(prepared, algorithm)
    assigned_mask = _fiberassign_prepared(prepared, target_pri, target_sub, algorithm, solver_state, thread = thread)
    patrolled_mask = prepared[2]
    nassigned = assigned_mask.astype(np.uint16)

    nworker = min(max(int(thread), 1), 128)
    if nworker == 1:
        for seed in tqdm(range(1, 129), total = 128, desc = 'IIP Weights'):
            target_sub_i = np.random.default_rng(seed).random(len(target_sub))
            nassigned += _fiberassign_prepared(prepared, target_pri, target_sub_i, algorithm, solver_state, thread = 1)
    else:
        batches = [seeds for seeds in np.array_split(np.arange(1, 129), nworker) if len(seeds)]
        jobs = [(prepared, target_pri, len(target_sub), algorithm, seeds) for seeds in batches]
        with ProcessPoolExecutor(max_workers = len(batches)) as executor, tqdm(total = 128, desc = 'IIP Weights') as progress:
            for count, nseed in executor.map(_iip_batch, jobs):
                nassigned += count
                progress.update(nseed)

    iip_weight = np.zeros(len(target_coord), dtype = float)
    iip_weight[assigned_mask] = 129.0 / nassigned[assigned_mask]

    return assigned_mask, iip_weight, patrolled_mask

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