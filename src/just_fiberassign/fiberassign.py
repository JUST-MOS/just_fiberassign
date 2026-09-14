import numpy as np
import os

from .utils import _get_radius_deg, radec2xy

from heapq import heapify, heappop, heappush
from ortools.sat.python import cp_model
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.spatial import KDTree

def _get_tile_edges(fiber_coord, tile_coord, tile_rotation, target_coord, r_patrol, r_exclude, telescope):
    tile_coord = np.atleast_2d(tile_coord)
    target_coord = np.asarray(target_coord)
    nf, nt = len(fiber_coord), len(target_coord)
    tile_rotation = np.broadcast_to(np.asarray(tile_rotation, dtype = float), len(tile_coord))

    ra, dec = np.radians(target_coord).T
    target_xyz = np.column_stack((np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)))
    ra, dec = np.radians(tile_coord).T
    tile_xyz = np.column_stack((np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)))

    r_edge = np.max(np.hypot(fiber_coord[:, 0], fiber_coord[:, 1])) + r_patrol
    r_chord = 2 * np.sin(np.radians(float(_get_radius_deg(r_edge, 0.0, telescope))) / 2)
    near_all = KDTree(target_xyz).query_ball_point(tile_xyz, r = r_chord)

    edge_f, edge_t, collision_groups = [], [], []
    pat_mask = np.zeros(nt, dtype = bool)

    for i, (tile, near) in enumerate(zip(tile_coord, near_all)):
        near = np.asarray(near, dtype = int)
        if not len(near):
            continue

        xy = radec2xy(tile[0], tile[1], target_coord[near, 0], target_coord[near, 1], telescope, rotation = tile_rotation[i])
        hits = KDTree(xy).query_ball_point(fiber_coord, r = r_patrol)
        lens = np.fromiter((len(c) for c in hits), dtype = int, count = nf)
        if not lens.sum():
            continue

        local = np.concatenate(hits).astype(int, copy = False)
        ef = i * nf + np.repeat(np.arange(nf), lens)
        et = near[local]
        base = len(edge_t)
        edge_f.extend(ef)
        edge_t.extend(et)

        pat = np.unique(local)
        pat_mask[near[pat]] = True
        order = np.argsort(et, kind = 'stable')
        cut = np.flatnonzero(np.diff(et[order])) + 1
        target_edges = {int(et[g[0]]): base + g for g in np.split(order, cut)}

        pairs = KDTree(xy[pat]).query_pairs(r = r_exclude, output_type = 'ndarray')
        for j1, j2 in pairs:
            t1, t2 = int(near[pat[j1]]), int(near[pat[j2]])
            collision_groups.append(np.r_[target_edges[t1], target_edges[t2]])

    return np.asarray(edge_f, dtype = int), np.asarray(edge_t, dtype = int), pat_mask, collision_groups

def _prepare_fiberassign(fiber_coord, tile_coord, tile_rotation, target_coord, r_patrol = 6.0, r_exclude = 1.6, telescope = 'just'):
    edge_f, edge_t, pat_mask, collision_groups = _get_tile_edges(fiber_coord, tile_coord, tile_rotation, target_coord,
                                                                 r_patrol, r_exclude, telescope)
    return edge_f, edge_t, pat_mask, collision_groups, len(np.atleast_2d(tile_coord)), len(fiber_coord), len(target_coord)

def _priority_score(target_pri, target_sub):
    return np.asarray(target_pri, dtype = float) + np.asarray(target_sub, dtype = float)

def _constraint_groups(edge_f, edge_t, collision_groups):
    groups = []
    for key in (edge_f, edge_t):
        order = np.argsort(key, kind = 'stable')
        cut = np.flatnonzero(np.diff(key[order])) + 1
        groups += [g for g in np.split(order, cut) if len(g) > 1]
    return groups + collision_groups

def _prepare_greedy(edge_f, edge_t, collision_groups, n_tiles, nf, n_targets):
    fiber_edges, target_edges = {}, {}
    for key, groups in ((edge_f, fiber_edges), (edge_t, target_edges)):
        order = np.argsort(key, kind = 'stable')
        cut = np.flatnonzero(np.diff(key[order])) + 1
        for group in np.split(order, cut):
            groups[int(key[group[0]])] = group

    tile_fibers = [[] for _ in range(n_tiles)]
    tile_targets = [set() for _ in range(n_tiles)]
    for f, group in fiber_edges.items():
        tile = f // nf
        tile_fibers[tile].append(f)
        tile_targets[tile].update(map(int, edge_t[group]))

    tile_fibers = [np.asarray(f, dtype = int) for f in tile_fibers]
    tile_targets = [np.fromiter(sorted(t), dtype = int, count = len(t)) for t in tile_targets]

    conflict_edges = [{} for _ in range(n_tiles)]
    for group in collision_groups:
        tile = int(edge_f[group[0]] // nf)
        targets = np.unique(edge_t[group])
        for target in targets:
            other = group[edge_t[group] != target]
            if len(other):
                conflict_edges[tile].setdefault(int(target), []).append(other)

    fiber_degree = np.bincount(edge_f, minlength = n_tiles * nf)
    target_degree = np.bincount(edge_t, minlength = n_targets)
    return fiber_edges, target_edges, tile_fibers, tile_targets, conflict_edges, fiber_degree, target_degree

def _prepare_solver(prepared, algorithm):
    edge_f, edge_t, _, collision_groups, n_tiles, nf, n_targets = prepared
    if algorithm == 'greedy':
        algorithm = 'greedy-classic'
    if algorithm not in ('greedy-classic', 'greedy-global', 'milp', 'cp-sat'):
        raise ValueError(f'Unknown fiber-assignment algorithm: {algorithm}')
    if not len(edge_t):
        return ()

    if algorithm in ('greedy-classic', 'greedy-global'):
        return _prepare_greedy(edge_f, edge_t, collision_groups, n_tiles, nf, n_targets)

    groups = _constraint_groups(edge_f, edge_t, collision_groups)
    if not groups:
        return ()

    used = np.unique(np.concatenate(groups))
    inv = np.full(len(edge_t), -1, dtype = int)
    inv[used] = np.arange(len(used))
    groups = [inv[g] for g in groups]
    free = np.ones(len(edge_t), dtype = bool)
    free[used] = False

    if algorithm == 'milp':
        length = np.fromiter((len(g) for g in groups), dtype = int, count = len(groups))
        col = np.concatenate(groups)
        row = np.repeat(np.arange(len(groups)), length)
        A = coo_matrix((np.ones(len(col)), (row, col)), shape = (len(groups), len(used))).tocsr()
        return used, free, Bounds(0, 1), np.ones(len(used)), LinearConstraint(A, 0, 1)

    model = cp_model.CpModel()
    x = [model.new_bool_var(f'x{i}') for i in range(len(used))]
    for group in groups:
        model.add(sum(x[int(j)] for j in group) <= 1)
    return used, free, model, x

def _deactivate_greedy(edges, active, edge_f, edge_t, fiber_degree, target_degree):
    edges = np.asarray(edges, dtype = int)
    edges = edges[active[edges]]
    if not len(edges):
        return np.empty(0, dtype = int)

    active[edges] = False
    fibers, n_fibers = np.unique(edge_f[edges], return_counts = True)
    targets, n_targets = np.unique(edge_t[edges], return_counts = True)
    fiber_degree[fibers] -= n_fibers
    target_degree[targets] -= n_targets
    return fibers

def _best_greedy_edge(group, active, edge_t, target_pri, target_sub, target_degree):
    edges = group[active[group]]
    if not len(edges):
        return -1
    return int(max(edges, key = lambda j: (target_pri[edge_t[j]], target_sub[edge_t[j]],
                                           -target_degree[edge_t[j]], -edge_t[j])))

def _solve_greedy_classic(prepared, solver_state, target_pri, target_sub):
    edge_f, edge_t, _, _, n_tiles, nf, n_targets = prepared
    ass_mask = np.zeros(n_targets, dtype = bool)
    if not len(edge_t):
        return ass_mask

    fiber_edges, target_edges, tile_fibers, tile_targets, conflict_edges, fiber_degree, target_degree = solver_state
    fiber_degree, target_degree = fiber_degree.copy(), target_degree.copy()
    active = np.ones(len(edge_t), dtype = bool)
    remaining_tiles = set(i for i in range(n_tiles) if len(tile_fibers[i]))

    while remaining_tiles:
        viable = [i for i in remaining_tiles if np.any(fiber_degree[tile_fibers[i]] > 0)]
        if not viable:
            break

        def tile_key(i):
            fibers = tile_fibers[i]
            n_fiber = np.count_nonzero(fiber_degree[fibers] > 0)
            n_target = np.count_nonzero(~ass_mask[tile_targets[i]])
            return n_target / n_fiber, n_target, i

        tile = min(viable, key = tile_key)
        remaining_tiles.remove(tile)
        heap = [(int(fiber_degree[f]), int(f)) for f in tile_fibers[tile] if fiber_degree[f] > 0]
        heapify(heap)

        while heap:
            degree, fiber = heappop(heap)
            if degree <= 0 or degree != fiber_degree[fiber]:
                continue

            best_edge = _best_greedy_edge(fiber_edges[fiber], active, edge_t, target_pri, target_sub, target_degree)
            if best_edge < 0:
                continue

            target = int(edge_t[best_edge])
            ass_mask[target] = True
            affected = set()

            for edges in (fiber_edges[fiber], target_edges[target]):
                affected.update(map(int, _deactivate_greedy(edges, active, edge_f, edge_t, fiber_degree, target_degree)))
            for edges in conflict_edges[tile].get(target, ()):
                affected.update(map(int, _deactivate_greedy(edges, active, edge_f, edge_t, fiber_degree, target_degree)))

            for f in affected:
                if f // nf == tile and fiber_degree[f] > 0:
                    heappush(heap, (int(fiber_degree[f]), f))

    return ass_mask

def _solve_greedy_global(prepared, solver_state, target_pri, target_sub):
    edge_f, edge_t, _, _, _, nf, n_targets = prepared
    ass_mask = np.zeros(n_targets, dtype = bool)
    if not len(edge_t):
        return ass_mask

    fiber_edges, target_edges, tile_fibers, _, conflict_edges, fiber_degree, target_degree = solver_state
    fiber_degree, target_degree = fiber_degree.copy(), target_degree.copy()
    active = np.ones(len(edge_t), dtype = bool)
    heap = [(int(fiber_degree[f]), int(f)) for fibers in tile_fibers for f in fibers if fiber_degree[f] > 0]
    heapify(heap)

    while heap:
        degree, fiber = heappop(heap)
        if degree <= 0 or degree != fiber_degree[fiber]:
            continue

        best_edge = _best_greedy_edge(fiber_edges[fiber], active, edge_t, target_pri, target_sub, target_degree)
        if best_edge < 0:
            continue

        tile = fiber // nf
        target = int(edge_t[best_edge])
        ass_mask[target] = True
        affected = set()

        for edges in (fiber_edges[fiber], target_edges[target]):
            affected.update(map(int, _deactivate_greedy(edges, active, edge_f, edge_t, fiber_degree, target_degree)))
        for edges in conflict_edges[tile].get(target, ()):
            affected.update(map(int, _deactivate_greedy(edges, active, edge_f, edge_t, fiber_degree, target_degree)))

        for f in affected:
            if fiber_degree[f] > 0:
                heappush(heap, (int(fiber_degree[f]), f))

    return ass_mask

def _solve_milp(prepared, solver_state, target_pri, target_sub):
    _, edge_t, _, _, _, _, n_targets = prepared
    ass_mask = np.zeros(n_targets, dtype = bool)
    if not len(edge_t):
        return ass_mask
    if not solver_state:
        ass_mask[edge_t] = True
        return ass_mask

    used, free, bounds, integrality, constraints = solver_state
    cost = -_priority_score(target_pri, target_sub)[edge_t[used]]
    res = milp(c = cost, integrality = integrality, bounds = bounds, constraints = constraints, options = {'mip_rel_gap': 0.0})
    if not res.success:
        raise RuntimeError(res.message)

    ass_mask[edge_t[free]] = True
    ass_mask[edge_t[used[res.x > 0.5]]] = True
    return ass_mask

def _solve_cpsat(prepared, solver_state, target_pri, target_sub, thread = None):
    _, edge_t, _, _, _, _, n_targets = prepared
    ass_mask = np.zeros(n_targets, dtype = bool)
    if not len(edge_t):
        return ass_mask
    if not solver_state:
        ass_mask[edge_t] = True
        return ass_mask

    used, free, model, x = solver_state
    score = _priority_score(target_pri, target_sub)[edge_t[used]]
    model.maximize(sum(float(score[i]) * x[i] for i in range(len(used))))

    n_cpu = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = max(1, min(72, n_cpu if thread is None else int(thread)))
    status = solver.solve(model)
    if status != cp_model.OPTIMAL:
        raise RuntimeError(f'CP-SAT did not prove optimality: {solver.status_name(status)}')

    selected = np.fromiter((solver.value(v) > 0 for v in x), dtype = bool, count = len(x))
    ass_mask[edge_t[free]] = True
    ass_mask[edge_t[used[selected]]] = True
    return ass_mask

def _fiberassign_prepared(prepared, target_pri, target_sub, algorithm = 'greedy-classic', solver_state = None, thread = None):
    target_pri, target_sub = np.asarray(target_pri), np.asarray(target_sub)
    solver_state = _prepare_solver(prepared, algorithm) if solver_state is None else solver_state

    if algorithm in ('greedy', 'greedy-classic'):
        return _solve_greedy_classic(prepared, solver_state, target_pri, target_sub)
    if algorithm == 'greedy-global':
        return _solve_greedy_global(prepared, solver_state, target_pri, target_sub)
    if algorithm == 'milp':
        return _solve_milp(prepared, solver_state, target_pri, target_sub)
    if algorithm == 'cp-sat':
        return _solve_cpsat(prepared, solver_state, target_pri, target_sub, thread)
    raise ValueError(f'Unknown fiber-assignment algorithm: {algorithm}')

def fiberpatrol(fiber_coord, tile_coord, tile_rotation, target_coord, r_patrol = 6.0, telescope = 'just'):
    tile_coord = np.atleast_2d(tile_coord)
    target_coord = np.asarray(target_coord)
    tile_rotation = np.broadcast_to(np.asarray(tile_rotation, dtype = float), len(tile_coord))
    pat_mask = np.zeros(len(target_coord), dtype = bool)

    ra, dec = np.radians(target_coord).T
    target_xyz = np.column_stack((np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)))
    ra, dec = np.radians(tile_coord).T
    tile_xyz = np.column_stack((np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)))

    r_edge = np.max(np.hypot(fiber_coord[:, 0], fiber_coord[:, 1])) + r_patrol
    r_chord = 2 * np.sin(np.radians(float(_get_radius_deg(r_edge, 0.0, telescope))) / 2)
    near_all = KDTree(target_xyz).query_ball_point(tile_xyz, r = r_chord)

    for i, (tile, near) in enumerate(zip(tile_coord, near_all)):
        near = np.asarray(near, dtype = int)
        if not len(near):
            continue

        xy = radec2xy(tile[0], tile[1], target_coord[near, 0], target_coord[near, 1], telescope, rotation = tile_rotation[i])
        hits = KDTree(xy).query_ball_point(fiber_coord, r = r_patrol)
        lens = np.fromiter((len(c) for c in hits), dtype = int, count = len(fiber_coord))
        if lens.sum():
            local = np.concatenate(hits).astype(int, copy = False)
            pat_mask[near[np.unique(local)]] = True

    return pat_mask

def fiberassign(fiber_coord, tile_coord, tile_rotation, target_coord, target_pri, target_sub,
                algorithm = 'greedy-classic', r_patrol = 6.0, r_exclude = 1.6, telescope = 'just'):
    prepared = _prepare_fiberassign(fiber_coord, tile_coord, tile_rotation, target_coord, r_patrol, r_exclude, telescope)
    solver_state = _prepare_solver(prepared, algorithm)
    return _fiberassign_prepared(prepared, target_pri, target_sub, algorithm, solver_state), prepared[2]