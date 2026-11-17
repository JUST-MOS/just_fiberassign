import numpy as np
import os

from .utils import _get_radius_deg, load_focalplane, radec2xy

from functools import lru_cache
from heapq import heapify, heappop, heappush
from numba import njit
from ortools.sat.python import cp_model
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.spatial import KDTree

def _build_tile_edges(fiber_coord, tile_coord, tile_rotation, target_coord, r_patrol, r_exclude, telescope, compact = False):
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
    if compact:
        tree = KDTree(target_xyz)
        near_all = (tree.query_ball_point(tile_xyz[i:i + 1], r = r_chord)[0] for i in range(len(tile_coord)))
    else:
        near_all = KDTree(target_xyz).query_ball_point(tile_xyz, r = r_chord)

    edge_f, edge_t, collision_groups = [], [], []
    n_edges = 0
    pat_mask = np.zeros(nt, dtype = bool)
    fiber_tree = KDTree(fiber_coord)
    if compact:
        fiber_dtype = np.int32 if len(tile_coord) * nf <= np.iinfo(np.int32).max else np.int64
        target_dtype = np.int32 if nt <= np.iinfo(np.int32).max else np.int64
        fiber_ptr = np.zeros(len(tile_coord) * nf + 1, dtype = np.int64)
        tile_ptr = np.zeros(len(tile_coord) + 1, dtype = np.int64)
        occurrence_t, occurrence_ptr, occurrence_edges, conflict_ptr, conflict_nodes = [], [], [], [], []
        n_occurrences, n_conflicts = 0, 0
        index, conflicts = _helper_greedy_global()[:2]

    for i, (tile, near) in enumerate(zip(tile_coord, near_all)):
        near = np.asarray(near, dtype = int)
        if not len(near):
            continue

        xy = radec2xy(tile[0], tile[1], target_coord[near, 0], target_coord[near, 1], telescope, rotation = tile_rotation[i])
        local_tree = KDTree(xy)
        hits = fiber_tree.query_ball_tree(local_tree, r = r_patrol)
        lens = np.fromiter((len(c) for c in hits), dtype = int, count = nf)
        if not lens.sum():
            continue

        local = np.concatenate(hits).astype(int, copy = False)
        del hits
        if compact:
            edge_f.append(np.repeat(np.arange(i * nf, (i + 1) * nf, dtype = fiber_dtype), lens))
            edge_t.append(near[local].astype(target_dtype, copy = False))
            fiber_ptr[i * nf + 1:(i + 1) * nf + 1] = lens
            ptr = np.zeros(len(near) + 1, dtype = np.int64)
            order = np.empty(len(local), dtype = np.int64)
            index(local, ptr, order)
            pat = np.flatnonzero(np.diff(ptr))
            pat_mask[near[pat]] = True
            tile_ptr[i + 1] = len(pat)
            occurrence_t.append(near[pat].astype(target_dtype, copy = False))
            occurrence_ptr.append(ptr[pat] + n_edges)
            order += n_edges
            occurrence_edges.append(order)

            pairs = (local_tree if len(pat) == len(near) else KDTree(xy[pat])).query_pairs(r = r_exclude, output_type = 'ndarray')
            ptr = np.zeros(len(pat) + 1, dtype = np.int64)
            other = np.empty(2 * len(pairs), dtype = np.int64)
            conflicts(pairs, ptr, other)
            conflict_ptr.append(ptr[:-1] + n_conflicts)
            other += n_occurrences
            conflict_nodes.append(other)
            n_edges += len(local)
            n_occurrences += len(pat)
            n_conflicts += len(other)
            continue

        ef = i * nf + np.repeat(np.arange(nf), lens)
        et = near[local]
        base = n_edges
        n_edges += len(et)
        edge_f.append(ef)
        edge_t.append(et)

        pat = np.unique(local)
        pat_mask[near[pat]] = True
        order = np.argsort(et, kind = 'stable')
        cut = np.flatnonzero(np.diff(et[order])) + 1
        target_edges = {int(et[order[a]]): base + order[a:b] for a, b in zip(np.r_[0, cut], np.r_[cut, len(order)])}

        pairs = (local_tree if len(pat) == len(near) else KDTree(xy[pat])).query_pairs(r = r_exclude, output_type = 'ndarray')
        for j1, j2 in pairs:
            t1, t2 = int(near[pat[j1]]), int(near[pat[j2]])
            collision_groups.append(np.concatenate((target_edges[t1], target_edges[t2])))

    if compact:
        del near_all, tree, target_xyz
        np.cumsum(fiber_ptr, out = fiber_ptr)
        np.cumsum(tile_ptr, out = tile_ptr)
        edge_f = np.concatenate(edge_f) if edge_f else np.empty(0, dtype = fiber_dtype)
        edge_t = np.concatenate(edge_t) if edge_t else np.empty(0, dtype = target_dtype)
        occurrence_t = np.concatenate(occurrence_t) if occurrence_t else np.empty(0, dtype = target_dtype)
        occurrence_ptr.append(np.array([n_edges], dtype = np.int64))
        occurrence_ptr = np.concatenate(occurrence_ptr)
        occurrence_edges = np.concatenate(occurrence_edges) if occurrence_edges else np.empty(0, dtype = np.int64)
        conflict_ptr.append(np.array([n_conflicts], dtype = np.int64))
        conflict_ptr = np.concatenate(conflict_ptr)
        conflict_nodes = np.concatenate(conflict_nodes) if conflict_nodes else np.empty(0, dtype = np.int64)
        collision_groups = fiber_ptr, tile_ptr, occurrence_t, occurrence_ptr, occurrence_edges, conflict_ptr, conflict_nodes
        return edge_f, edge_t, pat_mask, collision_groups

    return (np.concatenate(edge_f) if edge_f else np.empty(0, dtype = int)), \
           (np.concatenate(edge_t) if edge_t else np.empty(0, dtype = int)), pat_mask, collision_groups

def _prepare_fiberassign(fiber_coord, tile_coord, tile_rotation, target_coord, r_patrol = 6.0, r_exclude = 1.6, telescope = 'just',
                         compact = False):
    edge_f, edge_t, pat_mask, collision_groups = _build_tile_edges(fiber_coord, tile_coord, tile_rotation, target_coord,
                                                               r_patrol, r_exclude, telescope, compact = compact)
    return edge_f, edge_t, pat_mask, collision_groups, len(np.atleast_2d(tile_coord)), len(fiber_coord), len(target_coord)

def _score_priority(target_pri, target_sub):
    return np.asarray(target_pri, dtype = float) + np.asarray(target_sub, dtype = float)

def _build_constraint_groups(edge_f, edge_t, collision_groups):
    groups = []
    for key in (edge_f, edge_t):
        order = np.arange(len(key)) if np.all(key[1:] >= key[:-1]) else np.argsort(key, kind = 'stable')
        cut = np.flatnonzero(np.diff(key[order])) + 1
        groups.extend(order[a:b] for a, b in zip(np.r_[0, cut], np.r_[cut, len(order)]) if b - a > 1)
    return groups + collision_groups

def _helper_greedy():
    def deactivate_edges(edges, active, edge_f, edge_t, fiber_degree, target_degree):
        edges = edges[active[edges]]
        if not len(edges):
            return np.empty(0, dtype = int)

        active[edges] = False
        fibers = edge_f[edges]
        if len(edges) == 1:
            fiber_degree[fibers[0]] -= 1
            target_degree[edge_t[edges[0]]] -= 1
            return fibers
        np.subtract.at(fiber_degree, fibers, 1)
        np.subtract.at(target_degree, edge_t[edges], 1)
        return np.unique(fibers)

    def select_best_edge(group, active, edge_t, target_pri, target_sub, target_degree):
        edges = group[active[group]]
        if not len(edges):
            return -1
        return int(max(edges, key = lambda j: (target_pri[edge_t[j]], target_sub[edge_t[j]],
                                               -target_degree[edge_t[j]], -edge_t[j])))

    return deactivate_edges, select_best_edge

@lru_cache(maxsize = 1)
def _helper_greedy_global():
    @njit
    def build_index(key, ptr, order):
        for k in key:
            ptr[k + 1] += 1
        for i in range(1, len(ptr)):
            ptr[i] += ptr[i - 1]
        for i, k in enumerate(key):
            order[ptr[k]] = i
            ptr[k] += 1
        for i in range(len(ptr) - 1, 0, -1):
            ptr[i] = ptr[i - 1]
        ptr[0] = 0

    @njit
    def build_conflicts(pairs, ptr, other):
        for i in range(len(pairs)):
            ptr[pairs[i, 0] + 1] += 1
            ptr[pairs[i, 1] + 1] += 1
        for i in range(1, len(ptr)):
            ptr[i] += ptr[i - 1]
        for i in range(len(pairs)):
            a, b = pairs[i, 0], pairs[i, 1]
            other[ptr[a]] = b
            other[ptr[b]] = a
            ptr[a] += 1
            ptr[b] += 1
        for i in range(len(ptr) - 1, 0, -1):
            ptr[i] = ptr[i - 1]
        ptr[0] = 0

    @njit
    def sift_heap_up(heap, position, degree, i):
        fiber = heap[i]
        while i:
            parent = (i - 1) // 2
            other = heap[parent]
            if (degree[other], other) <= (degree[fiber], fiber):
                break
            heap[i] = other
            position[other] = i
            i = parent
        heap[i] = fiber
        position[fiber] = i

    @njit
    def sift_heap_down(heap, position, degree, i, size):
        fiber = heap[i]
        while 2 * i + 1 < size:
            child = 2 * i + 1
            if child + 1 < size:
                a, b = heap[child], heap[child + 1]
                if (degree[b], b) < (degree[a], a):
                    child += 1
            other = heap[child]
            if (degree[fiber], fiber) <= (degree[other], other):
                break
            heap[i] = other
            position[other] = i
            i = child
        heap[i] = fiber
        position[fiber] = i

    @njit
    def remove_heap(heap, position, degree, fiber, size):
        i = position[fiber]
        size -= 1
        position[fiber] = -1
        if i < size:
            other = heap[size]
            heap[i] = other
            position[other] = i
            if i and (degree[other], other) < (degree[heap[(i - 1) // 2]], heap[(i - 1) // 2]):
                sift_heap_up(heap, position, degree, i)
            else:
                sift_heap_down(heap, position, degree, i, size)
        return size

    @njit(inline = 'always')
    def deactivate_edge(edge, active, edge_f, edge_t, target_degree, removed, affected, count):
        if active[edge]:
            active[edge] = False
            fiber = edge_f[edge]
            if removed[fiber] == 0:
                affected[count] = fiber
                count += 1
            removed[fiber] += 1
            target_degree[edge_t[edge]] -= 1
        return count

    @njit
    def reset_assignment(state, work):
        ass_mask, active, blocked, fiber_degree, target_degree, removed, _, _, position, _, _, _ = work
        ass_mask[:] = False
        active[:] = True
        blocked[:] = False
        fiber_degree[:] = state[2]
        target_degree[:] = state[3]
        removed[:] = 0
        position[:] = -1

    @njit
    def select_target(fiber, fiber_ptr, edge_t, active, target_pri, target_sub, target_degree):
        target = -1
        for edge in range(fiber_ptr[fiber], fiber_ptr[fiber + 1]):
            if not active[edge]:
                continue
            t = edge_t[edge]
            if target < 0 or target_pri[t] > target_pri[target]:
                target = t
            elif target_pri[t] == target_pri[target]:
                if target_sub[t] > target_sub[target]:
                    target = t
                elif target_sub[t] == target_sub[target] and (target_degree[t], t) < (target_degree[target], target):
                    target = t
        return target

    @njit
    def assign_target(prepared, state, work, fiber, target, classic, counts, bitweights, word, bit):
        edge_f, edge_t, _, graph, _, nf, _ = prepared
        fiber_ptr, tile_ptr, _, occurrence_ptr, occurrence_edges, conflict_ptr, conflict_nodes = graph
        target_ptr, target_nodes, _, _, node_tile, _, _ = state
        ass_mask, active, blocked, _, target_degree, removed, affected, _, _, _, tile_n_target, _ = work
        lo, hi = target_ptr[target], target_ptr[target + 1]
        first = tile_ptr[fiber // nf]
        while lo < hi:
            mid = (lo + hi) // 2
            if target_nodes[mid] < first:
                lo = mid + 1
            else:
                hi = mid
        node = target_nodes[lo]
        ass_mask[target] = True
        if bit:
            counts[target] += 1
            bitweights[target, word] |= bit

        count = 0
        for edge in range(fiber_ptr[fiber], fiber_ptr[fiber + 1]):
            count = deactivate_edge(edge, active, edge_f, edge_t, target_degree, removed, affected, count)
        for kind in range(2):
            if kind == 0:
                nodes, start, stop = target_nodes, target_ptr[target], target_ptr[target + 1]
            else:
                nodes, start, stop = conflict_nodes, conflict_ptr[node], conflict_ptr[node + 1]
            for j in range(start, stop):
                other = nodes[j]
                # Classic counts unassigned patrolled targets, including blocked occurrences.
                if classic and kind == 0:
                    tile_n_target[node_tile[other]] -= 1
                if blocked[other]:
                    continue
                blocked[other] = True
                for k in range(occurrence_ptr[other], occurrence_ptr[other + 1]):
                    edge = occurrence_edges[k]
                    count = deactivate_edge(edge, active, edge_f, edge_t, target_degree, removed, affected, count)
        return count

    @njit(nogil = True)
    def solve_assignment(prepared, state, target_pri, target_sub, work, counts, bitweights, word, bit):
        reset_assignment(state, work)
        edge_t, fiber_ptr = prepared[1], prepared[3][0]
        ass_mask, active, _, fiber_degree, target_degree, removed, affected, heap, position, _, _, _ = work
        size = 0
        for fiber in range(len(fiber_degree)):
            if fiber_degree[fiber]:
                heap[size] = fiber
                position[fiber] = size
                size += 1
        for i in range(size // 2 - 1, -1, -1):
            sift_heap_down(heap, position, fiber_degree, i, size)

        while size:
            fiber = heap[0]
            target = select_target(fiber, fiber_ptr, edge_t, active, target_pri, target_sub, target_degree)
            size = remove_heap(heap, position, fiber_degree, fiber, size)
            count = assign_target(prepared, state, work, fiber, target, False, counts, bitweights, word, bit)
            for j in range(count):
                f = affected[j]
                fiber_degree[f] -= removed[f]
                removed[f] = 0
                if position[f] < 0:
                    continue
                if fiber_degree[f]:
                    sift_heap_up(heap, position, fiber_degree, position[f])
                else:
                    size = remove_heap(heap, position, fiber_degree, f, size)
        return ass_mask

    @njit(nogil = True)
    def solve_classic(prepared, state, target_pri, target_sub, work, counts, bitweights, word, bit):
        reset_assignment(state, work)
        edge_t, fiber_ptr, nf = prepared[1], prepared[3][0], prepared[5]
        ass_mask, active, _, fiber_degree, target_degree, removed, affected, heap, position, tile_n_fiber, tile_n_target, remaining = work
        tile_n_fiber[:] = state[5]
        tile_n_target[:] = state[6]
        remaining[:] = True

        while True:
            tile = -1
            key = (np.inf, 0, 0)
            for i in range(len(remaining)):
                if remaining[i] and tile_n_fiber[i]:
                    candidate = (tile_n_target[i] / tile_n_fiber[i], tile_n_target[i], i)
                    if tile < 0 or candidate < key:
                        tile, key = i, candidate
            if tile < 0:
                break
            remaining[tile] = False
            size = 0
            for fiber in range(tile * nf, (tile + 1) * nf):
                if fiber_degree[fiber]:
                    heap[size] = fiber
                    position[fiber] = size
                    size += 1
            for i in range(size // 2 - 1, -1, -1):
                sift_heap_down(heap, position, fiber_degree, i, size)

            while size:
                fiber = heap[0]
                target = select_target(fiber, fiber_ptr, edge_t, active, target_pri, target_sub, target_degree)
                size = remove_heap(heap, position, fiber_degree, fiber, size)
                count = assign_target(prepared, state, work, fiber, target, True, counts, bitweights, word, bit)
                for j in range(count):
                    f = affected[j]
                    fiber_degree[f] -= removed[f]
                    removed[f] = 0
                    if not fiber_degree[f]:
                        tile_n_fiber[f // nf] -= 1
                    if position[f] < 0:
                        continue
                    if fiber_degree[f]:
                        sift_heap_up(heap, position, fiber_degree, position[f])
                    else:
                        size = remove_heap(heap, position, fiber_degree, f, size)
        return ass_mask

    return build_index, build_conflicts, solve_assignment, solve_classic

def _use_compact_greedy(target_pri, target_sub, n_targets):
    return all(x.ndim == 1 and len(x) >= n_targets and x.dtype.isnative and
               (x.dtype.kind in 'biu' or x.dtype.kind == 'f' and x.dtype.itemsize in (4, 8))
               for x in (target_pri, target_sub))

def _prepare_greedy_workspace(prepared):
    edge_f, edge_t, _, graph, n_tiles, nf, n_targets = prepared
    n_fibers = n_tiles * nf
    return (np.empty(n_targets, dtype = bool), np.empty(len(edge_t), dtype = bool),
            np.empty(len(graph[2]), dtype = bool), np.empty(n_fibers, dtype = edge_t.dtype),
            np.empty(n_targets, dtype = edge_f.dtype), np.empty(n_fibers, dtype = edge_t.dtype),
            np.empty(n_fibers, dtype = edge_f.dtype), np.empty(n_fibers, dtype = edge_f.dtype),
            np.empty(n_fibers, dtype = edge_f.dtype), np.empty(n_tiles, dtype = np.int64),
            np.empty(n_tiles, dtype = np.int64), np.empty(n_tiles, dtype = bool))

def _prepare_greedy(edge_f, edge_t, collision_groups, n_tiles, nf, n_targets):
    fiber_edges, target_edges = {}, {}
    for key, groups in ((edge_f, fiber_edges), (edge_t, target_edges)):
        order = np.arange(len(key)) if np.all(key[1:] >= key[:-1]) else np.argsort(key, kind = 'stable')
        cut = np.flatnonzero(np.diff(key[order])) + 1
        for a, b in zip(np.r_[0, cut], np.r_[cut, len(order)]):
            groups[int(key[order[a]])] = order[a:b]

    tile_fibers = [[] for _ in range(n_tiles)]
    tile_targets = [set() for _ in range(n_tiles)]
    for f, group in fiber_edges.items():
        tile = f // nf
        tile_fibers[tile].append(f)
        tile_targets[tile].update(map(int, edge_t[group]))

    tile_fibers = [np.asarray(f, dtype = int) for f in tile_fibers]
    tile_targets = [np.fromiter(sorted(t), dtype = int, count = len(t)) for t in tile_targets]

    conflict_edges = [{} for _ in range(n_tiles)]
    if isinstance(collision_groups, tuple):
        _, tile_ptr, occurrence_t, occurrence_ptr, occurrence_edges, conflict_ptr, conflict_nodes = collision_groups
        for tile in range(n_tiles):
            for node in range(tile_ptr[tile], tile_ptr[tile + 1]):
                groups = [occurrence_edges[occurrence_ptr[other]:occurrence_ptr[other + 1]]
                          for other in conflict_nodes[conflict_ptr[node]:conflict_ptr[node + 1]]]
                if groups:
                    conflict_edges[tile][int(occurrence_t[node])] = groups
    else:
        for group in collision_groups:
            tile = int(edge_f[group[0]] // nf)
            targets = edge_t[group]
            for target in np.unique(targets):
                other = group[targets != target]
                if len(other):
                    conflict_edges[tile].setdefault(int(target), []).append(other)

    fiber_degree = np.bincount(edge_f, minlength = n_tiles * nf)
    target_degree = np.bincount(edge_t, minlength = n_targets)
    return fiber_edges, target_edges, tile_fibers, tile_targets, conflict_edges, fiber_degree, target_degree

def _prepare_algorithm(prepared, algorithm):
    edge_f, edge_t, _, collision_groups, n_tiles, nf, n_targets = prepared
    if algorithm == 'greedy':
        algorithm = 'greedy-classic'
    if algorithm not in ('greedy-classic', 'greedy-global', 'milp', 'cp-sat'):
        raise ValueError(f'Unknown fiber-assignment algorithm: {algorithm}')
    if not len(edge_t):
        return ()

    if algorithm in ('greedy-classic', 'greedy-global') and isinstance(collision_groups, tuple):
        fiber_ptr, tile_ptr, occurrence_t = collision_groups[:3]
        target_ptr = np.zeros(n_targets + 1, dtype = np.int64)
        target_nodes = np.empty(len(occurrence_t), dtype = np.int64)
        _helper_greedy_global()[0](occurrence_t, target_ptr, target_nodes)
        fiber_degree = np.diff(fiber_ptr).astype(edge_t.dtype, copy = False)
        target_degree = np.bincount(edge_t, minlength = n_targets).astype(edge_f.dtype, copy = False)
        node_tile = np.repeat(np.arange(n_tiles, dtype = edge_f.dtype), np.diff(tile_ptr))
        tile_n_fiber = np.count_nonzero(fiber_degree.reshape(n_tiles, nf), axis = 1)
        return target_ptr, target_nodes, fiber_degree, target_degree, node_tile, tile_n_fiber, np.diff(tile_ptr)
    if algorithm in ('greedy-classic', 'greedy-global'):
        return _prepare_greedy(edge_f, edge_t, collision_groups, n_tiles, nf, n_targets)

    groups = _build_constraint_groups(edge_f, edge_t, collision_groups)
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

def _solve_greedy_classic(prepared, solver_state, target_pri, target_sub):
    edge_f, edge_t, _, _, _, nf, n_targets = prepared
    ass_mask = np.zeros(n_targets, dtype = bool)
    if not len(edge_t):
        return ass_mask

    deactivate_edges, select_best_edge = _helper_greedy()
    fiber_edges, target_edges, tile_fibers, tile_targets, conflict_edges, fiber_degree, target_degree = solver_state
    fiber_degree, target_degree = fiber_degree.copy(), target_degree.copy()
    active = np.ones(len(edge_t), dtype = bool)
    remaining_tiles = {i for i, fibers in enumerate(tile_fibers) if len(fibers)}

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

            best_edge = select_best_edge(fiber_edges[fiber], active, edge_t, target_pri, target_sub, target_degree)
            if best_edge < 0:
                continue

            target = int(edge_t[best_edge])
            ass_mask[target] = True
            affected = set()

            for edges in (fiber_edges[fiber], target_edges[target]):
                affected.update(map(int, deactivate_edges(edges, active, edge_f, edge_t, fiber_degree, target_degree)))
            for edges in conflict_edges[tile].get(target, ()):
                affected.update(map(int, deactivate_edges(edges, active, edge_f, edge_t, fiber_degree, target_degree)))

            for f in affected:
                if f // nf == tile and fiber_degree[f] > 0:
                    heappush(heap, (int(fiber_degree[f]), f))

    return ass_mask

def _solve_greedy_global(prepared, solver_state, target_pri, target_sub):
    edge_f, edge_t, _, _, _, nf, n_targets = prepared
    ass_mask = np.zeros(n_targets, dtype = bool)
    if not len(edge_t):
        return ass_mask

    deactivate_edges, select_best_edge = _helper_greedy()
    fiber_edges, target_edges, tile_fibers, _, conflict_edges, fiber_degree, target_degree = solver_state
    fiber_degree, target_degree = fiber_degree.copy(), target_degree.copy()
    active = np.ones(len(edge_t), dtype = bool)
    heap = [(int(fiber_degree[f]), int(f)) for fibers in tile_fibers for f in fibers if fiber_degree[f] > 0]
    heapify(heap)

    while heap:
        degree, fiber = heappop(heap)
        if degree <= 0 or degree != fiber_degree[fiber]:
            continue

        best_edge = select_best_edge(fiber_edges[fiber], active, edge_t, target_pri, target_sub, target_degree)
        if best_edge < 0:
            continue

        tile = fiber // nf
        target = int(edge_t[best_edge])
        ass_mask[target] = True
        affected = set()

        for edges in (fiber_edges[fiber], target_edges[target]):
            affected.update(map(int, deactivate_edges(edges, active, edge_f, edge_t, fiber_degree, target_degree)))
        for edges in conflict_edges[tile].get(target, ()):
            affected.update(map(int, deactivate_edges(edges, active, edge_f, edge_t, fiber_degree, target_degree)))

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
    cost = -_score_priority(target_pri, target_sub)[edge_t[used]]
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
    score = _score_priority(target_pri, target_sub)[edge_t[used]]
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

def _solve_fiberassign(prepared, target_pri, target_sub, algorithm = 'greedy-global', solver_state = None, thread = None):
    target_pri, target_sub = np.asarray(target_pri), np.asarray(target_sub)
    solver_state = _prepare_algorithm(prepared, algorithm) if solver_state is None else solver_state

    if algorithm in ('greedy', 'greedy-classic', 'greedy-global'):
        if len(prepared[1]) and isinstance(solver_state[0], np.ndarray):
            if _use_compact_greedy(target_pri, target_sub, prepared[-1]):
                workspace = _prepare_greedy_workspace(prepared)
                counts, bitweights = np.empty(0, dtype = np.uint16), np.empty((0, 2), dtype = np.uint64)
                solve = _helper_greedy_global()[2 if algorithm == 'greedy-global' else 3]
                return solve(prepared, solver_state, target_pri, target_sub, workspace, counts, bitweights, 0, np.uint64(0))
            edge_f, edge_t, _, groups, n_tiles, nf, n_targets = prepared
            solver_state = _prepare_greedy(edge_f, edge_t, groups, n_tiles, nf, n_targets)
        solve = _solve_greedy_global if algorithm == 'greedy-global' else _solve_greedy_classic
        return solve(prepared, solver_state, target_pri, target_sub)
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

    fiber_tree = KDTree(fiber_coord)
    for i, (tile, near) in enumerate(zip(tile_coord, near_all)):
        near = np.asarray(near, dtype = int)
        near = near[~pat_mask[near]]
        if not len(near):
            continue

        xy = radec2xy(tile[0], tile[1], target_coord[near, 0], target_coord[near, 1], telescope, rotation = tile_rotation[i])
        pat_mask[near] = fiber_tree.query_ball_point(xy, r = r_patrol, return_length = True) > 0

    return pat_mask

def fiberassign(tile_coord, target_coord, target_pri, target_sub = None, tile_rotation = None,
                algorithm = 'greedy-global', r_patrol = 6.0, r_exclude = 1.6, telescope = 'just'):
    tile_rotation = np.zeros(len(np.atleast_2d(tile_coord))) if tile_rotation is None else tile_rotation
    target_pri = np.asarray(target_pri)
    target_sub = np.random.default_rng(0).random(len(target_coord)) if target_sub is None else np.asarray(target_sub)

    prepared = _prepare_fiberassign(load_focalplane(telescope), tile_coord, tile_rotation, target_coord, r_patrol, r_exclude, telescope,
                                    compact = algorithm in ('greedy', 'greedy-classic', 'greedy-global') and
                                    _use_compact_greedy(target_pri, target_sub, len(target_coord)))
    solver_state = _prepare_algorithm(prepared, algorithm)
    return _solve_fiberassign(prepared, target_pri, target_sub, algorithm, solver_state), prepared[2]
