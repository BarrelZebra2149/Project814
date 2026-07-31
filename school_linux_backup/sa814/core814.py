"""
Numba core for the sa814 solver: bitmask scorer, move operators, RNG,
acceptance rules (SA / LAHC / DLAS), and the parallel-tempering anneal loop.

This module is platform-agnostic and has no file I/O beyond what's passed to
it. It replaces:

  - The exponential-DFS scorer (`_has_path_fast` in 814_cpu_*.py) with a
    row-bitmask frontier-propagation scorer that computes the exact same
    answer (walk existence, revisits allowed) using O(digits * 8) word ops
    instead of a DFS into a fixed 500-slot stack that had no overflow check.
    See verify_scorer.py for the equivalence proof against the original.

  - The DEAP genetic algorithm (elitism + NSGA-II selection + roulette
    mutation, no temperature) with real simulated annealing: a Metropolis
    acceptance rule with a calibrated cooling schedule, plus optional
    LAHC / DLAS late-acceptance rules (DLAS ported directly from
    code/dlas.hpp, replacing the old subprocess+text-file bridge to the
    Linux-only `my_dlas` binary).

  - The `multiprocessing.Pool` per-generation fitness evaluation (which lost
    `GLOBAL_MAX_SCORE` / cache state across process boundaries) with
    `numba.prange` thread parallelism over independent replicas that runs
    entirely inside one process, nogil, no pickling.

Scoring semantics (score = largest K such that 1..K are all formable; a
number is formable if its digits can be read along an 8-adjacent walk that
may revisit cells) are unchanged from the original. Only the *implementation*
of the formability test changed.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

# ---------------------------------------------------------------------------
# Grid geometry
# ---------------------------------------------------------------------------
ROWS = 8
COLS = 14
COL_MASK = (1 << COLS) - 1  # 0x3FFF

UPPER = 50000          # original _evaluate_core_numba scans n in [1, UPPER)
DIGIT_BUF_LEN = 6      # max digits of any n < UPPER is 5; keep 1 spare like the original

# 8-connected deltas, same order as the original scripts.
DELTAS = np.array(
    [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
    dtype=np.int64,
)

# Move ids, must match config814.MOVE_NAMES order.
MOVE_SET_RANDOM = 0
MOVE_COPY_NEIGHBOR = 1
MOVE_SWAP_ADJACENT = 2
MOVE_AVOID_REPEAT = 3
MOVE_REMAP_PAIR = 4
MOVE_REMAP_FULL = 5
N_MOVES = 6

ACCEPT_SA = 0
ACCEPT_LAHC = 1
ACCEPT_DLAS = 2

ACCEPT_MODE_CODE = {"sa": ACCEPT_SA, "lahc": ACCEPT_LAHC, "dlas": ACCEPT_DLAS}


# ===========================================================================
# 1. Grid <-> bitmask state
# ===========================================================================

@njit(cache=True)
def build_dmask(grid, dmask):
    """dmask[d, r] bit c set  <=>  grid[r, c] == d. dmask must be shape (10, ROWS)."""
    dmask[:, :] = 0
    for r in range(ROWS):
        for c in range(COLS):
            v = grid[r, c]
            dmask[v, r] |= np.int64(1) << c


@njit(cache=True, inline="always")
def set_cell(grid, dmask, r, c, v):
    old = grid[r, c]
    if old == v:
        return
    dmask[old, r] &= ~(np.int64(1) << c)
    dmask[v, r] |= np.int64(1) << c
    grid[r, c] = v


# ===========================================================================
# 2. Bitmask formability scorer
# ===========================================================================

@njit(cache=True, inline="always")
def fill_digits(n, buf):
    """Writes the decimal digits of n (n >= 1) into buf, most-significant first.
    Returns the digit count. Mirrors the original _get_digits_math exactly."""
    nd = 0
    m = n
    while m > 0:
        buf[nd] = m % 10
        m //= 10
        nd += 1
    i = 0
    j = nd - 1
    while i < j:
        tmp = buf[i]
        buf[i] = buf[j]
        buf[j] = tmp
        i += 1
        j -= 1
    return nd


@njit(cache=True, inline="always")
def reverse_int(n):
    rev = 0
    while n > 0:
        rev = rev * 10 + (n % 10)
        n //= 10
    return rev


@njit(cache=True, nogil=True, inline="always")
def is_formable(dmask, buf, nd):
    """Walk-existence test: can the digit sequence buf[0..nd-1] be read by moving
    to an 8-adjacent cell each step (cells may be revisited)?

    Frontier propagation, unrolled over the 8 fixed rows so nothing escapes to
    the heap: c0..c7 holds, for the current digit prefix, the set of columns
    (as bits) in each row that are a valid walk endpoint so far.
    """
    d0 = buf[0]
    c0 = dmask[d0, 0]
    c1 = dmask[d0, 1]
    c2 = dmask[d0, 2]
    c3 = dmask[d0, 3]
    c4 = dmask[d0, 4]
    c5 = dmask[d0, 5]
    c6 = dmask[d0, 6]
    c7 = dmask[d0, 7]

    if nd == 1:
        return (c0 | c1 | c2 | c3 | c4 | c5 | c6 | c7) != 0

    for k in range(1, nd):
        dk = buf[k]

        n0 = (((c0 << 1) | (c0 >> 1)) | ((c1 << 1) | c1 | (c1 >> 1))) & COL_MASK
        n1 = (((c1 << 1) | (c1 >> 1)) | ((c0 << 1) | c0 | (c0 >> 1)) | ((c2 << 1) | c2 | (c2 >> 1))) & COL_MASK
        n2 = (((c2 << 1) | (c2 >> 1)) | ((c1 << 1) | c1 | (c1 >> 1)) | ((c3 << 1) | c3 | (c3 >> 1))) & COL_MASK
        n3 = (((c3 << 1) | (c3 >> 1)) | ((c2 << 1) | c2 | (c2 >> 1)) | ((c4 << 1) | c4 | (c4 >> 1))) & COL_MASK
        n4 = (((c4 << 1) | (c4 >> 1)) | ((c3 << 1) | c3 | (c3 >> 1)) | ((c5 << 1) | c5 | (c5 >> 1))) & COL_MASK
        n5 = (((c5 << 1) | (c5 >> 1)) | ((c4 << 1) | c4 | (c4 >> 1)) | ((c6 << 1) | c6 | (c6 >> 1))) & COL_MASK
        n6 = (((c6 << 1) | (c6 >> 1)) | ((c5 << 1) | c5 | (c5 >> 1)) | ((c7 << 1) | c7 | (c7 >> 1))) & COL_MASK
        n7 = (((c7 << 1) | (c7 >> 1)) | ((c6 << 1) | c6 | (c6 >> 1))) & COL_MASK

        c0 = n0 & dmask[dk, 0]
        c1 = n1 & dmask[dk, 1]
        c2 = n2 & dmask[dk, 2]
        c3 = n3 & dmask[dk, 3]
        c4 = n4 & dmask[dk, 4]
        c5 = n5 & dmask[dk, 5]
        c6 = n6 & dmask[dk, 6]
        c7 = n7 & dmask[dk, 7]

        if (c0 | c1 | c2 | c3 | c4 | c5 | c6 | c7) == 0:
            return False

    return True


@njit(cache=True, nogil=True)
def evaluate(dmask, stamp, gen, look_window, count_lo, count_hi, want_count, digit_buf):
    """Returns (score, look, count).

    score: largest K such that every integer 1..K is formable (identical
           semantics to the original _evaluate_core_numba's current_score).
    look:  number of formable values in (score, score+look_window] -- a NEW,
           purely additive smooth signal for SA's gradient. It is not part of
           the original scoring and never substitutes for it.
    count: number of formable values in [count_lo, count_hi), matching the
           original count_first "formable" metric, only computed if
           want_count is True.

    `stamp`/`gen` implement the found-cache without a per-call memset: a slot
    is considered "found" iff stamp[n] == gen. Since gen only increases, a
    fresh gen makes all previous marks automatically stale.
    """
    n = 1
    current_score = UPPER - 1
    while n < UPPER:
        rev_n = reverse_int(n)
        if stamp[n] == gen or (n % 10 != 0 and rev_n < UPPER and stamp[rev_n] == gen):
            n += 1
            continue
        nd = fill_digits(n, digit_buf)
        if is_formable(dmask, digit_buf, nd):
            stamp[n] = gen
            if rev_n < UPPER:
                stamp[rev_n] = gen
        else:
            current_score = n - 1
            break
        n += 1

    score = current_score

    look = 0
    if score < UPPER - 1:
        look_hi = score + 1 + look_window
        if look_hi > UPPER:
            look_hi = UPPER
        m = score + 1
        while m < look_hi:
            rev_m = reverse_int(m)
            if stamp[m] == gen or (m % 10 != 0 and rev_m < UPPER and stamp[rev_m] == gen):
                look += 1
                m += 1
                continue
            nd = fill_digits(m, digit_buf)
            if is_formable(dmask, digit_buf, nd):
                stamp[m] = gen
                if rev_m < UPPER:
                    stamp[rev_m] = gen
                look += 1
            m += 1

    count = 0
    if want_count:
        trivial_hi = score if score < count_hi - 1 else count_hi - 1
        if trivial_hi >= count_lo:
            count += trivial_hi - count_lo + 1
        start = score + 1 if score + 1 > count_lo else count_lo
        p = start
        while p < count_hi:
            rev_p = reverse_int(p)
            if stamp[p] == gen or (p % 10 != 0 and rev_p < UPPER and stamp[rev_p] == gen):
                count += 1
                p += 1
                continue
            nd = fill_digits(p, digit_buf)
            if is_formable(dmask, digit_buf, nd):
                stamp[p] = gen
                if rev_p < UPPER:
                    stamp[rev_p] = gen
                count += 1
            p += 1

    return score, look, count


# ===========================================================================
# 3. Optional legacy heuristic (chain-cluster bonus - digit-variance penalty)
#    Ported from calculate_advanced_fitness in 814_cpu_score_first.py.
#    Disabled by default (w_heur = 0); kept for parity / experimentation.
# ===========================================================================

@njit(cache=True, inline="always")
def _uf_find(parent, x):
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        nxt = parent[x]
        parent[x] = root
        x = nxt
    return root


@njit(cache=True, inline="always")
def _uf_union(parent, a, b):
    ra = _uf_find(parent, a)
    rb = _uf_find(parent, b)
    if ra != rb:
        parent[ra] = rb


@njit(cache=True)
def heur_chain_variance(grid):
    max_chains = ROWS * COLS
    chain_val = np.empty(max_chains, dtype=np.int64)
    n_chains = 0
    cell_chain_a = np.full((ROWS, COLS), -1, dtype=np.int64)
    cell_chain_b = np.full((ROWS, COLS), -1, dtype=np.int64)

    visited_bs = np.zeros((ROWS, COLS), dtype=np.bool_)
    cells_r = np.empty(ROWS, dtype=np.int64)
    cells_c = np.empty(ROWS, dtype=np.int64)
    for r in range(ROWS):
        for c in range(COLS):
            if not visited_bs[r, c]:
                val = grid[r, c]
                cr, cc, length = r, c, 0
                while cr < ROWS and cc < COLS and grid[cr, cc] == val:
                    cells_r[length] = cr
                    cells_c[length] = cc
                    visited_bs[cr, cc] = True
                    length += 1
                    cr += 1
                    cc += 1
                if length >= 2:
                    chain_val[n_chains] = val
                    for i in range(length):
                        cell_chain_a[cells_r[i], cells_c[i]] = n_chains
                    n_chains += 1

    visited_fs = np.zeros((ROWS, COLS), dtype=np.bool_)
    for r in range(ROWS):
        for c in range(COLS):
            if not visited_fs[r, c]:
                val = grid[r, c]
                cr, cc, length = r, c, 0
                while cr < ROWS and cc >= 0 and grid[cr, cc] == val:
                    cells_r[length] = cr
                    cells_c[length] = cc
                    visited_fs[cr, cc] = True
                    length += 1
                    cr += 1
                    cc -= 1
                if length >= 2:
                    chain_val[n_chains] = val
                    for i in range(length):
                        cell_chain_b[cells_r[i], cells_c[i]] = n_chains
                    n_chains += 1

    parent = np.arange(n_chains, dtype=np.int64)
    for r in range(ROWS):
        for c in range(COLS):
            val_here = grid[r, c]
            ids_here0 = cell_chain_a[r, c]
            ids_here1 = cell_chain_b[r, c]
            for di in range(8):
                nr = r + DELTAS[di, 0]
                nc = c + DELTAS[di, 1]
                if 0 <= nr < ROWS and 0 <= nc < COLS:
                    if grid[nr, nc] != val_here:
                        for a in (ids_here0, ids_here1):
                            if a < 0:
                                continue
                            for b in (cell_chain_a[nr, nc], cell_chain_b[nr, nc]):
                                if b < 0:
                                    continue
                                if chain_val[a] != chain_val[b]:
                                    _uf_union(parent, a, b)

    cluster_size = np.zeros(n_chains, dtype=np.int64)
    for i in range(n_chains):
        root = _uf_find(parent, i)
        cluster_size[root] += 1

    chain_bonus = 0
    for i in range(n_chains):
        if _uf_find(parent, i) == i:
            sz = cluster_size[i]
            if sz == 1:
                chain_bonus += 10
            elif sz == 2:
                chain_bonus += 30
            elif sz == 3:
                chain_bonus += 60
            elif sz == 4:
                chain_bonus += 5

    counts = np.zeros(10, dtype=np.int64)
    for r in range(ROWS):
        for c in range(COLS):
            counts[grid[r, c]] += 1
    variance_penalty = 0
    for d in range(10):
        diff = counts[d] - 11
        variance_penalty += diff * diff

    return float(chain_bonus) - 1.5 * float(variance_penalty)


@njit(cache=True)
def count_triple_chains(grid):
    """Counts 3-cell same-digit chains reachable via an 8-directional walk
    that is free to bend at each step (start -> mid -> end, mid and end each
    an 8-neighbor of the previous cell, end != start) -- NOT restricted to a
    straight line. A straight-line-only check (as an earlier version of this
    function did, checking only 4 fixed axis directions) misses bent chains
    like (1,1)->(1,2)->(2,1), which are exactly as wasteful as a straight
    one: since a walk may revisit cells, only TWO adjacent same-digit cells
    are ever needed to form an arbitrarily long run of that digit (the walk
    just bounces between them), so a THIRD reachable in any direction adds
    nothing to formability and is pure waste of a cell that could carry a
    more useful digit for some other number.

    Each undirected triple {start, mid, end} is found from both ends (once
    as start->mid->end, once as end->mid->start), so the raw count is halved
    to report the true number of distinct triples. For a simple straight
    run of length L >= 3 with no extra branching, this reduces to exactly
    the same (L - 2) as the old straight-line-only count; branching/blob
    shapes now correctly count additional bent triples through them too.
    """
    total = 0
    for r in range(ROWS):
        for c in range(COLS):
            d = grid[r, c]
            for di in range(8):
                r1 = r + DELTAS[di, 0]
                c1 = c + DELTAS[di, 1]
                if 0 <= r1 < ROWS and 0 <= c1 < COLS and grid[r1, c1] == d:
                    for dj in range(8):
                        r2 = r1 + DELTAS[dj, 0]
                        c2 = c1 + DELTAS[dj, 1]
                        if 0 <= r2 < ROWS and 0 <= c2 < COLS and grid[r2, c2] == d:
                            if r2 != r or c2 != c:
                                total += 1
    return total // 2


# ===========================================================================
# 4. RNG: xorshift128+ (per-replica state, checkpointable)
# ===========================================================================

@njit(cache=True, inline="always")
def rng_next_u64(state):
    s1 = state[0]
    s0 = state[1]
    state[0] = s0
    s1 ^= (s1 << np.uint64(23))
    s1 ^= (s1 >> np.uint64(17))
    s1 ^= s0
    s1 ^= (s0 >> np.uint64(26))
    state[1] = s1
    return s1 + s0


@njit(cache=True, inline="always")
def rng_next_double(state):
    x = rng_next_u64(state)
    return float(x >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True, inline="always")
def rng_next_bounded(state, bound):
    return np.int64(rng_next_u64(state) % np.uint64(bound))


def splitmix64_stream(seed, count):
    """Python-side (non-jit) generator of `count` well-mixed uint64 words from a
    single integer seed, used to initialize replica RNG states deterministically
    (e.g. for checkpoint/resume or reproducible test seeds)."""
    mask = (1 << 64) - 1
    z = seed & mask
    out = np.empty(count, dtype=np.uint64)
    for i in range(count):
        z = (z + 0x9E3779B97F4A7C15) & mask
        zz = z
        zz = ((zz ^ (zz >> 30)) * 0xBF58476D1CE4E5B9) & mask
        zz = ((zz ^ (zz >> 27)) * 0x94D049BB133111EB) & mask
        zz = zz ^ (zz >> 31)
        out[i] = zz
    return out


def make_rng_state(seed):
    """Returns a fresh uint64[2] RNG state seeded from an arbitrary python int."""
    words = splitmix64_stream(int(seed) & ((1 << 64) - 1), 2)
    state = words.copy()
    if state[0] == 0 and state[1] == 0:
        state[0] = np.uint64(0x9E3779B97F4A7C15)
    return state


# ===========================================================================
# 5. Move operators (in-place, with reversible undo records)
# ===========================================================================

@njit(cache=True, inline="always")
def _pick_edge_biased(state, edge_positions, p_edge):
    if rng_next_double(state) < p_edge:
        idx = rng_next_bounded(state, edge_positions.shape[0])
        return edge_positions[idx, 0], edge_positions[idx, 1]
    return rng_next_bounded(state, ROWS), rng_next_bounded(state, COLS)


@njit(cache=True, inline="always")
def apply_set_random(state, grid, dmask, edge_positions, p_edge, nbr_val_buf, params):
    """Adopts avoid_repeat's neighbor-awareness instead of picking a
    completely blind uniform digit: targets a random cell (edge-biased like
    the other single-cell moves), gathers its valid 8-directional neighbor
    values, and sets the cell to one of those neighbor values chosen
    uniformly at random -- excluding the cell's own current value from
    consideration, so the move always actually changes something. This is
    exactly copy_neighbor generalized: instead of always using one fixed
    random direction (which can land out of bounds and no-op at an edge),
    it samples uniformly over every valid neighbor that differs from the
    current value, via reservoir sampling (no extra buffer needed beyond
    nbr_val_buf, which avoid_repeat already uses).

    Falls back to a blind uniformly-random digit != old_v (the original
    set_random behavior) only in the degenerate case where every valid
    neighbor already shares the cell's own current value.
    """
    tr, tc = _pick_edge_biased(state, edge_positions, p_edge)
    old_v = grid[tr, tc]

    n = 0
    for di in range(8):
        nr = tr + DELTAS[di, 0]
        nc = tc + DELTAS[di, 1]
        if 0 <= nr < ROWS and 0 <= nc < COLS:
            nbr_val_buf[n] = grid[nr, nc]
            n += 1

    # Reservoir sample: uniformly pick one neighbor value != old_v, without
    # needing to materialize a filtered list.
    count = 0
    chosen = 0
    for t in range(n):
        v = nbr_val_buf[t]
        if v != old_v:
            count += 1
            if rng_next_bounded(state, count) == 0:
                chosen = v

    if count > 0:
        new_v = chosen
    else:
        new_v = rng_next_bounded(state, 9)
        if new_v >= old_v:
            new_v += 1

    set_cell(grid, dmask, tr, tc, new_v)
    params[0] = tr
    params[1] = tc
    params[2] = old_v


@njit(cache=True, inline="always")
def apply_avoid_repeat(state, grid, dmask, edge_positions, p_edge,
                       nbr_val_buf, flag_buf, allowed_buf, params):
    """Targets a random cell (edge-biased like the other single-cell moves)
    and looks at its valid 8-directional neighbors (3 at a corner, 5 on an
    edge, 8 in the interior -- always at least 3).

    If those neighbor values are already all pairwise distinct (no repeated
    digit among them to break up), there's nothing useful to force, so this
    falls back to simply copying a uniformly random neighbor's value (like
    MOVE_COPY_NEIGHBOR).

    Otherwise, a random subset of size k in [1, n_neighbors] of the neighbor
    values is sampled without replacement, and the new value is forced to be
    none of them -- deliberately breaking up same-digit runs among the
    neighbors. This directly targets what feeds count_triple_chains: since a
    walk may revisit cells, only two same-digit cells are ever needed to form
    an arbitrarily long run of that digit, so a third one in a straight line
    is pure waste.

    Undo is identical to MOVE_SET_RANDOM/MOVE_COPY_NEIGHBOR (single-cell
    revert via params), so this needs no dedicated undo function.
    """
    tr, tc = _pick_edge_biased(state, edge_positions, p_edge)

    n = 0
    for di in range(8):
        nr = tr + DELTAS[di, 0]
        nc = tc + DELTAS[di, 1]
        if 0 <= nr < ROWS and 0 <= nc < COLS:
            nbr_val_buf[n] = grid[nr, nc]
            n += 1

    old_v = grid[tr, tc]

    for d in range(10):
        flag_buf[d] = 0
    all_distinct = True
    for t in range(n):
        v = nbr_val_buf[t]
        if flag_buf[v] == 1:
            all_distinct = False
        flag_buf[v] = 1

    if all_distinct:
        new_v = nbr_val_buf[rng_next_bounded(state, n)]
    else:
        k = 1 + rng_next_bounded(state, n)
        # Partial Fisher-Yates shuffle of nbr_val_buf[0..n-1]; the first k
        # slots afterward are a uniformly random size-k subset without
        # replacement.
        for t in range(n - 1, 0, -1):
            j = rng_next_bounded(state, t + 1)
            tmp = nbr_val_buf[t]
            nbr_val_buf[t] = nbr_val_buf[j]
            nbr_val_buf[j] = tmp
        for d in range(10):
            flag_buf[d] = 0
        for t in range(k):
            flag_buf[nbr_val_buf[t]] = 1
        allowed_count = 0
        for d in range(10):
            if flag_buf[d] == 0:
                allowed_buf[allowed_count] = d
                allowed_count += 1
        # allowed_count >= 10 - n >= 2 always, since n <= 8.
        new_v = allowed_buf[rng_next_bounded(state, allowed_count)]

    set_cell(grid, dmask, tr, tc, new_v)
    params[0] = tr
    params[1] = tc
    params[2] = old_v


@njit(cache=True, inline="always")
def apply_remap_pair(grid, dmask, a, b):
    if a == b:
        return
    for r in range(ROWS):
        for c in range(COLS):
            v = grid[r, c]
            if v == a:
                grid[r, c] = b
            elif v == b:
                grid[r, c] = a
    for r in range(ROWS):
        tmp = dmask[a, r]
        dmask[a, r] = dmask[b, r]
        dmask[b, r] = tmp


@njit(cache=True, inline="always")
def apply_remap_full(grid, dmask, perm, dmask_scratch):
    """Relabels every digit d -> perm[d] across the whole grid: a full 10-digit
    permutation, not just a pairwise swap. Generalizes apply_remap_pair.

    Ported from the spirit of code/permutation.py, which brute-forces all
    10! = 3,628,800 relabelings of a FIXED grid to find the best-scoring
    digit assignment -- proof that a grid's underlying cluster/chain
    structure can score very differently depending purely on which digit
    labels which cluster. This move lets SA occasionally take that same kind
    of jump stochastically (one full reshuffle) instead of only reaching it
    through many small pairwise swaps (remap_pair).

    perm must be a permutation of 0..9 (perm[d] = new label for old digit d).
    dmask_scratch is scratch space, same shape as dmask (needed because a
    general permutation has cycles longer than 2, so naive in-place row
    reassignment would clobber a row before it's read).
    """
    for d in range(10):
        for r in range(ROWS):
            dmask_scratch[d, r] = dmask[d, r]
    for d in range(10):
        pd = perm[d]
        for r in range(ROWS):
            dmask[pd, r] = dmask_scratch[d, r]
    for r in range(ROWS):
        for c in range(COLS):
            grid[r, c] = perm[grid[r, c]]


@njit(cache=True, inline="always")
def invert_perm(perm, inv_out):
    for d in range(10):
        inv_out[perm[d]] = d


@njit(cache=True)
def apply_move(state, grid, dmask, edge_positions, move_probs, p_edge,
               nbr_val_buf, flag_buf, allowed_buf, params,
               perm_buf, dmask_scratch):
    """Applies one random move in-place. Fills `params` (int64[5]) with enough
    information for undo_move to reverse it exactly, and returns the move id.

    nbr_val_buf (int64[8]), flag_buf (int64[10]), allowed_buf (int64[10]) are
    scratch space used only by MOVE_AVOID_REPEAT. perm_buf (int64[10]) and
    dmask_scratch (int64[10, ROWS]) are scratch space used only by
    MOVE_REMAP_FULL; perm_buf also doubles as the undo record for that move
    (must survive unmodified until undo_move is called, which it does since
    undo always happens before the next apply_move on this replica)."""
    r = rng_next_double(state)
    cum = 0.0
    chosen = move_probs.shape[0] - 1
    for k in range(move_probs.shape[0]):
        cum += move_probs[k]
        if r < cum:
            chosen = k
            break

    if chosen == MOVE_SET_RANDOM:
        apply_set_random(state, grid, dmask, edge_positions, p_edge, nbr_val_buf, params)
        return MOVE_SET_RANDOM

    if chosen == MOVE_COPY_NEIGHBOR:
        tr, tc = _pick_edge_biased(state, edge_positions, p_edge)
        di = rng_next_bounded(state, 8)
        nr = tr + DELTAS[di, 0]
        nc = tc + DELTAS[di, 1]
        if 0 <= nr < ROWS and 0 <= nc < COLS:
            old_v = grid[tr, tc]
            new_v = grid[nr, nc]
            set_cell(grid, dmask, tr, tc, new_v)
            params[0] = tr
            params[1] = tc
            params[2] = old_v
        else:
            params[0] = tr
            params[1] = tc
            params[2] = grid[tr, tc]
        return MOVE_COPY_NEIGHBOR

    if chosen == MOVE_SWAP_ADJACENT:
        tr, tc = _pick_edge_biased(state, edge_positions, p_edge)
        di = rng_next_bounded(state, 8)
        nr = tr + DELTAS[di, 0]
        nc = tc + DELTAS[di, 1]
        if 0 <= nr < ROWS and 0 <= nc < COLS:
            v1 = grid[tr, tc]
            v2 = grid[nr, nc]
            set_cell(grid, dmask, tr, tc, v2)
            set_cell(grid, dmask, nr, nc, v1)
            params[0] = tr
            params[1] = tc
            params[2] = nr
            params[3] = nc
        else:
            params[0] = tr
            params[1] = tc
            params[2] = tr
            params[3] = tc
        return MOVE_SWAP_ADJACENT

    if chosen == MOVE_AVOID_REPEAT:
        apply_avoid_repeat(state, grid, dmask, edge_positions, p_edge,
                           nbr_val_buf, flag_buf, allowed_buf, params)
        return MOVE_AVOID_REPEAT

    if chosen == MOVE_REMAP_PAIR:
        a = rng_next_bounded(state, 10)
        b = rng_next_bounded(state, 9)
        if b >= a:
            b += 1
        apply_remap_pair(grid, dmask, a, b)
        params[0] = a
        params[1] = b
        return MOVE_REMAP_PAIR

    # MOVE_REMAP_FULL: uniformly random permutation of all 10 digits (Fisher-Yates)
    for i in range(10):
        perm_buf[i] = i
    for i in range(9, 0, -1):
        j = rng_next_bounded(state, i + 1)
        tmp = perm_buf[i]
        perm_buf[i] = perm_buf[j]
        perm_buf[j] = tmp
    apply_remap_full(grid, dmask, perm_buf, dmask_scratch)
    return MOVE_REMAP_FULL


@njit(cache=True)
def undo_move(move_id, params, grid, dmask, perm_buf, inv_buf, dmask_scratch):
    if move_id == MOVE_SET_RANDOM or move_id == MOVE_COPY_NEIGHBOR or move_id == MOVE_AVOID_REPEAT:
        set_cell(grid, dmask, params[0], params[1], params[2])
    elif move_id == MOVE_SWAP_ADJACENT:
        r1, c1, r2, c2 = params[0], params[1], params[2], params[3]
        if r1 != r2 or c1 != c2:
            v1 = grid[r1, c1]
            v2 = grid[r2, c2]
            set_cell(grid, dmask, r1, c1, v2)
            set_cell(grid, dmask, r2, c2, v1)
    elif move_id == MOVE_REMAP_PAIR:  # self-inverse
        apply_remap_pair(grid, dmask, params[0], params[1])
    else:  # MOVE_REMAP_FULL: undo with the inverse permutation
        invert_perm(perm_buf, inv_buf)
        apply_remap_full(grid, dmask, inv_buf, dmask_scratch)


@njit(cache=True)
def apply_kick(state, grid, dmask, k):
    """Applies k unconditional single-cell random rewrites, not tracked for undo.
    Used to diversify a replica after reseeding it from the best-known grid on
    a reheat/stagnation event, so the whole ladder doesn't collapse onto one
    exact configuration."""
    for _ in range(k):
        r = rng_next_bounded(state, ROWS)
        c = rng_next_bounded(state, COLS)
        old_v = grid[r, c]
        new_v = rng_next_bounded(state, 9)
        if new_v >= old_v:
            new_v += 1
        set_cell(grid, dmask, r, c, new_v)


# ===========================================================================
# 6. Edge positions (border ring), built once at startup
# ===========================================================================

def build_edge_positions():
    edges = []
    for c in range(COLS):
        edges.append((0, c))
    for c in range(COLS):
        edges.append((ROWS - 1, c))
    for r in range(1, ROWS - 1):
        edges.append((r, 0))
    for r in range(1, ROWS - 1):
        edges.append((r, COLS - 1))
    return np.array(edges, dtype=np.int64)


# ===========================================================================
# 7. Energy
# ===========================================================================

@njit(cache=True, inline="always")
def energy_of(score, look, count, heur, triples, w_score, w_look, w_count, w_heur, w_triple):
    return -(w_score * score + w_look * look + w_count * count + w_heur * heur
             - w_triple * triples)


# ===========================================================================
# 8. Single-replica annealing segment
# ===========================================================================

@njit(cache=True, nogil=True)
def _anneal_one(i, grids, dmasks, stamps, gens, energies, scores_arr, looks_arr, counts_arr,
                 temps, rng_states, digit_bufs,
                 hist, lahc_pos,
                 edge_positions, move_probs, p_edge,
                 w_score, w_look, w_count, w_heur, w_triple,
                 want_count, look_window, count_lo, count_hi,
                 accept_mode, iters, accept_counter, move_counter):
    grid = grids[i]
    dmask = dmasks[i]
    stamp = stamps[i]
    rng_state = rng_states[i]
    digit_buf = digit_bufs[i]
    my_hist = hist[i]
    lahc_len = my_hist.shape[0]

    T = temps[i]
    curE = energies[i]
    gen = gens[i]
    pos = lahc_pos[i]

    params = np.zeros(5, dtype=np.int64)
    perm_buf = np.zeros(10, dtype=np.int64)
    inv_buf = np.zeros(10, dtype=np.int64)
    dmask_scratch = np.zeros((10, ROWS), dtype=np.int64)
    nbr_val_buf = np.zeros(8, dtype=np.int64)
    flag_buf = np.zeros(10, dtype=np.int64)
    allowed_buf = np.zeros(10, dtype=np.int64)

    for _ in range(iters):
        gen += 1
        move_id = apply_move(rng_state, grid, dmask, edge_positions, move_probs, p_edge,
                              nbr_val_buf, flag_buf, allowed_buf, params,
                              perm_buf, dmask_scratch)

        score, look, count = evaluate(dmask, stamp, gen, look_window, count_lo, count_hi,
                                       want_count, digit_buf)
        heur = 0.0
        if w_heur != 0.0:
            heur = heur_chain_variance(grid)
        triples = 0.0
        if w_triple != 0.0:
            triples = float(count_triple_chains(grid))
        newE = energy_of(score, look, count, heur, triples, w_score, w_look, w_count, w_heur, w_triple)

        accept = False
        if accept_mode == ACCEPT_SA:
            dE = newE - curE
            if dE <= 0.0:
                accept = True
            elif T > 0.0:
                accept = rng_next_double(rng_state) < math.exp(-dE / T)
        elif accept_mode == ACCEPT_LAHC:
            v = pos % lahc_len
            if newE <= my_hist[v] or newE <= curE:
                accept = True
            candidateE = newE if accept else curE
            if candidateE < my_hist[v]:
                my_hist[v] = candidateE
            pos += 1
        else:  # ACCEPT_DLAS, ported from dlas.hpp
            v = pos % lahc_len
            hmax = my_hist[0]
            for h in range(1, lahc_len):
                if my_hist[h] > hmax:
                    hmax = my_hist[h]
            prvF = curE
            if newE == curE or newE < hmax:
                accept = True
            postF = newE if accept else curE
            fit = my_hist[v]
            if postF > fit or (postF < fit and postF < prvF):
                my_hist[v] = postF
            pos += 1

        if accept:
            curE = newE
            energies[i] = newE
            scores_arr[i] = score
            looks_arr[i] = look
            counts_arr[i] = count
            accept_counter[i] += 1
        else:
            undo_move(move_id, params, grid, dmask, perm_buf, inv_buf, dmask_scratch)

        move_counter[i, move_id] += 1

    gens[i] = gen
    lahc_pos[i] = pos


@njit(cache=True)
def _attempt_swaps(grids, dmasks, energies, scores_arr, looks_arr, counts_arr,
                    hist, lahc_pos, temps, swap_rng_state,
                    swap_accept_counter, swap_attempt_counter):
    R = grids.shape[0]
    for i in range(R - 1):
        j = i + 1
        swap_attempt_counter[i] += 1
        beta_i = 1.0 / temps[i] if temps[i] > 0.0 else 1.0e18
        beta_j = 1.0 / temps[j] if temps[j] > 0.0 else 1.0e18
        delta = (beta_i - beta_j) * (energies[i] - energies[j])
        p = 1.0 if delta >= 0.0 else math.exp(delta)
        if rng_next_double(swap_rng_state) < p:
            for r in range(ROWS):
                for c in range(COLS):
                    tg = grids[i, r, c]
                    grids[i, r, c] = grids[j, r, c]
                    grids[j, r, c] = tg
            for d in range(10):
                for r in range(ROWS):
                    tm = dmasks[i, d, r]
                    dmasks[i, d, r] = dmasks[j, d, r]
                    dmasks[j, d, r] = tm
            te = energies[i]; energies[i] = energies[j]; energies[j] = te
            ts = scores_arr[i]; scores_arr[i] = scores_arr[j]; scores_arr[j] = ts
            tl = looks_arr[i]; looks_arr[i] = looks_arr[j]; looks_arr[j] = tl
            tc_ = counts_arr[i]; counts_arr[i] = counts_arr[j]; counts_arr[j] = tc_
            for h in range(hist.shape[1]):
                th = hist[i, h]; hist[i, h] = hist[j, h]; hist[j, h] = th
            tp = lahc_pos[i]; lahc_pos[i] = lahc_pos[j]; lahc_pos[j] = tp
            swap_accept_counter[i] += 1


@njit(cache=True, parallel=True, nogil=True)
def run_block(grids, dmasks, stamps, gens, energies, scores_arr, looks_arr, counts_arr,
              temps, rng_states, digit_bufs,
              hist, lahc_pos, swap_rng_state,
              edge_positions, move_probs, p_edge,
              w_score, w_look, w_count, w_heur, w_triple,
              want_count, look_window, count_lo, count_hi,
              accept_mode, iters_per_segment, n_segments, do_swaps,
              accept_counter, move_counter, swap_accept_counter, swap_attempt_counter):
    """Runs n_segments * iters_per_segment SA iterations per replica, attempting
    a replica-exchange swap sweep between segments (if do_swaps)."""
    R = grids.shape[0]
    for _seg in range(n_segments):
        for i in prange(R):
            _anneal_one(i, grids, dmasks, stamps, gens, energies, scores_arr, looks_arr, counts_arr,
                        temps, rng_states, digit_bufs,
                        hist, lahc_pos,
                        edge_positions, move_probs, p_edge,
                        w_score, w_look, w_count, w_heur, w_triple,
                        want_count, look_window, count_lo, count_hi,
                        accept_mode, iters_per_segment, accept_counter, move_counter)
        if do_swaps and R > 1:
            _attempt_swaps(grids, dmasks, energies, scores_arr, looks_arr, counts_arr,
                           hist, lahc_pos, temps, swap_rng_state,
                           swap_accept_counter, swap_attempt_counter)


# ===========================================================================
# 9. Temperature calibration (python-driven, uses the same njit building blocks)
# ===========================================================================

@njit(cache=True)
def _calibrate_samples(grid, dmask, stamp, gen0, rng_state, edge_positions, move_probs, p_edge,
                        w_score, w_look, w_count, w_heur, w_triple,
                        want_count, look_window, count_lo, count_hi,
                        n_samples, digit_buf, samples_out):
    gen = gen0
    score, look, count = evaluate(dmask, stamp, gen, look_window, count_lo, count_hi, want_count, digit_buf)
    heur = heur_chain_variance(grid) if w_heur != 0.0 else 0.0
    triples = float(count_triple_chains(grid)) if w_triple != 0.0 else 0.0
    curE = energy_of(score, look, count, heur, triples, w_score, w_look, w_count, w_heur, w_triple)
    params = np.zeros(5, dtype=np.int64)
    perm_buf = np.zeros(10, dtype=np.int64)
    inv_buf = np.zeros(10, dtype=np.int64)
    dmask_scratch = np.zeros((10, ROWS), dtype=np.int64)
    nbr_val_buf = np.zeros(8, dtype=np.int64)
    flag_buf = np.zeros(10, dtype=np.int64)
    allowed_buf = np.zeros(10, dtype=np.int64)
    for k in range(n_samples):
        gen += 1
        move_id = apply_move(rng_state, grid, dmask, edge_positions, move_probs, p_edge,
                              nbr_val_buf, flag_buf, allowed_buf, params,
                              perm_buf, dmask_scratch)
        score, look, count = evaluate(dmask, stamp, gen, look_window, count_lo, count_hi,
                                       want_count, digit_buf)
        heur = heur_chain_variance(grid) if w_heur != 0.0 else 0.0
        triples = float(count_triple_chains(grid)) if w_triple != 0.0 else 0.0
        newE = energy_of(score, look, count, heur, triples, w_score, w_look, w_count, w_heur, w_triple)
        samples_out[k] = newE - curE
        undo_move(move_id, params, grid, dmask, perm_buf, inv_buf, dmask_scratch)
    return gen


def calibrate_temperature(grid, dmask, stamp, gen0, rng_state, edge_positions, move_probs, p_edge,
                           cfg, n_samples=2000, p_hot=0.5, p_cold=0.01):
    """Estimates T0 (~p_hot acceptance rate) and T_end (~p_cold acceptance rate)
    from the empirical distribution of positive dE over random moves from the
    given starting grid. Returns (T0, T_end, measured_up_moves_fraction).

    Uses the MEDIAN of positive dE, not the mean. Near a high-scoring grid the
    landscape is fragile: a small minority of moves hit a "load-bearing" cell
    shared by many numbers' paths and collapse the score by thousands, while
    most moves only cost a handful of points. Those catastrophic outliers blow
    up the mean (T0 computed from a seed scoring ~7666/8142 came out at ~8000,
    an accept-almost-anything temperature that reduced the annealer to a
    near-pure random walk and destroyed the seed within one block). The
    median is robust to that heavy right tail and calibrates T0 to the
    *typical* local move instead, so catastrophic collapses stay exponentially
    suppressed (exp(-dE/T) ~ 0) even while ordinary moves are freely explored.
    """
    digit_buf = np.zeros(DIGIT_BUF_LEN, dtype=np.int64)
    samples = np.zeros(n_samples, dtype=np.float64)

    _calibrate_samples(grid, dmask, stamp, gen0, rng_state, edge_positions, move_probs, p_edge,
                        cfg.w_score, cfg.w_look, cfg.w_count, cfg.w_heur, cfg.w_triple,
                        cfg.want_count, cfg.look_window, cfg.count_lo, cfg.count_hi,
                        n_samples, digit_buf, samples)

    up = samples[samples > 0.0]
    if up.size == 0:
        # No worsening move was ever sampled (e.g. a trivial/empty grid). Fall back
        # to a small constant schedule; the run will behave close to greedy.
        return 1e-6, 1e-9, 0.0

    median_up = float(np.median(up))
    q05 = float(np.quantile(up, 0.05))
    t0 = median_up / math.log(1.0 / p_hot) if p_hot < 1.0 else median_up
    t_end = q05 / math.log(1.0 / p_cold) if p_cold < 1.0 else q05
    t_end = max(t_end, 1e-9)
    t0 = max(t0, t_end * 1.01)
    return t0, t_end, up.size / float(n_samples)
