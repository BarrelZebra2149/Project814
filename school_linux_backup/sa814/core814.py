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
MOVE_COPY_NEIGHBOR = 0
MOVE_SWAP_ADJACENT = 1
MOVE_AVOID_REPEAT = 2
MOVE_REMAP = 3
N_MOVES = 4

MAX_CLUSTER = 9   # target cell + up to 8 neighbors, for swap_adjacent
MAX_DERANGE = 10  # largest derangement needed (remap: up to 10 digits)

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


@njit(cache=True, inline="always")
def weighted_index_choice(state, weights, n):
    """Draws an index in [0, n-1] from weights[0..n-1], renormalized over
    just that range (any entries at index >= n are ignored). Falls back to
    a uniform draw if the weights sum to ~0 (shouldn't happen once
    initialized to all-ones, but defensive). This is the single sampling
    path used for every k-selection in the solver: when all weights are
    equal (the default), it's exactly a uniform draw; --adaptive-k changes
    behavior purely by changing what's in `weights` between blocks, with no
    separate code path needed.
    """
    total = 0.0
    for i in range(n):
        total += weights[i]
    if total <= 1e-12:
        return rng_next_bounded(state, n)
    r = rng_next_double(state) * total
    cum = 0.0
    for i in range(n - 1):
        cum += weights[i]
        if r < cum:
            return i
    return n - 1


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
def apply_copy_neighbor(state, grid, dmask, edge_positions, p_edge, nbr_val_buf, params):
    """Targets a single random cell (edge-biased) and sets ONLY that cell to
    one value chosen from a random pool of size k in [1, n] of its valid
    8-directional neighbor values (n = 3 at a corner, 5 on an edge, 8 in the
    interior). Only the target cell changes -- k controls how many
    neighbors are considered as candidates before picking one, NOT how many
    cells get modified (unlike apply_swap_cluster / a homogenize-the-whole-
    cluster design, which would actively grow same-digit blobs and directly
    fight w_triple's whole purpose).

    Note the resulting value's distribution is uniform over all n
    neighbors regardless of k (for any fixed neighbor, P(chosen) =
    P(neighbor in the k-pool) * P(picked | in pool, size k) = (k/n)*(1/k) =
    1/n) -- k doesn't change what this move DOES today. It exists as an
    explicit, trackable parameter so a future acceptance-rate-based learning
    system (see README) has something to learn a preference over, even if
    the answer for this particular move turns out to be "k doesn't matter".

    This generalizes the original "copy one fixed random direction" idea
    (which could land out of bounds and no-op near an edge) into a sample
    over a random subset of neighbors. An earlier, separate "set_random"
    move did a similar thing under a different name and was merged in here.
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

    k = 1 + rng_next_bounded(state, n)  # uniform in [1, n]

    # Partial Fisher-Yates over nbr_val_buf[0..n-1]; the first k after
    # shuffling are a uniformly random size-k pool of neighbor values.
    for i in range(n - 1, 0, -1):
        j = rng_next_bounded(state, i + 1)
        tmp = nbr_val_buf[i]
        nbr_val_buf[i] = nbr_val_buf[j]
        nbr_val_buf[j] = tmp

    new_v = nbr_val_buf[rng_next_bounded(state, k)]

    set_cell(grid, dmask, tr, tc, new_v)
    params[0] = tr
    params[1] = tc
    params[2] = old_v


@njit(cache=True, inline="always")
def apply_avoid_repeat(state, grid, dmask, edge_positions, p_edge,
                       nbr_val_buf, flag_buf, allowed_buf, k_weights, params):
    """Targets a random cell (edge-biased like the other single-cell moves)
    and looks at its valid 8-directional neighbors (3 at a corner, 5 on an
    edge, 8 in the interior -- always at least 3).

    If those neighbor values are already all pairwise distinct (no repeated
    digit among them to break up), there's nothing useful to force, so this
    falls back to simply copying a uniformly random neighbor's value (like
    MOVE_COPY_NEIGHBOR). No k is drawn in this fallback case; params[3] is
    set to -1 to signal "not applicable" to the --adaptive-k tracker.

    Otherwise, a subset of size k in [1, n_neighbors] of the neighbor values
    is sampled without replacement (k drawn via weighted_index_choice from
    k_weights[postype], postype = 0 for an interior cell (n=8), 1 for a
    border cell (n=3 or 5); all-ones weights, the default, make this a
    uniform draw), and the new value is forced to be none of them --
    deliberately breaking up same-digit runs among the neighbors. This
    directly targets what feeds count_triple_chains: since a walk may
    revisit cells, only two same-digit cells are ever needed to form an
    arbitrarily long run of that digit, so a third one in a straight line
    is pure waste. params[3]/params[4] are set to the k index (k-1) and
    postype used, for the caller to record into --adaptive-k's counters.

    Undo is identical to MOVE_COPY_NEIGHBOR (single-cell revert via params),
    so this needs no dedicated undo function.
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
        params[3] = -1
        params[4] = -1
    else:
        postype = 0 if n == 8 else 1
        k_idx = weighted_index_choice(state, k_weights[postype], n)
        k = k_idx + 1
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
        params[3] = k_idx
        params[4] = postype

    set_cell(grid, dmask, tr, tc, new_v)
    params[0] = tr
    params[1] = tc
    params[2] = old_v


@njit(cache=True, inline="always")
def random_derangement_indices(state, m, perm_out):
    """Fills perm_out[0..m-1] with a uniformly random derangement of the
    identity permutation [0, 1, ..., m-1] (perm_out[i] != i for every i),
    via Fisher-Yates + rejection sampling. Requires m >= 2.

    A uniform random permutation of m elements is a derangement with
    probability exactly 1/2 at m=2, approaching 1/e (~36.8%) as m grows, so
    this finds one within ~2-3 tries on average. Bounded at 50 attempts as a
    defensive fallback for pathological RNG states; falls back to a plain
    rotation by 1 (always a valid derangement for m >= 2) if that bound is
    ever hit, which in practice never happens.
    """
    for _attempt in range(50):
        for i in range(m):
            perm_out[i] = i
        for i in range(m - 1, 0, -1):
            j = rng_next_bounded(state, i + 1)
            tmp = perm_out[i]
            perm_out[i] = perm_out[j]
            perm_out[j] = tmp
        ok = True
        for i in range(m):
            if perm_out[i] == i:
                ok = False
                break
        if ok:
            return
    for i in range(m):
        perm_out[i] = (i + 1) % m


@njit(cache=True, inline="always")
def random_value_derangement(state, orig_vals, m, perm_out, sort_idx_buf):
    """Fills perm_out[0..m-1] with a permutation of indices [0..m-1] such
    that orig_vals[perm_out[i]] != orig_vals[i] for EVERY i -- i.e. every
    position's VALUE actually changes, not merely its source index.

    This is a strictly stronger guarantee than random_derangement_indices:
    if two positions happen to already hold the same value (e.g. two
    neighbor cells that are both digit 5), a plain index derangement could
    still map one onto the other's slot and leave the visible value
    unchanged there, even though the abstract index "moved". Every cell
    apply_swap_cluster touches must end up with a genuinely different
    digit.

    A valid such permutation exists if and only if no single value occupies
    more than half of the m positions (max_freq <= m // 2) -- otherwise, by
    pigeonhole, there simply aren't enough "other value" slots to send every
    occurrence of the majority value to, no matter the arrangement.

    First tries rejection-sampling a plain index derangement and checking
    the actual values (bounded at 30 attempts; almost always succeeds
    immediately when feasible). If that's exhausted -- which empirically
    happens close to the max_freq == m // 2 boundary, where a random
    permutation satisfying the value condition becomes rare even though one
    exists -- falls back to a constructive method that is PROVEN correct
    whenever any valid arrangement exists at all (verified against 143,975
    randomized feasible cases, 0 failures): sort positions by value so
    same-valued positions land in one contiguous block, then rotate that
    sorted order by ceil(m/2). Same-value blocks can be at most m // 2
    long when feasible, so a rotation by ceil(m/2) always pushes every
    position past its own block into a differently-valued one.

    Only fails to change anything if a single value occupies MORE than half
    the cluster (mathematically impossible to derange then); in that
    genuinely infeasible case perm_out ends up as whatever the constructive
    step produces -- a harmless no-op for the caller, since no rearrangement
    could have changed those digits anyway. sort_idx_buf (int64[>=m]) is
    scratch space for the fallback only.
    """
    for _attempt in range(30):
        random_derangement_indices(state, m, perm_out)
        ok = True
        for i in range(m):
            if orig_vals[perm_out[i]] == orig_vals[i]:
                ok = False
                break
        if ok:
            return

    for i in range(m):
        sort_idx_buf[i] = i
    for i in range(1, m):
        key_idx = sort_idx_buf[i]
        key_val = orig_vals[key_idx]
        j = i - 1
        while j >= 0 and orig_vals[sort_idx_buf[j]] > key_val:
            sort_idx_buf[j + 1] = sort_idx_buf[j]
            j -= 1
        sort_idx_buf[j + 1] = key_idx

    shift = (m + 1) // 2
    for i in range(m):
        perm_out[sort_idx_buf[i]] = sort_idx_buf[(i + shift) % m]


@njit(cache=True, inline="always")
def apply_remap_full(grid, dmask, perm, dmask_scratch):
    """Relabels every digit d -> perm[d] across the whole grid.

    perm must be a permutation of 0..9 (perm[d] = new label for old digit d;
    a digit not covered by apply_remap's chosen subset simply maps to
    itself). dmask_scratch is scratch space, same shape as dmask (needed
    because a general permutation has cycles longer than 2, so naive
    in-place row reassignment would clobber a row before it's read).
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
def apply_remap(state, grid, dmask, perm_buf, subset_buf, derange_buf, dmask_scratch,
                 k_weights, params):
    """Generalizes the old remap_pair (swap exactly 2 digits) and remap_full
    (relabel all 10 digits via a random permutation) into one parameterized
    move: pick k in [2, 10] (drawn via weighted_index_choice from
    k_weights, an array of 9 entries for k=2..10; all-ones weights, the
    default, make this a uniform draw), choose k distinct digits, and
    derange (permute with no fixed points, so every chosen digit actually
    changes) just those k -- the other (10 - k) digits are left untouched.

    k=2 always produces a genuine digit swap (the only derangement of 2
    elements is the transposition) -- exactly the old remap_pair's
    behavior. k=10 deranges all 10 digits at once, close to the old
    remap_full, except now EVERY digit is guaranteed to actually change
    (remap_full's plain uniform permutation could leave a digit fixed by
    chance, ~1 on average). Intermediate k is a new, previously unreachable
    "partial reshuffle" of just some digits.

    Ported from the spirit of code/permutation.py, which brute-forces all
    10! = 3,628,800 relabelings of a FIXED grid to find the best-scoring
    digit assignment -- proof that a grid's underlying cluster/chain
    structure can score very differently depending purely on which digit
    labels which cluster.

    perm_buf (int64[10]) receives the full digit-relabeling permutation
    applied (identity outside the chosen subset) -- this also serves as the
    undo record, so it must survive unmodified until undo_move is called.
    subset_buf (int64[10]) and derange_buf (int64[>=10]) are pure scratch.
    dmask_scratch (int64[10, ROWS]) is apply_remap_full's usual scratch.
    params[3] is set to the k index (k-2), for --adaptive-k's counters.
    """
    k_idx = weighted_index_choice(state, k_weights, 9)
    k = k_idx + 2  # k in [2, 10]

    for d in range(10):
        subset_buf[d] = d
    for i in range(9, 0, -1):
        j = rng_next_bounded(state, i + 1)
        tmp = subset_buf[i]
        subset_buf[i] = subset_buf[j]
        subset_buf[j] = tmp
    # subset_buf[0..k-1] is now a uniformly random k-subset of digits 0..9.

    random_derangement_indices(state, k, derange_buf)

    for d in range(10):
        perm_buf[d] = d
    for i in range(k):
        perm_buf[subset_buf[i]] = subset_buf[derange_buf[i]]

    apply_remap_full(grid, dmask, perm_buf, dmask_scratch)
    params[3] = k_idx


@njit(cache=True, inline="always")
def invert_perm(perm, inv_out):
    for d in range(10):
        inv_out[perm[d]] = d


@njit(cache=True, inline="always")
def apply_swap_cluster(state, grid, dmask, edge_positions, p_edge,
                        cell_r_buf, cell_c_buf, orig_val_buf, derange_buf, sort_idx_buf,
                        k_weights, params):
    """Generalizes the old swap_adjacent (exchange values between exactly 2
    cells) into a parameterized move: target a random cell (edge-biased),
    pick a subset of size k in [1, n] of its valid 8-directional neighbors
    (n = 3 at a corner, 5 on an edge, 8 in the interior; k drawn via
    weighted_index_choice from k_weights[postype], postype = 0 for interior
    (n=8), 1 for border (n=3 or 5) -- all-ones weights, the default, make
    this a uniform draw), and derange the VALUES currently held by the
    target + those k neighbors (m = k+1 cells total) among themselves. The
    multiset of values in that cluster is preserved -- only which cell
    holds which value changes.

    Uses random_value_derangement, not the plain index-based one: two
    cells in the cluster can already hold the same digit, so a pure index
    derangement could still leave a cell's VISIBLE value unchanged even
    though its source index moved. Every touched cell is guaranteed to end
    up with a genuinely different digit than it started with, UNLESS a
    single digit occupies more than half of the cluster's cells -- then no
    rearrangement could ever change anything for the excess cells (proven
    impossible by pigeonhole; see random_value_derangement), and the move
    is a harmless no-op for those.

    k=1 (target + 1 neighbor, a cluster of exactly 2 cells) has only one
    possible derangement -- the pairwise exchange -- reproducing the
    original swap_adjacent exactly.

    cell_r_buf/cell_c_buf/orig_val_buf (int64[MAX_CLUSTER] each) record the
    cluster's positions and PRE-move values -- this directly doubles as the
    undo record (undo just writes orig_val_buf back to its recorded
    positions, no permutation inversion needed). derange_buf and
    sort_idx_buf (int64[>=MAX_CLUSTER] each) are pure scratch for
    random_value_derangement. params[0] is set to the cluster size m so
    undo_move knows how many entries to restore; params[3]/params[4] are
    set to the k index (k-1) and postype used, for --adaptive-k's counters.
    """
    tr, tc = _pick_edge_biased(state, edge_positions, p_edge)
    cell_r_buf[0] = tr
    cell_c_buf[0] = tc
    n = 0
    for di in range(8):
        nr = tr + DELTAS[di, 0]
        nc = tc + DELTAS[di, 1]
        if 0 <= nr < ROWS and 0 <= nc < COLS:
            cell_r_buf[1 + n] = nr
            cell_c_buf[1 + n] = nc
            n += 1

    postype = 0 if n == 8 else 1
    k_idx = weighted_index_choice(state, k_weights[postype], n)
    k = k_idx + 1

    # Partial Fisher-Yates over the n neighbor slots [1..n]; the first k
    # after shuffling are a uniformly random size-k subset without
    # replacement.
    for i in range(n - 1, 0, -1):
        j = rng_next_bounded(state, i + 1)
        tmp_r = cell_r_buf[1 + i]
        tmp_c = cell_c_buf[1 + i]
        cell_r_buf[1 + i] = cell_r_buf[1 + j]
        cell_c_buf[1 + i] = cell_c_buf[1 + j]
        cell_r_buf[1 + j] = tmp_r
        cell_c_buf[1 + j] = tmp_c

    m = k + 1
    for i in range(m):
        orig_val_buf[i] = grid[cell_r_buf[i], cell_c_buf[i]]

    random_value_derangement(state, orig_val_buf, m, derange_buf, sort_idx_buf)

    for i in range(m):
        set_cell(grid, dmask, cell_r_buf[i], cell_c_buf[i], orig_val_buf[derange_buf[i]])

    params[0] = m
    params[3] = k_idx
    params[4] = postype


@njit(cache=True)
def apply_move(state, grid, dmask, edge_positions, move_probs, p_edge,
               nbr_val_buf, flag_buf, allowed_buf, params,
               perm_buf, dmask_scratch,
               cell_r_buf, cell_c_buf, orig_val_buf,
               k_weights_avoid, k_weights_swap, k_weights_remap):
    """Applies one random move in-place. Fills `params` (int64[5]) with enough
    information for undo_move to reverse it exactly, and returns the move id.

    nbr_val_buf (int64[8]) is scratch space used by both MOVE_COPY_NEIGHBOR
    and MOVE_AVOID_REPEAT. flag_buf (int64[10]) and allowed_buf (int64[10])
    are used by MOVE_AVOID_REPEAT for their documented purpose, and reused
    as generic scratch (subset selection / derangement output respectively)
    by MOVE_REMAP, and allowed_buf/flag_buf are reused again as derangement
    scratch (derange_buf/sort_idx_buf) by MOVE_SWAP_ADJACENT -- safe since
    only one move executes per call. perm_buf (int64[10]) and dmask_scratch
    (int64[10, ROWS]) are scratch space used only by MOVE_REMAP; perm_buf
    also doubles as the undo record for that move. cell_r_buf/cell_c_buf/
    orig_val_buf (int64[MAX_CLUSTER] each) are used only by
    MOVE_SWAP_ADJACENT, doubling as its undo record too (all must survive
    unmodified until undo_move is called, which it does since undo always
    happens before the next apply_move on this replica).

    k_weights_avoid/k_weights_swap (float64[2, 8]) and k_weights_remap
    (float64[9]) drive each move's internal k-selection (see
    weighted_index_choice); all-ones (the default) makes every k draw
    uniform, matching the original behavior exactly. --adaptive-k changes
    these arrays between blocks based on observed accept rates; this
    function itself is unaware of whether that's happening. params[3] and
    params[4] are set by MOVE_AVOID_REPEAT/MOVE_SWAP_ADJACENT/MOVE_REMAP to
    the k index and (where applicable) position-type used, for the caller
    to feed into --adaptive-k's counters; MOVE_COPY_NEIGHBOR isn't tracked
    (its k provably doesn't affect the outcome, so there's nothing to
    learn) and resets them to -1.
    """
    r = rng_next_double(state)
    cum = 0.0
    chosen = move_probs.shape[0] - 1
    for k in range(move_probs.shape[0]):
        cum += move_probs[k]
        if r < cum:
            chosen = k
            break

    if chosen == MOVE_COPY_NEIGHBOR:
        apply_copy_neighbor(state, grid, dmask, edge_positions, p_edge, nbr_val_buf, params)
        params[3] = -1
        params[4] = -1
        return MOVE_COPY_NEIGHBOR

    if chosen == MOVE_SWAP_ADJACENT:
        apply_swap_cluster(state, grid, dmask, edge_positions, p_edge,
                            cell_r_buf, cell_c_buf, orig_val_buf, allowed_buf, flag_buf,
                            k_weights_swap, params)
        return MOVE_SWAP_ADJACENT

    if chosen == MOVE_AVOID_REPEAT:
        apply_avoid_repeat(state, grid, dmask, edge_positions, p_edge,
                           nbr_val_buf, flag_buf, allowed_buf, k_weights_avoid, params)
        return MOVE_AVOID_REPEAT

    # MOVE_REMAP
    apply_remap(state, grid, dmask, perm_buf, flag_buf, allowed_buf, dmask_scratch,
                k_weights_remap, params)
    return MOVE_REMAP


@njit(cache=True)
def undo_move(move_id, params, grid, dmask, perm_buf, inv_buf, dmask_scratch,
              cell_r_buf, cell_c_buf, orig_val_buf):
    if move_id == MOVE_COPY_NEIGHBOR or move_id == MOVE_AVOID_REPEAT:
        set_cell(grid, dmask, params[0], params[1], params[2])
    elif move_id == MOVE_SWAP_ADJACENT:
        m = params[0]
        for i in range(m):
            set_cell(grid, dmask, cell_r_buf[i], cell_c_buf[i], orig_val_buf[i])
    else:  # MOVE_REMAP: undo with the inverse permutation
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
                 accept_mode, iters, max_worsen, accept_counter, move_counter, move_accept_counter,
                 k_weights_avoid, k_weights_swap, k_weights_remap,
                 k_attempt_avoid, k_accept_avoid, k_attempt_swap, k_accept_swap,
                 k_attempt_remap, k_accept_remap):
    grid = grids[i]
    dmask = dmasks[i]
    stamp = stamps[i]
    rng_state = rng_states[i]
    digit_buf = digit_bufs[i]
    my_hist = hist[i]
    lahc_len = my_hist.shape[0]

    my_k_attempt_avoid = k_attempt_avoid[i]
    my_k_accept_avoid = k_accept_avoid[i]
    my_k_attempt_swap = k_attempt_swap[i]
    my_k_accept_swap = k_accept_swap[i]
    my_k_attempt_remap = k_attempt_remap[i]
    my_k_accept_remap = k_accept_remap[i]

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
    cell_r_buf = np.zeros(MAX_CLUSTER, dtype=np.int64)
    cell_c_buf = np.zeros(MAX_CLUSTER, dtype=np.int64)
    orig_val_buf = np.zeros(MAX_CLUSTER, dtype=np.int64)

    for _ in range(iters):
        gen += 1
        move_id = apply_move(rng_state, grid, dmask, edge_positions, move_probs, p_edge,
                              nbr_val_buf, flag_buf, allowed_buf, params,
                              perm_buf, dmask_scratch,
                              cell_r_buf, cell_c_buf, orig_val_buf,
                              k_weights_avoid, k_weights_swap, k_weights_remap)

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
            bar = my_hist[v]
            # Hard ceiling: a move that breaks the grid by thousands of
            # score points must never be accepted just because history has
            # drifted that high too. Without this, one such accept is
            # irrecoverable on score's cliff landscape -- see max_worsening
            # in config814.py for the measured evidence.
            if max_worsen > 0.0 and bar > curE + max_worsen:
                bar = curE + max_worsen
            if newE <= bar or newE <= curE:
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
            if max_worsen > 0.0 and hmax > curE + max_worsen:
                hmax = curE + max_worsen
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
            undo_move(move_id, params, grid, dmask, perm_buf, inv_buf, dmask_scratch,
                      cell_r_buf, cell_c_buf, orig_val_buf)

        move_counter[i, move_id] += 1
        if accept:
            move_accept_counter[i, move_id] += 1

        # --adaptive-k bookkeeping: only MOVE_AVOID_REPEAT/MOVE_SWAP_ADJACENT/
        # MOVE_REMAP report a real (k_idx, postype) via params[3]/params[4]
        # (MOVE_COPY_NEIGHBOR's k provably doesn't affect its outcome, so it
        # isn't tracked; MOVE_AVOID_REPEAT's all-distinct fallback also skips
        # a k draw and reports -1). These counters accumulate every call
        # regardless of whether --adaptive-k is enabled -- harmless when it
        # isn't, since driver.py simply never reads them in that case.
        if move_id == MOVE_AVOID_REPEAT:
            if params[3] >= 0:
                my_k_attempt_avoid[params[4], params[3]] += 1
                if accept:
                    my_k_accept_avoid[params[4], params[3]] += 1
        elif move_id == MOVE_SWAP_ADJACENT:
            my_k_attempt_swap[params[4], params[3]] += 1
            if accept:
                my_k_accept_swap[params[4], params[3]] += 1
        elif move_id == MOVE_REMAP:
            my_k_attempt_remap[params[3]] += 1
            if accept:
                my_k_accept_remap[params[3]] += 1

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
              accept_mode, iters_per_segment, n_segments, do_swaps, max_worsen,
              accept_counter, move_counter, move_accept_counter, swap_accept_counter, swap_attempt_counter,
              k_weights_avoid, k_weights_swap, k_weights_remap,
              k_attempt_avoid, k_accept_avoid, k_attempt_swap, k_accept_swap,
              k_attempt_remap, k_accept_remap):
    """Runs n_segments * iters_per_segment SA iterations per replica, attempting
    a replica-exchange swap sweep between segments (if do_swaps).

    k_weights_avoid/k_weights_swap (float64[2, 8]) and k_weights_remap
    (float64[9]) are shared (not per-replica) k-selection weights -- see
    apply_move. k_attempt_*/k_accept_* are per-replica counters (shape
    [R, 2, 8] or [R, 9]) that --adaptive-k pools across replicas between
    blocks to update the shared weights; harmless bookkeeping when
    --adaptive-k is off (driver.py just never reads them).
    """
    R = grids.shape[0]
    for _seg in range(n_segments):
        for i in prange(R):
            _anneal_one(i, grids, dmasks, stamps, gens, energies, scores_arr, looks_arr, counts_arr,
                        temps, rng_states, digit_bufs,
                        hist, lahc_pos,
                        edge_positions, move_probs, p_edge,
                        w_score, w_look, w_count, w_heur, w_triple,
                        want_count, look_window, count_lo, count_hi,
                        accept_mode, iters_per_segment, max_worsen, accept_counter, move_counter, move_accept_counter,
                        k_weights_avoid, k_weights_swap, k_weights_remap,
                        k_attempt_avoid, k_accept_avoid, k_attempt_swap, k_accept_swap,
                        k_attempt_remap, k_accept_remap)
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
    cell_r_buf = np.zeros(MAX_CLUSTER, dtype=np.int64)
    cell_c_buf = np.zeros(MAX_CLUSTER, dtype=np.int64)
    orig_val_buf = np.zeros(MAX_CLUSTER, dtype=np.int64)
    # Calibration always uses uniform k-selection (all-ones weights), even if
    # --adaptive-k is enabled: this only ever runs once, on a genuinely fresh
    # start before any learning has happened, so uniform is exactly correct.
    k_weights_avoid = np.ones((2, 8), dtype=np.float64)
    k_weights_swap = np.ones((2, 8), dtype=np.float64)
    k_weights_remap = np.ones(9, dtype=np.float64)
    for k in range(n_samples):
        gen += 1
        move_id = apply_move(rng_state, grid, dmask, edge_positions, move_probs, p_edge,
                              nbr_val_buf, flag_buf, allowed_buf, params,
                              perm_buf, dmask_scratch,
                              cell_r_buf, cell_c_buf, orig_val_buf,
                              k_weights_avoid, k_weights_swap, k_weights_remap)
        score, look, count = evaluate(dmask, stamp, gen, look_window, count_lo, count_hi,
                                       want_count, digit_buf)
        heur = heur_chain_variance(grid) if w_heur != 0.0 else 0.0
        triples = float(count_triple_chains(grid)) if w_triple != 0.0 else 0.0
        newE = energy_of(score, look, count, heur, triples, w_score, w_look, w_count, w_heur, w_triple)
        samples_out[k] = newE - curE
        undo_move(move_id, params, grid, dmask, perm_buf, inv_buf, dmask_scratch,
                  cell_r_buf, cell_c_buf, orig_val_buf)
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
