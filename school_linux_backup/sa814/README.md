# sa814 — SA / parallel-tempering rewrite of the 814-2 solver

Replaces the DEAP genetic algorithm in `../code/814_cpu_score_first.py` and
`../code/814_cpu_count_first.py` with real simulated annealing (temperature,
Metropolis acceptance, cooling, reheating) and adds actual mid-run
checkpointing. The old scripts are left untouched under `../code/` as a
reference / comparison baseline.

## Why this exists

The original GA had no temperature concept at all — only elitism, NSGA-II
selection, and roulette-wheel mutation. `814_cpu_score_first.py`'s registered
fitness never computed the real score in Python (it was hardwired to 0); the
actual scoring happened once per individual inside a Linux-only ELF binary
(`my_dlas`, built from `dlas.hpp` + `814_clang.cpp`) invoked via `subprocess`
with a text-file handoff — serial, and unusable on Windows. And neither
script had a checkpoint: only the best grid was ever appended to a seed file
on a new record, so killing the process lost all search state.

This rewrite:
- Computes the real score directly, in Python/numba, using a bitmask
  frontier-propagation scorer that is provably equivalent to the original
  (`verify_scorer.py`, 0 mismatches across 300 random grids × 1500 numbers
  and the entire 2242-grid `../data/*.txt` corpus) but with no fixed-size
  stack to overflow.
- Replaces GA selection with real SA / parallel tempering, and ports
  `dlas.hpp`'s late-acceptance rule natively (no more subprocess bridge).
- Checkpoints atomically every `--checkpoint-secs` (default 60s), on every
  new record, and on Ctrl-C / SIGTERM / SIGHUP — a killed run resumes from
  where it left off, not from scratch.

## Move set

Each iteration picks one move at random (`config814.SAConfig`'s `p_*` fields,
`core814.apply_move`):

| move | default weight | what changes |
|---|---|---|
| `copy_neighbor` | 40% | one cell (edge-biased 50% of the time) is set to one of its differing neighbor values, uniformly at random -- falls back to a blind random digit only if every neighbor already matches |
| `swap_adjacent` | 20% | derange the target cell + a random k in [1,n] of its neighbors as one cluster (see below) |
| `avoid_repeat` | 34% | one cell is forced away from a random subset of its neighbor values (or copies a neighbor, if they're already all distinct) |
| `remap` | 6% | pick k in [2,10], derange just those k digits everywhere in the grid (see below) |

(`avoid_repeat` was raised 10% -> 20% -> 34% after real runs showed
dramatically faster score climbs from fresh random seeds. An earlier
separate `set_random` move (20%) was merged into `copy_neighbor`: both
picked a value based on a cell's neighbors, so `set_random` was really just
`copy_neighbor` generalized under a different name -- 20% + 20% = 40%. The
two main levers to tune going forward are `p_copy_neighbor` and
`p_avoid_repeat`.)

### `remap`: unifying remap_pair and remap_full

There used to be two separate moves: `remap_pair` (swap exactly 2 digits,
1%) and `remap_full` (relabel all 10 via a uniformly random permutation,
5%). Naively, could `remap_full` produce `remap_pair`'s effect just by
chance? Only 1-in-80,640 (`C(10,2) / 10! = 45 / 3,628,800`) -- nowhere near
often enough to substitute for a dedicated move.

But parameterizing k directly (rather than hoping a uniform full-permutation
happens to reduce to one) unifies them for real: `core814.apply_remap` picks
k uniformly in `[2, 10]`, chooses k distinct digits, and **deranges** just
those k (a permutation with no fixed points, so every chosen digit is
guaranteed to actually change -- see below). `k=2` always produces a
genuine swap (the only derangement of 2 elements *is* the transposition) --
exactly the old `remap_pair`. `k=10` deranges all 10 at once, close to the
old `remap_full` except now every digit is guaranteed to change (a plain
uniform permutation could leave ~1 digit fixed by chance). Their shares
combine: `p_remap = 0.01 + 0.05 = 0.06`.

Ported from the spirit of `../code/permutation.py`, which brute-forces all
`10! = 3,628,800` relabelings of one fixed grid to find the best-scoring
digit assignment — proof that the same underlying cell/cluster structure can
score wildly differently purely depending on which digit labels which
cluster (since formability of a target number depends on which physical
cells carry *that* digit). `remap` lets SA reach that kind of relabeling
jump stochastically during the search itself. It's reversible in one step
(undo applies the inverse permutation), so it costs nothing extra to try
and reject.

### `avoid_repeat` and `copy_neighbor`

`avoid_repeat` replaced an earlier `line_shift` move (whole-line rotation,
removed). It targets a random cell, looks at its valid 8-directional
neighbors (3 at a corner, 5 on an edge, 8 in the interior), and: if those
neighbor values are already all pairwise distinct, it just copies a random
one (nothing useful to force); otherwise it samples a random subset of 1..n
of them and forces the new value to avoid every value in that subset. This
directly targets the waste that `w_triple` (below) penalizes — deliberately
breaking up same-digit runs among a cell's neighbors before they turn into a
3-in-a-row.

`copy_neighbor` itself was upgraded with this same neighbor-awareness: it
targets a single cell (only that one cell ever changes -- an earlier draft
of this move accidentally homogenized the whole target+neighbors cluster to
one value, which would have actively *grown* same-digit blobs and fought
`w_triple` head-on; that was caught before shipping) and picks a random pool
of size k in `[1,n]` of its valid neighbor values, then sets the cell to one
value drawn uniformly from that pool. Note k doesn't actually change the
resulting distribution here (for any fixed neighbor, `P(chosen) = P(in the
k-pool) * P(picked | pool size k) = (k/n)*(1/k) = 1/n`, independent of k) --
it's kept as an explicit, trackable parameter for consistency with
`avoid_repeat`/`remap`/`swap_adjacent`, not because it changes what this
particular move does.

(All of `avoid_repeat`/`copy_neighbor`/`swap_adjacent` currently draw k
**uniformly** over whatever range is available at each cell -- 1..n
neighbors, n = 3/5/8 for a corner/edge/interior cell -- and `remap` draws k
uniformly over `[2,10]`. A non-uniform, empirically-learned weighting over k
is a natural future refinement once real acceptance-rate data exists to
tune it from; see `--adaptive-k` below.)

### `swap_adjacent`: generalized to a k+1-cell cluster derangement

The original `swap_adjacent` exchanged values between exactly 2 cells (the
target and one random neighbor). `core814.apply_swap_cluster` generalizes
this: pick a random subset of size k in `[1, n]` of the target's valid
neighbors, then **derange the values** held by the target + those k
neighbors (m = k+1 cells total) among themselves -- the multiset of values
in the cluster is preserved, only which cell holds which value changes.
`k=1` (a cluster of exactly 2 cells) has only one possible derangement, the
pairwise exchange, exactly reproducing the original move.

**Guaranteeing every touched cell's value actually changes is harder than
it sounds.** A plain index derangement (`perm[i] != i` for every i) is not
enough: if two cells in the cluster already hold the same digit, a
derangement can still map one onto the other's slot and leave the *visible
value* unchanged there, even though the abstract index moved.
`core814.random_value_derangement` checks the actual values, not indices.

A valid rearrangement where every value changes exists **if and only if no
single digit occupies more than half the cluster** (`max_freq <= m // 2`,
by pigeonhole -- otherwise there aren't enough differently-valued slots to
send every occurrence of the majority digit to). The function first
rejection-samples plain index derangements and checks the values (usually
succeeds in 1-3 tries) but that alone was measured to fail on ~1% of
genuinely feasible near-threshold cases even at 50 attempts, since a random
permutation satisfying the *value* condition gets rare right at the
boundary. It falls back to a **constructive** method proven correct
whenever any valid arrangement exists at all (verified against 143,975
randomized feasible cases, 0 failures): sort positions by value so
same-valued positions land in one contiguous block, then rotate that sorted
order by `ceil(m/2)` -- since a same-value block can be at most `m // 2`
long when feasible, this rotation always pushes every position past its own
block into a differently-valued one. Only in the genuinely infeasible case
(one digit is the strict majority of the cluster) does the move end up a
harmless no-op for the excess cells, since no rearrangement could ever have
changed them anyway.

## Triple-chain penalty (`w_triple`)

Since a walk may revisit cells, only **two** adjacent same-digit cells are
ever needed to form an arbitrarily long run of that digit (the walk just
bounces between them to spell `11`, `111`, `1111`, ... as needed). A
**third** cell reachable from those two adds nothing to formability — it's a
wasted cell that could have carried a more useful digit for some other
number.

`core814.count_triple_chains(grid)` counts every 3-cell same-digit chain
reachable via an 8-directional walk that is free to **bend** at each step
(start -> mid -> end, each an 8-neighbor of the previous, end != start) --
not just straight lines. An earlier version only checked 4 fixed axis
directions and missed bent chains like `(1,1)->(1,2)->(2,1)`, which are
exactly as wasteful as a straight run; the corrected version catches those
too (verified against a brute-force Python mirror, 0 mismatches over 500
grids). For a simple straight run of length L >= 3 with no branching, the
count still reduces to exactly `L - 2`, matching the original formula;
branching/blob shapes now correctly count the extra bent triples through
them as well.

The energy function subtracts `w_triple * triple_count` (default
`w_triple = 0.005`) — see `core814.energy_of`. Calibrated against a
realistic worst case of ~200 triples a search might actually wander through
(not the ~2400 of a fully degenerate all-one-digit grid, which scores near 0
and is never seriously explored): `200 * 0.005 = 1.0`, so even that can only
just brush a single real score point, never flip it outright. `avoid_repeat`
is the move most directly aimed at reducing this count during the search.

## `--adaptive-k`: learning the k-distribution from observed data

`avoid_repeat`, `swap_adjacent`, and `remap` each draw a k (how many
neighbors, or digits, to touch) uniformly by default. `--adaptive-k` instead
learns a per-k weighting from this run's own observed accept rates, so the
search finds out for itself which k values tend to pay off rather than
treating them all as equally likely to help.

**What's tracked, and why not everything:** `copy_neighbor` is excluded --
its k provably doesn't change the resulting value distribution (see
`apply_copy_neighbor`'s docstring), so there's nothing to learn there.
`avoid_repeat` and `swap_adjacent` are tracked separately for **interior**
cells (8 neighbors) vs **border** cells (corner or edge, 3 or 5 neighbors)
-- two groups, not three, so a corner and an edge cell pool their statistics
together even though their actual neighbor counts differ (`weighted_index_
choice` just renormalizes over whichever range is valid at each cell).
`remap`'s k isn't position-based at all (it's about how many *digits* get
touched), so it has one shared distribution.

**How it works:** every replica accumulates `(k, accept?)` outcomes into
per-(move, position-type, k) counters as it runs (`core814._anneal_one`).
Every `--adaptive-k-update-iters` (default 50,000) total iterations,
`driver._update_k_weights` sums those counters across all replicas, folds
them into a decayed running total (`--adaptive-k-decay`, default 0.9, so
old evidence gradually fades and the learned preference can still shift
over a long run), and recomputes each k's weight as a Laplace-smoothed
acceptance rate: `(accepted + smoothing) / (attempted + 2*smoothing)`
(`--adaptive-k-smoothing`, default 2.0, so a k that hasn't been tried much
yet -- or got unlucky early -- isn't zeroed out). The updated weights feed
`core814.weighted_index_choice`, the single sampling path used for every
k-draw in the solver: with all-ones weights (the default, `--adaptive-k`
off) it's mathematically identical to a uniform draw, so enabling the flag
changes *only* what's in the weight arrays, not any code path.

This state (pooled counters + current weights) is included in the
checkpoint, so a resumed run picks up learning where it left off rather
than starting over.

```bash
python win_score_first.py --adaptive-k
# tune the update cadence / smoothing / decay if you want:
python win_score_first.py --adaptive-k --adaptive-k-update-iters 20000 --adaptive-k-smoothing 1.0 --adaptive-k-decay 0.95
```

Enabling or disabling `--adaptive-k` (or changing its hyperparameters)
changes `cfg_hash`, so resuming a checkpoint saved under different
`--adaptive-k` settings reseeds fresh from its best grid rather than
silently mixing old and new k-statistics.

## Layout

| File | Role |
|---|---|
| `core814.py` | numba kernel: bitmask scorer, moves, RNG, SA/LAHC/DLAS acceptance, the parallel-tempering loop. No file I/O. |
| `config814.py` | `SAConfig` dataclass, `score_first`/`count_first` presets, CLI parsing. |
| `checkpoint.py` | Atomic save/load, `best.txt`, `records.txt`, `progress.csv`. |
| `seeding.py` | Loads `../data/*.txt` (and previous run outputs) as starting grids. |
| `driver.py` | The outer Python loop: calls `core814.run_block` in chunks, updates temperature, checkpoints, logs, decides when to stop. |
| `runtime_win.py` / `runtime_linux.py` | Platform layer: signal handling, thread count, path anchoring. |
| `win_score_first.py` / `win_count_first.py` / `linux_score_first.py` / `linux_count_first.py` | Thin entry points. |
| `legacy_reference.py` | Consolidated test-only oracle: the original DFS scorer (numba, byte-for-byte, including its unbounded-stack bug) + a pure-Python unbounded reference scorer, in one place with no `deap` dependency. |
| `verify_scorer.py` | Proves the new scorer matches the original (three independent layers, using `legacy_reference.py`). |
| `bench.py` | Scorer throughput + SA thread-scaling benchmark. |

## Running it

```bash
# Windows, optimizing for the real consecutive score:
python win_score_first.py --seconds 3600

# Windows, optimizing for the formable-count secondary objective:
python win_count_first.py --seconds 3600

# Linux (same solvers, POSIX signal handling, auto-resume-friendly):
python3 linux_score_first.py --seconds 3600
nohup python3 linux_score_first.py --resume &
```

Re-running the same command resumes automatically from
`runs/<run_name>/checkpoint.npz` as long as the config hash matches (weights,
accept mode, move probabilities, replica count). Use `--fresh` to force a
clean start, or `--run-name` to keep multiple independent runs side by side.

`--fresh` alone still seeds every replica from `data/*.txt` (the existing
corpus, already scoring up to 7666/8142) plus any prior run's `best.txt`/
`records.txt` — it only skips resuming the *checkpoint* (temperature, RNG
state, iteration count, etc.), not the starting grids. For a genuine
from-scratch run with pure random starting grids, ignoring that corpus
entirely, add `--no-seed`:

```bash
python win_score_first.py --fresh --no-seed --run-name from_scratch
```

Useful flags: `--replicas N`, `--accept {sa,lahc,dlas}`, `--mode {pt,anneal}`,
`--target-score N`, `--iters N`, `--seed-file path.txt`, `--no-seed`,
`--adaptive-k` (see below). Run `python win_score_first.py --help` for the
full list.

### First-run compile cost

The first call into the numba kernel in a fresh process pays a one-time JIT
compile (roughly 30–90s for the `parallel=True` annealing loop). `cache=True`
persists the compiled code to disk, so subsequent runs are fast to start —
but a very short `--seconds` budget on a cold cache will overshoot it, since
the compile itself isn't interruptible. This is a one-time cost per machine
per numba/source version, not a per-run cost.

### Why the default acceptance rule is "dlas", not "sa"

The score is defined as "the largest K such that every integer 1..K is
formable" — breaking the walk for *any* single small number collapses the
whole score, no matter how good the rest of the grid is. Measured directly
on a real 7666/8142 seed grid: the 1st percentile of *worsening* moves' ΔE
was already ~690, and the median ~6129 — there is essentially no population
of "small, gentle" worsening moves near a high-quality grid to calibrate a
safe Boltzmann temperature against. A plain SA schedule hot enough to make
progress from a random start is, at the very same temperature, hot enough to
immediately destroy a near-optimal seed (verified: T=560, which was this
seed's own calibrated "near-frozen" T_end, dropped the grid from 7666 to 99
within one block). DLAS/LAHC's late-acceptance rule is self-relative to
recent history rather than an external energy scale, so it adapts
automatically — it stays close to greedy once history is already excellent,
while still accepting enough exploratory moves to make progress from a
mediocre or random start. `--accept sa` remains available (and does work
fine as a from-scratch explorer, e.g. climbing from 0 to 2000–3000 in ~10k
iterations/replica in testing), but is not a safe default once a run is
seeded from the (very good) existing `../data/*.txt` corpus, so it isn't one.

Note that `--mode anneal`'s cooling schedule only affects `--accept sa`;
`lahc`/`dlas` ignore the per-replica temperature entirely for the move-accept
decision (they still use it for the `pt`-mode replica-exchange *swap* step,
which is safe regardless of temperature magnitude since it only exchanges
already-valid configurations between replicas, never invents a mutation).

## Output files (per run, under `runs/<run_name>/`)

- `best.txt` — current best grid, 8 lines × 14 digits. This is the
  submission-format file.
- `records.txt` — append-only history of every new record, same 8×14 block
  format (+ a `# score=... iters=... t=...` comment line) as the original
  `../data/*.txt` corpus, so `../code/check_grids.py`, `make_new_gen.py`, and
  `permutation.py` can still read it unmodified.
- `progress.csv` — `iter,elapsed,best_score,mean_energy,accept_rate,T_min,T_max`.
- `checkpoint.npz` / `checkpoint.prev.npz` — full resumable state.
- `meta.json` — human-readable run summary + config hash.

## Verifying the rewrite yourself

```bash
python verify_scorer.py          # scorer equivalence: new bitmask vs original DFS
python bench.py                  # throughput: old vs new scorer, SA thread scaling
```

`verify_scorer.py` uses `legacy_reference.py` for all reference/oracle
functions (previously this logic was scattered: the safe oracle only existed
inside `../code/check_grids.py`, and the original DFS scorer had to be
pulled in via `importlib` from `../code/814_cpu_count_first.py`, which
executes `from deap import base, creator, tools` at import time just to
reach two small numba functions — so verification required a `deap` install
it didn't otherwise need). `legacy_reference.py` has no such dependency and
provides three layers of ground truth:

1. `original_has_path_fast` / `original_evaluate_core_numba` — numba,
   byte-for-byte copies of the pre-existing scorer, bug and all (the fixed
   500-slot stack with no overflow check).
2. `oracle_has_path` / `oracle_evaluate` — pure Python, unbounded, ported
   from `../code/check_grids.py`'s `_has_path`. Used as the tie-breaker
   whenever (1) and the new scorer disagree, since (1) can silently overflow
   on grids with enough digit repetition and give a wrong answer.
3. A direct new-scorer-vs-oracle check with no original numba code involved
   at all, on a small sample (the oracle is pure Python and slow, so this
   layer is intentionally not run over the whole corpus).

Any disagreement between the new scorer and (1) is checked against (2)
before being counted as a real bug.
