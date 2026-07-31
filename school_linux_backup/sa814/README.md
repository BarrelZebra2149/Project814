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
| `set_random` | 40% | one cell (edge-biased 50% of the time) becomes a different random digit |
| `copy_neighbor` | 22% | one cell copies the value of one of its 8 neighbors |
| `swap_adjacent` | 22% | one cell and one of its 8 neighbors swap values |
| `avoid_repeat` | 10% | one cell is forced away from a random subset of its neighbor values (or copies a neighbor, if they're already all distinct) |
| `remap_pair` | 1% | two digits (e.g. 3 and 7) swap everywhere in the grid |
| `remap_full` | 5% | **all 10 digits get relabeled at once via a random permutation** (e.g. `0123456789 -> 2938475610`), not just a pairwise swap |

`remap_full` generalizes `remap_pair` and is directly inspired by
`../code/permutation.py`, which brute-forces all `10! = 3,628,800` relabelings
of one fixed grid to find the best-scoring digit assignment — proof that the
same underlying cell/cluster structure can score wildly differently purely
depending on which digit labels which cluster (since formability of a target
number depends on which physical cells carry *that* digit). `remap_full` lets
SA reach that kind of relabeling jump stochastically during the search itself,
rather than only via a separate exhaustive post-processing pass. It's
reversible in one step (undo applies the inverse permutation) so it costs
nothing extra to try and reject.

`avoid_repeat` replaced an earlier `line_shift` move (whole-line rotation,
removed). It targets a random cell, looks at its valid 8-directional
neighbors (3 at a corner, 5 on an edge, 8 in the interior), and: if those
neighbor values are already all pairwise distinct, it just copies a random
one (nothing useful to force); otherwise it samples a random subset of 1..n
of them and forces the new value to avoid every value in that subset. This
directly targets the waste that `w_triple` (below) penalizes — deliberately
breaking up same-digit runs among a cell's neighbors before they turn into a
3-in-a-row.

## Triple-chain penalty (`w_triple`)

Since a walk may revisit cells, only **two** adjacent same-digit cells are
ever needed to form an arbitrarily long run of that digit (the walk just
bounces between them to spell `11`, `111`, `1111`, ... as needed). A
**third** cell continuing that same straight line adds nothing to
formability — it's a wasted cell that could have carried a more useful digit
for some other number.

`core814.count_triple_chains(grid)` counts every overlapping window of 3
consecutive identical digits along the 4 undirected axis directions
(horizontal, vertical, both diagonals — each axis counted once). A run of
length L >= 3 contributes `L - 2` overlapping triples, so longer redundant
runs are penalized more. The energy function subtracts
`w_triple * triple_count` (default `w_triple = 0.001`, small enough that even
a heavily-degenerate grid's worth of triples can never outweigh a single real
score point) — see `core814.energy_of`. `avoid_repeat` is the move most
directly aimed at reducing this count during the search.

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
`--target-score N`, `--iters N`, `--seed-file path.txt`, `--no-seed`. Run
`python win_score_first.py --help` for the full list.

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
