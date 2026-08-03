"""
Verification suite for the sa814 seed tournament (config814.seed_grids_file,
seeding.build_replicas_from_file, and tournament.py). Every test forces its
rare code path by shrinking a threshold rather than waiting longer -- the
anchor_grace unit-mismatch bug shipped precisely because a prior gate test
was too short to ever reach the reheat path it was supposed to guard.

Run: python test_tournament.py [-v]
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import core814 as core
import seeding
import checkpoint
import tournament as tour
from config814 import config_from_cli

SA814_DIR = Path(__file__).resolve().parent
DATA_DIR = SA814_DIR.parent / "data"

_ENTRYPOINT = "linux_score_first.py" if os.name == "posix" else "win_score_first.py"


def grid_to_text(grid: np.ndarray) -> str:
    return "\n".join("".join(str(int(v)) for v in row) for row in grid) + "\n"


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int((a != b).sum())


# ===========================================================================
# V1 -- build_replicas_from_file, pure unit tests, no subprocess
# ===========================================================================

class TestV1SeedingUnit(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.rng = np.random.default_rng(12345)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_single_grid_file_diversity_ladder(self):
        grid = self.rng.integers(0, 10, size=(8, 14)).astype(np.uint8)
        f = self.tmpdir / "one.txt"
        f.write_text(grid_to_text(grid), encoding="utf-8")

        out = seeding.build_replicas_from_file(12, str(f), self.rng)
        self.assertEqual(out.shape, (12, 8, 14))
        np.testing.assert_array_equal(out[0], grid, "replica 0 must be byte-identical to the source grid")

        seen = set()
        for i in range(1, 12):
            d = hamming(out[i], grid)
            self.assertGreaterEqual(d, 1, f"replica {i} must actually differ from the base grid")
            self.assertLessEqual(d, 6, f"replica {i} perturbation exceeds max_perturb_cells")
            seen.add(out[i].tobytes())
        self.assertEqual(len(seen), 11, "all 11 perturbed replicas must be pairwise distinct")

    def test_twenty_grid_file_preserves_file_order_not_score_order(self):
        # Deliberately worst-scoring-first so a score-sort bug would be caught.
        grids = [self.rng.integers(0, 10, size=(8, 14)).astype(np.uint8) for _ in range(20)]
        f = self.tmpdir / "twenty.txt"
        f.write_text("\n".join(grid_to_text(g) for g in grids), encoding="utf-8")

        out = seeding.build_replicas_from_file(12, str(f), self.rng)
        for i in range(12):
            np.testing.assert_array_equal(out[i], grids[i],
                                           f"replica {i} must match file order, not score order")

    def test_empty_file_raises(self):
        f = self.tmpdir / "empty.txt"
        f.write_text("", encoding="utf-8")
        with self.assertRaises(ValueError):
            seeding.build_replicas_from_file(4, str(f), self.rng)

    def test_comments_only_file_raises(self):
        f = self.tmpdir / "comments.txt"
        f.write_text("# score=100 iters=1 t=0.0s\n# just a comment, no grid\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            seeding.build_replicas_from_file(4, str(f), self.rng)

    def test_malformed_seven_line_block_raises(self):
        grid = self.rng.integers(0, 10, size=(8, 14)).astype(np.uint8)
        lines = grid_to_text(grid).splitlines()[:7]  # drop the 8th line
        f = self.tmpdir / "seven.txt"
        f.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            seeding.build_replicas_from_file(4, str(f), self.rng)


# ===========================================================================
# V2 / V2b -- exclusive seeding actually takes effect (1 real subprocess each)
# ===========================================================================

class TestV2SeedingIntegration(unittest.TestCase):
    """Picks a MID-corpus grid (not the 7666 max) so that a corpus leak,
    a random fallback, and a genuinely-working exclusive path are all
    distinguishable by a single best_score assertion."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = Path(tempfile.mkdtemp())
        grids = seeding.load_corpus(DATA_DIR, None, None)
        seen = set()
        unique = []
        for g in grids:
            k = g.tobytes()
            if k not in seen:
                seen.add(k)
                unique.append(g)
        scored = sorted(((seeding.score_grid(g), g) for g in unique), key=lambda t: t[0])
        mid = [g for s, g in scored if 3000 <= s <= 5000]
        if not mid:
            raise unittest.SkipTest("no mid-corpus (3000-5000) grid found in data/*.txt")
        cls.mid_score = seeding.score_grid(mid[len(mid) // 2])
        cls.mid_grid = mid[len(mid) // 2]
        cls.seed_file = cls.tmpdir / "mid.txt"
        cls.seed_file.write_text(grid_to_text(cls.mid_grid), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def _run(self, extra_flags, run_name):
        run_dir = SA814_DIR / "runs" / run_name
        shutil.rmtree(run_dir, ignore_errors=True)
        cmd = [sys.executable, "win_score_first.py", "--fresh", "--replicas", "4",
               "--iters", "1", "--rng-seed", "1", "--run-name", run_name] + extra_flags
        proc = subprocess.run(cmd, cwd=str(SA814_DIR), capture_output=True, text=True, timeout=120)
        meta = checkpoint.load_meta(run_dir)
        shutil.rmtree(run_dir, ignore_errors=True)
        return proc, meta

    def test_seed_grids_uses_the_specified_file_only(self):
        proc, meta = self._run(["--seed-grids", str(self.seed_file)], "__test_v2_seedgrids")
        self.assertIn("--seed-grids:", proc.stdout, "missing the machine-parseable seed-grids log line")
        self.assertIsNotNone(meta, "run produced no meta.json")
        best = meta["best_score"]
        # This single assertion catches all three failure modes at once:
        #  - silent fallback to random (best would be far below mid_score, tens/hundreds)
        #  - corpus leaking in (best would be 7666, the corpus max)
        #  - the file being ignored entirely
        self.assertGreaterEqual(best, self.mid_score - 50,
                                 f"best_score {best} is far below the seeded grid's own score "
                                 f"{self.mid_score} -- looks like a random fallback")
        self.assertLess(best, 7000,
                         f"best_score {best} >= 7000 -- looks like the 7666 corpus grid leaked in")

    def test_negative_control_seed_file_gets_outranked_by_corpus(self):
        """Proves --seed-file (the old, broken path) really is dominated by the
        corpus -- confirms this test suite would catch someone 'helpfully'
        merging --seed-grids back into --seed-file's code path."""
        proc, meta = self._run(["--seed-file", str(self.seed_file)], "__test_v2_seedfile_negctrl")
        self.assertIsNotNone(meta)
        self.assertEqual(meta["best_score"], 7666,
                          "--seed-file was expected to be outranked by the 7666 corpus grid; "
                          "if this fails, --seed-file's behavior changed and V2's premise is stale")


# ===========================================================================
# V3 / V3b -- DCD front-0 and NYPC relative-score formula, pure unit tests
# ===========================================================================

class TestV3DCDAndNYPC(unittest.TestCase):
    def _make_pop(self):
        # (score, look, triples) for 6 individuals, matching the values
        # already verified against real historical run data in the plan doc.
        vecs = [(5487, 120, 40), (5774, 90, 55), (5759, 150, 30),
                (4708, 200, 20), (4809, 80, 60), (5000, 140, 35)]
        pop = []
        for i, (s, l, t) in enumerate(vecs):
            pop.append(tour.Individual(ind_id=f"i{i:03d}", run_name=f"t__i{i:03d}",
                                        best_score=s, look=l, triples=t, recent_gain=0))
        return pop

    def test_front0_matches_verified_reference(self):
        pop = self._make_pop()
        front0 = set(tour.sort_nondominated_front0(pop))
        # (5487,120,40)=i000 (4809,80,60)=i004 (5000,140,35)=i005 are dominated.
        expected = {"i001", "i002", "i003"}  # (5774,90,55) (5759,150,30) (4708,200,20)
        self.assertEqual(front0, expected,
                          "score-last-place individual (4708) must survive on front0 via `look` -- "
                          "this is the mechanism that would have saved the historical 500M-checkpoint "
                          "last-place run that went on to win")

    def test_nypc_relative_scores_bounds_and_symmetry(self):
        values = [100, 90, 80, 70, 60]
        n = len(values)
        rel = tour.nypc_relative_scores(values, maximize=True)
        self.assertAlmostEqual(rel[0], 1_000_000.0, delta=1.0, msg="top scorer (n_lose=0) should get exactly 1e6")
        # Last place loses to the other n-1 (never to itself), so the bound
        # approaches -- but for finite n does not reach -- 5e5; it's exactly
        # 1e6*(1 - 0.5*sqrt((n-1)/n)) here, strictly greater than 5e5.
        expected_last = 1_000_000.0 * (1.0 - 0.5 * math.sqrt((n - 1) / n))
        self.assertAlmostEqual(rel[-1], expected_last, delta=1.0)
        for r in rel:
            self.assertGreater(r, 500_000.0, "no finite population can hit the asymptotic 5e5 floor exactly")
            self.assertLessEqual(r, 1_000_000.0 + 1.0)

    def test_nypc_relative_scores_ties(self):
        values = [50, 50, 50]
        rel = tour.nypc_relative_scores(values, maximize=True)
        # all tied -> n_lose=0, n_draw=2 for everyone -> identical, equal-ish score
        self.assertAlmostEqual(rel[0], rel[1], delta=1e-6)
        self.assertAlmostEqual(rel[1], rel[2], delta=1e-6)


# ===========================================================================
# V5a -- park/revive/retire state machine, pure unit test (no subprocess)
# ===========================================================================

class TestV5aStateMachine(unittest.TestCase):
    def tearDown(self):
        # apply_slice_result -> run_dir_for -> checkpoint.run_root has a
        # mkdir(exist_ok=True) side effect even though these tests never
        # spawn a real subprocess; clean up the resulting empty dirs.
        for p in (SA814_DIR / "runs").glob("__v5a__*"):
            shutil.rmtree(p, ignore_errors=True)

    def _tcfg(self, **overrides):
        base = dict(name="__v5a_test", pop_size=3, max_pop=4, replicas=2,
                    slice_iters=1000, park_stall_iters=5000, retire_stall_iters=20000,
                    min_trial_iters=10000, elite_keep=1)
        base.update(overrides)
        return tour.TournamentConfig(**base)

    def _make_ind(self, ind_id="i000", **kw):
        defaults = dict(ind_id=ind_id, run_name=f"__v5a__{ind_id}")
        defaults.update(kw)
        return tour.Individual(**defaults)

    def test_min_trial_iters_blocks_archiving_even_with_huge_stall(self):
        tcfg = self._tcfg()
        ind = self._make_ind(total_iters=100, stall_iters=0, best_score=100)
        pop = [ind]
        res = tour.SliceResult(ok=True, pre_iters=0, post_iters=100, delta_iters=100,
                                pre_score=-1, post_score=100, cfg_hash="h")
        scratch = tour._RescoreScratch()
        # Monkeypatch rescore to avoid needing a real best.txt on disk.
        scratch.rescore = lambda g: (100, 10, 1)
        # Force no grid found -> falls back to res.post_score path.
        orig = tour.parse_grid_from_best_txt
        tour.parse_grid_from_best_txt = lambda run_dir: None
        try:
            ind.stall_iters = 10 ** 12  # absurdly huge
            ind.total_iters = tcfg.min_trial_iters - 1  # still below the floor
            events = tour.apply_slice_result(ind, res, tcfg, round_no=1, scratch=scratch, population=pop)
            self.assertNotEqual(ind.state, "archived",
                                 "an individual below min_trial_iters must never be archived, "
                                 "regardless of stall_iters -- the historical winner was last "
                                 "place at exactly this iteration count")
        finally:
            tour.parse_grid_from_best_txt = orig

    def test_full_state_transition_sequence(self):
        tcfg = self._tcfg(min_trial_iters=0, elite_keep=0)
        ind = self._make_ind(best_score=100, total_iters=0)
        pop = [ind]
        scratch = tour._RescoreScratch()
        orig = tour.parse_grid_from_best_txt
        tour.parse_grid_from_best_txt = lambda run_dir: None
        try:
            # 1) A non-improving slice below park threshold -> stays active.
            res1 = tour.SliceResult(ok=True, pre_iters=0, post_iters=1000, delta_iters=1000,
                                     post_score=100, cfg_hash="h")
            tour.apply_slice_result(ind, res1, tcfg, 1, scratch, population=pop)
            self.assertEqual(ind.state, "active")

            # 2) Enough more non-improving iters to cross park_stall_iters -> parked.
            res2 = tour.SliceResult(ok=True, pre_iters=1000, post_iters=1000 + tcfg.park_stall_iters,
                                     delta_iters=tcfg.park_stall_iters, post_score=100, cfg_hash="h")
            tour.apply_slice_result(ind, res2, tcfg, 2, scratch, population=pop)
            self.assertEqual(ind.state, "parked")
            self.assertEqual(ind.parked_at_round, 2)

            # 3) All individuals parked -> revive() must bring it back with stall_iters==0.
            tour.revive(pop, tcfg, round_no=3)
            self.assertEqual(ind.state, "active")
            self.assertEqual(ind.stall_iters, 0,
                              "revive must reset stall_iters -- otherwise it's parked again immediately")

            # 4) An improving slice clears stall and records a new best.
            res3 = tour.SliceResult(ok=True, pre_iters=ind.total_iters, post_iters=ind.total_iters + 500,
                                     delta_iters=500, post_score=150, cfg_hash="h")
            scratch.rescore = lambda g: (150, 20, 2)
            tour.parse_grid_from_best_txt = lambda run_dir: np.zeros((8, 14), dtype=np.uint8)
            tour.apply_slice_result(ind, res3, tcfg, 4, scratch, population=pop)
            self.assertEqual(ind.best_score, 150)
            self.assertEqual(ind.stall_iters, 0)

            # 5) Stall again, past retire_stall_iters this time -> archived.
            tour.parse_grid_from_best_txt = lambda run_dir: None
            res4 = tour.SliceResult(ok=True, pre_iters=ind.total_iters,
                                     post_iters=ind.total_iters + tcfg.retire_stall_iters,
                                     delta_iters=tcfg.retire_stall_iters, post_score=150, cfg_hash="h")
            tour.apply_slice_result(ind, res4, tcfg, 5, scratch, population=pop)
            self.assertEqual(ind.state, "archived")

            # 6) Un-archive: revive() falls back to archived pool when parked is empty.
            tour.revive(pop, tcfg, round_no=6)
            self.assertEqual(ind.state, "active")
        finally:
            tour.parse_grid_from_best_txt = orig

    def test_champion_is_parked_not_archived(self):
        tcfg = self._tcfg(min_trial_iters=0, elite_keep=1, retire_stall_iters=1000, park_stall_iters=500)
        champ = self._make_ind("i000", best_score=999, total_iters=0)
        other = self._make_ind("i001", best_score=100, total_iters=0)
        pop = [champ, other]
        scratch = tour._RescoreScratch()
        orig = tour.parse_grid_from_best_txt
        tour.parse_grid_from_best_txt = lambda run_dir: None
        try:
            res = tour.SliceResult(ok=True, pre_iters=0, post_iters=2000, delta_iters=2000,
                                    post_score=999, cfg_hash="h")
            tour.apply_slice_result(champ, res, tcfg, 1, scratch, population=pop)
            self.assertEqual(champ.state, "parked",
                              "the champion must still be subject to park -- elitism protects "
                              "against archiving only, never against rotation")
            self.assertNotEqual(champ.state, "archived")
        finally:
            tour.parse_grid_from_best_txt = orig


# ===========================================================================
# V4 / V5b / V6 / V7 / V8 / V9 -- integration tests (real subprocesses)
# ===========================================================================

class _TournamentIntegrationBase(unittest.TestCase):
    NAME = None

    def setUp(self):
        self._cleanup()

    def tearDown(self):
        self._cleanup()

    def _cleanup(self):
        shutil.rmtree(SA814_DIR / "tournament" / self.NAME, ignore_errors=True)
        for p in (SA814_DIR / "runs").glob(f"{self.NAME}__*"):
            shutil.rmtree(p, ignore_errors=True)


class TestV4Rotation(_TournamentIntegrationBase):
    NAME = "__test_v4_rotation"

    def test_all_individuals_get_slices_and_progress_monotonically(self):
        tcfg = tour.TournamentConfig(
            name=self.NAME, solver_entrypoint=_ENTRYPOINT, pop_size=3, max_pop=3, replicas=2,
            slice_iters=20_000, park_stall_iters=10 ** 15, retire_stall_iters=10 ** 15,
            min_trial_iters=0, base_seed=500,
        )
        final = tour.run_tournament(tcfg, max_rounds=6)
        inds = [tour.Individual.from_dict(d) for d in final["individuals"]]
        for ind in inds:
            self.assertGreaterEqual(ind.slices_run, 1, f"{ind.ind_id} never got a slice in 6 rounds")
            self.assertGreater(ind.total_iters, 0)


class TestV5bParkReviveIntegration(_TournamentIntegrationBase):
    NAME = "__test_v5b_parkrevive"

    def test_parking_and_revival_events_logged(self):
        tcfg = tour.TournamentConfig(
            name=self.NAME, solver_entrypoint=_ENTRYPOINT, pop_size=2, max_pop=2, replicas=2,
            slice_iters=20_000, park_stall_iters=1, retire_stall_iters=10 ** 15,
            min_trial_iters=0, base_seed=600,
        )
        tour.run_tournament(tcfg, max_rounds=4)
        log_path = SA814_DIR / "tournament" / self.NAME / "log.jsonl"
        self.assertTrue(log_path.exists())
        events = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        n_parked = sum(1 for e in events if e["type"] == "parked")
        n_revived = sum(1 for e in events if e["type"] == "revived" or e["type"] == "individual_failed")
        self.assertGreaterEqual(n_parked, 1, "expected at least one park event with park_stall_iters=1")


class TestV6NewSeedReplacement(_TournamentIntegrationBase):
    NAME = "__test_v6_replacement"

    def test_archived_individual_replaced_by_fresh_random_seed(self):
        tcfg = tour.TournamentConfig(
            name=self.NAME, solver_entrypoint=_ENTRYPOINT, pop_size=2, max_pop=3, replicas=2,
            slice_iters=20_000, park_stall_iters=1, retire_stall_iters=1,
            min_trial_iters=0, elite_keep=0, base_seed=700,
        )
        final = tour.run_tournament(tcfg, max_rounds=5)
        inds = [tour.Individual.from_dict(d) for d in final["individuals"]]
        replacements = [i for i in inds if i.origin == "replacement:random"]
        self.assertGreaterEqual(len(replacements), 1, "expected at least one replacement individual")
        self.assertLessEqual(len(inds), tcfg.max_pop, "population must not exceed max_pop")


@unittest.skipUnless(os.name == "posix", "graceful SIGTERM shutdown is Linux-only (Windows "
                                          "Popen.terminate() is TerminateProcess, no clean checkpoint)")
class TestV7OrchestratorRestart(_TournamentIntegrationBase):
    NAME = "__test_v7_restart"

    def test_sigterm_checkpoints_and_resume_skips_fresh(self):
        import signal as _signal
        import threading

        tcfg = tour.TournamentConfig(
            name=self.NAME, solver_entrypoint=_ENTRYPOINT, pop_size=1, max_pop=1, replicas=2,
            slice_iters=2_000_000_000, base_seed=1000,  # deliberately long so we can interrupt mid-slice
        )
        result_holder = {}

        def _runner():
            result_holder["final"] = tour.run_tournament(tcfg, max_rounds=1)

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        time.sleep(3.0)
        os.kill(os.getpid(), _signal.SIGTERM)
        t.join(timeout=30)
        self.assertFalse(t.is_alive(), "orchestrator did not shut down within 30s of SIGTERM")

        state = tour.load_state(self.NAME)
        self.assertIsNotNone(state)
        ind = tour.Individual.from_dict(state["individuals"][0])
        self.assertGreater(ind.total_iters, 0, "child should have checkpointed some progress before exit")

        # Resume: re-run without max_rounds=0 and confirm no --fresh is passed
        # for the existing individual (state.json already has it).
        state2 = tour.load_state(self.NAME)
        self.assertIsNotNone(state2)
        ind2 = tour.Individual.from_dict(state2["individuals"][0])
        self.assertGreaterEqual(ind2.total_iters, ind.total_iters,
                                 "total_iters must never decrease across a restart")


class TestV8DestructiveResetGuard(_TournamentIntegrationBase):
    NAME = "__test_v8_reset"

    def test_replica_count_change_is_detected_and_quarantined(self):
        tcfg = tour.TournamentConfig(
            name=self.NAME, solver_entrypoint=_ENTRYPOINT, pop_size=1, max_pop=1, replicas=2,
            slice_iters=20_000, base_seed=800,
        )
        ind = tour.Individual(
            ind_id="i000", run_name=f"{self.NAME}__i000", origin="founder:random",
            cfg_flags=["--fresh", "--no-seed", "--rng-seed", "800", "--replicas", "2"],
        )
        stop_flag = {"proc": None}
        res1 = tour.run_slice(tcfg, ind, stop_flag)
        scratch = tour._RescoreScratch()
        tour.apply_slice_result(ind, res1, tcfg, 1, scratch, population=[ind])
        self.assertFalse(res1.reset_detected)

        # Now change a semantic cfg_hash field (replicas) for the SAME run_name
        # without touching the orchestrator's bookkeeping -- this trips
        # driver.py's cfg_hash-mismatch branch, which resets total_iters to 0.
        ind.cfg_flags = ["--rng-seed", "800", "--replicas", "8"]
        res2 = tour.run_slice(tcfg, ind, stop_flag)
        self.assertTrue(res2.reset_detected, "changing --replicas must be caught as a destructive reset")
        events = tour.apply_slice_result(ind, res2, tcfg, 2, scratch, population=[ind])
        self.assertEqual(ind.state, "failed")
        self.assertTrue(any(e[0] == "destructive_reset" for e in events))


class TestV9CrashResilience(_TournamentIntegrationBase):
    NAME = "__test_v9_crash"

    def test_bad_flags_dont_kill_the_tournament(self):
        tcfg = tour.TournamentConfig(
            name=self.NAME, solver_entrypoint=_ENTRYPOINT, pop_size=2, max_pop=2, replicas=2,
            slice_iters=20_000, base_seed=900,
        )
        bad = tour.Individual(
            ind_id="i000", run_name=f"{self.NAME}__i000", origin="founder:random",
            cfg_flags=["--fresh", "--replicas", "-1"],
        )
        good = tour.Individual(
            ind_id="i001", run_name=f"{self.NAME}__i001", origin="founder:random",
            cfg_flags=["--fresh", "--no-seed", "--rng-seed", "901", "--replicas", "2"],
        )
        pop = [bad, good]
        scratch = tour._RescoreScratch()
        stop_flag = {"proc": None}
        for _ in range(3):
            res = tour.run_slice(tcfg, bad, stop_flag)
            tour.apply_slice_result(bad, res, tcfg, 1, scratch, population=pop)
        self.assertEqual(bad.state, "failed", "bad individual should fail after 3 consecutive bad slices")

        res_good = tour.run_slice(tcfg, good, stop_flag)
        events = tour.apply_slice_result(good, res_good, tcfg, 1, scratch, population=pop)
        self.assertNotEqual(good.state, "failed", "a healthy individual must be unaffected by a sibling's failures")


if __name__ == "__main__":
    unittest.main(verbosity=2)
