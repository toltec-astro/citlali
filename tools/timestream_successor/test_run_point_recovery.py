"""Synthetic lifecycle tests: no native reduction, source fit or archive access."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run_point_recovery as runner
from point_recovery_products import ARRAYS, ROLES


class RunnerTest(unittest.TestCase):
    def run_fixture(self, failure=None):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        base = root / "base.json"
        base.write_text(
            json.dumps(dict(networks=[dict(root="original")], stage="cycle"))
        )
        geom = root / "geometry.npz"
        np.savez(geom, x=np.zeros((1, 1)), y=np.zeros((1, 1)))
        raster = root / "raster.json"
        raster.write_text(json.dumps(dict(values=[0.0] * 1681)))
        cases = {}
        for role in ROLES:
            p = root / (role + ".json")
            p.write_text(json.dumps(dict(arrays={})))
            cases[role] = dict(path=str(p))
        manifest = dict(
            output=str(root / "output"),
            base_request=dict(path=str(base)),
            max_seconds=100,
            minimum_free_bytes=0,
            max_output_bytes=10**8,
            geometry={a: dict(path=str(geom)) for a in ARRAYS},
            rasters={
                r: {a: dict(path=str(raster)) for a in ARRAYS}
                for r in ROLES
                if r != "SEED"
            },
            cases=cases,
            restoration={},
            native=dict(path="unused"),
            reference_work="unused",
        )
        calls = []
        contexts = []
        used = []

        def native(binary, path, stage, **kwargs):
            q = json.loads(path.read_text())
            role = stage.parent.name
            c = q["cycle_index"]
            calls.append((role, c, q))
            if failure == (role, c):
                raise ValueError("deliberate fixture failure")
            stage.mkdir()
            (stage / "products").mkdir()
            (stage / "products/cycle.yaml").write_text("fixture")
            return dict(seconds=0)

        guard = dict(
            accepted=True, accepted_parameters=[1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0]
        )

        def selection(map_path, ctx, dest):
            dest.mkdir()
            (dest / "candidate.fits").write_text("fixture")
            (dest / "feedback.json").write_text("{}")
            map_path.parent.mkdir(parents=True, exist_ok=True)
            (map_path.parent / "bundle.yaml").write_text("fixture")
            used.append((dest.parent.parent.name, ctx))
            return dict(
                status="ready", model_path=str(dest / "candidate.fits"), guard=guard
            )

        def initialize(map_path, dest):
            ctx = object()
            contexts.append(ctx)
            return ctx, selection(map_path, ctx, dest)

        def materialize(r, d, t):
            (d / "RESTORATION.json").write_text("{}")
            return {"original": "shadow"}

        def record_cycle(m, r, c, stage, *args):
            (stage / "measurements.json").write_text("{}")
            return {a: dict(total={}) for a in ARRAYS}

        with contextlib.ExitStack() as stack:
            mocks = {
                "check": lambda m: None,
                "native_call": native,
                "parity": lambda *args: None,
                "segment_check": lambda *args: {a: [] for a in ARRAYS},
                "materialize": materialize,
                "initialize": initialize,
                "update": selection,
                "measure_only": lambda *args: guard,
                "record_cycle": record_cycle,
                "publish_seed": lambda *args: {},
                "seed_initial": lambda *args: {},
                "load_map": lambda *args: (np.zeros((1, 1)), np.ones((1, 1), bool)),
                "measurements": lambda *args: {},
                "increments": lambda *args: {},
                "final_accounting": lambda *args: None,
            }
            for name, fn in mocks.items():
                stack.enter_context(patch.object(runner, name, fn))
            stack.enter_context(
                patch.object(runner.yaml, "safe_load", lambda *args: dict(segments=[]))
            )
            # SEED compare opens a real reference plan before the YAML mock.
            original_native = native

            def with_plan(binary, path, stage, **kwargs):
                result = original_native(binary, path, stage, **kwargs)
                for a in ARRAYS:
                    d = stage / "products" / a
                    d.mkdir(exist_ok=True)
                    (d / "plan.yaml").write_text("fixture")
                return result

            stack.enter_context(patch.object(runner, "native_call", with_plan))
            stack.enter_context(patch.object(runner.signal, "signal"))
            stack.enter_context(patch.object(runner.signal, "setitimer"))
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    runner.execute(manifest)
                except (RuntimeError, ValueError):
                    if failure is None:
                        raise
        return calls, contexts, used, json.loads((root / "output/RUN.json").read_text())

    def test_five_cases_no_seed_m0_no_cycle_11(self):
        calls, contexts, used, record = self.run_fixture()
        self.assertEqual(len(calls), 54)
        self.assertEqual(record["status"], "completed")
        self.assertEqual(len(contexts), 12)
        self.assertNotIn(("SEED", 0), [(r, c) for r, c, q in calls])
        self.assertTrue(all(c <= 10 for r, c, q in calls))
        self.assertEqual(
            {id(c) for r, c in used if r == "SEED"}, set(map(id, contexts[:3]))
        )
        for role, c, q in calls:
            self.assertEqual(q["weighting_control"], "inverse-noise")
            self.assertEqual(len(q["models"]), 0 if c == 0 else 3)

    def test_reference_failure_blocks_matrix(self):
        calls, _, _, record = self.run_fixture(("REF", 2))
        self.assertEqual(len(calls), 3)
        self.assertEqual(record["status"], "stopped")
        self.assertEqual(set(record["cases"]), {"REF"})

    def test_branch_failure_not_retried_other_roles_finish(self):
        calls, _, _, record = self.run_fixture(("CORE+", 2))
        self.assertEqual(record["cases"]["CORE+"]["status"], "stopped")
        self.assertEqual(record["cases"]["SEED"]["status"], "completed")
        self.assertEqual(record["status"], "completed_with_stops")
        self.assertEqual(sum(r == "CORE+" and c == 2 for r, c, q in calls), 1)


if __name__ == "__main__":
    unittest.main()
