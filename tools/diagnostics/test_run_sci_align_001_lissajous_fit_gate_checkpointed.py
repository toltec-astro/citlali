#!/usr/bin/env python3
"""Focused tests for atomic SCI-ALIGN-001 fit-gate model checkpoints."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run_sci_align_001_lissajous_fit_gate_checkpointed as runner


class CheckpointedFitGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.protocol = self.root / "protocol.json"
        self.selection = self.root / "selection.json"
        self.protocol.write_text("{}\n")
        self.selection.write_text("{}\n")
        self.args = argparse.Namespace(
            protocol=self.protocol,
            selection=self.selection,
            map_root=self.root / "maps",
            obsnum=150818,
            output=self.root / "fit",
            maximum_wall_seconds=2700.0,
        )
        self.identity = {
            "schema": "sci-align-001-fit-gate-model-checkpoint-identity-v1",
            "obsnum": 150818,
            "identity": "fixed-test-input",
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def fit(name: str) -> dict[str, object]:
        return {
            "status": "success",
            "model": name,
            "objective": 1.0,
            "tau_ms": 0.0,
            "parameters": {},
        }

    def test_checkpoint_rejects_identity_change(self) -> None:
        self.args.output.mkdir()
        fits = {"constant": self.fit("constant")}
        runner.save_checkpoint(self.args.output, self.identity, fits)
        self.assertEqual(
            runner.load_checkpoint(self.args.output, self.identity), fits
        )
        (self.args.output / runner.CHECKSUM_NAME).write_text(
            f"{'0' * 64}  {runner.CHECKPOINT_NAME}\n"
        )
        self.assertEqual(
            runner.load_checkpoint(self.args.output, self.identity), fits
        )
        runner.target.verify_sha256s(self.args.output, runner.CHECKSUM_NAME)
        changed = dict(self.identity, identity="changed")
        with self.assertRaisesRegex(
            runner.ContractError, "checkpoint identity changed"
        ):
            runner.load_checkpoint(self.args.output, changed)

    def test_interrupted_run_reuses_completed_models(self) -> None:
        observation = object()
        row = {"pointing_obsnum": 150818}
        coordinate_gate = {"status": "pass"}
        map_result = {"sha256": "map"}

        def fake_gate(*call_args: object) -> dict[str, object]:
            output = call_args[1]
            assert isinstance(output, Path)
            (output / "fit_gate.json").write_text("{}\n")
            runner.target.write_checksums(
                output, ["fit_gate.json"], "FIT_GATE_SHA256SUMS"
            )
            return {
                "quality_gate": {"automatic_structural_status": "pass"}
            }

        common = (
            mock.patch.object(runner.target, "load_protocol", return_value={
                "input_authority": {"selection_manifest_sha256": "selection"}
            }),
            mock.patch.object(runner.target, "load_selection", return_value={}),
            mock.patch.object(runner.target, "selected_row", return_value=row),
            mock.patch.object(
                runner.target, "prepare_observation", return_value=observation
            ),
            mock.patch.object(
                runner.target, "coordinate_reconstruction_gate",
                return_value=coordinate_gate,
            ),
            mock.patch.object(
                runner.target, "authenticated_map_result", return_value=map_result
            ),
            mock.patch.object(runner, "checkpoint_identity", return_value=self.identity),
            mock.patch.object(
                runner.target, "write_fit_gate_checkpoint", side_effect=fake_gate
            ),
        )
        with common[0], common[1], common[2], common[3], common[4], common[5], common[6], common[7]:
            first_models: list[str] = []

            def interrupted(
                unused_observation: object, model: str, **unused: object
            ) -> dict[str, object]:
                first_models.append(model)
                if model == "joint":
                    raise runner.target.ContractError("simulated interruption")
                return self.fit(model)

            with mock.patch.object(
                runner.target, "fit_observation_model", side_effect=interrupted
            ):
                with self.assertRaisesRegex(
                    runner.target.ContractError, "simulated interruption"
                ):
                    runner.run(self.args)
            self.assertEqual(
                first_models, ["constant", "lag", "hysteresis", "joint"]
            )
            checkpoint = runner.load_checkpoint(self.args.output, self.identity)
            self.assertEqual(list(checkpoint), ["constant", "lag", "hysteresis"])

            second_models: list[str] = []

            def complete(
                unused_observation: object, model: str, **unused: object
            ) -> dict[str, object]:
                second_models.append(model)
                return self.fit(model)

            with mock.patch.object(
                runner.target, "fit_observation_model", side_effect=complete
            ):
                runner.run(self.args)
            self.assertEqual(second_models, ["joint"])
            runner.target.verify_sha256s(
                self.args.output, "FIT_GATE_SHA256SUMS"
            )
            protected = (self.args.output / "FIT_GATE_SHA256SUMS").read_text()
            self.assertIn(runner.CHECKPOINT_NAME, protected)
            self.assertIn(runner.CHECKSUM_NAME, protected)


if __name__ == "__main__":
    unittest.main()
