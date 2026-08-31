#!/usr/bin/env python3
"""Focused tests for the WP-7 D2 PSD/line corpus aggregator."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.wp7 import rtc_filter_psd_line_corpus as corpus
from tools.wp7 import rtc_filter_psd_line_evidence as evidence
from tools.wp7 import test_rtc_filter_psd_line_evidence as evidence_test_helpers


class PsdLineCorpusTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = evidence_test_helpers.PsdLineEvidenceTest()

    def make_artifact(
        self,
        root: Path,
        *,
        case_id: str,
        route_family: str,
        stage: str,
        network: int,
        timing_domain: str = evidence.NATIVE_TIMING_DOMAIN,
        protected: bool = True,
    ) -> Path:
        root.mkdir()
        manifest = self.fixture.make_input(
            root,
            timing_domain=timing_domain,
            stage=stage,
            protected=protected,
        )
        document = json.loads(manifest.read_text())
        document["identity"]["case_id"] = case_id
        document["identity"]["route_family"] = route_family
        document["identity"]["network"] = network
        manifest.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        evidence.build_evidence(evidence.load_input(manifest), root / "out")
        return root / "out" / "evidence.json"

    def test_complete_routes_remain_an_unselected_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for index, route in enumerate(("beammap", "science", "oof")):
                paths.append(
                    self.make_artifact(
                        root / f"{route}-raw",
                        case_id=f"{route}-case",
                        route_family=route,
                        stage="native_prefilter",
                        network=index,
                    )
                )
                paths.append(
                    self.make_artifact(
                        root / f"{route}-residual",
                        case_id=f"{route}-case",
                        route_family=route,
                        stage="native_post_cleaning_residual",
                        network=index,
                    )
                )
            result = corpus.build_corpus(paths, root / "corpus")
            self.assertEqual(
                result["disposition"],
                "measurement_complete_owner_envelope_choice_not_selected",
            )
            self.assertEqual(result["missing_native_route_families"], [])
            self.assertEqual(len(result["residual_groups"]), 1)
            group = result["residual_groups"][0]
            self.assertEqual(group["detector_count"], 12)
            self.assertEqual(
                group["aggregate_row_order"],
                ["median", "q90", "q95", "q99", "maximum"],
            )

    def test_missing_route_is_explicitly_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [
                self.make_artifact(
                    root / "science-residual",
                    case_id="science-case",
                    route_family="science",
                    stage="native_post_cleaning_residual",
                    network=0,
                )
            ]
            result = corpus.build_corpus(paths, root / "corpus")
            self.assertEqual(
                result["disposition"], "incomplete_required_route_family_evidence"
            )
            self.assertEqual(result["missing_native_route_families"], ["beammap", "oof"])

    def test_legacy_artifact_never_enters_native_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = self.make_artifact(
                root / "legacy",
                case_id="legacy",
                route_family="science",
                stage="legacy_ptc_output",
                network=0,
                timing_domain=evidence.DISCOVERY_TIMING_DOMAIN,
            )
            result = corpus.build_corpus([legacy], root / "corpus")
            self.assertEqual(result["legacy_discovery_artifact_count"], 1)
            self.assertEqual(result["residual_groups"], [])

    def test_unmasked_residual_keeps_owner_input_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for index, route in enumerate(("beammap", "science", "oof")):
                paths.append(
                    self.make_artifact(
                        root / f"{route}-raw",
                        case_id=f"{route}-case",
                        route_family=route,
                        stage="native_prefilter",
                        network=index,
                    )
                )
                paths.append(
                    self.make_artifact(
                        root / f"{route}-residual",
                        case_id=f"{route}-case",
                        route_family=route,
                        stage="native_post_cleaning_residual",
                        network=index,
                        protected=(route != "science"),
                    )
                )
            result = corpus.build_corpus(paths, root / "corpus")
            self.assertEqual(
                result["disposition"],
                "measurement_complete_owner_envelope_input_pending",
            )

    def test_duplicate_case_network_stage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            one = self.make_artifact(
                root / "one",
                case_id="science-case",
                route_family="science",
                stage="native_prefilter",
                network=0,
            )
            two = self.make_artifact(
                root / "two",
                case_id="science-case",
                route_family="science",
                stage="native_prefilter",
                network=0,
            )
            with self.assertRaisesRegex(RuntimeError, "repeats"):
                corpus.build_corpus([one, two], root / "corpus")

    def test_integrated_power_does_not_bridge_a_line_mask(self) -> None:
        frequency = np.arange(5, dtype=float)
        psd = np.ones((1, 5), dtype=float)
        eligible = np.asarray([True, True, False, True, True])
        power = corpus._integrated_power(frequency, psd, eligible)
        self.assertEqual(power.tolist(), [2.0])


if __name__ == "__main__":
    unittest.main()
