from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from tools.baseline import audit_reduction_run as audit


def valid_processed_document() -> dict:
    requested = {
        "fruit_loops": {
            "enabled": False,
            "save_all_iters": False,
            "max_iters": 1,
        },
        "processed_time_chunk": {
            "clean": {"active": "standard_pca"},
            "weighting": {
                "source_mask_radius_arcsec": 30,
                "validation": {"enabled": True},
                "busy_row_suppression": {"enabled": False},
            },
            "flagging": {
                "second_pass_local": {
                    "enabled": True,
                    "source_protection": {
                        "enabled": True,
                        "active": False,
                    },
                }
            },
        },
    }
    effective = {
        "fruit_loops": {
            "enabled": False,
            "save_all_iters": True,
            "max_iters": 1,
        },
        "processed_time_chunk": {
            "clean": {"active": "standard_pca"},
            "weighting": {
                "source_mask_radius_arcsec": 30,
                "validation": {"enabled": True},
                "busy_row_suppression": {"enabled": False},
            },
            "flagging": {
                "second_pass_local": {
                    "enabled": True,
                    "source_protection": {
                        "enabled": True,
                        "active": True,
                    },
                }
            },
        },
    }
    return {
        "schema_version": "citlali-processed-timestream-provenance-v1",
        "initialized": True,
        "requested": requested,
        "effective": {
            "config": effective,
            "resolutions": {
                "cleaner_mode": {
                    "available": True,
                    "value": {"effective": "standard_pca"},
                },
                "weighting_source_mask": {
                    "available": True,
                    "value": {
                        "requested_present": True,
                        "requested": 30,
                        "effective": 30,
                        "inherited_from_cleaning": False,
                    },
                },
                "weighting_dependencies": {
                    "available": True,
                    "value": {
                        "validation_forced_by_weighting_type": False,
                        "busy_row_disabled_without_second_pass": False,
                    },
                },
                "fruit_loop_iterations": {
                    "available": True,
                    "value": {
                        "effective_max_iters": 1,
                        "effective_save_all_iters": True,
                        "forced_single_iteration_while_disabled": True,
                    },
                },
                "fruit_loop_interpolation": {
                    "available": True,
                    "value": {"effective": "jinc"},
                },
            },
        },
        "realized": {
            "source_protection": {
                "available": True,
                "value": {
                    "processed_activation_requested": True,
                    "source_aware_reduction": True,
                    "processed_active": True,
                },
            },
            "fruit_loop_iterations_completed": {
                "available": True,
                "value": 1,
            },
            "fruit_loops_converged": {
                "available": True,
                "value": False,
            },
        },
    }


def valid_raw_document() -> dict:
    return {
        "schema_version": "citlali-raw-timestream-provenance-v1",
        "initialized": True,
        "requested": {},
        "effective": {"config": {}, "resolutions": {}},
        "observation": {"available": True, "value": {}},
        "realized": {
            "execution_completed": True,
            "completed_scan_count": {"available": True, "value": 4},
            "flagged_sample_count": {"available": False},
            "dynamic_notch_count": {"available": False},
            "required_timestream_write_count": {
                "available": True,
                "value": 13,
            },
        },
    }


class ProvenanceAuditTest(unittest.TestCase):
    def test_accepts_raw_provenance_for_every_observation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            for obsnum in ("1", "2"):
                observation = redu / obsnum
                observation.mkdir()
                (observation / "raw_timestream_provenance.yaml").write_text(
                    yaml.safe_dump(valid_raw_document(), sort_keys=False),
                    encoding="utf-8",
                )

            raw = audit.audit_provenance_sidecars(
                redu, require_raw=True
            )["raw_timestream"]

            self.assertTrue(raw["required"])
            self.assertTrue(raw["valid"])
            self.assertEqual(raw["count"], 2)

    def test_rejects_incomplete_raw_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            document = valid_raw_document()
            document["realized"]["execution_completed"] = False
            document["realized"]["completed_scan_count"] = {
                "available": False
            }
            (redu / "raw_timestream_provenance.yaml").write_text(
                yaml.safe_dump(document, sort_keys=False),
                encoding="utf-8",
            )

            raw = audit.audit_provenance_sidecars(
                redu, require_raw=True
            )["raw_timestream"]

            self.assertFalse(raw["valid"])
            self.assertEqual(
                raw["files"][0]["semantic_errors"],
                [
                    "raw observation execution is not complete",
                    "realized completed_scan_count is unavailable",
                ],
            )

    def test_rejects_missing_required_raw_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records = audit.audit_provenance_sidecars(
                Path(directory), require_raw=True
            )

            self.assertFalse(records["raw_timestream"]["valid"])

    def test_accepts_complete_processed_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            (redu / "processed_timestream_provenance.yaml").write_text(
                yaml.safe_dump(valid_processed_document(), sort_keys=False),
                encoding="utf-8",
            )

            records = audit.audit_provenance_sidecars(
                redu, require_processed=True
            )

            processed = records["processed_timestream"]
            self.assertTrue(processed["present"])
            self.assertTrue(processed["required"])
            self.assertTrue(processed["valid"])
            self.assertEqual(len(processed["sha256"]), 64)

    def test_rejects_missing_processed_sections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            document = valid_processed_document()
            del document["effective"]["resolutions"]
            (redu / "processed_timestream_provenance.yaml").write_text(
                yaml.safe_dump(document, sort_keys=False),
                encoding="utf-8",
            )

            processed = audit.audit_provenance_sidecars(
                redu, require_processed=True
            )["processed_timestream"]

            self.assertFalse(processed["valid"])
            self.assertEqual(
                processed["missing_paths"],
                ["effective.resolutions"],
            )

    def test_rejects_missing_required_processed_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)

            records = audit.audit_provenance_sidecars(
                redu, require_processed=True
            )

            self.assertFalse(records["processed_timestream"]["valid"])
            self.assertFalse(audit.provenance_ok({"provenance": records}))

    def test_rejects_inconsistent_processed_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            document = valid_processed_document()
            document["effective"]["config"]["fruit_loops"]["max_iters"] = 2
            (redu / "processed_timestream_provenance.yaml").write_text(
                yaml.safe_dump(document, sort_keys=False),
                encoding="utf-8",
            )

            processed = audit.audit_provenance_sidecars(
                redu, require_processed=True
            )["processed_timestream"]

            self.assertFalse(processed["valid"])
            self.assertEqual(
                processed["files"][0]["semantic_errors"],
                ["iteration resolution does not match effective max_iters"],
            )

    def test_validates_every_observation_provenance_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            for obsnum, schema in (
                ("1", "citlali-timestream-output-provenance-v1"),
                ("2", "wrong-schema"),
            ):
                observation = redu / obsnum
                observation.mkdir()
                (observation / "timestream_output_provenance.yaml").write_text(
                    f"""\
schema_version: {schema}
requested: {{}}
effective: {{}}
realized: {{}}
""",
                    encoding="utf-8",
                )

            output = audit.audit_provenance_sidecars(redu)[
                "timestream_output"
            ]

            self.assertEqual(output["count"], 2)
            self.assertTrue(output["cardinality_ok"])
            self.assertFalse(output["valid"])
            self.assertEqual(len(output["files"]), 2)


if __name__ == "__main__":
    unittest.main()
