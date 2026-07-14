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
        "observation": {
            "available": True,
            "value": {
                "native_sample_rate_hz": {"available": True, "value": 100.0},
                "effective_sample_rate_hz": {"available": True, "value": 50.0},
                "downsample_factor": {"available": True, "value": 2},
                "filter_edge_guard_samples": {"available": True, "value": 4},
                "filter_outer_context_samples": {"available": True, "value": 8},
                "filter_edge_guard_parity_deferred": False,
                "source_protection_active": {"available": True, "value": False},
                "extinction_active": {"available": True, "value": False},
                "extinction_model": {"available": True, "value": "N/A"},
            },
        },
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


def valid_output_document() -> dict:
    return {
        "schema_version": "citlali-timestream-output-provenance-v1",
        "requested": {},
        "effective": {},
        "realized": {"n_scans": 4},
    }


def valid_mapmaking_document() -> dict:
    requested = {
        "enabled": True,
        "grouping": "auto",
        "cunit": "mJy/beam",
    }
    effective = {
        "enabled": True,
        "grouping": "array",
        "cunit": "mJy/beam",
        "method": "jinc",
    }
    return {
        "schema_version": "citlali-mapmaking-provenance-v2",
        "initialized": True,
        "requested": requested,
        "effective": {
            "config": effective,
            "resolution": {
                "reduction_type": "science",
                "requested_grouping": "auto",
                "effective_grouping": "array",
                "automatic_grouping_resolved": True,
                "detector_grouping_fell_back_to_array": False,
                "requested_unit": "mJy/beam",
                "effective_unit": "mJy/beam",
                "uncalibrated_unit_substituted": False,
            },
        },
        "observations": [
            {
                "observation_index": 0,
                "obsnum": 152389,
                "map_count": 3,
                "effective_pixel_size_rad": 9.696273622e-6,
                "required_map_write_count": 3,
                "outputs_completed": True,
            },
            {
                "observation_index": 1,
                "obsnum": 152390,
                "map_count": 3,
                "effective_pixel_size_rad": 9.696273622e-6,
                "required_map_write_count": 3,
                "outputs_completed": True,
            },
        ],
        "coadd": {
            "available": True,
            "map_count": 3,
            "required_map_write_count": 6,
            "outputs_completed": True,
        },
        "realized": {
            "reduction_completed": True,
            "mapmaking_executed": True,
            "completed_observation_count": {"available": True, "value": 2},
            "completed_coadd_count": {"available": True, "value": 1},
        },
    }


def valid_mapmaking_v1_document() -> dict:
    document = valid_mapmaking_document()
    document["schema_version"] = "citlali-mapmaking-provenance-v1"
    document["observation"] = {"available": False}
    document.pop("observations")
    document.pop("coadd")
    document["realized"]["completed_observation_count"] = {
        "available": False
    }
    document["realized"]["completed_coadd_count"] = {"available": False}
    return document


def valid_coadd_document(enabled: bool = True) -> dict:
    return {
        "schema_version": "citlali-coadd-provenance-v1",
        "initialized": True,
        "requested": {"enabled": enabled},
        "effective": {
            "config": {"enabled": enabled},
            "resolution": {
                "mapmaking_enabled": True,
                "requested_enabled": enabled,
                "effective_enabled": enabled,
                "disabled_by_mapmaking": False,
            },
        },
        "realized": {
            "reduction_completed": True,
            "coadd_executed": enabled,
            "map_count": (
                {"available": True, "value": 3}
                if enabled
                else {"available": False}
            ),
            "required_map_write_count": (
                {"available": True, "value": 6}
                if enabled
                else {"available": False}
            ),
            "outputs_completed": enabled,
        },
    }


def valid_noise_document(enabled: bool = True) -> dict:
    requested_count = 10
    effective_count = requested_count if enabled else 0
    requested = {
        "enabled": enabled,
        "n_noise_maps": requested_count,
        "randomize_dets": False,
        "write_realizations": False,
        "products": {
            "enabled": False,
            "apply_empirical_weights": True,
        },
    }
    effective = {
        **requested,
        "n_noise_maps": effective_count,
        "products": dict(requested["products"]),
    }
    unavailable = {"available": False}
    realized = {
        "reduction_completed": True,
        "generation_executed": enabled,
        "noise_maps_per_scientific_map": (
            {"available": True, "value": effective_count}
            if enabled else dict(unavailable)
        ),
        "observation_scientific_map_count": (
            {"available": True, "value": 6}
            if enabled else dict(unavailable)
        ),
        "observation_noise_realization_count": (
            {"available": True, "value": 60}
            if enabled else dict(unavailable)
        ),
        "coadd_scientific_map_count": (
            {"available": True, "value": 3}
            if enabled else dict(unavailable)
        ),
        "coadd_noise_realization_count": (
            {"available": True, "value": 30}
            if enabled else dict(unavailable)
        ),
        "total_noise_realization_count": (
            {"available": True, "value": 90}
            if enabled else dict(unavailable)
        ),
        "empirical_product_map_count": (
            {"available": True, "value": 0}
            if enabled else dict(unavailable)
        ),
        "realization_image_write_count": (
            {"available": True, "value": 0}
            if enabled else dict(unavailable)
        ),
        "outputs_completed": enabled,
    }
    return {
        "schema_version": "citlali-noise-products-provenance-v1",
        "initialized": True,
        "requested": requested,
        "effective": {
            "config": effective,
            "resolution": {
                "mapmaking_enabled": True,
                "requested_enabled": enabled,
                "effective_enabled": enabled,
                "disabled_by_mapmaking": False,
                "requested_n_noise_maps": requested_count,
                "effective_n_noise_maps": effective_count,
                "count_zeroed_while_disabled": not enabled,
                "randomization": {
                    "engine": "boost::random::mt19937",
                    "seed": 5489,
                    "seed_policy": "fixed_internal_default",
                    "generator_scope": "reduction_pipeline_invocation",
                },
            },
        },
        "realized": realized,
    }


class ProvenanceAuditTest(unittest.TestCase):
    def test_accepts_complete_mapmaking_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            (redu / "mapmaking_provenance.yaml").write_text(
                yaml.safe_dump(valid_mapmaking_document(), sort_keys=False),
                encoding="utf-8",
            )

            mapmaking = audit.audit_provenance_sidecars(
                redu, require_mapmaking=True
            )["mapmaking"]

            self.assertTrue(mapmaking["present"])
            self.assertTrue(mapmaking["required"])
            self.assertTrue(mapmaking["valid"])
            self.assertEqual(len(mapmaking["sha256"]), 64)

    def test_accepts_historical_mapmaking_v1_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            (redu / "mapmaking_provenance.yaml").write_text(
                yaml.safe_dump(
                    valid_mapmaking_v1_document(), sort_keys=False
                ),
                encoding="utf-8",
            )

            mapmaking = audit.audit_provenance_sidecars(
                redu, require_mapmaking=True
            )["mapmaking"]

            self.assertTrue(mapmaking["valid"])
            self.assertEqual(
                mapmaking["schema_version"],
                "citlali-mapmaking-provenance-v1",
            )

    def test_rejects_missing_required_mapmaking_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records = audit.audit_provenance_sidecars(
                Path(directory), require_mapmaking=True
            )

            self.assertFalse(records["mapmaking"]["valid"])
            self.assertFalse(audit.provenance_ok({"provenance": records}))

    def test_rejects_inconsistent_mapmaking_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            document = valid_mapmaking_document()
            document["effective"]["config"]["grouping"] = "detector"
            (redu / "mapmaking_provenance.yaml").write_text(
                yaml.safe_dump(document, sort_keys=False),
                encoding="utf-8",
            )

            mapmaking = audit.audit_provenance_sidecars(
                redu, require_mapmaking=True
            )["mapmaking"]

            self.assertFalse(mapmaking["valid"])
            self.assertEqual(
                mapmaking["files"][0]["semantic_errors"],
                [
                    "grouping resolution does not match effective config",
                ],
            )

    def test_rejects_incomplete_mapmaking_reduction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            document = valid_mapmaking_document()
            document["realized"]["reduction_completed"] = False
            (redu / "mapmaking_provenance.yaml").write_text(
                yaml.safe_dump(document, sort_keys=False),
                encoding="utf-8",
            )

            mapmaking = audit.audit_provenance_sidecars(
                redu, require_mapmaking=True
            )["mapmaking"]

            self.assertFalse(mapmaking["valid"])
            self.assertEqual(
                mapmaking["files"][0]["semantic_errors"],
                ["mapmaking reduction is not complete"],
            )

    def test_rejects_inconsistent_mapmaking_cardinality(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            document = valid_mapmaking_document()
            document["realized"]["completed_observation_count"]["value"] = 1
            (redu / "mapmaking_provenance.yaml").write_text(
                yaml.safe_dump(document, sort_keys=False),
                encoding="utf-8",
            )

            mapmaking = audit.audit_provenance_sidecars(
                redu, require_mapmaking=True
            )["mapmaking"]

            self.assertFalse(mapmaking["valid"])
            self.assertIn(
                "completed observation count does not match observations",
                mapmaking["files"][0]["semantic_errors"],
            )

    def test_accepts_digit_string_mapmaking_obsnums(self) -> None:
        document = valid_mapmaking_document()
        document["observations"][0]["obsnum"] = "00152389"

        self.assertEqual(
            audit.mapmaking_cardinality_semantic_errors(document), []
        )

    def test_rejects_duplicate_mapmaking_numeric_identity(self) -> None:
        document = valid_mapmaking_document()
        document["observations"][1]["obsnum"] = "0152389"

        self.assertIn(
            "duplicate mapmaking obsnum: 152389",
            audit.mapmaking_cardinality_semantic_errors(document),
        )

    def test_accepts_complete_coadd_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            (redu / "coadd_provenance.yaml").write_text(
                yaml.safe_dump(valid_coadd_document(), sort_keys=False),
                encoding="utf-8",
            )

            coadd = audit.audit_provenance_sidecars(
                redu, require_coadd=True
            )["coadd"]

            self.assertTrue(coadd["present"])
            self.assertTrue(coadd["required"])
            self.assertTrue(coadd["valid"])

    def test_accepts_effectively_disabled_coadd_provenance(self) -> None:
        document = valid_coadd_document(enabled=False)

        self.assertEqual(
            audit.coadd_provenance_semantic_errors(document), []
        )

    def test_accepts_complete_noise_products_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            (redu / "noise_products_provenance.yaml").write_text(
                yaml.safe_dump(valid_noise_document(), sort_keys=False),
                encoding="utf-8",
            )

            noise = audit.audit_provenance_sidecars(
                redu, require_noise_products=True
            )["noise_products"]

            self.assertTrue(noise["present"])
            self.assertTrue(noise["required"])
            self.assertTrue(noise["valid"])

    def test_accepts_effectively_disabled_noise_products(self) -> None:
        self.assertEqual(
            audit.noise_provenance_semantic_errors(
                valid_noise_document(enabled=False)
            ),
            [],
        )

    def test_rejects_missing_required_noise_products_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records = audit.audit_provenance_sidecars(
                Path(directory), require_noise_products=True
            )

            self.assertFalse(records["noise_products"]["valid"])

    def test_rejects_inconsistent_noise_activation(self) -> None:
        document = valid_noise_document()
        document["effective"]["config"]["enabled"] = False

        self.assertIn(
            "noise effective activation resolution is inconsistent",
            audit.noise_provenance_semantic_errors(document),
        )

    def test_rejects_inconsistent_noise_realization_count(self) -> None:
        document = valid_noise_document()
        document["realized"]["total_noise_realization_count"]["value"] = 89

        self.assertIn(
            "noise total realization count is inconsistent",
            audit.noise_provenance_semantic_errors(document),
        )

    def test_rejects_noise_rng_identity_drift(self) -> None:
        document = valid_noise_document()
        document["effective"]["resolution"]["randomization"]["seed"] = 1

        self.assertIn(
            "noise randomization identity is inconsistent",
            audit.noise_provenance_semantic_errors(document),
        )

    def test_rejects_noise_mapmaking_cardinality_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            mapmaking = valid_mapmaking_document()
            noise = valid_noise_document()
            noise["realized"]["coadd_scientific_map_count"]["value"] = 6
            noise["realized"]["coadd_noise_realization_count"]["value"] = 60
            noise["realized"]["total_noise_realization_count"]["value"] = 120
            (redu / "mapmaking_provenance.yaml").write_text(
                yaml.safe_dump(mapmaking, sort_keys=False),
                encoding="utf-8",
            )
            (redu / "noise_products_provenance.yaml").write_text(
                yaml.safe_dump(noise, sort_keys=False),
                encoding="utf-8",
            )

            records = audit.audit_provenance_sidecars(
                redu,
                require_mapmaking=True,
                require_noise_products=True,
            )

            self.assertTrue(records["mapmaking"]["valid"])
            self.assertFalse(records["noise_products"]["valid"])
            self.assertEqual(
                records["noise_products"]["cross_check_errors"],
                ["noise coadd map count differs from mapmaking provenance"],
            )

    def test_rejects_inconsistent_coadd_activation(self) -> None:
        document = valid_coadd_document()
        document["effective"]["config"]["enabled"] = False

        self.assertIn(
            "coadd effective resolution is inconsistent",
            audit.coadd_provenance_semantic_errors(document),
        )

    def test_rejects_non_boolean_coadd_resolution(self) -> None:
        document = valid_coadd_document()
        document["effective"]["resolution"]["mapmaking_enabled"] = 1

        self.assertIn(
            "coadd resolution values must be boolean",
            audit.coadd_provenance_semantic_errors(document),
        )

    def test_rejects_coadd_mapmaking_cardinality_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            mapmaking = valid_mapmaking_document()
            coadd = valid_coadd_document()
            coadd["realized"]["map_count"]["value"] = 6
            (redu / "mapmaking_provenance.yaml").write_text(
                yaml.safe_dump(mapmaking, sort_keys=False),
                encoding="utf-8",
            )
            (redu / "coadd_provenance.yaml").write_text(
                yaml.safe_dump(coadd, sort_keys=False),
                encoding="utf-8",
            )

            records = audit.audit_provenance_sidecars(
                redu, require_mapmaking=True, require_coadd=True
            )

            self.assertTrue(records["mapmaking"]["valid"])
            self.assertFalse(records["coadd"]["valid"])
            self.assertEqual(
                records["coadd"]["cross_check_errors"],
                ["coadd cardinality differs from mapmaking provenance"],
            )

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
                (observation / "timestream_output_provenance.yaml").write_text(
                    yaml.safe_dump(valid_output_document(), sort_keys=False),
                    encoding="utf-8",
                )

            raw = audit.audit_provenance_sidecars(
                redu, require_raw=True
            )["raw_timestream"]

            self.assertTrue(raw["required"])
            self.assertTrue(raw["valid"])
            self.assertEqual(raw["count"], 2)
            self.assertTrue(raw["observation_coverage_ok"])

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
            (redu / "timestream_output_provenance.yaml").write_text(
                yaml.safe_dump(valid_output_document(), sort_keys=False),
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

    def test_rejects_missing_raw_observation_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            for obsnum in ("1", "2"):
                observation = redu / obsnum
                observation.mkdir()
                (observation / "timestream_output_provenance.yaml").write_text(
                    yaml.safe_dump(valid_output_document(), sort_keys=False),
                    encoding="utf-8",
                )
            (redu / "1" / "raw_timestream_provenance.yaml").write_text(
                yaml.safe_dump(valid_raw_document(), sort_keys=False),
                encoding="utf-8",
            )

            raw = audit.audit_provenance_sidecars(
                redu, require_raw=True
            )["raw_timestream"]

            self.assertFalse(raw["valid"])
            self.assertFalse(raw["observation_coverage_ok"])
            self.assertEqual(raw["missing_observation_dirs"], [str(redu / "2")])

    def test_rejects_raw_scan_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            (redu / "raw_timestream_provenance.yaml").write_text(
                yaml.safe_dump(valid_raw_document(), sort_keys=False),
                encoding="utf-8",
            )
            output = valid_output_document()
            output["realized"]["n_scans"] = 5
            (redu / "timestream_output_provenance.yaml").write_text(
                yaml.safe_dump(output, sort_keys=False),
                encoding="utf-8",
            )

            raw = audit.audit_provenance_sidecars(
                redu, require_raw=True
            )["raw_timestream"]

            self.assertFalse(raw["valid"])
            self.assertEqual(
                raw["files"][0]["semantic_errors"],
                [
                    "completed scan count does not match "
                    "timestream-output provenance"
                ],
            )

    def test_rejects_inconsistent_raw_sample_rates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            redu = Path(directory)
            document = valid_raw_document()
            document["observation"]["value"]["effective_sample_rate_hz"][
                "value"
            ] = 40.0
            (redu / "raw_timestream_provenance.yaml").write_text(
                yaml.safe_dump(document, sort_keys=False),
                encoding="utf-8",
            )
            (redu / "timestream_output_provenance.yaml").write_text(
                yaml.safe_dump(valid_output_document(), sort_keys=False),
                encoding="utf-8",
            )

            raw = audit.audit_provenance_sidecars(
                redu, require_raw=True
            )["raw_timestream"]

            self.assertFalse(raw["valid"])
            self.assertIn(
                "effective sample rate does not match native "
                "rate/downsample factor",
                raw["files"][0]["semantic_errors"],
            )

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
