import csv
import json
import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import yaml
from netCDF4 import Dataset

from tools.diagnostics import generate_sci_align_001_3c273_slurm_array as scheduler
from tools.diagnostics import inventory_sci_align_001_3c273_corpus as inventory


class SyntheticCorpus:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.project = root / "project"
        self.reduced = self.project / "reduced"
        self.config_dir = self.project / "config"
        self.raw_root = root / "raw inputs"
        self.reduced.mkdir(parents=True)
        self.config_dir.mkdir(parents=True)
        self.raw_root.mkdir(parents=True)
        self.obsnums: set[int] = set()

    @staticmethod
    def _string_variable(dataset: Dataset, name: str, value: str) -> None:
        dimension = name.lower().replace(".", "_") + "_dim"
        dataset.createDimension(dimension, 1)
        variable = dataset.createVariable(name, str, (dimension,))
        variable[:] = np.asarray([value], dtype=object)

    def make_telescope(self, obsnum: int, source: str) -> Path:
        path = self.project / f"tel_toltec_2026-01-01_{obsnum}_00_0002.nc"
        with Dataset(path, "w") as dataset:
            dataset.createDimension("time", 4)
            self._string_variable(dataset, "Header.Source.SourceName", source)
            obs = dataset.createVariable("Header.Dcs.ObsNum", "i4")
            obs.assignValue(obsnum)
            for name, values in (
                ("Data.TelescopeBackend.TelTime", [0.0, 1.0, 2.0, 3.0]),
                ("Data.TelescopeBackend.TelAzAct", [0.0, 0.1, 0.2, 0.3]),
                ("Data.TelescopeBackend.TelElAct", [0.5, 0.5, 0.5, 0.5]),
                ("Data.TelescopeBackend.Hold", [0.0, 0.0, 0.0, 0.0]),
            ):
                variable = dataset.createVariable(name, "f8", ("time",))
                variable[:] = values
        return path

    def make_raw(
        self,
        obsnum: int,
        network: int,
        source: str,
        *,
        t0_values: tuple[int, ...] = (1700000000, 1700000000, 1700000000),
    ) -> Path:
        path = (
            self.raw_root
            / f"toltec{network}_{obsnum}_000_0002_2026_01_01_00_00_00.nc"
        )
        with Dataset(path, "w") as dataset:
            dataset.createDimension("time", len(t0_values))
            dataset.createDimension("timestamp_field", 6)
            self._string_variable(dataset, "Header.Source.SourceName", source)
            for name, dtype, value in (
                ("Header.Toltec.RoachIndex", "i4", network),
                ("Header.Toltec.FpgaFreq", "f8", 256000000.0),
                ("Header.Toltec.AccumLen", "i8", 2097152),
            ):
                variable = dataset.createVariable(name, dtype)
                variable.assignValue(value)
            timestamps = dataset.createVariable(
                "Data.Toltec.Ts", "i8", ("time", "timestamp_field")
            )
            rows = np.zeros((len(t0_values), 6), dtype=np.int64)
            rows[:, 0] = t0_values
            rows[:, 1] = np.arange(len(t0_values))
            rows[:, 2] = 100 + np.arange(len(t0_values))
            rows[:, 3] = 200 + np.arange(len(t0_values))
            rows[:, 4] = 90
            rows[:, 5] = 123456789
            timestamps[:] = rows
        return path

    def make_detector_tod(
        self, result: Path, obsnum: int, source: str, networks: tuple[int, ...]
    ) -> Path:
        directory = result / "raw/source_crossing_tod"
        directory.mkdir(parents=True)
        path = directory / f"beammap_{obsnum}_ptc_detector_tod.nc"
        with Dataset(path, "w") as dataset:
            dataset.createDimension("n_dets", len(networks))
            dataset.createDimension("slots", 1)
            dataset.createDimension("samples", 2)
            self._string_variable(dataset, "SOURCE", source)
            obs = dataset.createVariable("obsnum", "i4")
            obs.assignValue(obsnum)
            uid = dataset.createVariable("detector_tod_uid", "i4", ("n_dets",))
            network = dataset.createVariable(
                "detector_tod_network", "i4", ("n_dets",)
            )
            slot = dataset.createVariable(
                "detector_tod_slot_kind", "i4", ("n_dets", "slots")
            )
            count = dataset.createVariable(
                "detector_tod_n_samples", "i4", ("n_dets", "slots")
            )
            signal = dataset.createVariable(
                "signal", "f4", ("n_dets", "slots", "samples")
            )
            flags = dataset.createVariable(
                "flags", "i1", ("n_dets", "slots", "samples")
            )
            uid[:] = np.arange(len(networks))
            network[:] = networks
            slot[:] = 2
            count[:] = 2
            signal[:] = 0.0
            flags[:] = 0
        return path

    def make_candidate(
        self,
        obsnum: int,
        *,
        reduction_id: str = "redu00",
        detector_source: str = "3c273",
        telescope_source: str = "3c273",
        raw_source: str = "3c273",
        networks: tuple[int, ...] = (0, 1),
        raw_networks: tuple[int, ...] | None = None,
        include_scan_registry: bool = True,
        version: str = "v4.0.0-1-gabcdef12",
        t0_by_network: dict[int, tuple[int, ...]] | None = None,
    ) -> Path:
        self.obsnums.add(obsnum)
        result = self.reduced / reduction_id / str(obsnum)
        result.mkdir(parents=True)
        detector_tod = self.make_detector_tod(
            result, obsnum, detector_source, networks
        )
        del detector_tod
        telescope = self.make_telescope(obsnum, telescope_source)
        raw_paths = []
        for network in raw_networks if raw_networks is not None else networks:
            raw_paths.append(
                (
                    network,
                    self.make_raw(
                        obsnum,
                        network,
                        raw_source,
                        t0_values=(t0_by_network or {}).get(
                            network, (1700000000, 1700000000, 1700000000)
                        ),
                    ),
                )
            )
        config = {
            "inputs": [
                {
                    "data_items": [
                        {"filepath": str(telescope), "meta": {"interface": "lmt"}},
                        *[
                            {
                                "filepath": str(path),
                                "meta": {"interface": f"toltec{network}"},
                            }
                            for network, path in raw_paths
                        ],
                    ]
                }
            ]
        }
        config_path = self.config_dir / f"citlali_o{obsnum}_0_2_c1.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
        provenance = {
            "schema_version": "citlali-timestream-output-provenance-v2",
            "realized": {
                "sci_align_scan_plan": {
                    "records": [{"stable_id": 0}] if include_scan_registry else []
                },
                "sci_align_alignment": {
                    "producer": {"source_application_sha": "1" * 40}
                },
            },
        }
        (result / "timestream_output_provenance.yaml").write_text(
            yaml.safe_dump(provenance, sort_keys=True), encoding="utf-8"
        )
        (result / "index.yaml").write_text(
            yaml.safe_dump({"citlali_version": [version]}, sort_keys=True),
            encoding="utf-8",
        )
        apt = result / "raw" / f"apt_{obsnum}_citlali.ecsv"
        apt.parent.mkdir(exist_ok=True)
        apt.write_text("# synthetic apt\n", encoding="utf-8")
        return result

    def allowlist(self) -> Path:
        path = self.root / "authoritative_obsnums.json"
        path.write_text(json.dumps({
            "schema_version": inventory.ALLOWLIST_SCHEMA,
            "corpus_id": "synthetic-authoritative-corpus",
            "selection_authority": "test fixture",
            "obsnums": sorted(self.obsnums),
        }, indent=2) + "\n", encoding="utf-8")
        return path

    def run_inventory(
        self,
        output: Path,
        *,
        source_regex: str = inventory.DEFAULT_SOURCE_REGEX,
        threshold: int = inventory.DEFAULT_LARGE_FILE_THRESHOLD,
    ) -> dict:
        document, _ = inventory.inventory(
            [self.project],
            [self.raw_root],
            output=output,
            source_regex=source_regex,
            obsnum_allowlist=self.allowlist(),
            threshold=threshold,
        )
        return document


class InventoryTests(unittest.TestCase):
    def test_aliases_fullmatch_and_conflicting_identity_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticCorpus(Path(temporary))
            fixture.make_candidate(
                100001,
                detector_source="3C-273",
                telescope_source="3c-273",
                raw_source="3c-273",
            )
            document = fixture.run_inventory(Path(temporary) / "out")
            row = document["rows"][0]
            self.assertEqual(row["source_status"], "target")
            self.assertTrue(row["core_eligible"])

        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticCorpus(Path(temporary))
            fixture.make_candidate(100002, telescope_source="mars")
            row = fixture.run_inventory(Path(temporary) / "out")["rows"][0]
            self.assertEqual(row["source_status"], "ambiguous")
            self.assertFalse(row["core_eligible"])
            self.assertIn("source_identity_ambiguous", row["exclusion_reasons"])

    def test_duplicate_group_is_observation_identity_and_redu01_is_canonical(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticCorpus(Path(temporary))
            fixture.make_candidate(100010, reduction_id="redu00")
            fixture.make_candidate(100010, reduction_id="redu01")
            rows = fixture.run_inventory(Path(temporary) / "out")["rows"]
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["duplicate_group_id"] for row in rows}, {"obs:100010"})
            self.assertEqual(len({row["provenance_signature_id"] for row in rows}), 1)
            canonical = [row for row in rows if row["canonical_proposal"]]
            self.assertEqual(len(canonical), 1)
            self.assertEqual(canonical[0]["reduction_id"], "redu01")
            self.assertFalse(any(row["owner_selection_required"] for row in rows))

    def test_different_config_or_software_authority_is_never_auto_selected(self):
        rows = [
            {
                "observation_number": 42,
                "candidate_id": "a",
                "core_eligible": True,
                "canonical_quality_score": 10,
                "config_sha256": "a" * 64,
                "software_sha": "1" * 8,
                "software_version": "one",
                "canonical_proposal": False,
                "canonical_proposal_rule": None,
                "owner_selection_required": False,
            },
            {
                "observation_number": 42,
                "candidate_id": "b",
                "core_eligible": True,
                "canonical_quality_score": 5,
                "config_sha256": "b" * 64,
                "software_sha": "2" * 8,
                "software_version": "two",
                "canonical_proposal": False,
                "canonical_proposal_rule": None,
                "owner_selection_required": False,
            },
        ]
        inventory.apply_duplicate_policy(rows)
        self.assertFalse(any(row["canonical_proposal"] for row in rows))
        self.assertTrue(all(row["owner_selection_required"] for row in rows))

    def test_core_enhanced_missing_network_and_t0_field_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticCorpus(Path(temporary))
            fixture.make_candidate(100020, networks=(0, 1), raw_networks=(0,))
            row = fixture.run_inventory(Path(temporary) / "out")["rows"][0]
            self.assertTrue(row["core_eligible"])
            self.assertFalse(row["enhanced_eligible"])
            self.assertEqual(row["missing_raw_networks"], [1])
            self.assertEqual(row["network_t0_status"], "incomplete")
            fields = row["timestamp_counter_fields"][0]["fields"]
            self.assertEqual(fields["clock_time_integer_t0"]["column"], 0)
            self.assertEqual(fields["clock_time_integer_t0"]["distinct_value_count"], 1)
            self.assertEqual(fields["clock_time_nanosec"]["column"], 5)
            self.assertTrue(fields["pps_count"]["available"])
            self.assertTrue(fields["clock_count"]["available"])
            self.assertTrue(fields["fpga_freq"]["available"])
            self.assertTrue(fields["accum_len"]["available"])
            self.assertFalse(row["timestamp_semantics"]["common_phase_inferred"])
            self.assertEqual(
                row["timestamp_semantics"]["fpga_association_status"], "unproved"
            )
            self.assertEqual(
                row["timestamp_semantics"]["ntp_millisecond_error_hypothesis"],
                "strongly_disfavored",
            )

    def test_nonconstant_integer_t0_is_ambiguous_and_not_session_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticCorpus(Path(temporary))
            fixture.make_candidate(
                100021,
                networks=(0,),
                t0_by_network={0: (1700000000, 1700000001, 1700000000)},
            )
            row = fixture.run_inventory(Path(temporary) / "out")["rows"][0]
            self.assertEqual(row["network_t0_status"], "ambiguous")
            self.assertIsNone(row["network_t0_vector_sha256"])
            self.assertEqual(row["session_status"], "date_group_fallback")
            field = row["timestamp_counter_fields"][0]["fields"]["clock_time_integer_t0"]
            self.assertEqual(field["distinct_value_count"], 2)
            self.assertEqual(field["distinct_values"], [1700000000, 1700000001])

    def test_missing_scan_registry_is_ineligible_with_exact_reason(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticCorpus(Path(temporary))
            fixture.make_candidate(100030, include_scan_registry=False)
            row = fixture.run_inventory(Path(temporary) / "out")["rows"][0]
            self.assertFalse(row["core_eligible"])
            self.assertEqual(row["eligibility"], "ineligible")
            self.assertIn("scan_registry_missing", row["exclusion_reasons"])

    def test_required_product_identity_is_unique_and_output_apt_is_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticCorpus(Path(temporary))
            result = fixture.make_candidate(100031)
            (result / "raw/apt_alternate_citlali.ecsv").write_text(
                "# ambiguous synthetic apt\n", encoding="utf-8"
            )
            row = fixture.run_inventory(Path(temporary) / "out")["rows"][0]
            self.assertFalse(row["core_eligible"])
            self.assertIn(
                "output_apt_identity_ambiguous", row["exclusion_reasons"]
            )
            self.assertEqual(
                row["product_identity_resolutions"]["output_apt"]["status"],
                "ambiguous",
            )

        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticCorpus(Path(temporary))
            result = fixture.make_candidate(100032)
            (result / "raw/apt_100032_citlali.ecsv").unlink()
            row = fixture.run_inventory(Path(temporary) / "out")["rows"][0]
            self.assertFalse(row["core_eligible"])
            self.assertIn("output_apt_identity_missing", row["exclusion_reasons"])

    def test_config_source_manifest_selects_checksum_verified_reduction_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticCorpus(Path(temporary))
            result = fixture.make_candidate(100034)
            source = fixture.config_dir / "citlali_o100034_0_2_c1.yaml"
            copied = result.parent / source.name
            copied.write_bytes(source.read_bytes())
            manifest = {
                "schema_version": "citlali-config-source-manifest-v1",
                "sources": [
                    {
                        "copied_filename": copied.name,
                        "source_path": str(source),
                        "sha256": inventory.sha256_file(source),
                    }
                ],
            }
            (result.parent / "config_source_manifest.yaml").write_text(
                yaml.safe_dump(manifest, sort_keys=True), encoding="utf-8"
            )
            resolution = inventory.find_config(result, 100034)
            self.assertEqual(resolution.status, "unique")
            self.assertEqual(resolution.path, copied.resolve())

    def test_configured_raw_interface_must_match_producer_network_header(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticCorpus(Path(temporary))
            fixture.make_candidate(100033, networks=(0,))
            config_path = fixture.config_dir / "citlali_o100033_0_2_c1.yaml"
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config["inputs"][0]["data_items"][1]["meta"]["interface"] = "toltec1"
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
            )
            row = fixture.run_inventory(Path(temporary) / "out")["rows"][0]
            self.assertTrue(row["core_eligible"])
            self.assertFalse(row["enhanced_eligible"])
            self.assertIn(
                "configured_interface_header_conflict_toltec1_toltec0",
                row["raw_linkage_reasons"],
            )

    def test_freeze_only_manifest_binds_inventory_and_aliases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = SyntheticCorpus(root)
            fixture.make_candidate(100040)
            output = root / "output"
            document, cache = inventory.inventory(
                [fixture.project], [fixture.raw_root], output=output,
                source_regex=inventory.DEFAULT_SOURCE_REGEX,
                obsnum_allowlist=fixture.allowlist(),
            )
            inventory.emit(
                document, cache, output, commands=["inspect"],
                obsnum_allowlist=fixture.allowlist(),
            )
            selection = json.loads((output / "selection_template.json").read_text())
            selection["rows"][0]["selected"] = True
            selection_path = output / "owner_selection.json"
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            self.assertEqual(
                inventory.main(
                    [
                        "--inventory", str(output / "candidate_inventory.json"),
                        "--freeze-selection", str(selection_path),
                        "--output", str(output),
                    ]
                ),
                0,
            )
            selected = json.loads((output / "selected_manifest.json").read_text())
            row = selected["rows"][0]
            self.assertEqual(row["map_id"], row["candidate_id"])
            self.assertEqual(row["obsnum"], row["observation_number"])
            self.assertEqual(row["analysis_role"], "primary")
            self.assertEqual(
                selected["owner_selection_sha256"],
                inventory.sha256_file(selection_path),
            )
            base = {key: value for key, value in selected.items() if key != "manifest_sha256"}
            self.assertEqual(selected["manifest_sha256"], inventory.digest_object(base))
            checksum_lines = (output / "SHA256SUMS").read_text().splitlines()
            checksums = {
                line.split("  ", 1)[1]: line.split("  ", 1)[0]
                for line in checksum_lines
            }
            self.assertIn("selected_manifest.json", checksums)
            self.assertIn("candidate_inventory.json", checksums)
            self.assertIn("owner_selection.json", checksums)
            for relative, digest in checksums.items():
                self.assertEqual(inventory.sha256_file(output / relative), digest)

            tampered = json.loads((output / "candidate_inventory.json").read_text())
            tampered["rows"][0]["observation_number"] += 1
            tampered_path = output / "tampered_inventory.json"
            tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(inventory.InventoryError, "digest mismatch"):
                inventory.load_frozen_inventory(tampered_path)

            missing_digest = json.loads(json.dumps(selection))
            missing_digest.pop("source_inventory_sha256")
            missing_digest_path = output / "missing_digest_selection.json"
            missing_digest_path.write_text(json.dumps(missing_digest), encoding="utf-8")
            with self.assertRaisesRegex(
                inventory.InventoryError, "no source_inventory_sha256"
            ):
                inventory.freeze_selection(
                    document["rows"], document["inventory_sha256"], missing_digest_path,
                    obsnum_allowlist_sha256=document["obsnum_allowlist"]["sha256"],
                    obsnum_allowlist_schema_version=inventory.ALLOWLIST_SCHEMA,
                    obsnum_allowlist_filename=document["obsnum_allowlist"]["filename"],
                )

            truncated = json.loads(json.dumps(selection))
            truncated["rows"] = []
            truncated_path = output / "truncated_selection.json"
            truncated_path.write_text(json.dumps(truncated), encoding="utf-8")
            with self.assertRaisesRegex(
                inventory.InventoryError, "preserve every inventory candidate"
            ):
                inventory.freeze_selection(
                    document["rows"], document["inventory_sha256"], truncated_path,
                    obsnum_allowlist_sha256=document["obsnum_allowlist"]["sha256"],
                    obsnum_allowlist_schema_version=inventory.ALLOWLIST_SCHEMA,
                    obsnum_allowlist_filename=document["obsnum_allowlist"]["filename"],
                )

            wrong_observation = json.loads(json.dumps(selection))
            wrong_observation["rows"][0]["observation_number"] += 1
            wrong_observation_path = output / "wrong_observation_selection.json"
            wrong_observation_path.write_text(
                json.dumps(wrong_observation), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                inventory.InventoryError, "candidate/observation identity mismatch"
            ):
                inventory.freeze_selection(
                    document["rows"],
                    document["inventory_sha256"],
                    wrong_observation_path,
                    obsnum_allowlist_sha256=document["obsnum_allowlist"]["sha256"],
                    obsnum_allowlist_schema_version=inventory.ALLOWLIST_SCHEMA,
                    obsnum_allowlist_filename=document["obsnum_allowlist"]["filename"],
                )

            wrong_suffix = output / "owner_selection.txt"
            wrong_suffix.write_text(json.dumps(selection), encoding="utf-8")
            with self.assertRaisesRegex(inventory.InventoryError, r"\.csv or \.json"):
                inventory.read_selection(wrong_suffix)

            array_json = output / "selection_array.json"
            array_json.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(inventory.InventoryError, "must be an object"):
                inventory.read_selection(array_json)

    def test_freeze_keeps_nonprimary_eligible_duplicates_as_sensitivity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = SyntheticCorpus(root)
            fixture.make_candidate(100045, reduction_id="redu00")
            fixture.make_candidate(100045, reduction_id="redu01")
            document = fixture.run_inventory(root / "output")
            template = inventory.selection_template(
                document["rows"], document["inventory_sha256"]
            )
            selection_path = root / "owner_selection.json"
            selection_path.write_text(json.dumps(template), encoding="utf-8")
            selected = inventory.freeze_selection(
                document["rows"], document["inventory_sha256"], selection_path,
                obsnum_allowlist_sha256=document["obsnum_allowlist"]["sha256"],
                obsnum_allowlist_schema_version=inventory.ALLOWLIST_SCHEMA,
                obsnum_allowlist_filename=document["obsnum_allowlist"]["filename"],
            )
            self.assertEqual(len(selected["rows"]), 2)
            self.assertEqual(
                [row["analysis_role"] for row in selected["rows"]],
                ["primary", "sensitivity"],
            )
            self.assertEqual(
                {row["observation_number"] for row in selected["rows"]}, {100045}
            )

    def test_deterministic_replay_and_output_source_overlap_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = SyntheticCorpus(root)
            fixture.make_candidate(100050)
            first = fixture.run_inventory(root / "output-one")
            second = fixture.run_inventory(root / "output-two")
            self.assertEqual(inventory.canonical_json(first), inventory.canonical_json(second))
            with self.assertRaisesRegex(inventory.InventoryError, "overlaps"):
                inventory.inventory(
                    [fixture.project],
                    [fixture.raw_root],
                    output=fixture.project / "diagnostics",
                    source_regex=inventory.DEFAULT_SOURCE_REGEX,
                    obsnum_allowlist=fixture.allowlist(),
                )

    def test_authoritative_allowlist_and_excluded_run_root_are_inventory_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = SyntheticCorpus(root)
            fixture.make_candidate(100070)
            fixture.make_candidate(100071)
            allowlist = root / "allowlist.json"
            allowlist.write_text(json.dumps({
                "schema_version": inventory.ALLOWLIST_SCHEMA,
                "corpus_id": "test-authoritative-only",
                "selection_authority": "test",
                "obsnums": [100070, 100072],
            }), encoding="utf-8")
            run_root = fixture.project / "sci_align_001_corpus_run_2026-08-03"
            run_root.mkdir()
            document, _ = inventory.inventory(
                [fixture.project], [fixture.raw_root],
                output=run_root / "output", source_regex=inventory.DEFAULT_SOURCE_REGEX,
                obsnum_allowlist=allowlist, excluded_paths=[run_root],
            )
            self.assertEqual([row["observation_number"] for row in document["rows"]], [100070])
            self.assertEqual(
                [row["observation_number"] for row in document["out_of_scope_3c273_rows"]],
                [100071],
            )
            status = {row["observation_number"]: row["status"] for row in document["authoritative_obsnum_status"]}
            self.assertEqual(status[100072], "no_retained_reduction_found")
            self.assertEqual(document["excluded_paths"], [str(run_root.resolve())])

    def test_large_digest_status_and_physical_file_deduplication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "large.bin"
            path.write_bytes(b"0123456789")
            alias = root / "alias.bin"
            os.link(path, alias)
            cache = inventory.DigestCache(
                root / "cache.json", threshold=1, hash_large=False
            )
            first = cache.digest(path)
            second = cache.digest(alias)
            self.assertEqual(first.status, "not_hashed_large")
            self.assertIsNone(first.sha256)
            self.assertEqual(first, second)
            self.assertEqual(len(cache._memo), 1)


class SchedulerTests(unittest.TestCase):
    def selected_manifest(self, root: Path) -> Path:
        source = root / "source reduction" / "redu00" / "100060"
        source.mkdir(parents=True)
        row = {
            "candidate_id": "map:special-id",
            "map_id": "map:special-id",
            "observation_number": 100060,
            "obsnum": 100060,
            "analysis_role": "primary",
            "duplicate_group_id": "obs:100060",
            "session_id": "date:2026-01-01",
            "reduction_path": str(source),
            "reduction_run_path": str(source.parent),
            "project_path": str(source.parents[1]),
            "config_path": str(source / "config with spaces.yaml"),
            "detector_tod_path": str(source / "detector.nc"),
            "telescope_path": str(source / "telescope.nc"),
            "output_apt_path": str(source / "apt.ecsv"),
            "provenance_path": str(source / "provenance.yaml"),
            "raw_files": [],
            "core_eligible": True,
            "enhanced_eligible": False,
        }
        allowlist = root / "authoritative_obsnums.json"
        allowlist.write_text(json.dumps({
            "schema_version": inventory.ALLOWLIST_SCHEMA,
            "corpus_id": "scheduler-test",
            "selection_authority": "test",
            "obsnums": [100060],
        }), encoding="utf-8")
        base = {
            "schema_version": scheduler.SELECTED_MANIFEST_SCHEMA,
            "source_inventory_sha256": "a" * 64,
            "owner_selection_sha256": "b" * 64,
            "owner_selection_format": "csv",
            "obsnum_allowlist_sha256": scheduler.sha256_file(allowlist),
            "obsnum_allowlist_schema_version": inventory.ALLOWLIST_SCHEMA,
            "obsnum_allowlist_filename": allowlist.name,
            "rows": [row],
        }
        document = {**base, "manifest_sha256": scheduler.digest_object(base)}
        path = root / "selected manifest.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def test_slurm_argv_quoting_and_explicit_resource_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = self.selected_manifest(root)
            manifest = scheduler.load_selected_manifest(manifest_path)
            python = root / "python with spaces"
            runner = root / "runner's script.py"
            output_root = root / "analysis output"
            rows = scheduler.command_rows(
                manifest,
                python=python,
                runner=runner,
                protocol=root / "protocol.json",
                selected_manifest=manifest_path,
                output_root=output_root,
                resume=True,
            )
            argv = json.loads(rows[0]["argv_json"])
            self.assertEqual(argv[0], str(python))
            self.assertEqual(argv[1], str(runner))
            self.assertEqual(
                argv[argv.index("--protocol") + 1], str(root / "protocol.json")
            )
            self.assertEqual(argv[-1], "--resume")
            options = scheduler.parse_sbatch_options(
                ["partition=debug", "time=01:00:00", "cpus-per-task=4"]
            )
            script = scheduler.render_script(
                command_table=root / "commands with spaces.csv",
                python=python,
                row_count=1,
                job_name="sci-align-test",
                array_concurrency=2,
                sbatch_options=options,
                command_table_sha256="c" * 64,
                selected_manifest_sha256="d" * 64,
                obsnum_allowlist=root / "authoritative_obsnums.json",
                obsnum_allowlist_sha256="f" * 64,
                protocol_sha256="e" * 64,
            )
            self.assertIn("#SBATCH --array=0-0%2", script)
            self.assertIn("#SBATCH --partition=debug", script)
            self.assertLess(
                script.index("#SBATCH --partition=debug"),
                script.index("set -euo pipefail"),
            )
            self.assertIn("subprocess.run(argv, check=True)", script)
            self.assertIn("export OPENBLAS_NUM_THREADS=1", script)
            self.assertIn("command-table SHA-256 changed", script)
            self.assertIn("selected-manifest SHA-256 changed", script)
            self.assertIn("analysis-protocol SHA-256 changed", script)
            default_script = scheduler.render_script(
                command_table=root / "commands.csv",
                python=python,
                row_count=1,
                job_name="sci-align-test",
                array_concurrency=None,
                sbatch_options=[],
                command_table_sha256="c" * 64,
                selected_manifest_sha256="d" * 64,
                obsnum_allowlist=root / "authoritative_obsnums.json",
                obsnum_allowlist_sha256="f" * 64,
                protocol_sha256="e" * 64,
            )
            self.assertIn("#SBATCH --partition=toltec-cpu", default_script)
            self.assertNotIn("--account", default_script)
            self.assertIn("#SBATCH --time=48:00:00", default_script)
            self.assertIn("#SBATCH --cpus-per-task=6", default_script)
            with self.assertRaises(scheduler.SchedulerError):
                scheduler.parse_sbatch_options(["partition=debug\n#SBATCH --account=bad"])
            with self.assertRaises(scheduler.SchedulerError):
                scheduler.parse_sbatch_options(["account=project"])

    def test_bare_python_command_resolves_from_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "python"
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            executable.chmod(0o755)
            with mock.patch.object(scheduler.shutil, "which", return_value=str(executable)):
                self.assertEqual(
                    scheduler.resolve_python_executable(Path("python")), executable
                )
            with mock.patch.object(scheduler.shutil, "which", return_value=None):
                with self.assertRaisesRegex(scheduler.SchedulerError, "not available"):
                    scheduler.resolve_python_executable(Path("python"))

    def test_scheduler_main_writes_json_argv_command_table(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.selected_manifest(root)
            script = root / "scheduler output" / "array.sh"
            table = root / "scheduler output" / "commands.csv"
            output = root / "analysis output"
            serial = root / "scheduler output" / "serial.sh"
            runner = root / "runner.py"
            runner.write_text("# synthetic\n", encoding="utf-8")
            with mock.patch.object(
                scheduler.shutil, "which", return_value=os.sys.executable
            ):
                self.assertEqual(
                    scheduler.main(
                        [
                            "--selected-manifest", str(manifest),
                            "--output-script", str(script),
                            "--serial-script", str(serial),
                            "--command-table", str(table),
                            "--output-root", str(output),
                            "--runner", str(runner),
                            "--python", "python",
                            "--resume",
                        ]
                    ),
                    0,
                )
            with table.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            argv = json.loads(rows[0]["argv_json"])
            self.assertEqual(argv[0], os.sys.executable)
            self.assertEqual(argv[argv.index("--candidate-id") + 1], "map:special-id")
            self.assertTrue(script.stat().st_mode & 0o100)
            self.assertTrue(serial.stat().st_mode & 0o100)
            script_text = script.read_text(encoding="utf-8")
            self.assertIn(
                f"expected_command_table_sha256={scheduler.sha256_file(table)}",
                script_text,
            )
            self.assertIn(
                f"expected_selected_manifest_sha256={scheduler.sha256_file(manifest)}",
                script_text,
            )
            self.assertIn("ObsNum allowlist SHA-256 changed", script_text)
            self.assertIn("subprocess.run(argv, check=True)", serial.read_text(encoding="utf-8"))
            with self.assertRaisesRegex(
                scheduler.SchedulerError, "overlaps control input"
            ):
                scheduler.main(
                    [
                        "--selected-manifest", str(manifest),
                        "--output-script", str(manifest),
                        "--command-table", str(root / "other-commands.csv"),
                        "--output-root", str(root / "other-output"),
                        "--runner", str(runner),
                        "--python", str(Path(os.sys.executable)),
                        "--dry-run",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
