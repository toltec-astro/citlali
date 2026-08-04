#!/usr/bin/env python3
"""Focused local tests for the ED2 compact-evidence producer."""

from __future__ import annotations

import importlib.util
import copy
from datetime import datetime, timedelta, timezone
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import jsonschema
import numpy as np
import netCDF4


PACKAGE = Path(__file__).resolve().parents[1]
SCRIPT = PACKAGE / "scripts" / "compact-evidence.py"
SPEC = importlib.util.spec_from_file_location("sci_map_compact_evidence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
compact = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compact
SPEC.loader.exec_module(compact)


class CompactEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="compact-evidence-test-")
        self.root = Path(self.temporary.name).resolve()
        self.governed_roots = [
            self.root / "point-source-project",
            self.root / "science-source-project",
            self.root / "CAP-POINT",
            self.root / "CAP-SCIENCE",
            self.root / "compact",
        ]
        for governed_root in self.governed_roots:
            governed_root.mkdir()
        self.source = self.root / "source.npz"
        self.metadata = compact.write_self_check_fixture(self.source)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def source_arrays(self) -> dict[str, np.ndarray]:
        with np.load(self.source, allow_pickle=False) as archive:
            return {name: np.asarray(archive[name]).copy() for name in archive.files}

    def write_variant(self, name: str, arrays: dict[str, np.ndarray]) -> Path:
        path = self.root / name
        compact.deterministic_npz(path, arrays)
        return path

    def update_metadata(self, arrays: dict[str, np.ndarray], update) -> None:
        metadata = json.loads(str(np.asarray(arrays["metadata_json"]).item()))
        update(metadata)
        arrays["metadata_json"] = np.array(
            compact.canonical_json_bytes(metadata).decode("ascii"))

    def request(self, **changes):
        request = {
            "schema_version": compact.REQUEST_SCHEMA,
            "request_id": "F010-named-discrepancy",
            "candidate_sha": compact.CANDIDATE_SHA,
            "campaign_revision": compact.CAMPAIGN_REVISION,
            "raw_input_manifest_sha256": self.metadata["raw_input_manifest_sha256"],
            "trigger": {"kind": "named_discrepancy", "name": "F010-pixel-3-2"},
            "target": {
                "kind": "detector_sequence",
                "obsnum": 152389,
                "array": "a1100",
                "network": self.metadata["detector_order"][0]["network"],
                "scan_identity": self.metadata["scan_order"][1]["scan_identity"],
                "detector_identity":
                    self.metadata["detector_order"][0]["detector_identity"],
            },
            "max_terms": 32,
            "full_population": False,
        }
        request.update(changes)
        return request

    def write_request(self, name: str = "request.json", **changes) -> Path:
        path = self.root / name
        compact.write_json(path, self.request(**changes))
        return path

    def write_candidate_ptc(self, name: str = "candidate-ptc.nc", *,
                            omit_kernel: bool = False,
                            signal_dtype: str = "f8") -> tuple[Path, Path, dict]:
        ptc = self.governed_roots[2] / name
        metadata = copy.deepcopy(self.metadata)
        metadata["adapter"] = compact.NETCDF_ADAPTER
        nscan = len(metadata["scan_order"])
        ndet = len(metadata["detector_order"])
        sample_counts = [scan["sample_count"] for scan in metadata["scan_order"]]
        npts = sum(sample_counts)
        pixel_size = float.fromhex(metadata["map_pixel_size_rad"]["hex"])
        rows = metadata["map_shape"]["rows"]
        cols = metadata["map_shape"]["cols"]
        with netCDF4.Dataset(ptc, "w", format="NETCDF4") as dataset:
            dataset.createDimension("n_pts", npts)
            dataset.createDimension("n_dets", ndet)
            dataset.createDimension("n_scans", nscan)
            dataset.createDimension("n_scan_indices", 2)
            dataset.createDimension("n_tod_output_type", 1)
            output_type = dataset.createVariable(
                "tod_output_type", str, ("n_tod_output_type",))
            output_type[:] = np.asarray(["ptc"], dtype=object)
            dataset.createVariable("obsnum", "i4")[:] = metadata["obsnum"]
            dataset.createVariable("SAMPRATE", "f8")[:] = float.fromhex(
                metadata["native_fsmp_hz"]["hex"])
            signal = dataset.createVariable("signal", signal_dtype,
                                            ("n_pts", "n_dets"))
            flags = dataset.createVariable("flags", "f8", ("n_pts", "n_dets"))
            if not omit_kernel:
                kernel = dataset.createVariable("kernel", "f8",
                                                ("n_pts", "n_dets"))
            det_lat = dataset.createVariable("det_lat", "f8", ("n_pts", "n_dets"))
            det_lon = dataset.createVariable("det_lon", "f8", ("n_pts", "n_dets"))
            weights = dataset.createVariable("weights", "f8", ("n_scans", "n_dets"))
            scan_indices = dataset.createVariable(
                "scan_indices", "i4", ("n_scans", "n_scan_indices"))
            output_scan = dataset.createVariable(
                "output_scan_index", "i4", ("n_scans",))
            arrays = np.zeros(ndet, dtype=np.float64)
            apt_values = {
                "apt_flag": np.asarray(
                    [int(d["apt_flagged"]) for d in metadata["detector_order"]],
                    dtype=np.float64),
                "apt_array": arrays,
                "apt_nw": np.asarray([d["network"] for d in metadata["detector_order"]],
                                     dtype=np.float64),
                "apt_kids_tone": np.asarray(
                    [d["kids_tone"] for d in metadata["detector_order"]],
                    dtype=np.float64),
                "apt_uid": np.asarray(
                    [int(d["detector_uid"]) for d in metadata["detector_order"]],
                    dtype=np.float64),
            }
            for var_name, values in apt_values.items():
                dataset.createVariable(var_name, "f8", ("n_dets",))[:] = values
            signal_values = np.arange(npts * ndet, dtype=np.float64).reshape(npts, ndet) / 8.0
            kernel_values = 0.5 + signal_values / 16.0
            flag_values = np.zeros((npts, ndet), dtype=np.float64)
            flag_values[1, 1] = 1.0
            row_values = np.empty((npts, ndet), dtype=np.float64)
            col_values = np.empty((npts, ndet), dtype=np.float64)
            for sample in range(npts):
                for detector in range(ndet):
                    row_values[sample, detector] = (sample + detector) % rows
                    col_values[sample, detector] = (sample + 2 * detector) % cols
            # Exercise C++ half-away-from-zero: projected row -0.5 becomes -1.
            row_values[0, 0] = -0.5
            signal[:] = signal_values
            flags[:] = flag_values
            if not omit_kernel:
                kernel[:] = kernel_values
            det_lat[:] = (row_values - (rows - 1) / 2.0) * pixel_size
            det_lon[:] = (col_values - (cols - 1) / 2.0) * pixel_size
            weight_values = np.ones((nscan, ndet), dtype=np.float64)
            weight_values[2, 2] = 0.0
            weights[:] = weight_values
            start = 0
            for index, count in enumerate(sample_counts):
                scan_indices[index] = [start, start + count - 1]
                output_scan[index] = metadata["scan_order"][index]["output_scan_index"]
                start += count
        metadata["capture_ptc_sha256"] = compact.sha256_file(ptc)
        authority = self.root / f"{name}.authority.json"
        compact.write_json(authority, metadata)
        return ptc, authority, metadata

    def write_resource_gate(self, stage: str, target: Path, *,
                            projection_source: Path | None = None,
                            recorded_at: datetime | None = None,
                            roots: list[Path] | None = None
                            ) -> tuple[Path, Path, list[Path], datetime]:
        governed_roots = self.governed_roots if roots is None else roots
        inventory = compact._resource_inventory(governed_roots)
        entries = inventory["entries"]
        logical = sum(int(entry["logical_bytes"]) for entry in entries)
        allocated = sum(int(entry["allocated_bytes"]) for entry in entries)
        moment = (recorded_at or datetime.now(timezone.utc)).replace(microsecond=0)
        safe_stage = stage.replace(":", "-")
        inventory_path = self.root / "resource-records" / \
            f"{safe_stage}.pre.inventory.json"
        record_path = self.root / "resource-records" / f"{safe_stage}.pre.json"
        inventory_path.parent.mkdir(exist_ok=True)
        compact.write_json(inventory_path, inventory)
        if projection_source is None:
            projection_source = self.root / f"{safe_stage}.projection-source.json"
            if stage.startswith("compact-production:"):
                compact.write_json(projection_source, self.metadata)
            elif stage.startswith("focused-expansion"):
                compact.write_json(projection_source, self.request())
            else:
                self.fail(f"test helper has no projection rule for {stage}")
        source_node = compact.read_json(projection_source)
        if stage.startswith("compact-production:"):
            method = "primitive-count-two-bytes-plus-64mib-v1"
            fixed = 64 * 1024 * 1024
            unit_count = int(source_node["primitive_term_count"])
            bytes_per_unit = 2
        else:
            method = "bounded-request-max-terms-v1"
            fixed = 64 * 1024 * 1024
            unit_count = int(source_node["max_terms"])
            bytes_per_unit = 256 if stage.startswith(
                "focused-expansion-plan:") else 2048
        projected = fixed + unit_count * bytes_per_unit
        projection_path = self.root / "resource-records" / \
            f"{safe_stage}.projection.json"
        compact.write_json(projection_path, {
            "schema_version": "sci-map-001-resource-projection-v1",
            "request_id": compact.REQUEST_ID,
            "revision": compact.CAMPAIGN_REVISION,
            "candidate_sha": compact.CANDIDATE_SHA,
            "stage": stage,
            "method": method,
            "source": {
                "path": str(projection_source.resolve()),
                "size_bytes": projection_source.stat().st_size,
                "sha256": compact.sha256_file(projection_source),
                "schema_version": source_node["schema_version"],
            },
            "fixed_overhead_bytes": fixed,
            "unit_count": unit_count,
            "bytes_per_unit": bytes_per_unit,
            "projected_incremental_bytes": projected,
        })
        record = {
            "schema_version": compact.RESOURCE_SCHEMA,
            "request_id": compact.REQUEST_ID,
            "revision": compact.CAMPAIGN_REVISION,
            "candidate_sha": compact.CANDIDATE_SHA,
            "stage": stage,
            "recorded_at_utc": moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "governed_roots": [str(path.resolve()) for path in governed_roots],
            "ceiling_bytes": compact.RESOURCE_CEILING_BYTES,
            "filesystem_root": str(self.root.resolve()),
            "filesystem_device": int(self.root.stat().st_dev),
            "current_logical_bytes": logical,
            "current_allocated_bytes": allocated,
            "projected_incremental_bytes": projected,
            "filesystem_available_bytes": max(
                projected, shutil.disk_usage(target.parent).free),
            "logical_plus_projected_bytes": logical + projected,
            "allocated_plus_projected_bytes": allocated + projected,
            "projection_authority": {
                "path": str(projection_path.resolve()),
                "sha256": compact.sha256_file(projection_path),
                "method": method,
            },
            "inventory": {
                "path_count": len(entries),
                "total_logical_bytes": logical,
                "total_allocated_bytes": allocated,
                "sha256": compact.hashlib.sha256(
                    compact.canonical_json_bytes(inventory)).hexdigest(),
            },
            "passed": True,
            "retention": {
                "automatic_cleanup": False,
                "capture_point_retained": True,
                "capture_science_retained": True,
            },
        }
        compact.write_json(record_path, record)
        return record_path, inventory_path, governed_roots, moment

    def test_chunk_invariance_and_small_fixture_full_parity(self) -> None:
        first = compact.produce_compact_group(self.source, self.root / "chunk-1", 1)
        second = compact.produce_compact_group(self.source, self.root / "chunk-17", 17)
        for filename in ("group.json", "sufficient-statistics.npz",
                         "deterministic-trace.npz", "trace-selection.json"):
            self.assertEqual((first.parent / filename).read_bytes(),
                             (second.parent / filename).read_bytes())
        reconstructed = compact.reconstruct_compact_group(first, 0.1)
        reference = compact.full_fixture_reference(self.source, 0.1)
        compact.assert_compact_parity(reconstructed, reference)
        self.assertEqual(reconstructed["noise"].shape[-1], 64)

    def test_direct_candidate_full_ptc_adapter_and_missing_full_fields(self) -> None:
        ptc, authority, metadata = self.write_candidate_ptc()
        group_dir = self.governed_roots[4] / "ptc-group"
        record, inventory, roots, _ = self.write_resource_gate(
            "compact-production:152389:a1100", group_dir)
        group_path = compact.produce_compact_group(
            ptc, group_dir, 2, authority_path=authority,
            resource_record_path=record, resource_inventory_path=inventory,
            governed_roots=roots)
        loaded = compact.load_compact_group(group_path)
        self.assertEqual(loaded.group["source_stream_sha256"],
                         compact.sha256_file(ptc))
        self.assertEqual(
            loaded.group["realized_raw_timestream_provenance_sha256"],
            metadata["realized_raw_timestream_provenance_sha256"],
        )
        self.assertEqual(loaded.group["realized_mapmaking_provenance_sha256"],
                         metadata["realized_mapmaking_provenance_sha256"])
        self.assertEqual(loaded.group["population"]["primitive_term_count"], 35)
        # The selected -0.5 projected row is llround(-0.5)=-1, hence out of bounds.
        self.assertEqual(int(loaded.stats["geometric_hits"].sum()), 34)

        request_path = self.write_request("direct-request.json")
        plan_path = self.governed_roots[4] / "direct-plan.json"
        expansion_path = self.governed_roots[4] / "direct-expansion.npz"
        record, inventory, roots, _ = self.write_resource_gate(
            "focused-expansion-plan:F010-named-discrepancy", plan_path)
        compact.plan_expansion(
            ptc, request_path, plan_path, 2, authority_path=authority,
            resource_record_path=record, resource_inventory_path=inventory,
            governed_roots=roots)
        record, inventory, roots, _ = self.write_resource_gate(
            "focused-expansion:F010-named-discrepancy", expansion_path)
        compact.emit_expansion(
            ptc, plan_path, expansion_path, 3, authority_path=authority,
            resource_record_path=record, resource_inventory_path=inventory,
            governed_roots=roots)
        with np.load(expansion_path, allow_pickle=False) as expansion:
            self.assertEqual(np.asarray(expansion["ordinal"]).size, 3)

        missing, missing_authority, _ = self.write_candidate_ptc(
            "missing-kernel.nc", omit_kernel=True)
        missing_output = self.governed_roots[4] / "missing-kernel-group"
        record, inventory, roots, _ = self.write_resource_gate(
            "compact-production:152389:a1100", missing_output)
        with self.assertRaisesRegex(compact.EvidenceError, "lacks required primitive.*kernel"):
            compact.produce_compact_group(
                missing, missing_output, 2, authority_path=missing_authority,
                resource_record_path=record, resource_inventory_path=inventory,
                governed_roots=roots)

        mini, mini_authority, _ = self.write_candidate_ptc(
            "mini-signal.nc", signal_dtype="f4")
        mini_output = self.governed_roots[4] / "mini-signal-group"
        record, inventory, roots, _ = self.write_resource_gate(
            "compact-production:152389:a1100", mini_output)
        with self.assertRaisesRegex(compact.EvidenceError, "signal is not full binary64"):
            compact.produce_compact_group(
                mini, mini_output, 2, authority_path=mini_authority,
                resource_record_path=record, resource_inventory_path=inventory,
                governed_roots=roots)

        cli_output = self.governed_roots[4] / "cli-ptc-group"
        record, inventory, roots, _ = self.write_resource_gate(
            "compact-production:152389:a1100", cli_output)
        command = [
            sys.executable, str(SCRIPT), "produce", "--source", str(ptc),
            "--authority", str(authority), "--resource-record", str(record),
            "--resource-inventory", str(inventory), "--output-dir", str(cli_output),
            "--chunk-size", "3",
        ]
        for root in roots:
            command.extend(("--governed-root", str(root)))
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(compact.load_compact_group(
            cli_output / "group.json").group["obsnum"], 152389)

    def test_resource_gate_rejects_stale_wrong_root_capacity_and_mutation(self) -> None:
        target = self.governed_roots[4] / "new-stage-output"
        validation_now = datetime.now(timezone.utc).replace(microsecond=0)
        record, inventory, roots, _ = self.write_resource_gate(
            "compact-production:152389:a1100", target,
            recorded_at=validation_now)
        compact.validate_resource_gate(
            record, inventory, roots, "compact-production:152389:a1100",
            target, now=validation_now)

        with self.assertRaisesRegex(compact.EvidenceError, "stage differs"):
            compact.validate_resource_gate(
                record, inventory, roots, "compact-production:152390:a1100",
                target, now=validation_now)
        with self.assertRaisesRegex(compact.EvidenceError, "governed roots differ"):
            compact.validate_resource_gate(
                record, inventory, list(reversed(roots)),
                "compact-production:152389:a1100", target,
                now=validation_now)

        stale_record, stale_inventory, roots, _ = self.write_resource_gate(
            "compact-production:152389:a1100", target,
            recorded_at=validation_now - timedelta(
                seconds=compact.RESOURCE_MAX_AGE_SECONDS + 1))
        with self.assertRaisesRegex(compact.EvidenceError, "stale"):
            compact.validate_resource_gate(
                stale_record, stale_inventory, roots,
                "compact-production:152389:a1100", target,
                now=validation_now)

        record, inventory, roots, _ = self.write_resource_gate(
            "compact-production:152389:a1100", target,
            recorded_at=validation_now)
        node = compact.read_json(record)
        node["filesystem_available_bytes"] = \
            node["projected_incremental_bytes"] - 1
        compact.write_json(record, node)
        with self.assertRaisesRegex(compact.EvidenceError, "capacity"):
            compact.validate_resource_gate(
                record, inventory, roots, "compact-production:152389:a1100",
                target, now=validation_now)

        record, inventory, roots, _ = self.write_resource_gate(
            "compact-production:152389:a1100", target,
            recorded_at=validation_now)
        node = compact.read_json(record)
        test_ceiling = node["projected_incremental_bytes"] - 1
        node["ceiling_bytes"] = test_ceiling
        compact.write_json(record, node)
        with mock.patch.object(compact, "RESOURCE_CEILING_BYTES", test_ceiling):
            with self.assertRaisesRegex(compact.EvidenceError, "200-GiB ceiling"):
                compact.validate_resource_gate(
                    record, inventory, roots, "compact-production:152389:a1100",
                    target, now=validation_now)

        record, inventory, roots, _ = self.write_resource_gate(
            "compact-production:152389:a1100", target,
            recorded_at=validation_now)
        node = compact.read_json(record)
        node["retention"]["automatic_cleanup"] = True
        compact.write_json(record, node)
        with self.assertRaisesRegex(compact.EvidenceError, "retention proof"):
            compact.validate_resource_gate(
                record, inventory, roots, "compact-production:152389:a1100",
                target, now=validation_now)

        record, inventory, roots, _ = self.write_resource_gate(
            "compact-production:152389:a1100", target,
            recorded_at=validation_now)
        (self.governed_roots[1] / "changed-after-preflight.txt").write_text(
            "changed\n", encoding="utf-8")
        with self.assertRaisesRegex(compact.EvidenceError, "changed after"):
            compact.validate_resource_gate(
                record, inventory, roots, "compact-production:152389:a1100",
                target, now=validation_now)

        outside = self.root / "canonical-raw-target.bin"
        outside.write_bytes(b"x" * 4096)
        link = self.governed_roots[0] / "raw-link.nc"
        link.symlink_to(outside)
        live = compact._resource_inventory(roots)
        link_entry = next(entry for entry in live["entries"]
                          if entry["relative_path"] == "raw-link.nc")
        self.assertEqual(link_entry["kind"], "symlink")
        self.assertNotEqual(link_entry["logical_bytes"], outside.stat().st_size)
        self.assertEqual(link_entry["sha256"], compact.hashlib.sha256(
            str(outside).encode("utf-8")).hexdigest())

    def test_trace_spans_every_network_and_records_absent_class(self) -> None:
        group_path = compact.produce_compact_group(self.source, self.root / "group", 5)
        loaded = compact.load_compact_group(group_path)
        selected_scans = [item["scan_index"]
                          for item in loaded.trace_selection["selected_scans"]]
        self.assertEqual(selected_scans, [0, 1, 2])
        entries = loaded.trace_selection["entries"]
        observed = {(e["scan_index"], e["network"], e["detector_state"])
                    for e in entries}
        expected = {(scan, network, state)
                    for scan in selected_scans
                    for network in loaded.group["active_networks"]
                    for state in ("valid", "flagged")}
        self.assertEqual(observed, expected)
        self.assertGreater(loaded.group["trace_selection"]["absence_fact_count"], 0)
        self.assertTrue(any(not entry["present"] for entry in entries))
        self.assertTrue(any(entry["present"] and entry["detector_state"] == "valid"
                            for entry in entries))
        self.assertTrue(any(entry["present"] and entry["detector_state"] == "flagged"
                            for entry in entries))
        self.assertTrue({"scan_index", "detector_index", "sample_index", "network"}
                        .isdisjoint(loaded.trace))

        selection, _ = compact.select_trace(self.metadata,
                                             compact.boost_mt19937_scan_signs(3))
        for entry in selection["entries"]:
            if not entry["present"]:
                continue
            candidates = [detector for detector in self.metadata["detector_order"]
                          if detector["network"] == entry["network"] and
                          detector["apt_flagged"] ==
                          (entry["detector_state"] == "flagged")]
            scan = self.metadata["scan_order"][entry["scan_index"]]
            hashes = [compact._selection_digest(
                self.metadata, scan, entry["scan_roles"], entry["network"],
                entry["detector_state"], detector)
                for detector in candidates]
            self.assertEqual(entry["selection_hash"], min(hashes))

    def test_rejects_mini_partial_and_missing_primitive_authority(self) -> None:
        arrays = self.source_arrays()
        self.update_metadata(arrays,
                             lambda metadata: metadata.update(capture_output_mode="mini"))
        mini = self.write_variant("mini.npz", arrays)
        with self.assertRaisesRegex(compact.EvidenceError, "capture_output_mode"):
            compact.produce_compact_group(mini, self.root / "mini-group")

        arrays = self.source_arrays()
        arrays.pop("sample_kernel")
        missing = self.write_variant("missing.npz", arrays)
        with self.assertRaisesRegex(compact.EvidenceError, "members differ"):
            compact.produce_compact_group(missing, self.root / "missing-group")

        arrays = self.source_arrays()
        self.update_metadata(
            arrays, lambda metadata: metadata.update(capture_detector_selection="indices"))
        partial = self.write_variant("partial.npz", arrays)
        with self.assertRaisesRegex(compact.EvidenceError,
                                    "capture_detector_selection"):
            compact.produce_compact_group(partial, self.root / "partial-group")

    def test_rejects_reordered_repeated_and_nonfinite_valid_term(self) -> None:
        arrays = self.source_arrays()
        for name in (*compact.TERM_INT64, *compact.TERM_UINT8, *compact.TERM_FLOAT64):
            arrays[name][[0, 1]] = arrays[name][[1, 0]]
        reordered = self.write_variant("reordered.npz", arrays)
        with self.assertRaisesRegex(compact.EvidenceError, "reorders Cartesian"):
            compact.produce_compact_group(reordered, self.root / "reordered-group", 3)

        arrays = self.source_arrays()
        upstream = np.flatnonzero(arrays["upstream_eligible"])
        self.assertGreater(upstream.size, 0)
        arrays["coefficient"][upstream[0]] = np.nan
        nonfinite = self.write_variant("nonfinite.npz", arrays)
        with self.assertRaisesRegex(compact.EvidenceError, "coefficient is non-finite"):
            compact.produce_compact_group(nonfinite, self.root / "nonfinite-group")

        arrays = self.source_arrays()
        index = int(upstream[0])
        arrays["sample_signal"][index] = np.inf
        nonfinite_signal = self.write_variant("nonfinite-signal.npz", arrays)
        with self.assertRaisesRegex(compact.EvidenceError, "signal is non-finite"):
            compact.produce_compact_group(nonfinite_signal,
                                          self.root / "nonfinite-signal-group")

    def test_native_and_effective_rate_authorities_are_separate_and_exact(self) -> None:
        group_path = compact.produce_compact_group(self.source, self.root / "group", 9)
        loaded = compact.load_compact_group(group_path)
        rates = loaded.group["rates"]
        self.assertEqual(rates["native_fsmp_hz"]["authority"], "telescope.fsmp")
        self.assertEqual(rates["effective_d_fsmp_hz"]["authority"],
                         "telescope.d_fsmp")
        native = float.fromhex(rates["native_fsmp_hz"]["hex"])
        effective = float.fromhex(rates["effective_d_fsmp_hz"]["hex"])
        interval = float.fromhex(rates["sample_interval_s"]["hex"])
        self.assertNotEqual(native, effective)
        self.assertEqual(np.float64(interval).view(np.uint64),
                         np.float64(1.0 / effective).view(np.uint64))

        arrays = self.source_arrays()
        self.update_metadata(
            arrays,
            lambda metadata: metadata["effective_d_fsmp_hz"].update(
                decimal="0", hex=float(0.0).hex()))
        zero = self.write_variant("zero-rate.npz", arrays)
        with self.assertRaisesRegex(compact.EvidenceError,
                                    "telescope.d_fsmp .*positive binary64"):
            compact.produce_compact_group(zero, self.root / "zero-rate-group")

        arrays = self.source_arrays()
        self.update_metadata(
            arrays,
            lambda metadata: metadata["native_fsmp_hz"].update(
                authority="telescope.d_fsmp"))
        relabeled = self.write_variant("relabeled-native-rate.npz", arrays)
        with self.assertRaisesRegex(compact.EvidenceError, "rate authority differs"):
            compact.produce_compact_group(relabeled,
                                          self.root / "relabeled-native-group")

    def test_digest_tamper_and_accidental_full_term_axis_are_rejected(self) -> None:
        group_path = compact.produce_compact_group(self.source, self.root / "group", 4)
        stats_path = group_path.parent / "sufficient-statistics.npz"
        tampered = bytearray(stats_path.read_bytes())
        tampered[-1] ^= 1
        stats_path.write_bytes(tampered)
        with self.assertRaisesRegex(compact.EvidenceError, "size or digest differs"):
            compact.load_compact_group(group_path)

        group_path = compact.produce_compact_group(self.source, self.root / "axis", 4)
        stats_path = group_path.parent / "sufficient-statistics.npz"
        with np.load(stats_path, allow_pickle=False) as archive:
            stats = {name: np.asarray(archive[name]).copy() for name in archive.files}
        stats["row"] = np.zeros(self.metadata["primitive_term_count"], dtype=np.int64)
        compact.deterministic_npz(stats_path, stats)
        group = compact.read_json(group_path)
        group["artifacts"]["sufficient_statistics"]["sha256"] = \
            compact.sha256_file(stats_path)
        group["artifacts"]["sufficient_statistics"]["stored_bytes"] = \
            stats_path.stat().st_size
        compact.write_json(group_path, group)
        with self.assertRaisesRegex(compact.EvidenceError, "members differ"):
            compact.load_compact_group(group_path)

    def test_two_pass_named_bounded_detector_and_pixel_expansion(self) -> None:
        request_path = self.write_request()
        plan_path = self.root / "plan.json"
        output_path = self.root / "expanded.npz"
        compact.plan_expansion(self.source, request_path, plan_path, 2)
        plan = compact.read_json(plan_path)
        self.assertEqual(plan["planned_terms"], 3)
        self.assertLessEqual(plan["planned_terms"], plan["maximum_terms"])
        compact.emit_expansion(self.source, plan_path, output_path, 11)
        with np.load(output_path, allow_pickle=False) as archive:
            self.assertEqual(np.asarray(archive["ordinal"]).size, 3)
            metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
            self.assertFalse(metadata["full_population"])
            self.assertTrue(metadata["bounded"])

        arrays = self.source_arrays()
        geometric = np.flatnonzero(arrays["geometric_in_bounds"])
        index = int(geometric[0])
        target = {
            "kind": "pixel", "obsnum": 152389, "array": "a1100",
            "network": int(arrays["network"][index]),
            "row": int(arrays["row"][index]), "col": int(arrays["col"][index]),
        }
        pixel_request = self.write_request("pixel-request.json", target=target)
        pixel_plan = self.root / "pixel-plan.json"
        compact.plan_expansion(self.source, pixel_request, pixel_plan, 6)
        self.assertGreater(compact.read_json(pixel_plan)["planned_terms"], 0)

    def test_unbounded_unnamed_and_changed_source_expansion_fail_closed(self) -> None:
        broad = self.write_request(
            "broad.json",
            target={"kind": "all", "obsnum": 152389, "array": "a1100",
                    "network": 0})
        with self.assertRaisesRegex(compact.EvidenceError, "must be detector_sequence or pixel"):
            compact.plan_expansion(self.source, broad, self.root / "broad-plan.json")

        unnamed = self.write_request(
            "unnamed.json", trigger={"kind": "named_discrepancy", "name": " "})
        with self.assertRaisesRegex(compact.EvidenceError, "requires one named"):
            compact.plan_expansion(self.source, unnamed, self.root / "unnamed-plan.json")

        too_small = self.write_request("too-small.json", max_terms=2)
        with self.assertRaisesRegex(compact.EvidenceError, "exceeds"):
            compact.plan_expansion(self.source, too_small,
                                   self.root / "too-small-plan.json", 1)

        request_path = self.write_request("two-pass.json")
        plan_path = self.root / "two-pass-plan.json"
        compact.plan_expansion(self.source, request_path, plan_path)
        changed = self.source_arrays()
        changed["sample_signal"][0] += 1.0
        compact.deterministic_npz(self.source, changed)
        with self.assertRaisesRegex(compact.EvidenceError, "changed between pass"):
            compact.emit_expansion(self.source, plan_path, self.root / "must-not-exist.npz")

    def test_exact_nine_group_mapping_and_missing_group_rejection(self) -> None:
        groups = self.root / "nine"
        mapping = {}
        for key in compact.REQUIRED_GROUP_KEYS:
            obs_text, array = key.split(":", 1)
            source = groups / "sources" / f"{obs_text}-{array}.npz"
            compact.write_self_check_fixture(source, int(obs_text), array)
            destination = groups / "groups" / f"{obs_text}-{array}"
            compact.produce_compact_group(source, destination, 8)
            mapping[key] = str(destination.relative_to(groups) / "group.json")
        collection = groups / "collection.json"
        compact.write_json(collection, {"request_root": str(groups.resolve()),
                                        "compact_groups": mapping})
        self.assertEqual(set(compact.verify_nine_group_mapping(collection)),
                         set(compact.REQUIRED_GROUP_KEYS))
        collection_link = self.root / "collection-link.json"
        collection_link.symlink_to(collection)
        with self.assertRaisesRegex(compact.EvidenceError, "nonsymlink"):
            compact.verify_nine_group_mapping(collection_link)

        first_key = compact.REQUIRED_GROUP_KEYS[0]
        original = mapping[first_key]
        linked_group = groups / "groups" / "linked-group.json"
        linked_group.symlink_to(groups / original)
        mapping[first_key] = str(linked_group.relative_to(groups))
        compact.write_json(collection, {"request_root": str(groups.resolve()),
                                        "compact_groups": mapping})
        with self.assertRaisesRegex(compact.EvidenceError, "symlinked"):
            compact.verify_nine_group_mapping(collection)
        mapping[first_key] = original
        mapping.pop(compact.REQUIRED_GROUP_KEYS[-1])
        compact.write_json(collection, {"request_root": str(groups.resolve()),
                                        "compact_groups": mapping})
        with self.assertRaisesRegex(compact.EvidenceError, "exact nine"):
            compact.verify_nine_group_mapping(collection)

    def test_cli_self_check_and_schema_documents(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "self-check"],
            check=False, capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["nine_group_count"], 9)
        schemas = {}
        for filename in ("compact-evidence-contract.json", "compact-group.schema.json",
                         "producer-stream.schema.json", "discrepancy-request.schema.json"):
            with (PACKAGE / filename).open("r", encoding="utf-8") as stream:
                document = json.load(stream)
            if "$schema" in document:
                jsonschema.Draft202012Validator.check_schema(document)
                schemas[filename] = document
        jsonschema.validate(self.metadata, schemas["producer-stream.schema.json"])
        group = compact.produce_compact_group(
            self.source, self.root / "schema-validation-group", 7)
        jsonschema.validate(compact.read_json(group),
                            schemas["compact-group.schema.json"])
        jsonschema.validate(self.request(),
                            schemas["discrepancy-request.schema.json"])


if __name__ == "__main__":
    unittest.main()
