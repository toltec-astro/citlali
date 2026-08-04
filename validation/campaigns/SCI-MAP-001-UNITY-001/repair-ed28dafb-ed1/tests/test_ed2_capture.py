#!/usr/bin/env python3
"""Focused local tests for the file-only MAP-UNITY-ED2 capture helper."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import types
import unittest
from unittest import mock

import jsonschema
import numpy as np
import yaml


PACKAGE = Path(__file__).resolve().parents[1]
PROGRAM = PACKAGE / "scripts" / "ed2-capture.py"
SPEC = importlib.util.spec_from_file_location("sci_map_ed2_capture", PROGRAM)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import bootstrap
    raise RuntimeError(f"cannot import {PROGRAM}")
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")


def namespace(**values: object) -> argparse.Namespace:
    return argparse.Namespace(**values)


class CaptureToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="sci-map-ed2-capture-")
        self.root = Path(self.temporary.name).resolve()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def selection(self, capture_id: str, records: list[dict[str, object]],
                  name: str = "selection.json") -> Path:
        path = self.root / name
        write_json(path, {
            "schema_version": "sci-map-001-raw-selection-v1",
            "capture_id": capture_id,
            "records": records,
        })
        return path

    def authority_selection(self, capture_id: str,
                            records: list[dict[str, object]]) -> Path:
        path = self.root / "authority-selection.json"
        normalized = [{**row, "source_path": row.get(
            "source_path", str(self.root / "authorities" / str(row["basename"]))) }
            for row in records]
        write_json(path, {
            "schema_version": "sci-map-001-authority-selection-v1",
            "capture_id": capture_id,
            "records": normalized,
        })
        return path

    def test_raw_selection_exact_observation_identity_and_coverage(self) -> None:
        point = self.selection("CAP-POINT", [{
            "observation": 152389,
            "basename": "toltec0_152389_000_0002.nc",
        }])
        self.assertEqual(capture._raw_selection(point, "CAP-POINT"), [{
            "observation": 152389,
            "basename": "toltec0_152389_000_0002.nc",
        }])

        cases = [
            ([{"observation": 152390,
               "basename": "toltec0_152390_000_0002.nc"}],
             "does not cover|identity/basename"),
            ([{"observation": 152389,
               "basename": "toltec0_152390_000_0002.nc"}],
             "identity/basename"),
            ([{"observation": 152389, "basename": "../escape.nc"}],
             "identity/basename"),
            ([{"observation": 152389,
               "basename": "toltec0_152389_000_0002.nc"},
              {"observation": 152389,
               "basename": "toltec0_152389_000_0002.nc"}],
             "identity/basename"),
        ]
        for index, (records, message) in enumerate(cases):
            with self.subTest(index=index), self.assertRaisesRegex(
                    capture.CaptureError, message):
                capture._raw_selection(
                    self.selection("CAP-POINT", records, f"bad-{index}.json"),
                    "CAP-POINT")

    def _redirect_canonical_path(self, canonical_root: Path):
        real_path = Path

        def redirected(value: object = ".") -> Path:
            if os.fspath(value) == "/work/toltec":
                return canonical_root
            return real_path(value)

        return mock.patch.object(capture, "Path", new=redirected)

    def test_raw_manifest_resolves_one_regular_canonical_source(self) -> None:
        canonical = self.root / "canonical"
        (canonical / "nested").mkdir(parents=True)
        basename = "toltec0_152389_000_0002.nc"
        source = canonical / "nested" / basename
        source.write_bytes(b"raw-authority")
        selection = self.selection(
            "CAP-POINT", [{"observation": 152389, "basename": basename}])
        output = self.root / "raw-link-manifest.json"
        with self._redirect_canonical_path(canonical):
            result = capture.command_raw_manifest(namespace(
                capture_id="CAP-POINT", canonical_root=canonical,
                selection=selection, output=output))
        record = result["records"][0]
        self.assertEqual(Path(record["resolved_target"]), source.resolve())
        self.assertEqual(record["size_bytes"], len(b"raw-authority"))
        self.assertEqual(record["sha256"], hashlib.sha256(b"raw-authority").hexdigest())
        self.assertEqual(result["staging_policy"], "individual-file-symlinks-only")
        self.assertEqual(result["tolproj_copy_raw_after_staging"], "prohibited")

    def test_raw_manifest_rejects_missing_ambiguous_symlink_and_preexisting(self) -> None:
        canonical = self.root / "canonical"
        canonical.mkdir()
        basename = "toltec0_152389_000_0002.nc"
        selection = self.selection(
            "CAP-POINT", [{"observation": 152389, "basename": basename}])

        with self._redirect_canonical_path(canonical), self.assertRaisesRegex(
                capture.CaptureError, "0 matches"):
            capture.command_raw_manifest(namespace(
                capture_id="CAP-POINT", canonical_root=canonical,
                selection=selection, output=self.root / "missing.json"))

        for name in ("one", "two"):
            (canonical / name).mkdir()
            (canonical / name / basename).write_bytes(name.encode("ascii"))
        with self._redirect_canonical_path(canonical), self.assertRaisesRegex(
                capture.CaptureError, "2 matches"):
            capture.command_raw_manifest(namespace(
                capture_id="CAP-POINT", canonical_root=canonical,
                selection=selection, output=self.root / "ambiguous.json"))

        shutil.rmtree(canonical)
        canonical.mkdir()
        target = self.root / "target.nc"
        target.write_bytes(b"target")
        (canonical / basename).symlink_to(target)
        with self._redirect_canonical_path(canonical), self.assertRaisesRegex(
                capture.CaptureError, "nonsymlink regular file"):
            capture.command_raw_manifest(namespace(
                capture_id="CAP-POINT", canonical_root=canonical,
                selection=selection, output=self.root / "symlink.json"))

        (canonical / basename).unlink()
        (canonical / basename).write_bytes(b"raw")
        output = self.root / "exists.json"
        output.write_text("do not replace", encoding="utf-8")
        with self._redirect_canonical_path(canonical), self.assertRaisesRegex(
                capture.CaptureError, "already exists"):
            capture.command_raw_manifest(namespace(
                capture_id="CAP-POINT", canonical_root=canonical,
                selection=selection, output=output))
        self.assertEqual(output.read_text(encoding="utf-8"), "do not replace")

    def _raw_link_manifest(self, source: Path, basename: str) -> dict[str, object]:
        status = source.stat()
        return {
            "schema_version": "sci-map-001-raw-link-manifest-v1",
            "request_id": capture.REQUEST_ID,
            "revision": capture.REVISION,
            "candidate_sha": capture.CANDIDATE,
            "capture_id": "CAP-POINT",
            "mode": "point",
            "canonical_raw_root": "/work/toltec",
            "records": [{
                "observation": 152389,
                "basename": basename,
                "resolved_target": str(source),
                "size_bytes": status.st_size,
                "device": status.st_dev,
                "inode": status.st_ino,
                "mtime_ns": status.st_mtime_ns,
                "sha256": capture.sha256(source),
            }],
            "staging_policy": "individual-file-symlinks-only",
            "tolproj_copy_raw_after_staging": "prohibited",
        }

    def test_raw_staging_creates_only_individual_absolute_links(self) -> None:
        basename = "toltec0_152389_000_0002.nc"
        canonical = self.root / "canonical"
        canonical.mkdir()
        source = canonical / basename
        source.write_bytes(b"immutable raw")
        manifest_path = self.root / "raw-manifest.json"
        manifest = self._raw_link_manifest(source, basename)
        write_json(manifest_path, manifest)
        destination = self.root / "project-data"
        destination.mkdir()
        output_path = self.root / "raw-staging.json"
        with self._redirect_canonical_path(canonical):
            result = capture.command_stage_raw(namespace(
                manifest=manifest_path, destination=destination, output=output_path))
        link = destination / basename
        self.assertTrue(link.is_symlink())
        self.assertTrue(Path(os.readlink(link)).is_absolute())
        self.assertEqual(link.resolve(strict=True), source)
        self.assertEqual(source.read_bytes(), b"immutable raw")
        self.assertFalse(result["directory_symlinks"])
        self.assertFalse(result["copied_raw_files"])
        self.assertEqual(result["tolproj_copy_raw_after_staging"], "prohibited")

    def test_raw_staging_rejects_nonfresh_or_changed_state(self) -> None:
        basename = "toltec0_152389_000_0002.nc"
        canonical = self.root / "canonical"
        canonical.mkdir()
        source = canonical / basename
        source.write_bytes(b"before")
        manifest_path = self.root / "raw-manifest.json"
        write_json(manifest_path, self._raw_link_manifest(source, basename))

        nonfresh = self.root / "nonfresh"
        nonfresh.mkdir()
        (nonfresh / "unrelated").write_text("present", encoding="utf-8")
        with self._redirect_canonical_path(canonical), self.assertRaisesRegex(
                capture.CaptureError, "must be empty"):
            capture.command_stage_raw(namespace(
                manifest=manifest_path, destination=nonfresh,
                output=self.root / "nonfresh-result.json"))

        source.write_bytes(b"after")
        fresh = self.root / "fresh"
        fresh.mkdir()
        with self._redirect_canonical_path(canonical), self.assertRaisesRegex(
                capture.CaptureError, "identity|digest"):
            capture.command_stage_raw(namespace(
                manifest=manifest_path, destination=fresh,
                output=self.root / "changed-result.json"))
        self.assertEqual(list(fresh.iterdir()), [])

    @staticmethod
    def science_authorities() -> list[dict[str, object]]:
        return [
            {"role": "apt", "observation": 152390,
             "basename": "apt_152390_matched.ecsv"},
            {"role": "apt", "observation": 152392,
             "basename": "apt_152392_matched.ecsv"},
            {"role": "ppt", "observation": 152389,
             "basename": "ppt_pointing_152389.ecsv"},
            {"role": "ppt", "observation": 152391,
             "basename": "ppt_pointing_152391.ecsv"},
            {"role": "ppt", "observation": 152393,
             "basename": "ppt_pointing_152393.ecsv"},
        ]

    def test_authority_copy_exact_rules_and_digest_binding(self) -> None:
        authority_root = self.root / "authorities"
        authority_root.mkdir()
        records = self.science_authorities()
        for index, row in enumerate(records):
            (authority_root / str(row["basename"])).write_bytes(
                f"authority-{index}".encode("ascii"))
        selection = self.authority_selection("CAP-SCIENCE", records)
        apt = self.root / "apt"
        ppt = self.root / "ppt"
        apt.mkdir()
        ppt.mkdir()
        output = self.root / "authority-staging.json"
        result = capture.command_stage_authorities(namespace(
            capture_id="CAP-SCIENCE", authority_root=authority_root,
            selection=selection, apt_destination=apt,
            ppt_destination=ppt, output=output))
        self.assertEqual(len(result["records"]), 5)
        self.assertFalse(result["wholesale_legacy_reduction_used"])
        for row in result["records"]:
            copied = Path(row["destination_path"])
            self.assertTrue(copied.is_file())
            self.assertFalse(copied.is_symlink())
            self.assertEqual(capture.sha256(copied), row["sha256"])
            self.assertEqual(copied.read_bytes(), Path(row["source_path"]).read_bytes())

    def test_authority_copy_rejects_missing_wrong_order_and_preexisting(self) -> None:
        records = self.science_authorities()
        wrong = [records[1], records[0], *records[2:]]
        with self.assertRaisesRegex(capture.CaptureError, "fixed policy"):
            capture._authority_selection(
                self.authority_selection("CAP-SCIENCE", wrong), "CAP-SCIENCE")

        authority_root = self.root / "authorities"
        authority_root.mkdir()
        for row in records[:-1]:
            (authority_root / str(row["basename"])).write_bytes(b"authority")
        apt = self.root / "apt"
        apt.mkdir()
        ppt = self.root / "ppt"
        ppt.mkdir()
        selection = self.authority_selection("CAP-SCIENCE", records)
        with self.assertRaisesRegex(
                capture.CaptureError, "regular file"):
            capture.command_stage_authorities(namespace(
                capture_id="CAP-SCIENCE", authority_root=authority_root,
                selection=selection, apt_destination=apt,
                ppt_destination=ppt, output=self.root / "missing-authority.json"))

        final = records[-1]
        (authority_root / str(final["basename"])).write_bytes(b"authority")
        (apt / "apt_152390_matched.ecsv").write_bytes(b"do not replace")
        with self.assertRaisesRegex(capture.CaptureError, "already exists"):
            capture.command_stage_authorities(namespace(
                capture_id="CAP-SCIENCE", authority_root=authority_root,
                selection=selection, apt_destination=apt,
                ppt_destination=ppt, output=self.root / "preexisting-authority.json"))
        self.assertEqual((apt / "apt_152390_matched.ecsv").read_bytes(),
                         b"do not replace")

    def test_authority_copy_uses_selected_exact_ppt_source(self) -> None:
        """An unselected duplicate cannot change the explicitly selected PPT."""
        authority_root = self.root / "authorities"
        authority_root.mkdir()
        records = self.science_authorities()
        for row in records:
            (authority_root / str(row["basename"])).write_bytes(b"authority")
        (authority_root / "ppt_alternate_152389.ecsv").write_bytes(b"ambiguous")
        selection = self.authority_selection("CAP-SCIENCE", records)
        apt = self.root / "apt"
        apt.mkdir()
        ppt = self.root / "ppt"
        ppt.mkdir()
        result = capture.command_stage_authorities(namespace(
            capture_id="CAP-SCIENCE", authority_root=authority_root,
            selection=selection, apt_destination=apt,
            ppt_destination=ppt, output=self.root / "selected-ppt.json"))
        selected = next(row for row in result["records"]
                        if row["observation"] == 152389)
        self.assertEqual(Path(selected["source_path"]).name,
                         "ppt_pointing_152389.ecsv")

    def test_authority_copy_rejects_reference_only_source(self) -> None:
        legacy = self.root / "citlali-validation" / "v1"
        legacy.mkdir(parents=True)
        apt_name = "apt_152389_matched.ecsv"
        source = legacy / apt_name
        source.write_bytes(b"reference only")
        selection = self.authority_selection("CAP-POINT", [{
            "role": "apt", "observation": 152389, "basename": apt_name,
            "source_path": str(source),
        }])
        destination = self.root / "apt"
        destination.mkdir()
        with self.assertRaisesRegex(capture.CaptureError, "reference-only"):
            capture.command_stage_authorities(namespace(
                capture_id="CAP-POINT", selection=selection,
                apt_destination=destination, ppt_destination=None,
                output=self.root / "reference-only.json"))

    def _config_fixture(self, mode: str = "point") -> tuple[Path, Path, Path, Path]:
        fixed_root = self.root / "fixed-numbered"
        capture_root = self.root / "capture-numbered"
        fixed_root.mkdir()
        capture_root.mkdir()
        names = capture.NUMBERED[mode]
        expert = f"99_{'pointing' if mode == 'point' else 'science'}_expert_overrides.yaml"
        fixed_output = {"enabled": False, "mode": "compact", "indices": "selected"}
        capture_output = {"enabled": True, "mode": "full", "indices": "all"}
        for name in names:
            if name == expert:
                fixed_value = {"timestream": {"processed_time_chunk": {
                    "output": fixed_output}}}
                capture_value = {"timestream": {"processed_time_chunk": {
                    "output": capture_output}}}
            else:
                fixed_value = capture_value = {"fixed": name}
            (fixed_root / name).write_text(yaml.safe_dump(fixed_value), encoding="utf-8")
            (capture_root / name).write_text(
                yaml.safe_dump(capture_value), encoding="utf-8")
        fixed_merged = self.root / "fixed-merged.yaml"
        capture_merged = self.root / "capture-merged.yaml"
        fixed_merged.write_text(yaml.safe_dump({"timestream": {
            "processed_time_chunk": {"output": fixed_output},
            "unchanged": {"leaf": 7}}}), encoding="utf-8")
        capture_merged.write_text(yaml.safe_dump({"timestream": {
            "processed_time_chunk": {"output": capture_output},
            "unchanged": {"leaf": 7}}}), encoding="utf-8")
        return fixed_root, capture_root, fixed_merged, capture_merged

    def _write_config_inventories(self, fixed_root: Path, capture_root: Path,
                                  mode: str = "point") -> tuple[Path, Path]:
        fixed = self.root / "fixed-inventory.json"
        realized = self.root / "capture-inventory.json"
        capture.command_config_inventory(namespace(
            mode=mode, numbered_dir=fixed_root, included_fragment=[], output=fixed))
        capture.command_config_inventory(namespace(
            mode=mode, numbered_dir=capture_root, included_fragment=[], output=realized))
        return fixed, realized

    def test_complete_config_diff_proves_exact_three_leaf_allowlist(self) -> None:
        fixed_root, realized_root, fixed_merged, realized_merged = \
            self._config_fixture()
        fixed_inventory, realized_inventory = self._write_config_inventories(
            fixed_root, realized_root)
        output = self.root / "config-proof.json"
        proof = capture.command_config_proof(namespace(
            capture_id="CAP-POINT", fixed_config=fixed_merged,
            capture_config=realized_merged, fixed_inventory=fixed_inventory,
            capture_inventory=realized_inventory, output=output))
        self.assertTrue(proof["passed"])
        self.assertEqual(proof["allowlist"], capture.ALLOWLIST)
        self.assertEqual(set(proof["differences"]), set(capture.ALLOWLIST))
        self.assertEqual(proof["numbered_order"], capture.NUMBERED["point"])

    def test_capture_overlay_is_mechanical_three_leaf_only(self) -> None:
        binary = self.root / "citlali"
        binary.write_bytes(b"ordinary-candidate-binary")
        reference = self.root / "fixed-overlay.yaml"
        reference.write_text(yaml.safe_dump({
            "reduce": {"jobkey": "P-SEQ", "steps": {0: {
                "path": str(binary), "config": {"low_level": {
                    "runtime": {"n_threads": 1},
                    "timestream": {"processed_time_chunk": {"output": {
                        "enabled": False, "mode": "compact",
                        "indices": "selected"}}},
                }}}}},
        }, sort_keys=False), encoding="utf-8")
        output = self.root / "capture-overlay.yaml"
        record = capture.command_capture_overlay(namespace(
            reference_overlay=reference, candidate_binary=binary,
            output=output))
        self.assertEqual(set(record["differences"]), set(capture.ALLOWLIST))
        rendered = yaml.safe_load(output.read_text(encoding="utf-8"))
        self.assertEqual(capture.flatten(capture.low_level(rendered))[
            "runtime.n_threads"], 1)
        for key, expected in capture.ALLOWLIST.items():
            self.assertEqual(capture.flatten(capture.low_level(rendered))[key],
                             expected)

    def test_config_diff_rejects_extra_leaf_order_and_fragment_tamper(self) -> None:
        fixed_root, realized_root, fixed_merged, realized_merged = \
            self._config_fixture()
        fixed_inventory, realized_inventory = self._write_config_inventories(
            fixed_root, realized_root)
        value = yaml.safe_load(realized_merged.read_text(encoding="utf-8"))
        value["timestream"]["unchanged"]["leaf"] = 8
        realized_merged.write_text(yaml.safe_dump(value), encoding="utf-8")
        with self.assertRaisesRegex(capture.CaptureError, "escapes allowlist"):
            capture.command_config_proof(namespace(
                capture_id="CAP-POINT", fixed_config=fixed_merged,
                capture_config=realized_merged, fixed_inventory=fixed_inventory,
                capture_inventory=realized_inventory,
                output=self.root / "extra-proof.json"))

        # A difference outside the wrapped low-level subtree is also forbidden.
        value["timestream"]["unchanged"]["leaf"] = 7
        fixed_value = yaml.safe_load(fixed_merged.read_text(encoding="utf-8"))
        fixed_wrapped = {
            "reduce": {"steps": {0: {"config": {"low_level": fixed_value}}}},
            "runtime_wrapper": {"identity": "same"},
        }
        realized_wrapped = {
            "reduce": {"steps": {0: {"config": {"low_level": value}}}},
            "runtime_wrapper": {"identity": "changed"},
        }
        fixed_merged.write_text(yaml.safe_dump(fixed_wrapped), encoding="utf-8")
        realized_merged.write_text(yaml.safe_dump(realized_wrapped), encoding="utf-8")
        with self.assertRaisesRegex(capture.CaptureError, "escapes allowlist"):
            capture.command_config_proof(namespace(
                capture_id="CAP-POINT", fixed_config=fixed_merged,
                capture_config=realized_merged, fixed_inventory=fixed_inventory,
                capture_inventory=realized_inventory,
                output=self.root / "wrapper-proof.json"))

        # Explicit null is a present leaf, not equivalent to an absent leaf.
        fixed_merged.write_text(yaml.safe_dump(fixed_value), encoding="utf-8")
        realized_merged.write_text(yaml.safe_dump(value), encoding="utf-8")
        null_merged = dict(value)
        null_merged["unapproved_null"] = None
        realized_merged.write_text(yaml.safe_dump(null_merged), encoding="utf-8")
        with self.assertRaisesRegex(capture.CaptureError, "escapes allowlist"):
            capture.command_config_proof(namespace(
                capture_id="CAP-POINT", fixed_config=fixed_merged,
                capture_config=realized_merged, fixed_inventory=fixed_inventory,
                capture_inventory=realized_inventory,
                output=self.root / "null-merged-proof.json"))

        realized_merged.write_text(yaml.safe_dump(value), encoding="utf-8")
        expert = realized_root / "99_pointing_expert_overrides.yaml"
        expert_value = yaml.safe_load(expert.read_text(encoding="utf-8"))
        expert_value["unapproved_null"] = None
        expert.write_text(yaml.safe_dump(expert_value), encoding="utf-8")
        realized_inventory = self.root / "capture-null-inventory.json"
        capture.command_config_inventory(namespace(
            mode="point", numbered_dir=realized_root,
            included_fragment=[], output=realized_inventory))
        with self.assertRaisesRegex(capture.CaptureError, "expert source"):
            capture.command_config_proof(namespace(
                capture_id="CAP-POINT", fixed_config=fixed_merged,
                capture_config=realized_merged, fixed_inventory=fixed_inventory,
                capture_inventory=realized_inventory,
                output=self.root / "null-source-proof.json"))

        # Restore the expert source, then tamper the declared numbered order.
        expert_value.pop("unapproved_null")
        expert.write_text(yaml.safe_dump(expert_value), encoding="utf-8")
        realized_inventory = self.root / "capture-restored-inventory.json"
        capture.command_config_inventory(namespace(
            mode="point", numbered_dir=realized_root,
            included_fragment=[], output=realized_inventory))
        inventory = json.loads(realized_inventory.read_text(encoding="utf-8"))
        inventory["ordered_numbered_sources"][0], \
            inventory["ordered_numbered_sources"][1] = \
            inventory["ordered_numbered_sources"][1], \
            inventory["ordered_numbered_sources"][0]
        realized_inventory.chmod(0o644)
        write_json(realized_inventory, inventory)
        with self.assertRaisesRegex(capture.CaptureError, "order.*differ"):
            capture.command_config_proof(namespace(
                capture_id="CAP-POINT", fixed_config=fixed_merged,
                capture_config=realized_merged, fixed_inventory=fixed_inventory,
                capture_inventory=realized_inventory,
                output=self.root / "order-proof.json"))

    def test_config_inventory_rejects_tenth_numbered_source(self) -> None:
        fixed_root, _, _, _ = self._config_fixture()
        (fixed_root / "98_unapproved.yaml").write_text("extra: true\n", encoding="utf-8")
        with self.assertRaisesRegex(capture.CaptureError, "inventory differs"):
            capture.command_config_inventory(namespace(
                mode="point", numbered_dir=fixed_root,
                included_fragment=[], output=self.root / "inventory.json"))

    def _five_roots(self) -> list[Path]:
        roots = []
        for name in ("point-project", "science-project", "cap-point",
                     "cap-science", "compact"):
            path = self.root / name
            path.mkdir()
            roots.append(path)
        return roots

    def test_resource_inventory_is_deterministic_and_excludes_symlink_target(self) -> None:
        roots = self._five_roots()
        (roots[0] / "small.ecsv").write_bytes(b"abc")
        outside = self.root / "canonical-raw-large.nc"
        outside.write_bytes(b"x" * 1024 * 1024)
        link = roots[0] / "raw.nc"
        link.symlink_to(outside)
        first = capture.resource_inventory(roots)
        second = capture.resource_inventory(roots)
        self.assertEqual(first, second)
        self.assertEqual([row["kind"] for row in first["entries"]],
                         ["directory", "symlink", "regular-file",
                          "directory", "directory", "directory", "directory"])
        self.assertEqual(sum(row["kind"] == "directory"
                             for row in first["entries"]), 5)
        symlink_row = next(row for row in first["entries"]
                           if row["kind"] == "symlink")
        self.assertEqual(symlink_row["logical_bytes"], link.lstat().st_size)
        self.assertNotEqual(symlink_row["logical_bytes"], outside.stat().st_size)
        self.assertEqual(symlink_row["sha256"], hashlib.sha256(
            os.readlink(link).encode()).hexdigest())

    def test_resource_record_arithmetic_schema_and_capacity_failures(self) -> None:
        roots = self._five_roots()
        (roots[2] / "capture.bin").write_bytes(b"capture")
        records_root = roots[4] / "_campaign" / "resource-records"
        records_root.mkdir(parents=True)
        inventory_path = records_root / "CAP-POINT.pre.inventory.json"
        record_path = records_root / "CAP-POINT.pre.json"
        projection_path = self.root / "CAP-POINT.projection.json"
        projection = capture.command_resource_projection(namespace(
            stage="CAP-POINT", source=PACKAGE / "resource-report.json",
            output=projection_path))
        record = capture.command_resource_record(namespace(
            stage="CAP-POINT", phase="pre", governed_root=roots,
            filesystem_root=self.root, projection_authority=projection_path,
            inventory=inventory_path, record=record_path))
        self.assertEqual(record["logical_plus_projected_bytes"],
                         record["current_logical_bytes"] +
                         projection["projected_incremental_bytes"])
        self.assertEqual(record["allocated_plus_projected_bytes"],
                         record["current_allocated_bytes"] +
                         projection["projected_incremental_bytes"])
        self.assertEqual(record["filesystem_root"], str(self.root.resolve()))
        self.assertEqual(record["filesystem_device"], self.root.stat().st_dev)
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        self.assertEqual(record["inventory"]["sha256"], hashlib.sha256(
            capture.canonical_json(inventory)).hexdigest())
        schema = json.loads((PACKAGE / "resource-record.schema.json").read_text(
            encoding="utf-8"))
        jsonschema.Draft202012Validator(schema).validate(record)

        fake_over = {"schema_version": "sci-map-001-resource-inventory-v1",
                     "governed_roots": [str(path) for path in roots],
                     "entries": [{"logical_bytes": capture.CEILING,
                                  "allocated_bytes": 0}]}
        over_records = roots[4] / "_campaign" / "resource-records"
        with mock.patch.object(capture, "resource_inventory", return_value=fake_over), \
                self.assertRaisesRegex(capture.CaptureError, "resource gate fails"):
            capture.command_resource_record(namespace(
                stage="CAP-POINT", phase="pre", governed_root=roots,
                filesystem_root=self.root,
                projection_authority=projection_path,
                inventory=over_records / "CAP-POINT.pre.inventory.json",
                record=over_records / "CAP-POINT.pre.json"))

        disk = types.SimpleNamespace(total=1000, used=999, free=0)
        capacity_records = roots[4] / "_campaign" / "resource-records"
        with mock.patch.object(capture.shutil, "disk_usage", return_value=disk), \
                self.assertRaisesRegex(capture.CaptureError, "resource gate fails"):
            capture.command_resource_record(namespace(
                stage="CAP-POINT", phase="pre", governed_root=roots,
                filesystem_root=self.root, projection_authority=projection_path,
                inventory=capacity_records / "CAP-POINT.pre.inventory.json",
                record=capacity_records / "CAP-POINT.pre.json"))

        with self.assertRaisesRegex(capture.CaptureError, "cannot accept"):
            capture.command_resource_record(namespace(
                stage="post-nonzero", phase="post", governed_root=roots,
                filesystem_root=self.root, projection_authority=projection_path,
                inventory=records_root / "post-nonzero.post.inventory.json",
                record=records_root / "post-nonzero.post.json"))

        forged = json.loads(projection_path.read_text(encoding="utf-8"))
        forged["projected_incremental_bytes"] = 1
        forged_path = self.root / "forged.projection.json"
        write_json(forged_path, forged)
        with self.assertRaisesRegex(capture.CaptureError, "reconstruction differs"):
            capture._validate_resource_projection(forged_path, "CAP-POINT")

    def test_resource_gate_requires_exact_five_distinct_roots(self) -> None:
        roots = self._five_roots()
        with self.assertRaisesRegex(capture.CaptureError, "requires five roots"):
            capture.canonical_roots(roots[:4])
        with self.assertRaisesRegex(capture.CaptureError, "repeat"):
            capture.canonical_roots([*roots[:4], roots[0]])

    def test_capture_helper_has_no_network_submission_or_delete_calls(self) -> None:
        source = PROGRAM.read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden_imports = {
            "socket", "subprocess", "requests", "paramiko", "fabric",
            "urllib", "http", "ftplib",
        }
        imported = set()
        forbidden_calls = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and \
                        isinstance(node.func.value, ast.Name):
                    qualified = f"{node.func.value.id}.{node.func.attr}"
                    if qualified in {
                        "os.remove", "os.unlink", "os.rmdir",
                        "shutil.rmtree", "Path.unlink", "Path.rmdir",
                    }:
                        forbidden_calls.add(qualified)
        self.assertEqual(imported & forbidden_imports, set())
        self.assertEqual(forbidden_calls, set())
        self.assertNotRegex(source, r"\b(?:ssh|scp|rsync|sbatch|srun|scancel)\b")
        self.assertNotIn("tolproj copy-raw", source)

    def _write_ptc(self, path: Path, *, obsnum: int = 152389,
                   native_rate: float = 20.0,
                   scan_indices: np.ndarray | None = None) -> None:
        import netCDF4

        scans = np.asarray(
            [[0, 1], [2, 3]] if scan_indices is None else scan_indices,
            dtype=np.int64,
        )
        with netCDF4.Dataset(path, mode="w") as dataset:
            dataset.createDimension("time", 4)
            dataset.createDimension("detector", 3)
            dataset.createDimension("scan", 2)
            dataset.createDimension("bound", 2)
            for name in ("signal", "flags", "kernel", "det_lat", "det_lon"):
                variable = dataset.createVariable(
                    name, "f8", ("time", "detector"))
                variable[:] = np.arange(12, dtype=np.float64).reshape(4, 3)
            weights = dataset.createVariable(
                "weights", "f8", ("scan", "detector"))
            weights[:] = np.ones((2, 3), dtype=np.float64)
            apt_values = {
                "apt_flag": [0.0, 0.0, 0.0],
                "apt_array": [0.0, 1.0, 2.0],
                "apt_nw": [0.0, 7.0, 11.0],
                "apt_kids_tone": [10.0, 20.0, 30.0],
                "apt_uid": [100.0, 101.0, 102.0],
            }
            for name, values in apt_values.items():
                variable = dataset.createVariable(name, "f8", ("detector",))
                variable[:] = np.asarray(values, dtype=np.float64)
            scan_variable = dataset.createVariable(
                "scan_indices", "i8", ("scan", "bound"))
            scan_variable[:] = scans
            output_scan = dataset.createVariable(
                "output_scan_index", "i8", ("scan",))
            output_scan[:] = np.asarray([1, 2], dtype=np.int64)
            output_type = dataset.createVariable("tod_output_type", str)
            output_type[...] = "ptc"
            rate = dataset.createVariable("SAMPRATE", "f8")
            rate.assignValue(native_rate)
            observation = dataset.createVariable("obsnum", "i8")
            observation.assignValue(obsnum)
            for array, value in zip(capture.ARRAYS, (5.0, 6.0, 7.0)):
                major = dataset.createVariable(f"BMAJ_{array}", "f8")
                minor = dataset.createVariable(f"BMIN_{array}", "f8")
                major.assignValue(value)
                minor.assignValue(value + 1.0)

    def _write_raw_provenance(
            self, path: Path, *, native_rate: float = 20.0,
            effective_rate: float = 10.0, downsample_factor: int = 2,
            scan_count: int = 2) -> None:
        path.write_text(yaml.safe_dump({
            "schema_version": "citlali-raw-timestream-provenance-v2",
            "initialized": True,
            "observation": {
                "available": True,
                "value": {
                    "native_sample_rate_hz": {
                        "available": True, "value": native_rate,
                    },
                    "effective_sample_rate_hz": {
                        "available": True, "value": effective_rate,
                    },
                    "downsample_factor": {
                        "available": True, "value": downsample_factor,
                    },
                },
            },
            "realized": {
                "execution_completed": True,
                "completed_scan_count": {
                    "available": True, "value": scan_count,
                },
                "required_timestream_write_count": {
                    "available": True, "value": scan_count,
                },
            },
        }), encoding="utf-8")

    def _write_map_provenance(self, path: Path, *, obsnum: int = 152389) -> None:
        bundle = {
            "identity_digest": "canonical-hexfloat-sha256-v1:" + "1" * 64,
            "grouping": "array",
            "shape": {"rows": 3, "cols": 4},
            "wcs": {
                "coordinate_frame": "altaz",
                "reference_world": [
                    capture.raw_exact_float(12.5),
                    capture.raw_exact_float(45.0),
                ],
                "axis_units": ["deg", "deg"],
            },
            "ordered_slots": [
                {"grouping": "array", "stokes_identity": "I",
                 "array_identity": array}
                for array in capture.ARRAYS
            ],
        }
        path.write_text(yaml.safe_dump({
            "schema_version": "citlali-mapmaking-provenance-v3",
            "initialized": True,
            "realized": {
                "reduction_completed": True,
                "mapmaking_executed": True,
            },
            "observations": [{
                "obsnum": obsnum,
                "outputs_completed": True,
                "effective_pixel_size_rad": 0.001,
                "science_state": {
                    "available": True,
                    "bundle_identity": {
                        "available": True,
                        "value": bundle,
                    },
                },
            }],
        }), encoding="utf-8")

    def _source_selection_fixture(
            self, capture_root: Path, raw_sources: list[Path],
            apt_source: Path) -> Path:
        all_networks = [0, 7, 11]
        records = [{
            "id": f"staged-raw-152389-{array}", "role": "raw_timestream",
            "path": str(raw_source), "obsnums": [152389],
            "arrays": [array], "networks": [network],
        } for raw_source, array, network in zip(
            raw_sources, capture.ARRAYS, all_networks)]
        for array, network in zip(capture.ARRAYS, all_networks):
            source = capture_root / f"kids-{array}.dat"
            source.write_text(f"kids-{array}\n", encoding="utf-8")
            records.append({
                "id": f"kids-{array}", "role": "kids_fit_report",
                "path": str(source),
                "obsnums": [152389],
                "arrays": [array], "networks": [network],
            })
        calibration = capture_root / "calibration.dat"
        calibration.write_text("calibration\n", encoding="utf-8")
        records.extend((
            {"id": "staged-apt-152389", "role": "apt", "path": str(apt_source),
             "obsnums": [152389], "arrays": list(capture.ARRAYS),
             "networks": all_networks},
            {"id": "capture-calibration", "role": "calibration",
             "path": str(calibration), "obsnums": [152389],
             "arrays": list(capture.ARRAYS), "networks": all_networks},
            {"id": "staged-pointing-152389", "role": "pointing_support",
             "path": str(apt_source), "obsnums": [152389],
             "arrays": list(capture.ARRAYS), "networks": all_networks},
        ))
        selection = capture_root / "source-selection.json"
        write_json(selection, {
            "schema_version": "sci-map-001-source-selection-v1",
            "capture_id": "CAP-POINT",
            "records": records,
        })
        return selection

    def _ptc_authority_fixture(self) -> tuple[Path, Path, Path, Path]:
        capture_root = self.root / "capture-root"
        capture_root.mkdir()
        ptc = capture_root / "point-152389-ptc.nc"
        raw_provenance = capture_root / "raw_timestream_provenance.yaml"
        map_provenance = capture_root / "mapmaking_provenance.yaml"
        self._write_ptc(ptc)
        self._write_raw_provenance(raw_provenance)
        self._write_map_provenance(map_provenance)
        return capture_root, ptc, raw_provenance, map_provenance

    def test_ptc_and_rate_provenance_are_separate_exact_authorities(self) -> None:
        _, ptc_path, raw_path, _ = self._ptc_authority_fixture()
        ptc = capture.inspect_ptc(ptc_path, 152389)
        provenance = capture.realized_raw_provenance(raw_path, 152389, ptc)
        self.assertEqual(ptc["native_fsmp_hz"], 20.0)
        self.assertEqual(provenance["native_fsmp_hz"], 20.0)
        self.assertEqual(provenance["effective_d_fsmp_hz"], 10.0)
        self.assertEqual(provenance["downsample_factor"], 2)
        self.assertEqual(ptc["sample_count"], 4)
        self.assertEqual([row["sample_count"] for row in ptc["scan_order"]],
                         [2, 2])

        bad_rate = self.root / "bad-rate.yaml"
        self._write_raw_provenance(bad_rate, effective_rate=0.0)
        with self.assertRaisesRegex(capture.CaptureError, "finite positive"):
            capture.realized_raw_provenance(bad_rate, 152389, ptc)

        bad_native = self.root / "bad-native.yaml"
        self._write_raw_provenance(bad_native, native_rate=19.0)
        with self.assertRaisesRegex(capture.CaptureError, "not bit-equal"):
            capture.realized_raw_provenance(bad_native, 152389, ptc)

        bad_factor = self.root / "bad-factor.yaml"
        self._write_raw_provenance(bad_factor, downsample_factor=4)
        with self.assertRaisesRegex(capture.CaptureError,
                                    "fsmp/downsample_factor"):
            capture.realized_raw_provenance(bad_factor, 152389, ptc)

    def test_ptc_rejects_noncontiguous_full_all_scan_authority(self) -> None:
        path = self.root / "bad-scans.nc"
        self._write_ptc(path, scan_indices=np.asarray([[0, 1], [3, 3]]))
        with self.assertRaisesRegex(capture.CaptureError, "contiguous full timebase"):
            capture.inspect_ptc(path, 152389)

    def test_raw_manifest_producer_authority_and_capture_record_end_to_end(self) -> None:
        capture_root, ptc, raw_provenance, map_provenance = \
            self._ptc_authority_fixture()
        canonical = self.root / "canonical"
        canonical.mkdir()
        raw_basenames = [
            f"toltec{network}_152389_000_0002.nc"
            for network in (0, 7, 11)]
        raw_sources = [canonical / basename for basename in raw_basenames]
        for raw_source in raw_sources:
            raw_source.write_bytes(f"canonical raw {raw_source.name}".encode("ascii"))
        point_project = self.root / "point-source-project"
        point_data = point_project / "data"
        point_logs = point_project / "logs"
        point_apt = point_project / "apt"
        for path in (point_data, point_logs, point_apt):
            path.mkdir(parents=True, exist_ok=True)
        raw_link_manifest = point_logs / "raw-link-manifest.json"
        with self._redirect_canonical_path(canonical):
            capture.command_raw_manifest(namespace(
                capture_id="CAP-POINT", canonical_root=canonical,
                selection=self.selection("CAP-POINT", [{
                    "observation": 152389, "basename": basename}
                    for basename in raw_basenames],
                    "point-raw-selection.json"), output=raw_link_manifest))
            raw_link_staging = point_logs / "raw-link-staging.json"
            capture.command_stage_raw(namespace(
                manifest=raw_link_manifest, destination=point_data,
                output=raw_link_staging))

        authority_root = self.root / "authorities"
        authority_root.mkdir()
        apt_name = "apt_152389_matched.ecsv"
        (authority_root / apt_name).write_bytes(b"point apt authority")
        authority_manifest = point_logs / "authority-staging.json"
        capture.command_stage_authorities(namespace(
            capture_id="CAP-POINT", authority_root=authority_root,
            selection=self.authority_selection("CAP-POINT", [{
                "role": "apt", "observation": 152389, "basename": apt_name}]),
            apt_destination=point_apt, ppt_destination=None,
            output=authority_manifest))
        staged_apt = point_apt / apt_name
        source_selection = self._source_selection_fixture(
            capture_root, raw_sources, staged_apt)
        with self._redirect_canonical_path(canonical):
            raw_manifest = capture.command_build_raw_input_manifest(namespace(
                capture_id="CAP-POINT", capture_root=capture_root,
                ptc=[f"152389={ptc}"],
                raw_provenance=[f"152389={raw_provenance}"],
                map_provenance=[f"152389={map_provenance}"],
                source_selection=source_selection,
                raw_link_manifest=raw_link_manifest,
                raw_link_staging=raw_link_staging,
                authority_manifest=authority_manifest,
            ))
        raw_manifest_path = capture_root / "raw-input-manifest.json"
        self.assertTrue(raw_manifest_path.is_file())
        self.assertEqual(len(raw_manifest["memberships"]), 3)
        self.assertEqual(
            {(row["obsnum"], row["array"]) for row in raw_manifest["memberships"]},
            {(152389, array) for array in capture.ARRAYS},
        )
        for membership in raw_manifest["memberships"]:
            projection = membership["projection"]
            self.assertEqual(
                capture.parse_exact(projection["native_fsmp_hz"], "native"),
                20.0,
            )
            self.assertEqual(
                capture.parse_exact(projection["effective_d_fsmp_hz"], "effective"),
                10.0,
            )
            self.assertEqual(
                capture.parse_exact(projection["sample_interval_s"], "interval"),
                0.1,
            )

        producer_path = capture_root / "152389-a1100-authority.json"
        with self._redirect_canonical_path(canonical):
            producer = capture.command_producer_authority(namespace(
                raw_input_manifest=raw_manifest_path,
                obsnum=152389, array="a1100", output=producer_path))
        self.assertEqual(producer["capture_output_mode"], "full")
        self.assertEqual(producer["capture_detector_selection"], "all")
        self.assertEqual(producer["primitive_term_count"], 4)
        self.assertEqual(producer["native_fsmp_hz"]["authority"],
                         "telescope.fsmp")
        self.assertEqual(producer["effective_d_fsmp_hz"]["authority"],
                         "telescope.d_fsmp")
        self.assertEqual(
            producer["realized_raw_timestream_provenance_sha256"],
            capture.sha256(raw_provenance),
        )
        self.assertEqual(
            producer["realized_mapmaking_provenance_sha256"],
            capture.sha256(map_provenance),
        )
        self.assertNotEqual(
            producer["realized_raw_timestream_provenance_sha256"],
            producer["realized_mapmaking_provenance_sha256"],
        )

        binary = self.root / "citlali"
        binary.write_bytes(b"ordinary exact candidate binary")
        binary.chmod(0o755)
        compiler = self.root / "compiler"
        compiler.write_bytes(b"synthetic compiler")
        build_manifest = self.root / "candidate-build.json"
        version_output = self.root / "candidate-version.txt"
        version_output.write_text("citlali synthetic version\n", encoding="utf-8")
        write_json(build_manifest, {
            "schema_version": "sci-map-unity-build-state-v1",
            "started_at": "2026-08-02T12:00:00Z",
            "completed_at": "2026-08-02T12:01:00Z",
            "candidate_sha": capture.CANDIDATE,
            "candidate_tree": capture.CANDIDATE_TREE,
            "build_preset": "unity_release", "build_target": "citlali_cli",
            "binary": str(binary),
            "binary_sha256": capture.sha256(binary),
            "built_binary": str(binary),
            "built_binary_sha256": capture.sha256(binary),
            "cmake_cache_sha256": "1" * 64,
            "compile_commands_sha256": "2" * 64,
            "compiler": str(compiler), "compiler_sha256": capture.sha256(compiler),
            "version_output": str(version_output),
            "version_output_sha256": capture.sha256(version_output),
            "binary_count": 1, "ordinary": True, "instrumented": False,
            "dependencies": {},
        })
        fixed_root, realized_root, fixed_merged, realized_merged = \
            self._config_fixture()
        fixed_inventory, realized_inventory = self._write_config_inventories(
            fixed_root, realized_root)
        proof_root = capture_root / "capture-authority"
        proof_root.mkdir()
        config_proof = proof_root / "config-proof.json"
        capture.command_config_proof(namespace(
            capture_id="CAP-POINT", fixed_config=fixed_merged,
            capture_config=realized_merged, fixed_inventory=fixed_inventory,
            capture_inventory=realized_inventory, output=config_proof))
        with self._redirect_canonical_path(canonical):
            record = capture.command_capture_record(namespace(
                capture_id="CAP-POINT", capture_root=capture_root,
                binary=binary, build_manifest=build_manifest,
                version_output=version_output,
                raw_link_manifest=raw_link_manifest,
                raw_link_staging=raw_link_staging,
                authority_manifest=authority_manifest,
                config_proof=config_proof, ptc=[f"152389={ptc}"],
                raw_provenance=[f"152389={raw_provenance}"],
            ))
        record_path = capture_root / "capture-record.json"
        self.assertTrue(record_path.is_file())
        self.assertEqual(record["binary_sha256"], capture.sha256(binary))
        self.assertEqual(record["raw_input_manifest"], {
            "path": str(raw_manifest_path),
            "sha256": capture.sha256(raw_manifest_path),
        })
        self.assertEqual(record["rates"]["native_fsmp_hz"]["decimal"], "20")
        self.assertEqual(record["rates"]["effective_d_fsmp_hz"]["decimal"], "10")
        with self._redirect_canonical_path(canonical):
            verified = capture.command_verify_capture_record(namespace(
                capture_record=record_path))
        self.assertEqual(verified["status"], "pass")
        self.assertEqual(verified["full_ptc_count"], 1)

        original_record_bytes = record_path.read_bytes()
        tampered = json.loads(original_record_bytes)
        tampered["rates"]["sample_interval_s"] = capture.exact_float(
            0.2, "binary64(1/telescope.d_fsmp)")
        record_path.chmod(0o644)
        write_json(record_path, tampered)
        with self._redirect_canonical_path(canonical), self.assertRaisesRegex(
                capture.CaptureError, "sample interval differs"):
            capture.command_verify_capture_record(namespace(
                capture_record=record_path))
        record_path.write_bytes(original_record_bytes)
        record_path.chmod(0o444)

        tampered = json.loads(original_record_bytes)
        tampered["ptc_outputs"][0]["scan_order"][0]["exposure_s"] = \
            capture.exact_float(
                0.4, "binary64(sample_count/telescope.d_fsmp)")
        record_path.chmod(0o644)
        write_json(record_path, tampered)
        with self._redirect_canonical_path(canonical), self.assertRaisesRegex(
                capture.CaptureError, "scan exposure differs"):
            capture.command_verify_capture_record(namespace(
                capture_record=record_path))
        record_path.write_bytes(original_record_bytes)
        record_path.chmod(0o444)

        def assert_live_tamper_rejected(path: Path, pattern: str) -> None:
            original = path.read_bytes()
            original_stat = path.stat()
            mode = original_stat.st_mode & 0o777
            path.chmod(mode | 0o200)
            path.write_bytes(original + b"tamper")
            try:
                with self._redirect_canonical_path(canonical), \
                        self.assertRaisesRegex(capture.CaptureError, pattern):
                    capture.command_verify_capture_record(namespace(
                        capture_record=record_path))
            finally:
                path.write_bytes(original)
                path.chmod(mode)
                os.utime(path, ns=(original_stat.st_atime_ns,
                                   original_stat.st_mtime_ns))

        assert_live_tamper_rejected(binary, "does not bind the exact ordinary binary")
        assert_live_tamper_rejected(
            version_output, "does not bind the exact ordinary binary")
        assert_live_tamper_rejected(raw_sources[0], "raw.*digest differs")
        assert_live_tamper_rejected(staged_apt, "authority.*digest differs")
        assert_live_tamper_rejected(
            config_proof, "config proof digest/outcome differs")

        extra_ptc = capture_root / "unrecorded-full-ptc.nc"
        shutil.copy2(ptc, extra_ptc)
        with self._redirect_canonical_path(canonical), self.assertRaisesRegex(
                capture.CaptureError, "full-PTC inventory differs"):
            capture.command_verify_capture_record(namespace(
                capture_record=record_path))
        self.assertEqual(record["rates"]["sample_interval_s"]["decimal"], "0.10000000000000001")
        self.assertEqual(record["ptc_outputs"][0]["scan_order"][0]
                         ["exposure_s"]["decimal"], "0.20000000000000001")
        self.assertTrue(record["retained"])
        self.assertFalse(record["retention"]["automatic_cleanup"])
        schema = json.loads((PACKAGE / "capture-record.schema.json").read_text(
            encoding="utf-8"))
        jsonschema.Draft202012Validator(schema).validate(record)

    def test_parser_exposes_only_local_capture_commands(self) -> None:
        expected = {
            "raw-manifest", "stage-raw", "stage-authorities",
            "config-inventory", "capture-overlay", "config-proof",
            "resource-projection", "resource-record",
            "build-raw-input-manifest", "producer-authority",
            "capture-record", "verify-capture-record", "self-check",
        }
        subparser_action = next(
            action for action in capture.parser()._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        self.assertEqual(set(subparser_action.choices), expected)


if __name__ == "__main__":
    unittest.main()
