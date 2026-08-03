#!/usr/bin/env python3
"""Run the read-only SCI-ALIGN-001 diagnostic for retained 3C273 Beammaps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

try:
    from tools.diagnostics.sci_align_001_3c273_common import (
        AnalysisProducts,
        AnalysisProtocol,
        ContractError,
        RawLinkageError,
        ReductionInputs,
        atomic_write_csv,
        atomic_write_json,
        atomic_write_text,
        canonical_json,
        checksum_lines,
        parse_manifest,
        resume_binding_digest,
        resume_is_valid,
        safe_candidate_component,
        sha256_file,
        source_write_guard,
        table_fields,
        analyze_reduction,
    )
except ModuleNotFoundError:  # direct execution from tools/diagnostics
    from sci_align_001_3c273_common import (  # type: ignore[no-redef]
        AnalysisProducts,
        AnalysisProtocol,
        ContractError,
        RawLinkageError,
        ReductionInputs,
        atomic_write_csv,
        atomic_write_json,
        atomic_write_text,
        canonical_json,
        checksum_lines,
        parse_manifest,
        resume_binding_digest,
        resume_is_valid,
        safe_candidate_component,
        sha256_file,
        source_write_guard,
        table_fields,
        analyze_reduction,
    )


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_PROTOCOL = (
    REPO
    / "validation/sci_align_001_3c273_corpus_tooling_2026-08-03"
    / "frozen_analysis_protocol.json"
)


class DigestCache:
    """Owner-output cache that prevents repeated hashing of immutable large inputs."""

    def __init__(self, output_root: Path) -> None:
        self.path = output_root / "_input_digest_cache.json"
        self.values: dict[str, str] = {}
        if self.path.is_file():
            try:
                value = json.loads(self.path.read_text())
            except json.JSONDecodeError as error:
                raise ContractError(f"invalid digest cache {self.path}: {error}") from error
            if value.get("schema") != "sci-align-001-input-digest-cache-v1":
                raise ContractError(f"unsupported digest cache schema in {self.path}")
            self.values = {
                str(key): str(digest)
                for key, digest in value.get("entries", {}).items()
                if len(str(digest)) == 64
                and all(character in "0123456789abcdef" for character in str(digest))
            }
        self.changed = False

    @staticmethod
    def key(path: Path) -> str:
        stat = path.stat()
        return canonical_json(
            {
                "path": str(path.resolve()),
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
            }
        )

    def digest(self, path: Path, supplied: str | None = None) -> tuple[str, str]:
        expected = None
        if supplied:
            expected = supplied.lower()
            if len(expected) != 64 or any(
                character not in "0123456789abcdef" for character in expected
            ):
                raise ContractError(f"malformed supplied SHA-256 for {path}")
        key = self.key(path)
        if key in self.values:
            measured = self.values[key]
        else:
            measured = sha256_file(path)
            self.values[key] = measured
            self.changed = True
        if expected is not None and measured != expected:
            raise ContractError(
                f"current file SHA-256 differs from supplied authority for {path}: "
                f"expected {expected}, measured {measured}"
            )
        source = (
            "runner_sha256_validated_against_supplied_authority"
            if expected is not None
            else "runner_sha256_digest_cache"
        )
        return measured, source

    def publish(self) -> None:
        if self.changed or not self.path.exists():
            atomic_write_json(
                self.path,
                {
                    "schema": "sci-align-001-input-digest-cache-v1",
                    "physical_identity": "path:device:inode:size:mtime_ns:ctime_ns",
                    "entries": dict(sorted(self.values.items())),
                },
            )


def inherited_input_digests(inputs: ReductionInputs) -> dict[str, str]:
    result = dict(inputs.supplied_sha256)
    for ancestor in [inputs.reduction_path, *inputs.reduction_path.parents[:6]]:
        preparation = ancestor / "evidence/preparation.json"
        if not preparation.is_file():
            continue
        try:
            document = json.loads(preparation.read_text())
        except json.JSONDecodeError:
            continue
        for row in document.get("selected_inputs", []):
            path = row.get("path")
            digest = row.get("sha256")
            if path and digest:
                result[str(Path(path).expanduser().resolve())] = str(digest)
    return result


def build_input_manifest(
    inputs: ReductionInputs,
    enhanced: bool,
    cache: DigestCache,
    selected_manifest: Path | None = None,
) -> list[dict[str, Any]]:
    inherited = inherited_input_digests(inputs)
    rows = []
    for role, path in inputs.required_paths(enhanced):
        if role == "reduction":
            continue
        resolved = path.resolve()
        digest, source = cache.digest(resolved, inherited.get(str(resolved)))
        rows.append(
            {
                "role": role,
                "path": str(resolved),
                "size_bytes": resolved.stat().st_size,
                "sha256": digest,
                "digest_source": source,
                "mutated": False,
            }
        )
    if selected_manifest is not None:
        path = selected_manifest.resolve()
        digest, source = cache.digest(path, inherited.get(str(path)))
        rows.append(
            {
                "role": "selected_manifest",
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": digest,
                "digest_source": source,
                "mutated": False,
            }
        )
    return sorted(rows, key=lambda row: (row["role"], row["path"]))


def tool_digests() -> dict[str, str]:
    common = HERE / "sci_align_001_3c273_common.py"
    return {
        str(common.relative_to(REPO)): sha256_file(common),
        str(Path(__file__).resolve().relative_to(REPO)): sha256_file(Path(__file__).resolve()),
    }


def _write_table(path: Path, rows: list[Mapping[str, Any]], preferred: list[str]) -> None:
    atomic_write_csv(path, rows, table_fields(rows, preferred))


def publish_products(
    directory: Path,
    products: AnalysisProducts,
    input_rows: list[dict[str, Any]],
    binding: Mapping[str, Any],
    log: list[str],
    enhanced_failure: Mapping[str, Any] | None = None,
) -> None:
    atomic_write_json(directory / "map_summary.json", products.map_summary)
    _write_table(
        directory / "map_summary.csv",
        [products.map_summary],
        ["map_id", "observation_number", "analysis_mode", "status", "quality"],
    )
    atomic_write_json(directory / "map_result.json", products.map_result)
    _write_table(
        directory / "network_map_results.csv",
        products.network_rows,
        [
            "map_id",
            "observation_number",
            "network_id",
            "array",
            "available",
            "status",
            "detector_count",
            "timing_residual_sec",
            "timing_se_sec",
        ],
    )
    _write_table(
        directory / "timing_phase_results.csv",
        products.timing_rows,
        [
            "map_id",
            "observation_number",
            "model_id",
            "time_basis",
            "row_shift_k",
            "phase_phi_samples",
            "support",
            "group",
            "quality",
            "timing_residual_sec",
            "timing_se_sec",
        ],
    )
    _write_table(
        directory / "fit_controls.csv",
        products.fit_control_rows,
        [
            "map_id",
            "model_id",
            "support",
            "group",
            "uid",
            "network_id",
            "array",
            "direction",
            "quality",
            "reason",
        ],
    )
    atomic_write_json(directory / "fit_controls.json", products.fit_controls)
    _write_table(
        directory / "scan_registry.csv",
        products.scan_registry,
        [
            "stable_scan_id",
            "compatibility_ordinal_1based",
            "classification",
            "selected",
            "exclusion_reason",
        ],
    )
    _write_table(
        directory / "raw_counter_transitions.csv",
        products.raw_counter_rows,
        table_fields(products.raw_counter_rows),
    )
    _write_table(
        directory / "raw_phase_summary.csv",
        products.raw_phase_rows,
        table_fields(products.raw_phase_rows),
    )
    _write_table(
        directory / "raw_pps_time_increment_anomalies.csv",
        products.raw_pps_time_increment_anomaly_rows,
        table_fields(products.raw_pps_time_increment_anomaly_rows),
    )
    _write_table(
        directory / "input_manifest.csv",
        input_rows,
        ["role", "path", "size_bytes", "sha256", "digest_source", "mutated"],
    )
    atomic_write_json(
        directory / "input_manifest.json",
        {"schema": "sci-align-001-3c273-input-manifest-v1", "inputs": input_rows},
    )
    atomic_write_json(directory / "resume_binding.json", binding)
    if enhanced_failure is not None:
        atomic_write_json(directory / "enhanced_failure.json", enhanced_failure)
    atomic_write_text(directory / "run.log", "\n".join(log) + "\n")
    atomic_write_text(directory / "SHA256SUMS", checksum_lines(directory))


def resolved_mode(inputs: ReductionInputs, requested: str) -> str:
    if requested == "auto":
        return "enhanced" if inputs.enhanced_eligible else "core"
    return requested


def dry_run_plan(
    candidates: list[ReductionInputs], output_root: Path, requested_mode: str
) -> dict[str, Any]:
    rows = []
    for inputs in candidates:
        mode = resolved_mode(inputs, requested_mode)
        inputs.validate(mode == "enhanced")
        source_write_guard(inputs, output_root)
        rows.append(
            {
                "candidate_id": inputs.candidate_id,
                "observation_number": inputs.observation_number,
                "mode": mode,
                "output_directory": str(
                    output_root.resolve() / safe_candidate_component(inputs.candidate_id)
                ),
                "inputs": inputs.identity(),
            }
        )
    return {
        "schema": "sci-align-001-3c273-dry-run-plan-v1",
        "writes_performed": False,
        "citlali_reductions_launched": False,
        "candidates": rows,
    }


def run_one(
    inputs: ReductionInputs,
    output_root: Path,
    protocol: AnalysisProtocol,
    requested_mode: str,
    resume: bool,
    cache: DigestCache,
    selected_manifest: Path | None = None,
) -> str:
    requested_resolved_mode = resolved_mode(inputs, requested_mode)
    inputs.validate(False)
    analysis_mode = requested_resolved_mode
    enhanced_failure: dict[str, Any] | None = None
    if requested_resolved_mode == "enhanced":
        try:
            inputs.validate(True)
        except ContractError as error:
            analysis_mode = "core"
            enhanced_failure = {
                "schema": "sci-align-001-3c273-enhanced-failure-v1",
                "status": "enhanced_failed_core_retained",
                "stage": "enhanced_input_validation",
                "error_type": type(error).__name__,
                "error": str(error),
                "silent_downgrade": False,
                "core_analysis_executed": True,
            }
    source_write_guard(inputs, output_root)
    directory = output_root / safe_candidate_component(inputs.candidate_id)
    if directory.exists() and any(directory.iterdir()) and not resume:
        raise ContractError(
            f"output directory is not empty (use --resume): {directory}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    input_rows = build_input_manifest(
        inputs,
        analysis_mode == "enhanced",
        cache,
        selected_manifest,
    )
    selected_manifest_sha256 = next(
        (
            str(row["sha256"])
            for row in input_rows
            if row["role"] == "selected_manifest"
        ),
        None,
    )
    cache.publish()
    tools = tool_digests()
    binding_protocol = {
        "analysis_protocol": protocol.to_dict(),
        "requested_mode": requested_resolved_mode,
        "candidate_identity": inputs.identity(),
    }
    binding_sha = resume_binding_digest(binding_protocol, input_rows, tools)
    binding = {
        "schema": "sci-align-001-3c273-resume-binding-v1",
        "binding_sha256": binding_sha,
        "protocol": binding_protocol,
        "selected_manifest_sha256": selected_manifest_sha256,
        "input_manifest_sha256": sha256_file_from_rows(input_rows),
        "tool_sha256": tools,
    }
    binding_path = directory / "resume_binding.json"
    if resume and binding_path.is_file():
        previous = json.loads(binding_path.read_text())
        if previous.get("binding_sha256") != binding_sha:
            raise ContractError(
                f"resume binding differs for {inputs.candidate_id}; refusing stale output reuse"
            )
        if resume_is_valid(directory, binding_sha):
            summary = json.loads((directory / "map_summary.json").read_text())
            return (
                "resumed_partial_core_success_enhanced_failed"
                if summary.get("enhanced_status") == "failed"
                else "resumed_complete"
            )
    elif resume and any(directory.iterdir()):
        raise ContractError(
            f"incomplete output lacks a valid resume binding: {directory}"
        )
    atomic_write_json(binding_path, binding)
    log = [
        "SCI-ALIGN-001 3C273 retained-product runner",
        f"candidate_id={inputs.candidate_id}",
        f"observation_number={inputs.observation_number}",
        f"requested_mode={requested_resolved_mode}",
        f"analysis_mode={analysis_mode}",
        f"resume_binding_sha256={binding_sha}",
        "source_products_modified=false",
        "citlali_reduction_launched=false",
    ]
    try:
        try:
            products = analyze_reduction(inputs, protocol, analysis_mode, log)
        except RawLinkageError as error:
            if requested_resolved_mode != "enhanced":
                raise
            enhanced_failure = {
                "schema": "sci-align-001-3c273-enhanced-failure-v1",
                "status": "enhanced_failed_core_retained",
                "stage": "raw_linkage_proof",
                "error_type": type(error).__name__,
                "error": str(error),
                "silent_downgrade": False,
                "core_analysis_executed": True,
            }
            log.append(f"enhanced_status=failure error={error}")
            log.append("core_fallback=explicit")
            analysis_mode = "core"
            products = analyze_reduction(inputs, protocol, "core", log)
        products.map_summary["analysis_role"] = inputs.analysis_role
        products.map_result.setdefault("identity", inputs.identity())
        products.map_result["identity"]["analysis_role"] = inputs.analysis_role
        products.map_result["summary"] = products.map_summary
        if enhanced_failure is not None:
            products.map_summary["analysis_mode"] = "core"
            products.map_summary["enhanced_status"] = "failed"
            products.map_summary["enhanced_failure_stage"] = enhanced_failure["stage"]
            products.map_summary["enhanced_failure_reason"] = enhanced_failure["error"]
            products.map_summary["status"] = "partial_core_success_enhanced_failed"
            products.map_result["summary"] = products.map_summary
            products.map_result["enhanced_failure"] = enhanced_failure
            products.map_result["scope"]["enhanced_analysis_complete"] = False
            for row in products.network_rows:
                row["enhanced_status"] = "failed"
                row["enhanced_failure_reason"] = enhanced_failure["error"]
        products.map_result["selected_manifest_sha256"] = selected_manifest_sha256
        log.append("status=success")
        publish_products(
            directory,
            products,
            input_rows,
            binding,
            log,
            enhanced_failure,
        )
    except Exception as error:
        log.append(f"status=failure error_type={type(error).__name__}")
        log.append(f"error={error}")
        atomic_write_text(directory / "run.log", "\n".join(log) + "\n")
        atomic_write_json(
            directory / "failure.json",
            {
                "schema": "sci-align-001-3c273-failure-v1",
                "candidate_id": inputs.candidate_id,
                "error_type": type(error).__name__,
                "error": str(error),
                "source_products_modified": False,
                "citlali_reduction_launched": False,
            },
        )
        raise
    return (
        "partial_core_success_enhanced_failed"
        if enhanced_failure is not None
        else "completed"
    )


def sha256_file_from_rows(rows: list[Mapping[str, Any]]) -> str:
    import hashlib

    return hashlib.sha256(canonical_json(rows).encode("utf-8")).hexdigest()


def parse_raw_argument(values: list[str]) -> dict[int, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ContractError("--raw must be NETWORK=PATH")
        identity, path = value.split("=", 1)
        network = int(identity.removeprefix("toltec"))
        resolved = Path(path).expanduser().resolve()
        if network in result and result[network] != resolved:
            raise ContractError(f"duplicate --raw value for network {network}")
        result[network] = resolved
    return dict(sorted(result.items()))


def direct_candidate(args: argparse.Namespace) -> ReductionInputs:
    required = {
        "--candidate-id": args.candidate_id,
        "--observation-number": args.observation_number,
        "--reduction-root": args.reduction_root,
        "--config": args.config,
    }
    missing = [name for name, value in required.items() if value in (None, "")]
    if missing:
        raise ContractError(f"direct one-map mode requires {missing}")
    row = {
        "candidate_id": args.candidate_id,
        "observation_number": args.observation_number,
        "reduction_path": args.reduction_root,
        "project_path": args.project_path,
        "config_path": args.config,
        "detector_tod_path": args.detector_tod,
        "output_apt_path": args.output_apt,
        "provenance_path": args.provenance,
        "telescope_path": args.telescope,
        "raw_files_json": {
            str(network): str(path)
            for network, path in parse_raw_argument(args.raw).items()
        },
        "analysis_role": "primary",
        "core_eligible": True,
        "enhanced_eligible": bool(args.raw) or args.mode == "enhanced",
    }
    return ReductionInputs.from_mapping(row)


def select_candidates(args: argparse.Namespace) -> list[ReductionInputs]:
    if args.manifest:
        candidates = parse_manifest(args.manifest.resolve())
        if args.candidate_id:
            candidates = [
                item for item in candidates if item.candidate_id == args.candidate_id
            ]
            if not candidates:
                raise ContractError(
                    f"candidate_id {args.candidate_id!r} is absent from {args.manifest}"
                )
        return candidates
    return [direct_candidate(args)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read retained Beammap products; never launch Citlali or write sources."
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--candidate-id")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("auto", "core", "enhanced"), default="auto")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--observation-number", "--obsnum", type=int)
    parser.add_argument("--reduction-root", type=Path)
    parser.add_argument("--project-path", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--detector-tod", type=Path)
    parser.add_argument("--output-apt", type=Path)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--telescope", type=Path)
    parser.add_argument("--raw", action="append", default=[], metavar="NETWORK=PATH")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        candidates = select_candidates(args)
        if not candidates:
            raise ContractError("no manifest candidates selected")
        protocol_path = args.protocol
        if protocol_path is None:
            if not DEFAULT_PROTOCOL.is_file():
                raise ContractError(
                    f"required frozen corpus protocol is missing: {DEFAULT_PROTOCOL}"
                )
            protocol_path = DEFAULT_PROTOCOL
        protocol = AnalysisProtocol.from_json(protocol_path)
        if (
            protocol.authority_schema_version
            != "sci-align-001-3c273-corpus-protocol-v2"
        ):
            raise ContractError(
                "per-map execution requires the frozen corpus protocol schema"
            )
        output_root = args.output_root.expanduser().resolve()
        if args.dry_run:
            print(canonical_json(dry_run_plan(candidates, output_root, args.mode)))
            return 0
        output_root.mkdir(parents=True, exist_ok=True)
        cache = DigestCache(output_root)
        failures = []
        for inputs in candidates:
            try:
                status = run_one(
                    inputs,
                    output_root,
                    protocol,
                    args.mode,
                    args.resume,
                    cache,
                    args.manifest.resolve() if args.manifest else None,
                )
                print(f"{inputs.candidate_id}: {status}", flush=True)
                if "partial_core_success_enhanced_failed" in status:
                    failures.append(
                        (
                            inputs.candidate_id,
                            "EnhancedAnalysisFailed",
                            "core result retained; inspect enhanced_failure.json",
                        )
                    )
            except Exception as error:
                failures.append((inputs.candidate_id, type(error).__name__, str(error)))
                print(
                    f"{inputs.candidate_id}: failure {type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
        if failures:
            print(canonical_json({"failures": failures}), file=sys.stderr)
            return 1
        return 0
    except (ContractError, OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
