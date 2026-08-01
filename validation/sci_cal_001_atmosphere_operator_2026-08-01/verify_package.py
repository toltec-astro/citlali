#!/usr/bin/env python3
"""Verify schemas, frozen provenance, generated tables, and package digests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import jsonschema


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
RAW_SOURCE_DIR = Path("/Users/gwilson/GitHub/toltec_beammap/src/toltec_sensitivity")
AM_ROOT = Path("/Users/gwilson/work_toltec/local_data/AM")
TOLTECA_REPOSITORY = Path("/Users/gwilson/GitHub/tolteca")
TOLTECA_REVISION = "2791e6a1e6349ad1d3ac549a648f41cbc51b98c7"
OWNER_SUPPLIED_SCHEMA_NAME = "owner_supplied_manifest.schema.json"
OWNER_SUPPLIED_SCHEMA_SHA256 = (
    "c4d534a89f5b8b3a6441db6142addca1063b6a6ef3db126c2b727668a5634146"
)
OWNER_INPUT_REQUEST_SHA256 = (
    "e8ba1c641d4acd71a4983fd4528d453a2b8078b3f45a87a803ad4523f3f942ef"
)

FROZEN_PATHS = {
    REPO_ROOT
    / "include/citlali/core/timestream/rtc/calibrate.h": "d70a55278227b43cdd7de19bc67e4ddb332524d40e1455c5fa7a80ae5e2d11ee",
    REPO_ROOT
    / "include/citlali/core/timestream/extinction_model_selection.h": "45cf86bbb2318c22514411f6d2a0e0371e22e9e355e61b293d93c628d9f3469d",
    REPO_ROOT
    / "validation/sci_cal_001_phase0_2026-07-31/generate_q_model_continuity.py": "a46211c007bdc1fa11d1408c6db4c4a68264ca00cd383806fd421ba978fffe78",
    REPO_ROOT
    / "validation/sci_cal_001_phase0_2026-07-31/Q_MODEL_CONTINUITY_REPORT.md": "1a5a5cfe057e8871f299c9216e69826e393ac518a6198bdbb44f2a48d11cac00",
    REPO_ROOT
    / "validation/sci_cal_001_phase0_2026-07-31/q_model_continuity_table.csv": "6de859fbc3e3f91376e2a6ad841f6f4f5d1eac0b773c25b521b8d5fffa5ec50f",
    REPO_ROOT
    / "validation/sci_cal_001_phase0_2026-07-31/SHA256SUMS": "c9db2008c9f9e1b0d27a98c108e6139025ef939c0323d471cce1b9d317b7e6c6",
    Path(
        "/private/tmp/citlali-scientific-audit-framework/doc/audits/packages/SCI-CAL-001_ACCURACY_AND_MODEL_SCOPE_AMENDMENT_2026-08-01.md"
    ): "ae295b717da3f9ed7129661e812b05847ff988500bb3d6f159b7ffae45cb2780",
    Path(
        "/private/tmp/citlali-scientific-audit-framework/doc/audits/packages/SCI-CAL-001_OPACITY_DECISION_AMENDMENT_2026-07-31.md"
    ): "64fd3ae9788c6a8e3db18ac5ea4f04799586b548f9e7ec12cc8c18f9cbf96e09",
    Path(
        "/private/tmp/citlali-scientific-audit-framework/doc/audits/packages/SCI-CAL-001_COORDINATOR_DECISION_2026-07-31.md"
    ): "1e2d44b76138734aaae5a7b241ae621deb1510c2de763b74be31a87e106d175c",
    Path(
        "/private/tmp/citlali-scientific-audit-framework/doc/audits/packages/SCI-CAL-001_BOUNDED_REPAIR_REAUDIT_HANDOFF_2026-07-31.md"
    ): "9d2c0ae89244d355070d6b300f431ac1799787b835c7e4cb76c8d7fc51cde106",
    Path(
        "/private/tmp/citlali-scientific-audit-framework/doc/audits/handoffs/SCI-CAL-001/SCI-CAL-001-XAUD-001.yaml"
    ): "2248422a507455e972c70c221c214b40fec68566011d27a9d8827952e43087d5",
    Path(
        "/Users/gwilson/work_toltec/local_data/doc/mmccrackan_dissertation.pdf"
    ): "2aa4373aaa0394f1a79e6668047a7aecd07d4914ce162c931f495d5502a49be0",
}

FROZEN_RAW_SOURCE_FILES = {
    "amLMT25.npz": "6ddffcd2c68bbc0f6d8f6470eba0d1aa81457dcc2f348fd2d7e44c9dfe48c87b",
    "amLMT50.npz": "1fe6dd2ab7a4d65f445e20c5a8f438eb42884836e7932d86f80c30e235710f81",
    "amLMT75.npz": "adbb8eb974c4e2744c3efb0f627708565f954c4029d9345e4f434699e8843f8e",
    "LMTAtmosphere.py": "66f580b85ccbfff9152519ec644df363e4571b9263fe06849dc89aa1858e52d0",
    "Detector.py": "82105317865ae1182d88d0874ed96c36a2b8c79c56d7fc6bb1990f008bd81d1a",
    "model_passbands.npz": (
        "861e6ce7af55b18c14a800defaf0b9a11099a16c307da08e391e1d8f79a39765"
    ),
}

CSV_ARTIFACTS = (
    "candidate_disagreement_metrics.csv",
    "candidate_surface_metrics.csv",
    "copied_am_annual_fit_coefficients.csv",
    "copied_am_legacy_comparison.csv",
    "copied_am_operator_stress_metrics.csv",
    "copied_am_operator_stress_rows.csv",
    "copied_am_product_inventory.csv",
    "copied_am_raw_output_inventory.csv",
    "frequency_resolution_metrics.csv",
    "leave_one_anchor_out_metrics.csv",
    "legacy_anchor_metrics.csv",
    "legacy_anchor_surface.csv",
    "raw_anchor_fit_metrics.csv",
    "raw_anchor_operator_metrics.csv",
    "raw_grid_physical_metrics.csv",
    "raw_q50_holdout_metrics.csv",
    "raw_q50_operator_holdout_metrics.csv",
    "recovered_fit_coefficients.csv",
    "recovered_raw_nominal_grid.csv",
)

FOLLOWUP_REQUIRED_FILES = (
    "COPIED_AM_FOLLOWUP_REPORT.md",
    "FOLLOWUP_STUDY_PREREGISTRATION.md",
    "FOLLOWUP_STUDY_PROTOCOL_ADDENDUM.md",
    "FOLLOWUP_STUDY_DEVIATION_LOG.md",
    "FREQUENCY_RESOLUTION_REPORT.md",
    "analyze_copied_am_suite.py",
    "copied_am_manifest.json",
    "frequency_resolution_manifest.json",
    "probe_am12_h2o_scale_hypotheses.py",
    "run_am12_native_regeneration_check.py",
    "run_am12_resolution_convergence.py",
)

NATIVE_REGENERATION_ARTIFACTS = (
    "native_regeneration_metrics.csv",
    "native_regeneration_manifest.json",
    "NATIVE_REGENERATION_REPORT.md",
)

H2O_HYPOTHESIS_ARTIFACTS = (
    "h2o_scale_hypothesis_scales.csv",
    "h2o_scale_hypothesis_metrics.csv",
    "h2o_scale_hypothesis_coefficients.csv",
    "h2o_scale_hypothesis_manifest.json",
    "H2O_SCALE_HYPOTHESIS_REPORT.md",
)

H2O_FINAL_ARTIFACT_IDENTITIES = {
    "h2o_scale_hypothesis_scales.csv": {
        "bytes": 93701,
        "sha256": ("7a43c01563855518fb1a6b51985ab6c43d41c96bf4723df2ae73b047799f9dbe"),
    },
    "h2o_scale_hypothesis_metrics.csv": {
        "bytes": 1120084,
        "sha256": ("31210cbf3c1f5ab202fdcb8de579dc5dcfedd0ace1e36ba9ce9bfa64039f1f8b"),
    },
    "h2o_scale_hypothesis_coefficients.csv": {
        "bytes": 386823,
        "sha256": ("31e7958d1ddc96c7642ba626cf6c349e5d6161830c9a9d09eacf064fc28a7d00"),
    },
    "h2o_scale_hypothesis_manifest.json": {
        "bytes": 99719,
        "sha256": ("1316b92a06edc7dc1eb7a6752e271a7b80eb409192ad9f7bf2882cc12928d14c"),
    },
    "H2O_SCALE_HYPOTHESIS_REPORT.md": {
        "bytes": 6568,
        "sha256": ("1519928944075689f07e3b041fcba35a9f4f2c1042345df06cd14a6d47e2c5b6"),
    },
}

H2O_FINAL_RUN_SUMMARY_IDENTITY = {
    "unique_referenced_run_count": 13667,
    "normalized_numeric_text_aggregate_sha256": (
        "343acc6878062a433b665b0c80516212dc3b338fc77337bc9b6d1ade8196d1e1"
    ),
    "normalized_warning_bearing_output_aggregate_sha256": (
        "3fcfe769fab3490e7067876a55c75a06e6d17e8990f137238399d02ab246728f"
    ),
    "return_code_counts": {"0": 9792, "1": 3875},
    "am_version_identity_counts": {
        "am version 12.2 (build date Aug  1 2026 11:20:29)": 13667,
    },
    "diagnostic_totals": {
        "warning_bearing_run_count": 3875,
        "unresolved_line_warning_count_sum": 335885,
        "unresolved_column_warning_line_count": 139655,
        "unresolved_summary_warning_line_count": 3875,
        "other_warning_line_count": 0,
        "error_line_count": 0,
    },
}

H2O_FINAL_DIRECT_RANK1_PROFILES = {
    "am_q25": {"transmission": "LMT_MAM_5", "trj": "LMT_DJF_5"},
    "am_q50": {"transmission": "LMT_MAM_25", "trj": "LMT_DJF_25"},
    "am_q75": {"transmission": "LMT_DJF_50", "trj": "LMT_DJF_75"},
    "am_q95": {"transmission": "LMT_DJF_25"},
}

COPIED_AM_GENERATED_ARTIFACTS = {
    "COPIED_AM_FOLLOWUP_REPORT.md",
    "copied_am_annual_fit_coefficients.csv",
    "copied_am_legacy_comparison.csv",
    "copied_am_operator_stress_metrics.csv",
    "copied_am_operator_stress_rows.csv",
    "copied_am_product_inventory.csv",
    "copied_am_raw_output_inventory.csv",
}

R1_NORMALIZED_OUTPUT_ALGORITHM = (
    "decode UTF-8; normalize line endings to LF; replace '# run time ...' with "
    "'# run time <volatile>' and '# dcache hit: ...' with "
    "'# dcache counters <volatile>'; append one LF"
)

FROZEN_GIT_OBJECTS = {
    "tolteca/simu/lmt/__init__.py": "f2fbf70dff7a355e70188e11e97f50e059c8104a8fb29953d24de4f1a23235d5",
    "tolteca/common/lmt/__init__.py": "56113ab1ab9326c65ea07a24d8374f1c2ad6bd577ad1ba0785c01fa41d36d5fa",
    "tolteca/data/cal/toltec_passband/data/a1100_passband.ecsv": "13b8fd009bb8d7c375d3c46d21e26d0a779f7f00a949a2a5ccd619d1fe56fd72",
    "tolteca/data/cal/toltec_passband/data/a1400_passband.ecsv": "a7b671d9f659cbc98dad99d3015ce81a3d7a3486c702819d9b3305703e7c682e",
    "tolteca/data/cal/toltec_passband/data/a2000_passband.ecsv": "77e4b33c7bbc2c345ef94d41480d5fee5cb096d789f4fe78e1b4f80a37e0d6ff",
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_sha256_hex(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


def verify_frozen_provenance_paths() -> None:
    for path, expected in FROZEN_PATHS.items():
        if not path.is_file():
            raise RuntimeError(f"missing frozen provenance input: {path}")
        actual = sha256_path(path)
        if actual != expected:
            raise RuntimeError(
                f"frozen provenance digest mismatch: {path}: {actual} != {expected}"
            )

    for relative, expected in FROZEN_GIT_OBJECTS.items():
        result = subprocess.run(
            [
                "git",
                "-C",
                str(TOLTECA_REPOSITORY),
                "show",
                f"{TOLTECA_REVISION}:{relative}",
            ],
            check=True,
            stdout=subprocess.PIPE,
        )
        actual = sha256_bytes(result.stdout)
        if actual != expected:
            raise RuntimeError(
                "frozen repository-object digest mismatch: "
                f"{TOLTECA_REVISION}:{relative}: {actual} != {expected}"
            )


def verify_frozen_raw_sources(raw_source_dir: Path) -> None:
    for filename, expected in FROZEN_RAW_SOURCE_FILES.items():
        path = raw_source_dir / filename
        if not path.is_file():
            raise RuntimeError(f"missing frozen raw-source input: {path}")
        actual = sha256_path(path)
        if actual != expected:
            raise RuntimeError(
                f"frozen raw-source digest mismatch: {path}: {actual} != {expected}"
            )


def verify_owner_supplied_manifest_schema(request: dict[str, object]) -> None:
    schema_path = PACKAGE_DIR / OWNER_SUPPLIED_SCHEMA_NAME
    if (
        not schema_path.is_file()
        or sha256_path(schema_path) != OWNER_SUPPLIED_SCHEMA_SHA256
    ):
        raise RuntimeError("owner-supplied manifest schema digest changed")
    schema = json.loads(schema_path.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    )

    requested_items = request["required_items"]
    requested_ids = {item["id"] for item in requested_items}
    schema_fact_ids = set(schema["$defs"]["fact_id"]["enum"])
    expected_paths = {"faithful_generic_lineage", "versioned_am12_successor"}
    if (
        schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema"
        or schema.get("$id")
        != (
            "https://toltec.astro.umass.edu/schemas/sci-cal-001/"
            "owner-supplied-manifest-v1.json"
        )
        or schema["properties"]["owner_path"]["enum"]
        != [
            "faithful_generic_lineage",
            "versioned_am12_successor",
        ]
        or schema_fact_ids != requested_ids
    ):
        raise RuntimeError("owner-supplied schema identity/path/fact contract changed")

    faithful_sample = {
        "schema_version": "sci-cal-001-owner-supplied-manifest-v1",
        "package_id": "SCI-CAL-001",
        "request_id": "SCI-CAL-001-ATM-INPUT-001",
        "submission_date": "2026-08-01",
        "owner_path": "faithful_generic_lineage",
        "historical_generic_lineage_custody_status": "unresolved",
        "responses": [
            {
                "fact_id": "Q95-001",
                "disposition": "unresolved_retained",
                "statement": "Original registered q95 bytes remain unavailable.",
                "artifact_ids": [],
            }
        ],
        "artifacts": [],
    }
    successor_selection = {
        "model_id": "toltec_am12_successor",
        "model_version": "owner-version-required",
        "basis": "copied_am_12_2",
        "profile_family_rule": "owner rule required",
        "frequency_grid_policy": "owner policy required",
        "spectral_convention": "legacy_monochromatic",
        "unresolved_line_status_policy": "owner policy required",
        "historical_generic_products_replaced": False,
    }
    successor_sample = {
        **faithful_sample,
        "owner_path": "versioned_am12_successor",
        "historical_generic_lineage_custody_status": "unresolved_retained",
        "successor_selection": successor_selection,
    }
    for label, sample in (
        ("faithful path", faithful_sample),
        ("versioned successor path", successor_sample),
    ):
        errors = list(validator.iter_errors(sample))
        if errors:
            raise RuntimeError(
                f"owner-supplied schema rejects minimal valid {label}: "
                f"{errors[0].message}"
            )
    successor_without_selection = dict(successor_sample)
    successor_without_selection.pop("successor_selection")
    successor_with_false_custody = {
        **successor_sample,
        "historical_generic_lineage_custody_status": "resolved",
    }
    if not list(validator.iter_errors(successor_without_selection)):
        raise RuntimeError("owner schema accepts successor without versioned selection")
    if not list(validator.iter_errors(successor_with_false_custody)):
        raise RuntimeError("owner schema lets successor claim resolved generic custody")

    decision_paths = request.get("decision_paths")
    if not isinstance(decision_paths, list):
        raise RuntimeError("owner request has no explicit decision paths")
    decision_by_id = {item["id"]: item for item in decision_paths}
    q95 = next(item for item in requested_items if item["id"] == "Q95-001")
    delivery = request["delivery_layout"]
    if (
        request.get("schema_version") != "sci-cal-001-owner-input-request-v2"
        or set(decision_by_id) != expected_paths
        or len(decision_by_id) != len(decision_paths)
        or decision_by_id["faithful_generic_lineage"]["recommendation"]
        != "recommended_faithful_closure"
        or decision_by_id["versioned_am12_successor"]["recommendation"]
        != "allowed_new_model_alternative"
        or q95["priority"] != "blocking_for_faithful_generic_lineage_only"
        or delivery["required_manifest"] != "owner_supplied_manifest.json"
        or delivery["manifest_schema"]
        != (
            "validation/sci_cal_001_atmosphere_operator_2026-08-01/"
            "owner_supplied_manifest.schema.json"
        )
    ):
        raise RuntimeError("owner request decision-path/Q95/schema routing changed")

    delivered_path = PACKAGE_DIR / "inputs/owner_supplied/owner_supplied_manifest.json"
    if not delivered_path.is_file():
        return
    delivered = json.loads(delivered_path.read_text())
    errors = sorted(validator.iter_errors(delivered), key=lambda item: list(item.path))
    if errors:
        rendered = "\n".join(f"{list(error.path)}: {error.message}" for error in errors)
        raise RuntimeError(f"owner-supplied manifest schema errors:\n{rendered}")
    response_ids = [item["fact_id"] for item in delivered["responses"]]
    artifact_ids = [item["id"] for item in delivered["artifacts"]]
    referenced_ids = {
        artifact_id
        for response in delivered["responses"]
        for artifact_id in response["artifact_ids"]
    }
    if (
        len(response_ids) != len(set(response_ids))
        or len(artifact_ids) != len(set(artifact_ids))
        or not referenced_ids.issubset(artifact_ids)
        or any(
            response["disposition"]
            in {"evidence_supplied", "partial_evidence_supplied"}
            and not response["artifact_ids"]
            for response in delivered["responses"]
        )
    ):
        raise RuntimeError(
            "owner-supplied manifest has duplicate/unknown references or "
            "an evidence response without an artifact"
        )
    if delivered["owner_path"] == "versioned_am12_successor":
        q95_responses = [
            item for item in delivered["responses"] if item["fact_id"] == "Q95-001"
        ]
        if (
            len(q95_responses) != 1
            or q95_responses[0]["disposition"] != "unresolved_retained"
        ):
            raise RuntimeError("successor submission did not retain Q95-001 unresolved")
    elif "successor_selection" in delivered:
        raise RuntimeError("faithful-lineage submission included successor selection")
    delivery_root = delivered_path.parent.resolve()
    for artifact in delivered["artifacts"]:
        relative = Path(artifact["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("owner-supplied artifact path is not delivery-relative")
        path = (delivery_root / relative).resolve()
        try:
            path.relative_to(delivery_root)
        except ValueError as error:
            raise RuntimeError(
                "owner-supplied artifact escapes delivery root"
            ) from error
        if (
            not path.is_file()
            or path.stat().st_size != artifact["bytes"]
            or sha256_path(path) != artifact["sha256"]
            or artifact["original_bytes_preserved"] is not True
        ):
            raise RuntimeError(f"owner-supplied artifact identity changed: {path}")


def verify_package_level_tau_provenance_clarification(
    manifest: dict[str, object], request: dict[str, object]
) -> None:
    artifacts = manifest.get("artifacts")
    profiles = manifest.get("atmospheric_profiles")
    if not isinstance(artifacts, list) or not isinstance(profiles, dict):
        raise RuntimeError("regeneration manifest lost P1 provenance structure")
    h2o_artifacts = [
        item
        for item in artifacts
        if isinstance(item, dict)
        and item.get("id") == "am12-h2o-scale-hypothesis-manifest"
    ]
    if len(h2o_artifacts) != 1:
        raise RuntimeError("regeneration manifest lost the canonical P1 artifact")
    h2o_artifact = h2o_artifacts[0]
    h2o_notes = h2o_artifact.get("notes")
    if (
        h2o_artifact.get("availability") != "task_package"
        or h2o_artifact.get("bytes") != 99719
        or h2o_artifact.get("path")
        != (
            "validation/sci_cal_001_atmosphere_operator_2026-08-01/"
            "h2o_scale_hypothesis_manifest.json"
        )
        or h2o_artifact.get("sha256")
        != "1316b92a06edc7dc1eb7a6752e271a7b80eb409192ad9f7bf2882cc12928d14c"
        or h2o_artifact.get("role")
        != "post-hoc H2O-scale provenance-hypothesis diagnostic"
        or not isinstance(h2o_notes, str)
        or any(
            statement not in h2o_notes
            for statement in (
                "Correction tau is asymmetric",
                (
                    "q25/q50/q75 candidate uses direct atmTaun but generic truth "
                    "uses -log(atmTtx)"
                ),
                "q95 uses -log(Tband/T225) on both sides",
                "supersedes the frozen P1 report/manifest's overbroad",
            )
        )
    ):
        raise RuntimeError("canonical P1 artifact/provenance clarification changed")
    input_ids = profiles.get("input_artifact_ids")
    construction = profiles.get("construction_method")
    if (
        not isinstance(input_ids, list)
        or input_ids.count("am12-h2o-scale-hypothesis-manifest") != 1
        or not isinstance(construction, str)
        or "candidate tau is direct AM atmTaun" not in construction
        or "generic truth tau is reconstructed as -log(atmTtx)" not in construction
        or "q95 reconstructs both nominal ratio sides as -log(Tband/T225)"
        not in construction
        or "superseded by this package-level clarification" not in construction
    ):
        raise RuntimeError("P1 profile-input linkage/tau clarification changed")

    required_document_fragments = {
        "README.md": (
            "direct copied-AM `atmTaun` is authoritative on the candidate side",
            "reconstructs generic truth-side line-of-sight tau as `-log(atmTtx)`",
            "q95 is necessarily ratio-only and reconstructs both sides",
            "package-level clarification supersedes",
        ),
        "OWNER_DECISION_BRIEF.md": (
            "direct copied-AM `atmTaun` is authoritative on the candidate side",
            "generic truth NPZs contain only `atmTtx`",
            "q95 is ratio-only and reconstructs both sides",
            "superseded by this package-level clarification",
        ),
        "LOCAL_PROVENANCE_INVENTORY.md": (
            "candidate side uses direct AM `atmTaun`",
            "generic truth NPZs contain no tau array",
            "both candidate and repair-literal truth sides",
            "package-level clarification supersedes",
        ),
        "REGENERATION_SPEC.md": (
            "candidate side uses direct AM `atmTaun`",
            "generic truth NPZs have no tau member",
            "both candidate and repair-literal truth sides",
            "package-level specification supersedes",
        ),
    }
    for filename, fragments in required_document_fragments.items():
        content = (PACKAGE_DIR / filename).read_text(encoding="utf-8")
        if any(fragment not in content for fragment in fragments):
            raise RuntimeError(
                f"package-level tau-provenance clarification changed: {filename}"
            )
    recovered_facts = request.get("facts_already_recovered_do_not_request_again")
    if (
        not isinstance(recovered_facts, list)
        or not all(isinstance(item, str) for item in recovered_facts)
        or "\n".join(recovered_facts).count(
            "q25/q50/q75 candidate uses direct AM atmTaun while generic truth uses "
            "-log(atmTtx); q95 uses -log(Tband/T225) on both sides"
        )
        != 1
    ):
        raise RuntimeError("owner request lost the P1 tau-provenance clarification")


def verify_json_schema() -> None:
    schema = json.loads(
        (PACKAGE_DIR / "atmosphere_regeneration_manifest.schema.json").read_text()
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    manifest = json.loads((PACKAGE_DIR / "regeneration_manifest.json").read_text())
    validator = jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    )
    errors = sorted(validator.iter_errors(manifest), key=lambda item: list(item.path))
    if errors:
        rendered = "\n".join(f"{list(error.path)}: {error.message}" for error in errors)
        raise RuntimeError(f"regeneration manifest schema errors:\n{rendered}")

    incomplete_as_complete = json.loads(json.dumps(manifest))
    incomplete_as_complete["status"] = "complete_regeneration_input"
    if not list(validator.iter_errors(incomplete_as_complete)):
        raise RuntimeError(
            "schema accepted the unresolved partial manifest as complete input"
        )

    artifact_ids = [item["id"] for item in manifest["artifacts"]]
    if len(artifact_ids) != len(set(artifact_ids)):
        raise RuntimeError("regeneration manifest has duplicate artifact IDs")
    referenced_artifact_ids = set(
        manifest["atmospheric_profiles"]["input_artifact_ids"]
    )
    referenced_artifact_ids.update(
        manifest["bandpasses"]["available_not_used_artifact_ids"]
    )
    referenced_artifact_ids.update(
        item["raw_grid_artifact_id"]
        for item in manifest["tau225_anchors"]
        if item["raw_grid_artifact_id"] is not None
    )
    unknown_artifact_ids = referenced_artifact_ids - set(artifact_ids)
    if unknown_artifact_ids:
        raise RuntimeError(
            "regeneration manifest has unknown artifact references: "
            f"{sorted(unknown_artifact_ids)}"
        )

    request = json.loads((PACKAGE_DIR / "owner_input_request.json").read_text())
    if sha256_path(PACKAGE_DIR / "owner_input_request.json") != (
        OWNER_INPUT_REQUEST_SHA256
    ):
        raise RuntimeError("owner input request digest changed")
    verify_owner_supplied_manifest_schema(request)
    manifest_ids = set(manifest["unresolved_fact_ids"])
    requested_ids = {item["id"] for item in request["required_items"]}
    if len(requested_ids) != len(request["required_items"]):
        raise RuntimeError("owner request has duplicate fact IDs")
    if manifest_ids != requested_ids:
        raise RuntimeError(
            "owner request IDs do not exactly match manifest unresolved facts: "
            f"manifest_only={sorted(manifest_ids - requested_ids)}, "
            f"request_only={sorted(requested_ids - manifest_ids)}"
        )
    verify_package_level_tau_provenance_clarification(manifest, request)


def verify_manifest_artifact_files(*, include_external: bool) -> None:
    manifest = json.loads((PACKAGE_DIR / "regeneration_manifest.json").read_text())
    for artifact in manifest["artifacts"]:
        availability = artifact["availability"]
        if availability == "repository_revision":
            continue
        if availability == "local_read_only" and not include_external:
            continue
        if availability not in {"task_package", "local_read_only"}:
            raise RuntimeError(
                f"unexpected artifact availability for {artifact['id']}: {availability}"
            )

        declared_path = Path(artifact["path"]).expanduser()
        if not declared_path.is_absolute():
            declared_path = REPO_ROOT / declared_path
        resolved_path = declared_path.resolve()
        if availability == "task_package":
            try:
                resolved_path.relative_to(REPO_ROOT.resolve())
            except ValueError as error:
                raise RuntimeError(
                    f"task-package artifact resolves outside repository: "
                    f"{artifact['id']}: {resolved_path}"
                ) from error
        if not resolved_path.is_file():
            raise RuntimeError(
                f"missing {availability} manifest artifact: "
                f"{artifact['id']}: {resolved_path}"
            )
        if "bytes" not in artifact:
            raise RuntimeError(f"manifest artifact has no byte count: {artifact['id']}")
        actual_bytes = resolved_path.stat().st_size
        if actual_bytes != artifact["bytes"]:
            raise RuntimeError(
                f"manifest artifact byte mismatch: {artifact['id']}: "
                f"{actual_bytes} != {artifact['bytes']}"
            )
        actual_sha256 = sha256_path(resolved_path)
        if actual_sha256 != artifact["sha256"]:
            raise RuntimeError(
                f"manifest artifact digest mismatch: {artifact['id']}: "
                f"{actual_sha256} != {artifact['sha256']}"
            )


def verify_csvs() -> None:
    for name in CSV_ARTIFACTS:
        read_csv_rows(name)


def read_csv_rows(name: str) -> list[dict[str, str]]:
    path = PACKAGE_DIR / name
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise RuntimeError(f"CSV has no header: {path}")
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"CSV has no data rows: {path}")
    if any(None in row for row in rows):
        raise RuntimeError(f"CSV has malformed rows: {path}")
    return rows


def value_counts(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row[key]
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def package_basename_path(filename: str, *, label: str) -> Path:
    relative = Path(filename)
    if (
        relative.is_absolute()
        or relative.name != filename
        or filename in {"", ".", ".."}
    ):
        raise RuntimeError(f"{label} is not a package basename: {filename!r}")
    resolved = (PACKAGE_DIR / relative).resolve()
    try:
        resolved.relative_to(PACKAGE_DIR.resolve())
    except ValueError as error:
        raise RuntimeError(
            f"{label} resolves outside the evidence package: {filename!r}"
        ) from error
    return resolved


def require_artifact_group(names: tuple[str, ...], *, label: str) -> bool:
    present = [name for name in names if (PACKAGE_DIR / name).is_file()]
    if present and len(present) != len(names):
        missing = sorted(set(names) - set(present))
        raise RuntimeError(f"incomplete {label} artifact group: missing {missing}")
    return bool(present)


def verify_generated_artifact_digests(manifest: dict[str, object]) -> None:
    artifacts = manifest.get("generated_artifacts")
    if not isinstance(artifacts, list):
        raise RuntimeError("copied-AM manifest has no generated-artifact inventory")
    filenames: list[str] = []
    for item in artifacts:
        if not isinstance(item, dict):
            raise RuntimeError("malformed copied-AM generated-artifact inventory")
        filename = item.get("filename")
        if not isinstance(filename, str):
            raise RuntimeError("copied-AM generated artifact has no filename")
        filenames.append(filename)
        path = package_basename_path(
            filename, label="copied-AM generated artifact filename"
        )
        if not path.is_file():
            raise RuntimeError(f"missing copied-AM generated artifact: {path}")
        if path.stat().st_size != item.get("bytes"):
            raise RuntimeError(f"copied-AM generated artifact byte mismatch: {path}")
        if sha256_path(path) != item.get("sha256"):
            raise RuntimeError(f"copied-AM generated artifact digest mismatch: {path}")
    if len(filenames) != len(set(filenames)):
        raise RuntimeError("copied-AM generated-artifact inventory has duplicate names")
    if set(filenames) != COPIED_AM_GENERATED_ARTIFACTS:
        raise RuntimeError(
            "copied-AM generated-artifact inventory is incomplete or unexpected: "
            f"{sorted(filenames)}"
        )


def verify_copied_am_evidence() -> None:
    manifest = json.loads((PACKAGE_DIR / "copied_am_manifest.json").read_text())
    identity = manifest["identity"]
    if identity["copied_suite_identity"] != "am_12_2_not_historical_legacy_q_identity":
        raise RuntimeError(
            "copied AM suite was conflated with legacy generic-q identity"
        )
    if identity["operator_authorization"] != "none":
        raise RuntimeError("copied-AM stress must not authorize an operator")
    if identity["operational_domain_authorization"] != "none":
        raise RuntimeError("copied-AM stress must not authorize an operational domain")

    deviation = manifest["protocol_deviation"]
    deviation_path = package_basename_path(
        deviation["filename"], label="protocol-deviation filename"
    )
    if (
        deviation["filename"] != "FOLLOWUP_STUDY_DEVIATION_LOG.md"
        or deviation["bytes"] != 2066
        or deviation["sha256"]
        != "a3df86366c7869579b3255d9ea8f95cf6827e78018e0a2a83a1640360be1b036"
        or deviation["status"]
        != "clarification_only_no_candidate_or_numeric_reinterpretation"
        or deviation["stopped_study_c_identities"]
        != ["piecewise_linear_los_tau_v1", "pchip_los_tau_v1"]
        or deviation["diagnostic_c1_evaluated_identities"]
        != ["piecewise_linear_los_tau_v0", "pchip_los_tau_v0"]
        or not deviation_path.is_file()
        or deviation_path.stat().st_size != deviation["bytes"]
        or sha256_path(deviation_path) != deviation["sha256"]
    ):
        raise RuntimeError("copied-AM protocol-deviation clarification changed")

    suite = manifest["copied_suite"]
    products = suite["products"]
    expected_product_names = {
        f"LMT_{family}_{percentile}.npz"
        for family in ("annual", "DJF", "MAM", "JJA", "SON")
        for percentile in (5, 25, 50, 75, 95)
    }
    product_names = {item["filename"] for item in products}
    if (
        suite["product_count"] != 25
        or len(products) != 25
        or len(product_names) != len(products)
        or product_names != expected_product_names
        or suite["canonical_manifest_definition"]
        != "sorted basename\\tbytes\\tsha256\\tmd5\\n"
        or any(
            item["bytes"] <= 0
            or not is_sha256_hex(item["sha256"])
            or len(item["md5"]) != 32
            for item in products
        )
    ):
        raise RuntimeError("copied-AM suite must contain exactly 25 products")
    copied_suite_records = "".join(
        f"{item['filename']}\t{item['bytes']}\t{item['sha256']}\t{item['md5']}\n"
        for item in sorted(products, key=lambda value: value["filename"])
    ).encode("utf-8")
    if (
        sum(item["bytes"] for item in products) != suite["total_bytes"]
        or sha256_bytes(copied_suite_records)
        != "18dfd96f4438151197d3b6be5201476f7a71710363d81ec49c801101fa12b3ac"
        or suite["canonical_manifest_sha256"] != sha256_bytes(copied_suite_records)
    ):
        raise RuntimeError("copied-AM suite aggregate identity changed")
    product_by_id = {Path(product["filename"]).stem: product for product in products}
    seasonal = [item for item in products if "_annual_" not in item["filename"]]
    annual = [item for item in products if "_annual_" in item["filename"]]
    if len(seasonal) != 20 or len(annual) != 5:
        raise RuntimeError(
            "copied-AM suite must contain 20 seasonal and five annual products"
        )
    amc_inputs = manifest["amc_inputs"]
    amc_files = amc_inputs["files"]
    expected_amc_names = {
        f"{Path(product['filename']).stem}.amc" for product in products
    }
    if (
        amc_inputs["file_count"] != 25
        or len(amc_files) != 25
        or amc_inputs["total_bytes"] != 121065
        or {item["filename"] for item in amc_files} != expected_amc_names
        or len({item["filename"] for item in amc_files}) != len(amc_files)
        or any(
            item["path_relative_to_am_root"] != f"LMT_am_inputs/{item['filename']}"
            or item["release_path_relative_to_am_source_root"]
            != f"cookbook/sites/LMT/{item['filename']}"
            or item["release_copy_exact"] is not True
            or item["bytes"] <= 0
            or not is_sha256_hex(item["sha256"])
            for item in amc_files
        )
    ):
        raise RuntimeError("copied-AM AMC identity inventory changed")
    ordered_amcs = sorted(amc_files, key=lambda item: item["filename"].encode())
    sha256sum_records = "".join(
        f"{item['sha256']}  cookbook/sites/LMT/{item['filename']}\n"
        for item in ordered_amcs
    ).encode("utf-8")
    nul_records = b"".join(
        item["filename"].encode("utf-8") + b"\0" + bytes.fromhex(item["sha256"]) + b"\0"
        for item in ordered_amcs
    )
    if (
        sum(item["bytes"] for item in amc_files) != 121065
        or amc_inputs["canonical_sha256sum_records"]["algorithm"]
        != (
            "UTF-8 concatenation in AMC-basename bytewise sort order of "
            "sha256<TWO_SPACES>cookbook/sites/LMT/basename<LF>"
        )
        or amc_inputs["canonical_nul_records"]["algorithm"]
        != "sha256(basename UTF-8 NUL raw 32-byte SHA-256 NUL)"
        or sha256_bytes(sha256sum_records)
        != "d3e4d9e1c095ffafb77b22a7d72a988335f36e476e240aadc27b8c23ef0f3bde"
        or sha256_bytes(nul_records)
        != "b7dd766852b4f422bdc861337e04d8f0184732045ea1a06a962560e86d2ce87c"
        or amc_inputs["canonical_sha256sum_records"]["sha256"]
        != sha256_bytes(sha256sum_records)
        or amc_inputs["canonical_nul_records"]["sha256"] != sha256_bytes(nul_records)
    ):
        raise RuntimeError("copied-AM AMC aggregate identity changed")
    amc_by_name = {item["filename"]: item for item in amc_files}
    if any(
        product["amc_input"]
        != {
            "filename": amc_by_name[f"{Path(product['filename']).stem}.amc"][
                "filename"
            ],
            "bytes": amc_by_name[f"{Path(product['filename']).stem}.amc"]["bytes"],
            "sha256": amc_by_name[f"{Path(product['filename']).stem}.amc"]["sha256"],
            "release_copy_exact": True,
        }
        for product in products
    ):
        raise RuntimeError("copied-AM product/AMC manifest identities changed")
    if any(
        item["tolteca_registry_identity"]["status"]
        != "exact_seasonal_registry_md5_match"
        or item["generic_q_registry_relation"]
        != "seasonal_registry_identity_generic_q_artifacts_separate"
        for item in seasonal
    ):
        raise RuntimeError(
            "not all 20 seasonal products retain exact registry identity"
        )
    if any(
        item["tolteca_registry_identity"]["status"]
        != "no_matching_generic_registry_identity"
        or item["generic_q_registry_relation"]
        != "no_matching_generic_registry_identity"
        for item in annual
    ):
        raise RuntimeError(
            "an annual product was assigned an unsupported generic-q identity"
        )

    registry = manifest["tolteca_registry_provenance"]
    if registry["seasonal_identity_count"] != 20:
        raise RuntimeError(
            "copied-AM registry evidence must retain 20 seasonal matches"
        )
    generic = registry["generic_q_artifacts"]
    if {item["name"] for item in generic} != {
        "am_q25",
        "am_q50",
        "am_q75",
        "am_q95",
    } or any(
        item["relation_to_copied_suite"] != "separate_registry_artifact"
        for item in generic
    ):
        raise RuntimeError("generic q registry artifacts are not explicitly separate")

    inventory = read_csv_rows("copied_am_product_inventory.csv")
    if (
        len(inventory) != 25
        or len({row["profile_id"] for row in inventory}) != len(inventory)
        or {row["profile_id"] for row in inventory} != set(product_by_id)
    ):
        raise RuntimeError("copied-AM inventory must have 25 rows")
    if any(
        row["npz_path_relative_to_am_root"]
        != f"LMT_am_npz/{product_by_id[row['profile_id']]['filename']}"
        or int(row["bytes"]) != product_by_id[row["profile_id"]]["bytes"]
        or row["sha256"] != product_by_id[row["profile_id"]]["sha256"]
        or row["md5"] != product_by_id[row["profile_id"]]["md5"]
        or row["amc_filename"] != f"{row['profile_id']}.amc"
        or row["amc_filename"] not in amc_by_name
        or row["amc_path_relative_to_am_root"]
        != amc_by_name[row["amc_filename"]]["path_relative_to_am_root"]
        or int(row["amc_bytes"]) != amc_by_name[row["amc_filename"]]["bytes"]
        or row["amc_sha256"] != amc_by_name[row["amc_filename"]]["sha256"]
        or row["amc_release_copy_exact"] != "true"
        or row["npz_members"] != "el;atmFreq;atmTRJ;atmTtx;atmTaun"
        or row["elevation_grid_deg"] != "10:80:2"
        or row["frequency_grid_ghz"] != "0:500:0.01"
        or row["spectral_array_shape"] != "50001x36"
        or row["scientific_identity"]
        != "copied_am_12_2_profile_not_historical_legacy_q_identity"
        for row in inventory
    ):
        raise RuntimeError("copied-AM product/AMC cross-identities changed")
    inventory_seasonal = [row for row in inventory if row["season"] != "annual"]
    inventory_annual = [row for row in inventory if row["season"] == "annual"]
    if len(inventory_seasonal) != 20 or any(
        row["tolteca_registry_status"] != "exact_seasonal_registry_md5_match"
        for row in inventory_seasonal
    ):
        raise RuntimeError("copied-AM inventory lost an exact seasonal registry match")
    if len(inventory_annual) != 5 or any(
        row["tolteca_registry_status"] != "no_matching_generic_registry_identity"
        for row in inventory_annual
    ):
        raise RuntimeError(
            "copied-AM annual inventory claims a generic registry identity"
        )

    comparisons = read_csv_rows("copied_am_legacy_comparison.csv")
    expected_comparison_matrix = {
        (family, percentile, band)
        for family in ("annual", "DJF", "MAM", "JJA", "SON")
        for percentile in (25, 50, 75, 95)
        for band in ("a1100", "a1400", "a2000")
    }
    actual_comparison_matrix = {
        (
            row["copied_profile_family"],
            int(row["legacy_model"].removeprefix("am_q")),
            row["band"],
        )
        for row in comparisons
    }
    if (
        len(comparisons) != 60
        or len(actual_comparison_matrix) != len(comparisons)
        or actual_comparison_matrix != expected_comparison_matrix
        or manifest["stress_scope"]["comparison_row_count"] != 60
        or manifest["stress_scope"]["comparison_cartesian_contract"]
        != {
            "profile_families": ["annual", "DJF", "MAM", "JJA", "SON"],
            "percentiles": [25, 50, 75, 95],
            "bands": ["a1100", "a1400", "a2000"],
            "family_ranking_claim": "none_metric_values_only",
        }
        or any(
            row["copied_profile_id"]
            != (
                f"LMT_{row['copied_profile_family']}_"
                f"{row['legacy_model'].removeprefix('am_q')}"
            )
            or row["copied_sha256"] != product_by_id[row["copied_profile_id"]]["sha256"]
            or row["copied_md5"] != product_by_id[row["copied_profile_id"]]["md5"]
            or row["family_ranking_claim"] != "none_metric_values_only"
            or row["legacy_identity_match"] != "false"
            or row["disposition"]
            != "identity_diagnostic_only_copied_am12_2_not_substitute"
            for row in comparisons
        )
    ):
        raise RuntimeError(
            "copied-AM comparison no longer keeps legacy identity distinct"
        )

    coefficient_rows = read_csv_rows("copied_am_annual_fit_coefficients.csv")
    expected_coefficient_matrix = {
        (percentile, band, index, 6 - index)
        for percentile in (25, 50, 75, 95)
        for band in ("a1100", "a1400", "a2000")
        for index in range(7)
    }
    actual_coefficient_matrix = {
        (
            int(row["legacy_model_comparison_identity"].removeprefix("am_q")),
            row["band"],
            int(row["coefficient_index_descending"]),
            int(row["polynomial_power"]),
        )
        for row in coefficient_rows
    }
    if (
        len(coefficient_rows) != 84
        or len(actual_coefficient_matrix) != len(coefficient_rows)
        or actual_coefficient_matrix != expected_coefficient_matrix
        or manifest["stress_scope"]["annual_fit_coefficient_row_count"] != 84
        or any(
            row["copied_profile_id"]
            != (
                "LMT_annual_"
                f"{row['legacy_model_comparison_identity'].removeprefix('am_q')}"
            )
            or row["copied_profile_sha256"]
            != product_by_id[row["copied_profile_id"]]["sha256"]
            or row["fit_elevation_grid_deg"] != "20:80:2"
            or row["fit_elevation_coordinate"] != "repair_base_radians"
            or row["fit_degree"] != "6"
            or len(row["rounded_8_decimal_coefficient"].partition(".")[2]) != 8
            or len(row["legacy_source_8_decimal_coefficient"].partition(".")[2]) != 8
            or row["rounded_8_decimal_matches_legacy_source"]
            != str(
                row["rounded_8_decimal_coefficient"]
                == row["legacy_source_8_decimal_coefficient"]
            ).lower()
            or not math.isclose(
                float(row["signed_rounded_coefficient_difference"]),
                float(row["rounded_8_decimal_coefficient"])
                - float(row["legacy_source_8_decimal_coefficient"]),
                rel_tol=0.0,
                abs_tol=1e-16,
            )
            or not math.isfinite(float(row["unrounded_binary64_coefficient"]))
            or row["identity_disposition"]
            != "annual_am12_2_fit_diagnostic_not_generic_q_substitute"
            for row in coefficient_rows
        )
    ):
        raise RuntimeError("copied-AM annual coefficient table contract changed")

    raw_summary = manifest["copied_raw_outputs"]
    expected_raw_digest = (
        "b9bcdb36952444f4db33549fa621318c5f757dbe36c4b6a11addceb46ec95053"
    )
    expected_warning_distribution = {"86": 324, "87": 540, "88": 36}
    if (
        raw_summary["file_count"] != 900
        or raw_summary["total_bytes"] != 2983517161
        or raw_summary["canonical_manifest_sha256"] != expected_raw_digest
        or raw_summary["canonical_manifest_algorithm"]
        != (
            "UTF-8 concatenation in relative-path bytewise sort order of "
            "relative_path<TAB>bytes<TAB>sha256<LF>"
        )
        or raw_summary["expected_rows_per_file"] != 50001
        or raw_summary["am_identity"]
        != "am version 12.2 (build date Aug 26 2022 19:20:13)"
        or raw_summary["za_to_elevation_mapping"] != "elevation_deg=90-zenith_angle_deg"
        or raw_summary["all_four_retained_fields_exact_npz"] is not True
        or raw_summary["retained_npz_fields"] != ["f", "tau", "tx", "Trj"]
        or raw_summary["unresolved_line_warning_count_distribution"]
        != expected_warning_distribution
        or raw_summary["return_footer_status_distribution"]
        != {"complete_numeric_grid_then_slurm_exit_code_1": 900}
        or raw_summary["slurm_retry_file_count"] != 3
        or raw_summary["historical_return_disposition"]
        != "complete_numeric_outputs_with_nonzero_footer_not_clean_run_proof"
        or raw_summary["tb_status"]
        != {
            "npz_comparison": "not_applicable_npz_omits_tb",
            "present_and_finite_in_all_dat_files": True,
        }
    ):
        raise RuntimeError("copied raw-output aggregate contract changed")

    raw_rows = read_csv_rows("copied_am_raw_output_inventory.csv")
    if len(raw_rows) != 900:
        raise RuntimeError("copied raw-output inventory must contain 900 rows")
    if len({row["relative_path"] for row in raw_rows}) != 900:
        raise RuntimeError("copied raw-output inventory contains duplicate paths")
    expected_cases = {
        (profile["filename"].removesuffix(".npz"), angle)
        for profile in products
        for angle in range(10, 82, 2)
    }
    actual_cases = {
        (row["profile_id"], int(row["zenith_angle_deg"])) for row in raw_rows
    }
    if actual_cases != expected_cases:
        raise RuntimeError("copied raw-output profile/zenith-angle matrix changed")
    required_true_fields = (
        "frequency_exact_npz",
        "direct_tau_exact_npz",
        "transmission_exact_npz",
        "trj_exact_npz",
        "tb_column_present",
        "tb_all_finite",
    )
    if any(
        row["numeric_row_count"] != "50001"
        or int(row["elevation_deg"]) != 90 - int(row["zenith_angle_deg"])
        or row["am_identity"] != raw_summary["am_identity"]
        or row["return_footer_status"] != "complete_numeric_grid_then_slurm_exit_code_1"
        or row["footer_exit_code"] != "1"
        or any(row[field] != "true" for field in required_true_fields)
        or row["tb_npz_comparison_status"]
        != "not_applicable_npz_omits_tb_no_comparison_invented"
        or row["validation_status"] != "complete_grid_four_retained_fields_exact_npz"
        or not row["relative_path"].startswith("LMT_am_outputs/")
        or not row["relative_path"].endswith(".dat")
        for row in raw_rows
    ):
        raise RuntimeError("copied raw-output row contract changed")
    warning_distribution = {
        warning: sum(
            row["unresolved_line_warning_count"] == warning for row in raw_rows
        )
        for warning in ("86", "87", "88")
    }
    if warning_distribution != expected_warning_distribution:
        raise RuntimeError("copied raw-output warning distribution changed")
    if sum(row["slurm_retry_flag"] == "true" for row in raw_rows) != 3:
        raise RuntimeError("copied raw-output Slurm-retry count changed")
    if sum(int(row["bytes"]) for row in raw_rows) != 2983517161:
        raise RuntimeError("copied raw-output byte total changed")
    canonical_bytes = "".join(
        f"{row['relative_path']}\t{row['bytes']}\t{row['sha256']}\n"
        for row in sorted(raw_rows, key=lambda item: item["relative_path"].encode())
    ).encode("utf-8")
    if sha256_bytes(canonical_bytes) != expected_raw_digest:
        raise RuntimeError("copied raw-output canonical aggregate digest mismatch")

    excluded = manifest["excluded_inputs"]
    if excluded != [
        {
            "pattern": "LMT_am_npz/DVUploaderLog*.log",
            "read_by_generator": False,
            "reason": (
                "not_scientific_input_may_contain_credentials_and_not_upload_proof"
            ),
        }
    ] or any("DVUploaderLog" in row["relative_path"] for row in raw_rows):
        raise RuntimeError("uploader-log exclusion contract changed")

    report = (PACKAGE_DIR / "COPIED_AM_FOLLOWUP_REPORT.md").read_text()
    required_report_statements = (
        "All `900` copied raw DAT outputs parse as `50001` five-column numeric rows",
        expected_raw_digest,
        "no Tb equality comparison is claimed or invented",
        'The unresolved-line count distribution is `{"86": 324, "87": 540, "88": 36}`',
        "not reclassified as clean successful runs",
        "never reads the nearby Dataverse uploader logs",
    )
    if any(statement not in report for statement in required_report_statements):
        raise RuntimeError("copied-AM follow-up report lost raw-output qualifications")

    stress = read_csv_rows("copied_am_operator_stress_metrics.csv")
    if len(stress) != 6:
        raise RuntimeError(
            "copied-AM stress metrics must have two candidates by three bands"
        )
    expected_candidates = {"piecewise_linear_los_tau_v0", "pchip_los_tau_v0"}
    if (
        {row["candidate"] for row in stress} != expected_candidates
        or set(manifest["candidates"]) != expected_candidates
        or set(deviation["diagnostic_c1_evaluated_identities"]) != expected_candidates
    ):
        raise RuntimeError("copied-AM stress candidate set changed")
    for row in stress:
        error = float(row["max_abs_fractional_correction_error"])
        should_pass = row["band"] != "a1100"
        if should_pass != (error <= 0.01):
            raise RuntimeError(
                f"unexpected one-percent stress result for {row['candidate']}/{row['band']}"
            )
        if row["passes_post_discovery_am12_2_stress_1pct"] != str(should_pass).lower():
            raise RuntimeError(
                f"inconsistent stress verdict for {row['candidate']}/{row['band']}"
            )
        if row["operator_authorization"] != "none":
            raise RuntimeError(
                "post-discovery stress was treated as operator authorization"
            )
    if {row["band"] for row in stress} != {"a1100", "a1400", "a2000"}:
        raise RuntimeError("copied-AM stress band set changed")
    verify_generated_artifact_digests(manifest)


def verify_frequency_resolution_evidence() -> None:
    manifest = json.loads(
        (PACKAGE_DIR / "frequency_resolution_manifest.json").read_text()
    )
    if manifest["schema_version"] != "sci-cal-001-am12-frequency-resolution-v2":
        raise RuntimeError("unexpected frequency-resolution manifest schema")
    threshold = float(manifest["diagnostic_threshold_fraction"])
    if threshold <= 0.0 or threshold > 0.001:
        raise RuntimeError(
            "frequency-resolution diagnostic threshold exceeds 0.1 percent"
        )
    if manifest["ten_mhz_resolution_diagnostic_passes"] is not True:
        raise RuntimeError("10 MHz frequency-resolution diagnostic no longer passes")
    if manifest["study_status"] != "diagnostic_not_operator_authorization":
        raise RuntimeError(
            "frequency-resolution diagnostic was treated as authorization"
        )
    native_manifest = json.loads(
        (PACKAGE_DIR / "native_regeneration_manifest.json").read_text()
    )
    copied_manifest = json.loads((PACKAGE_DIR / "copied_am_manifest.json").read_text())
    copied_build = native_manifest["builds"]["copied_linux_reference"]
    regenerated_build = native_manifest["builds"]["regeneration"]
    inputs = manifest["inputs"]
    native_input = inputs["native_executable"]
    copied_input = inputs["copied_linux_executable"]
    if (
        copied_input["sha256"]
        != "3fc1f71b3a025ac79f5559bdd2fbf40cf5de2aa7598cabf474f74f9a6c3b290c"
        or copied_input["sha256"] != copied_build["sha256"]
        or copied_input["bytes"] != copied_build["size_bytes"]
        or copied_input["path_relative_to_am_root"] != "am-12.2/bin/am"
        or native_input["sha256"]
        != "78e721d45b08990069a2d67a5fb337446bcbfb728046940c0d473bea340205fb"
        or native_input["bytes"] != 58435360
        or native_input["sha256"] != regenerated_build["sha256"]
        or native_input["bytes"] != regenerated_build["size_bytes"]
        or Path(native_input["path"]).resolve()
        != Path(regenerated_build["resolved_path"]).resolve()
        or set(native_manifest["results"]["regeneration_am_identity_counts"])
        != {"am version 12.2 (build date Aug  1 2026 11:20:29)"}
    ):
        raise RuntimeError("frequency-resolution AM 12.2/build identity changed")

    expected_profile_inputs = {
        "LMT_DJF_5": {
            "amc": {
                "path_relative_to_am_root": (
                    "Big_Atmosphere/LMT_am_inputs/LMT_DJF_5.amc"
                ),
                "bytes": 4837,
                "sha256": (
                    "fcb3b70f44cad98cf0586fede9dcd3b2e35f3cb45023d0485c782c108b25b474"
                ),
            },
            "npz": {
                "path_relative_to_am_root": ("Big_Atmosphere/LMT_am_npz/LMT_DJF_5.npz"),
                "bytes": 57602678,
                "sha256": (
                    "214d9fa975c73afa01a4e1b5c5f068245779989578acd8574831b7fe2b6ed6cc"
                ),
            },
        },
        "LMT_DJF_95": {
            "amc": {
                "path_relative_to_am_root": (
                    "Big_Atmosphere/LMT_am_inputs/LMT_DJF_95.amc"
                ),
                "bytes": 4841,
                "sha256": (
                    "b87b918b302425ef3d85aeedc285863a987579923289a37b97c6de5c935175e6"
                ),
            },
            "npz": {
                "path_relative_to_am_root": (
                    "Big_Atmosphere/LMT_am_npz/LMT_DJF_95.npz"
                ),
                "bytes": 57602678,
                "sha256": (
                    "3dd961143e31a8db8182c35dd55472ad9ec943a711f652f6d55d485ee5ddb42d"
                ),
            },
        },
    }
    profile_inputs = {item["profile"]: item for item in inputs["profiles"]}
    copied_products = {
        Path(item["filename"]).stem: item
        for item in copied_manifest["copied_suite"]["products"]
    }
    copied_amcs = {
        Path(item["filename"]).stem: item
        for item in copied_manifest["amc_inputs"]["files"]
    }
    if (
        len(inputs["profiles"]) != 2
        or len(profile_inputs) != 2
        or set(profile_inputs) != set(expected_profile_inputs)
        or any(
            {
                "amc": profile_inputs[profile]["amc"],
                "npz": profile_inputs[profile]["npz"],
            }
            != expected
            or expected["amc"]["bytes"] != copied_amcs[profile]["bytes"]
            or expected["amc"]["sha256"] != copied_amcs[profile]["sha256"]
            or expected["npz"]["bytes"] != copied_products[profile]["bytes"]
            or expected["npz"]["sha256"] != copied_products[profile]["sha256"]
            for profile, expected in expected_profile_inputs.items()
        )
    ):
        raise RuntimeError("frequency-resolution frozen profile inputs changed")

    execution = manifest["execution"]
    if (
        execution["run_count"] != 16
        or execution["omp_threads"] != 14
        or execution["am_cache_path_relative_to_cache"] != "am_cache"
        or execution["am_identity"]
        != "am version 12.2 (build date Aug  1 2026 11:20:29)"
        or execution["check_mode"]
        != "cache_only_no_process_execution_no_directory_creation"
        or "return code 1" not in execution["known_warning_policy"]
        or "cache" not in execution["known_warning_policy"]
    ):
        raise RuntimeError("frequency-resolution execution/warning contract changed")
    cache_evidence = manifest["cache_evidence"]
    sidecar_digests = cache_evidence["execution_sidecar_digests"]
    if (
        cache_evidence["raw_outputs"]
        != "stored below --cache-dir/raw_outputs and not committed"
        or cache_evidence["execution_sidecars"]
        != "stored below --cache-dir/execution_records and not committed"
        or cache_evidence["normalized_output_algorithm"]
        != R1_NORMALIZED_OUTPUT_ALGORITHM
        or cache_evidence["raw_output_digest_algorithm"]
        != "SHA-256 of exact combined stdout/stderr bytes"
        or cache_evidence["numeric_output_digest_algorithm"]
        != "SHA-256 of parsed numeric lines joined by LF with one final LF"
        or cache_evidence["execution_sidecar_digest_algorithm"]
        != (
            "SHA-256 of deterministic sorted-key UTF-8 JSON with two-space "
            "indentation and one final LF"
        )
        or len(sidecar_digests) != 16
        or len({item["case_id"] for item in sidecar_digests}) != 16
        or any(
            item["bytes"] <= 0
            or not is_sha256_hex(item["sha256"])
            or item["path_relative_to_cache"]
            != f"execution_records/{item['case_id']}.run.json"
            for item in sidecar_digests
        )
    ):
        raise RuntimeError("frequency-resolution cache-evidence contract changed")
    maxima = manifest["maximum_fractional_correction_difference_by_step"]
    if set(maxima) != {"1", "2", "5", "10"}:
        raise RuntimeError("frequency-resolution step set changed")
    if max(float(value) for value in maxima.values()) > threshold:
        raise RuntimeError(
            "frequency-resolution result exceeds its diagnostic threshold"
        )
    if (
        float(
            manifest[
                "ten_mhz_140to280_vs_copied_0to500_maximum_fractional_correction_difference"
            ]
        )
        > threshold
    ):
        raise RuntimeError("10 MHz range-comparison result exceeds 0.1 percent")

    rows = read_csv_rows("frequency_resolution_metrics.csv")
    if len(rows) != 64:
        raise RuntimeError("frequency-resolution table must have 64 rows")
    expected_matrix = {
        (profile, zenith_angle, step_mhz, frequency)
        for profile in ("LMT_DJF_5", "LMT_DJF_95")
        for zenith_angle in (10, 70)
        for step_mhz in (10, 5, 2, 1)
        for frequency in (150.0, 214.29, 225.0, 272.73)
    }
    actual_matrix = {
        (
            row["profile"],
            int(row["zenith_angle_deg"]),
            int(row["step_mhz"]),
            float(row["frequency_ghz"]),
        )
        for row in rows
    }
    if len(actual_matrix) != len(rows) or actual_matrix != expected_matrix:
        raise RuntimeError("frequency-resolution CSV is not the unique Cartesian grid")
    if (
        manifest["profiles"] != ["LMT_DJF_5", "LMT_DJF_95"]
        or manifest["zenith_angles_deg"] != [10, 70]
        or manifest["grid"]
        != {
            "minimum_ghz": 140,
            "maximum_ghz": 280,
            "steps_mhz": [10, 5, 2, 1],
            "center_frequencies_ghz": [150.0, 214.29, 225.0, 272.73],
        }
    ):
        raise RuntimeError("frequency-resolution manifest Cartesian grid changed")

    runs = manifest["runs"]
    expected_run_matrix = {
        (profile, zenith_angle, step_mhz)
        for profile in ("LMT_DJF_5", "LMT_DJF_95")
        for zenith_angle in (10, 70)
        for step_mhz in (10, 5, 2, 1)
    }
    actual_run_matrix = {
        (
            record["profile"],
            int(record["zenith_angle_deg"]),
            int(record["step_mhz"]),
        )
        for record in runs
    }
    if (
        len(runs) != 16
        or len(actual_run_matrix) != len(runs)
        or actual_run_matrix != expected_run_matrix
        or {item["case_id"] for item in sidecar_digests}
        != {record["case_id"] for record in runs}
    ):
        raise RuntimeError("frequency-resolution run inventory is not Cartesian")
    native_executable_path = Path(native_input["path"]).resolve()
    run_cache_dirs = {
        Path(record["environment_overrides"]["AM_CACHE_PATH"]).resolve().parent
        for record in runs
    }
    warning_class_names = {
        "unresolved_column",
        "unresolved_summary",
        "cache_insert_as_mru",
        "cache_promote_to_mru",
        "other",
    }
    if len(run_cache_dirs) != 1:
        raise RuntimeError("frequency-resolution runs do not share one cache root")
    cache_dir = next(iter(run_cache_dirs))
    if (
        Path(next(iter(runs))["environment_overrides"]["AM_CACHE_PATH"]).resolve()
        != cache_dir / "am_cache"
    ):
        raise RuntimeError("frequency-resolution AM cache identity changed")
    for record in runs:
        profile = record["profile"]
        zenith_angle = int(record["zenith_angle_deg"])
        step_mhz = int(record["step_mhz"])
        expected_case_id = f"{profile}_za{zenith_angle:02d}_{step_mhz}mhz"
        expected_amc = expected_profile_inputs[profile]["amc"]
        argv = record["argv"]
        warnings = record["warning_class_counts"]
        if (
            record["schema_version"] != "sci-cal-001-am12-frequency-run-v1"
            or record["case_id"] != expected_case_id
            or int(record["elevation_deg"]) != 90 - zenith_angle
            or len(argv) != 11
            or Path(argv[0]).resolve() != native_executable_path
            or not Path(argv[1])
            .as_posix()
            .endswith(f"/{expected_amc['path_relative_to_am_root']}")
            or argv[2:]
            != [
                "140",
                "GHz",
                "280",
                "GHz",
                str(step_mhz),
                "MHz",
                str(zenith_angle),
                "deg",
                "1.0",
            ]
            or record["environment_overrides"]["OMP_NUM_THREADS"] != "14"
            or Path(record["environment_overrides"]["AM_CACHE_PATH"]).resolve()
            != cache_dir / "am_cache"
            or record["return_code"] != 1
            or record["am_identity"]
            != "am version 12.2 (build date Aug  1 2026 11:20:29)"
            or record["row_count"] != (280 - 140) * 1000 // step_mhz + 1
            or record["unresolved_line_warning_count"] <= 0
            or set(warnings) != warning_class_names
            or warnings["unresolved_column"] <= 0
            or warnings["unresolved_summary"] != 1
            or warnings["cache_insert_as_mru"] != 0
            or warnings["cache_promote_to_mru"] != 0
            or warnings["other"] != 0
            or record["unexpected_warning_count"] != 0
            or record["error_line_count"] != 0
            or record["raw_output_path_relative_to_cache"]
            != f"raw_outputs/{expected_case_id}.dat"
            or record["sidecar_path_relative_to_cache"]
            != f"execution_records/{expected_case_id}.run.json"
            or record["raw_output_bytes"] <= 0
            or not is_sha256_hex(record["raw_output_sha256"])
            or not is_sha256_hex(record["normalized_output_sha256"])
            or not is_sha256_hex(record["numeric_output_sha256"])
            or record["normalized_output_algorithm"] != R1_NORMALIZED_OUTPUT_ALGORITHM
        ):
            raise RuntimeError(
                f"frequency-resolution run contract changed: {expected_case_id}"
            )

    recomputed_warning_totals = {
        name: sum(record["warning_class_counts"][name] for record in runs)
        for name in sorted(warning_class_names)
    }
    if (
        manifest["warning_class_totals"] != recomputed_warning_totals
        or manifest["warning_class_totals"]["cache_insert_as_mru"] != 0
        or manifest["warning_class_totals"]["cache_promote_to_mru"] != 0
        or manifest["warning_class_totals"]["other"] != 0
        or manifest["error_line_total"] != 0
        or sum(record["error_line_count"] for record in runs) != 0
    ):
        raise RuntimeError("frequency-resolution diagnostics are not clean")

    if any(
        int(row["elevation_deg"]) != 90 - int(row["zenith_angle_deg"])
        or int(row["row_count"]) != (280 - 140) * 1000 // int(row["step_mhz"]) + 1
        for row in rows
    ):
        raise RuntimeError("frequency-resolution geometry or row count changed")
    if {row["return_code"] for row in rows} != {"1"}:
        raise RuntimeError(
            "frequency-resolution table must explicitly retain AM status 1"
        )
    run_by_key = {
        (
            record["profile"],
            int(record["zenith_angle_deg"]),
            int(record["step_mhz"]),
        ): record
        for record in runs
    }
    for row in rows:
        key = (
            row["profile"],
            int(row["zenith_angle_deg"]),
            int(row["step_mhz"]),
        )
        record = run_by_key[key]
        warnings = record["warning_class_counts"]
        if (
            row["am_identity"] != record["am_identity"]
            or row["return_code"] != str(record["return_code"])
            or row["unresolved_line_warning_count"]
            != str(record["unresolved_line_warning_count"])
            or row["unresolved_column_warning_line_count"]
            != str(warnings["unresolved_column"])
            or row["unresolved_summary_warning_line_count"]
            != str(warnings["unresolved_summary"])
            or row["cache_warning_line_count"]
            != str(warnings["cache_insert_as_mru"] + warnings["cache_promote_to_mru"])
            or row["cache_insert_as_mru_warning_line_count"]
            != str(warnings["cache_insert_as_mru"])
            or row["cache_promote_to_mru_warning_line_count"]
            != str(warnings["cache_promote_to_mru"])
            or row["other_warning_line_count"] != str(warnings["other"])
            or row["error_line_count"] != str(record["error_line_count"])
            or row["row_count"] != str(record["row_count"])
            or row["raw_output_sha256"] != record["raw_output_sha256"]
            or row["normalized_output_sha256"] != record["normalized_output_sha256"]
            or row["numeric_output_sha256"] != record["numeric_output_sha256"]
            or float(row["fractional_correction_difference_vs_1mhz"]) < 0.0
            or float(row["fractional_correction_difference_vs_copied_0to500ghz_10mhz"])
            < 0.0
        ):
            raise RuntimeError(
                f"frequency-resolution CSV/run provenance mismatch: {record['case_id']}"
            )
    if (
        max(abs(float(row["fractional_correction_difference_vs_1mhz"])) for row in rows)
        > threshold
    ):
        raise RuntimeError("frequency-resolution CSV exceeds the 0.1-percent threshold")
    recomputed_maxima = {
        str(step): max(
            float(row["fractional_correction_difference_vs_1mhz"])
            for row in rows
            if int(row["step_mhz"]) == step
        )
        for step in (10, 5, 2, 1)
    }
    if any(
        recomputed_maxima[step]
        != float(manifest["maximum_fractional_correction_difference_by_step"][step])
        for step in recomputed_maxima
    ):
        raise RuntimeError("frequency-resolution maxima do not match the CSV")
    ten_mhz_rows = [row for row in rows if row["step_mhz"] == "10"]
    non_ten_mhz_rows = [row for row in rows if row["step_mhz"] != "10"]
    if any(row["exact_copied_match_when_10mhz"] for row in non_ten_mhz_rows):
        raise RuntimeError("non-10-MHz rows claim a copied-grid exact-match verdict")
    recomputed_exact = all(
        row["exact_copied_match_when_10mhz"] == "true" for row in ten_mhz_rows
    )
    recomputed_range_maximum = max(
        float(row["fractional_correction_difference_vs_copied_0to500ghz_10mhz"])
        for row in ten_mhz_rows
    )
    if (
        recomputed_exact != manifest["ten_mhz_exactly_matches_copied_centers"]
        or recomputed_range_maximum
        != float(
            manifest[
                "ten_mhz_140to280_vs_copied_0to500_maximum_fractional_correction_difference"
            ]
        )
        or (recomputed_maxima["10"] <= threshold)
        != manifest["ten_mhz_resolution_diagnostic_passes"]
    ):
        raise RuntimeError("frequency-resolution booleans/maxima do not match the CSV")


def verify_native_regeneration_evidence() -> None:
    if not require_artifact_group(
        NATIVE_REGENERATION_ARTIFACTS, label="native-regeneration"
    ):
        return
    manifest = json.loads(
        (PACKAGE_DIR / "native_regeneration_manifest.json").read_text()
    )
    if manifest["schema_version"] != "sci-cal-001-am12-native-regeneration-v2":
        raise RuntimeError("unexpected native-regeneration manifest schema")
    for path, _value in walk_json(manifest):
        key = path.rsplit(".", 1)[-1]
        if is_volatile_output_digest_name(key):
            raise RuntimeError(
                f"native manifest committed a volatile raw-output digest at {path}"
            )

    execution_context_record = manifest["cache_execution_context"]
    execution_context = execution_context_record["content"]
    canonical_context_bytes = (
        json.dumps(execution_context, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    native_runner_path = PACKAGE_DIR / "run_am12_native_regeneration_check.py"
    if (
        execution_context_record["filename"] != "execution_context.json"
        or execution_context_record["sha256"] != sha256_bytes(canonical_context_bytes)
        or execution_context_record["sha256"]
        != "8ff9af2fa844db88f94ca27585e2f33854dc38fe5422935dc57865a669e60093"
        or execution_context.get("schema_version")
        != "sci-cal-001-am12-native-regeneration-v2-execution-context-v1"
        or execution_context.get("runner")
        != {
            "filename": native_runner_path.name,
            "sha256": sha256_path(native_runner_path),
        }
        or execution_context.get("run_scope") != "full_annual_matrix"
    ):
        raise RuntimeError(
            "native regeneration cache execution-context identity changed"
        )

    context_host = execution_context.get("execution_host")
    if (
        not isinstance(context_host, dict)
        or set(context_host)
        != {
            "node",
            "system",
            "release",
            "machine",
            "python",
            "python_executable",
            "numpy",
        }
        or any(
            not isinstance(value, str) or not value for value in context_host.values()
        )
        or not Path(context_host["python_executable"]).is_absolute()
    ):
        raise RuntimeError("native regeneration execution host is incomplete")

    context_parameters = execution_context.get("execution_parameters")
    if context_parameters != {
        "jobs": 7,
        "omp_threads_per_process": 2,
        "locale": {"LANG": "C", "LC_ALL": "C"},
        "argv_template": [
            "<am-executable>",
            "LMT_am_inputs/LMT_annual_<percentile>.amc",
            "0",
            "GHz",
            "500",
            "GHz",
            "10",
            "MHz",
            "<zenith-angle-deg>",
            "deg",
            "1.0",
        ],
        "slurm_wrapper_used": False,
        "working_directory_role": "Big_Atmosphere",
        "am_cache_sharding": {
            "shard_count": 7,
            "assignment": (
                "percentile-major matrix index with zenith angle minor, "
                "modulo shard_count"
            ),
            "phase_order": [
                {
                    "phase": "smoke_gate",
                    "cases": ["LMT_annual_95_za10", "LMT_annual_95_za70"],
                    "completion_barrier": (
                        "both cases must exactly match before any subsequent phase"
                    ),
                },
                {
                    "phase": "remaining_matrix",
                    "order": (
                        "all non-smoke cases in percentile-major order with zenith "
                        "angle ascending before shard assignment"
                    ),
                },
            ],
            "within_shard_order": (
                "encounter order inside each phase; phases do not overlap"
            ),
            "process_ownership": (
                "one ordered worker queue per shard per phase inside one process"
            ),
        },
        "cache_lock": {
            "filename": ".native_regeneration.lock",
            "writer_mode": "nonblocking whole-cache POSIX exclusive lock",
            "reader_mode": "nonblocking whole-cache POSIX shared lock",
        },
    }:
        raise RuntimeError("native regeneration immutable execution parameters changed")

    def verify_inventory(
        inventory: object,
        *,
        label: str,
        expected_paths: list[str] | None = None,
    ) -> list[dict[str, object]]:
        if not isinstance(inventory, dict) or not isinstance(
            inventory.get("files"), list
        ):
            raise RuntimeError(f"malformed native {label} inventory")
        files = inventory["files"]
        paths: list[str] = []
        total_bytes = 0
        aggregate = hashlib.sha256()
        for entry in files:
            if not isinstance(entry, dict) or set(entry) != {
                "path",
                "size_bytes",
                "sha256",
            }:
                raise RuntimeError(f"malformed native {label} inventory entry")
            relative = entry["path"]
            size = entry["size_bytes"]
            digest = entry["sha256"]
            if (
                not isinstance(relative, str)
                or not relative
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or not isinstance(size, int)
                or size <= 0
                or not is_sha256_hex(digest)
            ):
                raise RuntimeError(f"unsafe or incomplete native {label} entry")
            paths.append(relative)
            total_bytes += size
            aggregate.update(relative.encode("utf-8"))
            aggregate.update(b"\0")
            aggregate.update(bytes.fromhex(digest))
            aggregate.update(b"\0")
        if (
            len(paths) != len(set(paths))
            or inventory.get("file_count") != len(files)
            or inventory.get("total_bytes") != total_bytes
            or inventory.get("aggregate_algorithm")
            != "sha256(relative_path NUL file_sha256_bytes NUL)"
            or inventory.get("aggregate_sha256") != aggregate.hexdigest()
            or (expected_paths is not None and paths != expected_paths)
        ):
            raise RuntimeError(f"native {label} inventory aggregate changed")
        return files

    context_inputs = execution_context.get("inputs")
    if not isinstance(context_inputs, dict) or set(context_inputs) != {
        "am_source_inventory",
        "annual_profile_inventory",
        "copied_reference_output_inventory",
    }:
        raise RuntimeError("native regeneration execution-context inputs changed")
    source_files = verify_inventory(
        context_inputs["am_source_inventory"], label="AM-source"
    )
    if not source_files or [entry["path"] for entry in source_files] != sorted(
        entry["path"] for entry in source_files
    ):
        raise RuntimeError("native AM-source inventory ordering changed")
    expected_profile_paths = [
        f"LMT_annual_{percentile}.amc" for percentile in (5, 25, 50, 75, 95)
    ]
    verify_inventory(
        context_inputs["annual_profile_inventory"],
        label="annual-profile",
        expected_paths=expected_profile_paths,
    )
    expected_reference_paths = [
        f"LMT_annual_{percentile}_{angle}.dat"
        for percentile in (5, 25, 50, 75, 95)
        for angle in range(10, 82, 2)
    ]
    reference_inventory_files = verify_inventory(
        context_inputs["copied_reference_output_inventory"],
        label="copied-reference-output",
        expected_paths=expected_reference_paths,
    )

    expected_workflow = {
        "historical_command_printer": {
            "path_relative_to_am_root": "Big_Atmosphere/generateAmModels.py",
            "sha256": "29b5445f18463fee872cfa863e6c7799647980294ca2c85432aceb10ed8262a6",
        },
        "historical_packer": {
            "path_relative_to_am_root": "Big_Atmosphere/make_npz.py",
            "sha256": "3a1c7b5283f03230a0d572620b4eca1a4859d61ca8c2b9786a67f4026e2717b5",
        },
        "historical_run_script": {
            "path_relative_to_am_root": "Big_Atmosphere/01_do_am_runs.sh",
            "sha256": "02d64a26c85f615bb194abd6102206f5cef29267599c78d4318dc327b7ce12a3",
        },
    }
    if execution_context.get("historical_workflow") != expected_workflow:
        raise RuntimeError("native regeneration frozen historical workflow changed")
    expected_normalization = {
        "purpose": (
            "preserve warning-bearing combined output while replacing only "
            "volatile runtime and dcache-counter header values"
        ),
        "algorithm": (
            "UTF-8 splitlines; replace lines beginning '# run time ' with "
            "'# run time <volatile>' and lines beginning '# dcache hit: ' "
            "with '# dcache counters <volatile>'; join with LF and append LF"
        ),
    }
    expected_security = {
        "uploader_logs_read": False,
        "uploader_logs_or_credentials_copied": False,
        "network_access": False,
        "unity_access": False,
    }
    if (
        execution_context.get("output_normalization") != expected_normalization
        or execution_context.get("security") != expected_security
    ):
        raise RuntimeError("native execution-context normalization/security changed")
    scope = manifest["scope"]
    if (
        scope["season"] != "annual"
        or scope["water_profile_percentiles"] != [5, 25, 50, 75, 95]
        or scope["zenith_angle_deg"]
        != {"minimum": 10, "maximum": 80, "step": 2, "count_per_profile": 36}
        or scope["derived_elevation_deg"] != {"minimum": 10, "maximum": 80, "step": 2}
        or scope["frequency_ghz"]
        != {
            "minimum": "0.00000000000000000e+00",
            "maximum": "5.00000000000000000e+02",
            "step": "1.00000000000000002e-02",
            "count": 50001,
        }
        or scope["fields_compared"] != ["f_GHz", "tau_neper", "tx", "Trj_K", "Tb_K"]
    ):
        raise RuntimeError(
            "native-regeneration scientific grid or compared fields changed"
        )
    results = manifest["results"]
    if scope["case_count"] != 180 or results["case_count"] != 180:
        raise RuntimeError("native regeneration must contain the full 180-case matrix")
    if (
        results["exact_case_count"] != 180
        or results["mismatch_case_count"] != 0
        or results["all_cases_exact"] is not True
    ):
        raise RuntimeError("native regeneration is not an exact 180-case result")
    if any(
        float(value) != 0.0
        for value in results["maximum_absolute_differences"].values()
    ):
        raise RuntimeError("native regeneration has a nonzero parsed-field difference")
    if (
        results["numeric_data_line_byte_exact_count"] != 180
        or results["numeric_data_line_byte_mismatch_count"] != 0
        or results["return_code_counts"] != {"1": 180}
        or results["unresolved_line_warning_counts"] != {"86": 72, "87": 108}
        or results["warning_class_totals"]
        != {
            "unresolved_column_warning_line_count": 6480,
            "unresolved_summary_warning_line_count": 180,
            "cache_insert_as_mru_warning_line_count": 0,
            "cache_promote_to_mru_warning_line_count": 0,
            "other_warning_line_count": 0,
        }
        or results["error_line_total"] != 0
        or results["normalized_numeric_output_aggregate_algorithm"]
        != "case_id NUL numeric_text_sha256_bytes NUL in case_id bytewise order"
        or results["normalized_numeric_output_aggregate_sha256"]
        != "18abf7fb57f335637c7cb2e105aea910f491d74dcd485df01c63ef759a28cd5c"
        or results["normalized_full_output_aggregate_algorithm"]
        != ("case_id NUL normalized_output_sha256_bytes NUL in case_id bytewise order")
        or results["normalized_full_output_aggregate_sha256"]
        != "fc465133e1cc2ac7458f593209dd8b0adbf320ba79a233fcf852f018883aefaf"
    ):
        raise RuntimeError(
            "native regeneration lost byte identity or warning-status evidence"
        )
    if "not the exact legacy" not in manifest["identity"]["scientific_scope"]:
        raise RuntimeError(
            "native annual regeneration was conflated with legacy-q lineage"
        )
    if manifest["rejected_predecessor_attempt"] != {
        "status": "excluded_from_canonical_evidence",
        "reason": (
            "concurrent processes shared one AM_CACHE_PATH and emitted cache "
            "mutation warnings"
        ),
        "external_cache_basename": ("sci_cal_001_am12_2_native_matrix_20260801_root"),
        "case_count": 180,
        "cases_with_cache_warning": 28,
        "cache_insert_as_mru_warning_line_count": 22,
        "cache_promote_to_mru_warning_line_count": 9,
        "all_numeric_data_lines_exact": True,
        "canonical_artifacts_use_this_attempt": False,
    }:
        raise RuntimeError("native rejected-predecessor disposition changed")
    if manifest["superseded_predecessor_attempt"] != {
        "status": "superseded_by_stronger_provenance_contract",
        "external_cache_basename": (
            "sci_cal_001_am12_2_native_matrix_clean_sharded_20260801_root"
        ),
        "reason": (
            "the matrix was numerically exact and diagnostic-clean, but its "
            "cache did not bind the runner/source/reference/compiler/host "
            "context or committed normalized warning-bearing output"
        ),
        "case_count": 180,
        "all_numeric_data_lines_exact": True,
        "canonical_artifacts_use_this_attempt": False,
    }:
        raise RuntimeError("native superseded-predecessor disposition changed")

    execution = manifest["execution"]
    if (
        execution["argv_template"] != context_parameters["argv_template"]
        or execution["slurm_wrapper_used"] is not False
        or execution["working_directory_role"] != "Big_Atmosphere"
        or execution["jobs"] != 7
        or execution["omp_threads_per_process"] != 2
        or execution["cache_path_recorded_as"] != "paths relative to --cache-dir"
        or execution["am_cache_root_relative_to_cache"] != "am_cache"
        or execution["cache_concurrency_policy"]
        != (
            "one process holds a whole-cache exclusive POSIX lock while "
            "running; one ordered worker queue owns each deterministic "
            "AM_CACHE_PATH shard"
        )
        or execution["am_cache_sharding"]
        != {
            "shard_count": 7,
            "assignment": (
                "(percentile-major matrix index with zenith angle minor) "
                "modulo shard_count"
            ),
            "within_shard_order": (
                "smoke-gate encounter order, completion barrier, then "
                "remaining percentile-major/zenith-angle encounter order"
            ),
            "ownership": (
                "one ordered worker queue per shard per phase; phases do not overlap"
            ),
        }
        or execution["whole_cache_lock"]
        != {
            "filename": ".native_regeneration.lock",
            "run_mode": "nonblocking POSIX exclusive lock",
            "cache_only_mode": "nonblocking POSIX shared lock",
        }
        or execution["environment_overrides"]
        != {
            "OMP_NUM_THREADS": "2",
            "LANG": "C",
            "LC_ALL": "C",
            "AM_CACHE_PATH": {
                "root_path_relative_to_cache": "am_cache",
                "runtime_value_policy": (
                    "absolute resolution of --cache-dir/am_cache/"
                    "shard_<matrix-index-mod-jobs>"
                ),
            },
        }
        or "cache, unknown-warning, and error diagnostics are rejected"
        not in execution["accepted_return_contract"]
        or execution["committed_output_digest_policy"]
        != (
            "commit per-case numeric-text and normalized warning-bearing "
            "combined-output SHA-256 values plus aggregates; raw combined-"
            "output SHA-256 values remain in execution sidecars"
        )
        or execution["combined_output_normalization"] != expected_normalization
        or execution["host"] != context_host
        or execution["staged_smoke_gate"]
        != {
            "cases": ["LMT_annual_95_za10", "LMT_annual_95_za70"],
            "requirement": (
                "all five parsed fields exactly equal before remaining "
                "178 cases execute"
            ),
        }
    ):
        raise RuntimeError("native regeneration cache/warning execution policy changed")

    context_builds = execution_context.get("builds")
    manifest_builds = manifest["builds"]
    if (
        not isinstance(context_builds, dict)
        or set(context_builds) != {"copied_linux_reference", "regeneration"}
        or context_builds["copied_linux_reference"]["sha256"]
        != manifest_builds["copied_linux_reference"]["sha256"]
        or context_builds["copied_linux_reference"]["size_bytes"]
        != manifest_builds["copied_linux_reference"]["size_bytes"]
        or context_builds["copied_linux_reference"]["binary_format"]
        != manifest_builds["copied_linux_reference"]["binary_format"]
        or context_builds["regeneration"]["classification"]
        != manifest_builds["regeneration"]["classification"]
        or context_builds["regeneration"]["native_build_command"]
        != manifest_builds["regeneration"]["build_command_supplied_by_operator"]
        or context_builds["regeneration"]["compiler"]
        != manifest_builds["regeneration"]["compiler"]
        or any(
            context_builds["regeneration"][field]
            != manifest_builds["regeneration"][field]
            for field in (
                "supplied_path",
                "resolved_path",
                "sha256",
                "size_bytes",
                "binary_format",
            )
        )
        or manifest_builds["regeneration"]["same_bytes_as_copied_linux_reference"]
        != (
            manifest_builds["regeneration"]["sha256"]
            == manifest_builds["copied_linux_reference"]["sha256"]
        )
    ):
        raise RuntimeError("native build provenance is not execution-context bound")
    compiler = manifest_builds["regeneration"]["compiler"]
    if (
        manifest_builds["copied_linux_reference"]
        != {
            "path_relative_to_am_root": "am-12.2/bin/am",
            "sha256": (
                "3fc1f71b3a025ac79f5559bdd2fbf40cf5de2aa7598cabf474f74f9a6c3b290c"
            ),
            "size_bytes": 57995352,
            "binary_format": "elf",
            "am_identity": "am version 12.2 (build date Aug 26 2022 19:20:13)",
            "compiler_identity_embedded_in_binary": (
                "GCC 9.4.0 (Ubuntu 9.4.0-1ubuntu1~20.04.1)"
            ),
        }
        or manifest_builds["regeneration"]["classification"]
        != "native_macos_build_distinct_from_copied_linux_binary"
        or manifest_builds["regeneration"]["binary_format"] != "mach-o"
        or not is_sha256_hex(manifest_builds["regeneration"]["sha256"])
        or not isinstance(
            manifest_builds["regeneration"]["build_command_supplied_by_operator"],
            str,
        )
        or not manifest_builds["regeneration"]["build_command_supplied_by_operator"]
        or compiler.get("status") != "supplied_by_operator_as_build_compiler"
        or any(
            not is_sha256_hex(compiler.get(field))
            for field in ("sha256", "version_output_sha256")
        )
        or compiler.get("version_command_return_code") != 0
        or not compiler.get("version_output")
    ):
        raise RuntimeError("native compiler/build identity is incomplete")

    if (
        manifest["historical_workflow"] != expected_workflow
        or manifest["am_source_inventory"] != context_inputs["am_source_inventory"]
        or manifest["annual_profile_inventory"]
        != context_inputs["annual_profile_inventory"]
        or manifest["copied_reference_output_inventory"]
        != context_inputs["copied_reference_output_inventory"]
    ):
        raise RuntimeError("native manifest inputs diverge from execution context")

    copied_manifest = json.loads((PACKAGE_DIR / "copied_am_manifest.json").read_text())
    copied_amcs = {
        item["filename"]: item for item in copied_manifest["amc_inputs"]["files"]
    }
    native_profiles = manifest["annual_profile_inventory"]
    expected_native_amcs = {
        f"LMT_annual_{percentile}.amc" for percentile in (5, 25, 50, 75, 95)
    }
    if (
        native_profiles["file_count"] != 5
        or len(native_profiles["files"]) != 5
        or {item["path"] for item in native_profiles["files"]} != expected_native_amcs
        or any(
            item["size_bytes"] != copied_amcs[item["path"]]["bytes"]
            or item["sha256"] != copied_amcs[item["path"]]["sha256"]
            for item in native_profiles["files"]
        )
    ):
        raise RuntimeError("native regeneration annual AMC identities changed")

    rows = read_csv_rows("native_regeneration_metrics.csv")
    if any(is_volatile_output_digest_name(name) for name in rows[0]):
        raise RuntimeError(
            "native metrics committed a volatile raw-output or sidecar SHA-256"
        )
    if "generated_normalized_output_sha256" not in rows[0]:
        raise RuntimeError(
            "native metrics omit normalized warning-bearing output identity"
        )
    if len({row["case_id"] for row in rows}) != len(rows):
        raise RuntimeError("native regeneration CSV has duplicate case IDs")
    if len(rows) != 180 or any(
        row["all_fields_exact_equal"] != "true"
        or row["numeric_data_lines_byte_equal"] != "true"
        or row["status"] != "exact_match"
        or row["reference_row_count"] != "50001"
        or row["generated_row_count"] != "50001"
        or int(row["unresolved_column_warning_line_count"]) <= 0
        or row["unresolved_summary_warning_line_count"] != "1"
        or row["cache_insert_as_mru_warning_line_count"] != "0"
        or row["cache_promote_to_mru_warning_line_count"] != "0"
        or row["other_warning_line_count"] != "0"
        or row["error_line_count"] != "0"
        or not is_sha256_hex(row["reference_numeric_text_sha256"])
        or not is_sha256_hex(row["generated_numeric_text_sha256"])
        or row["reference_numeric_text_sha256"] != row["generated_numeric_text_sha256"]
        or not is_sha256_hex(row["generated_normalized_output_sha256"])
        or any(
            row[f"{field}_exact_equal"] != "true"
            or float(row[f"{field}_max_abs_difference"]) != 0.0
            for field in ("frequency", "tau", "tx", "trj", "tb")
        )
        for row in rows
    ):
        raise RuntimeError("native regeneration CSV is not 180 exact cases")
    expected_cases = {
        f"LMT_annual_{percentile}_za{angle:02d}"
        for percentile in (5, 25, 50, 75, 95)
        for angle in range(10, 82, 2)
    }
    if {row["case_id"] for row in rows} != expected_cases:
        raise RuntimeError("native regeneration case grid changed")

    reference_inventory_by_path = {
        entry["path"]: entry for entry in reference_inventory_files
    }
    copied_raw_rows = read_csv_rows("copied_am_raw_output_inventory.csv")
    copied_annual_raw_by_path = {
        Path(row["relative_path"]).name: row
        for row in copied_raw_rows
        if row["season"] == "annual"
    }
    if (
        set(reference_inventory_by_path) != set(expected_reference_paths)
        or set(copied_annual_raw_by_path) != set(expected_reference_paths)
        or any(
            entry["sha256"] != copied_annual_raw_by_path[path]["sha256"]
            or entry["size_bytes"] != int(copied_annual_raw_by_path[path]["bytes"])
            for path, entry in reference_inventory_by_path.items()
        )
    ):
        raise RuntimeError(
            "native copied-reference inventory diverges from copied-AM custody"
        )
    if any(
        row["case_id"]
        != f"LMT_annual_{row['percentile']}_za{int(row['zenith_angle_deg']):02d}"
        or row["profile"] != f"LMT_annual_{row['percentile']}"
        or int(row["elevation_deg"]) != 90 - int(row["zenith_angle_deg"])
        or row["reference_path_relative_to_am_root"]
        != (
            f"Big_Atmosphere/LMT_am_outputs/LMT_annual_{row['percentile']}_"
            f"{int(row['zenith_angle_deg'])}.dat"
        )
        or row["generated_path_relative_to_cache"]
        != (
            f"raw_outputs/LMT_annual_{row['percentile']}_"
            f"{int(row['zenith_angle_deg'])}.dat"
        )
        or not is_sha256_hex(row["reference_sha256"])
        or row["reference_sha256"]
        != reference_inventory_by_path[
            Path(row["reference_path_relative_to_am_root"]).name
        ]["sha256"]
        for row in rows
    ):
        raise RuntimeError("native regeneration row identity or geometry changed")
    if any(
        row["return_code"] not in {"0", "1"}
        or (
            row["return_code"] == "1"
            and row["unresolved_line_warning_count"] not in {"86", "87", "88"}
        )
        for row in rows
    ):
        raise RuntimeError(
            "native regeneration violates the recorded AM return contract"
        )
    recomputed_maxima = {
        field: max(float(row[f"{field}_max_abs_difference"]) for row in rows)
        for field in ("frequency", "tau", "tx", "trj", "tb")
    }
    if any(
        recomputed_maxima[field]
        != float(results["maximum_absolute_differences"][field])
        for field in recomputed_maxima
    ):
        raise RuntimeError("native regeneration maxima do not match the CSV")
    if value_counts(rows, "return_code") != results["return_code_counts"]:
        raise RuntimeError("native regeneration return-code distribution mismatch")
    if (
        value_counts(rows, "unresolved_line_warning_count")
        != results["unresolved_line_warning_counts"]
    ):
        raise RuntimeError("native regeneration warning-count distribution mismatch")
    if (
        value_counts(rows, "regeneration_am_identity")
        != results["regeneration_am_identity_counts"]
    ):
        raise RuntimeError("native regeneration AM-identity distribution mismatch")
    native_warning_fields = (
        "unresolved_column_warning_line_count",
        "unresolved_summary_warning_line_count",
        "cache_insert_as_mru_warning_line_count",
        "cache_promote_to_mru_warning_line_count",
        "other_warning_line_count",
    )
    recomputed_warning_totals = {
        field: sum(int(row[field]) for row in rows) for field in native_warning_fields
    }
    normalized_numeric_digest = hashlib.sha256()
    normalized_full_output_digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["case_id"].encode("utf-8")):
        case_id = row["case_id"].encode("utf-8")
        normalized_numeric_digest.update(case_id)
        normalized_numeric_digest.update(b"\0")
        normalized_numeric_digest.update(
            bytes.fromhex(row["generated_numeric_text_sha256"])
        )
        normalized_numeric_digest.update(b"\0")
        normalized_full_output_digest.update(case_id)
        normalized_full_output_digest.update(b"\0")
        normalized_full_output_digest.update(
            bytes.fromhex(row["generated_normalized_output_sha256"])
        )
        normalized_full_output_digest.update(b"\0")
    if (
        recomputed_warning_totals != results["warning_class_totals"]
        or sum(int(row["error_line_count"]) for row in rows)
        != results["error_line_total"]
        or normalized_numeric_digest.hexdigest()
        != results["normalized_numeric_output_aggregate_sha256"]
        or normalized_full_output_digest.hexdigest()
        != results["normalized_full_output_aggregate_sha256"]
    ):
        raise RuntimeError(
            "native regeneration warning/digest aggregates do not match the CSV"
        )
    if (
        sum(row["all_fields_exact_equal"] == "true" for row in rows)
        != results["exact_case_count"]
        or sum(row["numeric_data_lines_byte_equal"] == "true" for row in rows)
        != results["numeric_data_line_byte_exact_count"]
    ):
        raise RuntimeError(
            "native regeneration exact-match counts do not match the CSV"
        )
    if manifest["security"] != expected_security:
        raise RuntimeError("native regeneration security boundary changed")
    metrics_artifact = manifest["artifacts"]["metrics_csv"]
    metrics_path = package_basename_path(
        metrics_artifact["filename"],
        label="native-regeneration metrics artifact filename",
    )
    if (
        metrics_artifact["filename"] != "native_regeneration_metrics.csv"
        or metrics_artifact["sha256"] != sha256_path(metrics_path)
        or metrics_artifact["sha256"]
        != "1d6f099383880207bca94cc0f0236a379a158a0be17e4a365b62371cb1ebca87"
    ):
        raise RuntimeError("native-regeneration metrics digest mismatch")
    if (
        set(manifest["artifacts"])
        != {"metrics_csv", "raw_outputs", "execution_sidecars"}
        or manifest["artifacts"]["raw_outputs"]
        != "stored below --cache-dir and not committed"
        or manifest["artifacts"]["execution_sidecars"]
        != "stored below --cache-dir and not committed"
    ):
        raise RuntimeError("native-regeneration artifact disposition changed")

    report_text = (PACKAGE_DIR / "NATIVE_REGENERATION_REPORT.md").read_text()
    manifest_digest = sha256_path(PACKAGE_DIR / "native_regeneration_manifest.json")
    if (
        manifest_digest
        != "128d2b8481d64120be2fac020658f9f6abbe3de620438563572e6d40d8493ac4"
        or sha256_path(PACKAGE_DIR / "NATIVE_REGENERATION_REPORT.md")
        != "a1f370251cd9e4ecb27b717225094b34a9c6fc5067ae7266a62df8d884906b9b"
    ):
        raise RuntimeError("native-regeneration final artifact identity changed")
    required_report_fragments = (
        "The complete `180`-case annual AM 12.2 matrix",
        "does not establish that these profiles are the exact legacy",
        "did not yet bind the complete execution context",
        (
            "unresolved-line warnings and status 1 remain explicit and are not "
            "described as a clean software success"
        ),
        "with pinned `LANG=C`, `LC_ALL=C`",
        "One whole-cache POSIX writer lock",
        "Within each of two nonoverlapping phases, one ordered worker queue",
        "normalized warning-bearing combined output",
        results["normalized_numeric_output_aggregate_sha256"],
        results["normalized_full_output_aggregate_sha256"],
        metrics_artifact["sha256"],
        manifest_digest,
        execution_context_record["sha256"],
        "No network or Unity access is part of this workflow.",
    )
    if any(fragment not in report_text for fragment in required_report_fragments):
        raise RuntimeError("native-regeneration report lost a required disclosure")


def walk_json(value: object, prefix: str = "") -> list[tuple[str, object]]:
    items: list[tuple[str, object]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            items.append((child_prefix, child))
            items.extend(walk_json(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            items.extend(walk_json(child, f"{prefix}[{index}]"))
    return items


def is_volatile_output_digest_name(name: object) -> bool:
    """Identify raw/combined-output digests while allowing normalized science IDs."""
    if not isinstance(name, str):
        return False
    normalized = name.lower().replace("-", "_")
    has_output_kind = "raw_output" in normalized or "combined_output" in normalized
    has_sha = "sha" in normalized
    allowed_normalized = "normalized" in normalized or "numeric" in normalized
    describes_location = normalized.endswith("_location")
    return (
        has_output_kind
        and has_sha
        and not allowed_normalized
        and not describes_location
    )


def verify_h2o_hypothesis_evidence() -> None:
    if not require_artifact_group(H2O_HYPOTHESIS_ARTIFACTS, label="H2O-hypothesis"):
        raise RuntimeError("missing required all-direct P1 H2O-hypothesis artifacts")
    if set(H2O_FINAL_ARTIFACT_IDENTITIES) != set(H2O_HYPOTHESIS_ARTIFACTS):
        raise RuntimeError("internal H2O final-artifact pin inventory is incomplete")
    for name, expected in H2O_FINAL_ARTIFACT_IDENTITIES.items():
        path = package_basename_path(name, label="H2O final artifact filename")
        if (
            path.stat().st_size != expected["bytes"]
            or sha256_path(path) != expected["sha256"]
        ):
            raise RuntimeError(f"H2O final artifact identity changed: {name}")
    manifest = json.loads(
        (PACKAGE_DIR / "h2o_scale_hypothesis_manifest.json").read_text()
    )
    if manifest.get("schema_version") != "sci-cal-001-am12-h2o-scale-hypothesis-v3":
        raise RuntimeError("unexpected H2O-hypothesis manifest schema")
    context_record = manifest["cache_execution_context"]
    context = context_record["content"]
    canonical_context_bytes = (
        json.dumps(context, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    generator_path = PACKAGE_DIR / "probe_am12_h2o_scale_hypotheses.py"
    if (
        context_record["filename"] != "execution_context.json"
        or context_record["sha256"] != sha256_bytes(canonical_context_bytes)
        or context_record["sha256"]
        != "05148050e96e73577ec75be525b026b5bf37bbd2a8753f8e3702fc0b6dfb2bee"
        or context.get("schema_version")
        != "sci-cal-001-am12-h2o-scale-hypothesis-v2-execution-context-v1"
        or context.get("runner")
        != {
            "filename": generator_path.name,
            "sha256": sha256_path(generator_path),
        }
        or context["runner"]["sha256"]
        != "caa41ca105eec6df99f31d982ca69910ef2d7e1ebcbad86c96faa7d0e4cd3c2c"
    ):
        raise RuntimeError("H2O cache execution-context identity changed")
    generator_text = generator_path.read_text(encoding="utf-8")
    if any(
        statement not in generator_text
        for statement in (
            'truth_tau = -np.log(truth["tx"])',
            'predicted_tau=direct["tau"]',
            "truth_tau=truth_tau",
            "predicted_tau=-np.log(ratio)",
            "truth_tau=-np.log(source_ratio)",
        )
    ):
        raise RuntimeError("H2O candidate/truth tau provenance computation changed")
    context_host = context.get("execution_host")
    if (
        not isinstance(context_host, dict)
        or set(context_host)
        != {
            "node",
            "system",
            "release",
            "machine",
            "python",
            "python_executable",
            "numpy",
        }
        or any(
            not isinstance(value, str) or not value for value in context_host.values()
        )
        or not Path(context_host["python_executable"]).is_absolute()
    ):
        raise RuntimeError("H2O execution-context host is incomplete")
    context_execution = context.get("execution_parameters")
    expected_context_argv = [
        "<am-executable>",
        "LMT_am_inputs/<immutable-profile>.amc",
        "<fmin-binary64-17e>",
        "GHz",
        "<fmax-binary64-17e>",
        "GHz",
        "10",
        "MHz",
        "<integer-zenith-angle-deg>",
        "deg",
        "<frozen-h2o-scale-decimal>",
    ]
    if context_execution != {
        "jobs": 8,
        "omp_threads_per_process": 1,
        "locale": {"LANG": "C", "LC_ALL": "C"},
        "argv_template": expected_context_argv,
        "working_directory_role": "Big_Atmosphere",
        "slurm_wrapper_used": False,
        "am_cache_sharding": {
            "shard_count": 8,
            "cache_id_identity": (
                "canonical RunSpec request, AM executable/profile SHA-256, "
                "OMP threads, shard count, and execution-context SHA-256"
            ),
            "assignment": (
                "big-endian first 64 bits of sha256(cache_id) modulo shard_count"
            ),
            "within_process_locking": ("one lock per shard around each AM subprocess"),
        },
        "cache_lock": {
            "filename": ".h2o_scale_hypothesis.lock",
            "writer_mode": "nonblocking whole-cache POSIX exclusive lock",
            "reader_mode": "nonblocking whole-cache POSIX shared lock",
        },
        "in_process_observation_retention": {
            "record_type": "frozen lightweight RunObservation",
            "retained_fields": [
                "cache_id",
                "RunSpec",
                "return_code",
                "AM_version_identity",
                "diagnostic_counts",
                "numeric_text_sha256",
                "normalized_output_sha256",
            ],
            "explicitly_not_retained": [
                "ParsedOutput.samples",
                "raw_combined_output",
                "execution_sidecar_payload",
            ],
            "purpose": (
                "keep final all-run digest aggregation memory-bounded without "
                "changing its scientific identity or digest semantics"
            ),
        },
    }:
        raise RuntimeError("H2O immutable execution-context parameters changed")
    identity = manifest["identity"]
    if (
        identity["study"] != "diagnostic_P1_documented_h2o_scale_provenance_hypothesis"
        or identity["study_status"] != "post_hoc_provenance_hypothesis"
        or identity["custody_proof"] is not False
        or identity["holdout_evidence"] is not False
        or identity["operator_authorization"] != "none"
        or identity["operational_domain_authorization"] != "none"
    ):
        raise RuntimeError("H2O hypothesis identity/authorization boundary changed")
    expected_target_literals = {
        "am_q25": "0.9500275",
        "am_q50": "0.9142065",
        "am_q75": "0.8515054",
        "am_q95": "0.7337698",
    }
    expected_targets = set(expected_target_literals)
    expected_profiles = {
        f"LMT_{family}_{percentile}"
        for family in ("annual", "DJF", "MAM", "JJA", "SON")
        for percentile in (5, 25, 50, 75, 95)
    }
    expected_profile_order = [
        f"LMT_{family}_{percentile}"
        for family in ("annual", "DJF", "MAM", "JJA", "SON")
        for percentile in (5, 25, 50, 75, 95)
    ]
    scope = manifest["scope"]
    if (
        set(scope["generic_registry_targets"]) != expected_targets
        or len(scope["generic_registry_targets"]) != 4
        or set(scope["copied_profile_families"]) != expected_profiles
        or len(scope["copied_profile_families"]) != 25
        or scope["profile_count"] != 25
        or scope["hypothesis_count"] != 100
        or scope["frequency_grid_ghz"]["count"] != 50001
        or scope["elevation_grid_deg"]["count"] != 31
        or scope["passband_integration"] != "none_legacy_monochromatic_contract"
    ):
        raise RuntimeError("H2O hypothesis scope is not the frozen 100-case P1")

    expected_context_normalization = {
        "purpose": (
            "preserve warning-bearing combined output while replacing only "
            "volatile runtime and dcache-counter header values"
        ),
        "algorithm": (
            "UTF-8 splitlines; replace lines beginning '# run time ' with "
            "'# run time <volatile>' and lines beginning '# dcache hit: ' "
            "with '# dcache counters <volatile>'; join with LF and append LF"
        ),
    }
    expected_context_security = {
        "uploader_logs_read": False,
        "uploader_logs_or_credentials_copied": False,
        "network_access": False,
        "unity_access": False,
        "citlali_application_code_modified": False,
    }
    if (
        context.get("protocol")
        != {
            "study": "diagnostic_P1_documented_h2o_scale_provenance_hypothesis",
            "repair_base_sha": "9aae0e669384c5c0c0dda93debc194d6b8dac787",
            "repair_line_evidence_head": ("ae99be1cef8c390d0e7490835ffca1f31da7ebc0"),
            "profile_stems": expected_profile_order,
            "target_transmission_literals": expected_target_literals,
            "frequency_grid_ghz": {
                "minimum": "0.00000000000000000e+00",
                "maximum": "5.00000000000000000e+02",
                "step": "1.00000000000000002e-02",
                "count": 50001,
            },
            "elevation_grid_deg": list(range(20, 82, 2)),
            "root_iterations": 48,
            "maximum_bracket_expansions": 64,
            "only_varying_parameter": (
                "Nscale troposphere h2o through immutable AMC argv %9"
            ),
        }
        or context.get("output_normalization") != expected_context_normalization
        or context.get("security") != expected_context_security
    ):
        raise RuntimeError("H2O execution-context protocol/security changed")

    input_provenance = manifest["input_provenance"]
    copied_manifest_path = PACKAGE_DIR / "copied_am_manifest.json"
    copied_manifest = json.loads(copied_manifest_path.read_text())
    native_manifest = json.loads(
        (PACKAGE_DIR / "native_regeneration_manifest.json").read_text()
    )
    h2o_executable = input_provenance["am_executable"]
    native_build = native_manifest["builds"]["regeneration"]
    if (
        h2o_executable["sha256"] != native_build["sha256"]
        or h2o_executable["size_bytes"] != native_build["size_bytes"]
        or Path(h2o_executable["resolved_path"]).resolve()
        != Path(native_build["resolved_path"]).resolve()
        or input_provenance["copied_suite_manifest"]
        != {
            "filename": "copied_am_manifest.json",
            "sha256": sha256_path(copied_manifest_path),
            "canonical_product_manifest_sha256": (
                "18dfd96f4438151197d3b6be5201476f7a71710363d81ec49c801101fa12b3ac"
            ),
        }
    ):
        raise RuntimeError("H2O hypothesis executable/copied-suite identity changed")

    copied_amcs = {
        item["filename"]: item for item in copied_manifest["amc_inputs"]["files"]
    }
    h2o_amcs = input_provenance["immutable_amc_profiles"]
    h2o_amc_by_name = {
        item["path_relative_to_profile_root"]: item for item in h2o_amcs["files"]
    }
    if (
        h2o_amcs["file_count"] != 25
        or len(h2o_amcs["files"]) != 25
        or set(h2o_amc_by_name) != set(copied_amcs)
        or h2o_amcs["total_bytes"] != copied_manifest["amc_inputs"]["total_bytes"]
        or h2o_amcs["aggregate_sha256"]
        != copied_manifest["amc_inputs"]["canonical_nul_records"]["sha256"]
        or any(
            item["size_bytes"] != copied_amcs[name]["bytes"]
            or item["sha256"] != copied_amcs[name]["sha256"]
            for name, item in h2o_amc_by_name.items()
        )
    ):
        raise RuntimeError("H2O hypothesis immutable AMC inventory changed")

    copied_products = {
        Path(item["filename"]).stem: item
        for item in copied_manifest["copied_suite"]["products"]
    }
    h2o_products = {
        item["profile"]: item for item in input_provenance["copied_scale1_npz_products"]
    }
    if (
        len(input_provenance["copied_scale1_npz_products"]) != 25
        or set(h2o_products) != set(copied_products)
        or any(
            item["filename"] != copied_products[profile]["filename"]
            or item["size_bytes"] != copied_products[profile]["bytes"]
            or item["sha256"] != copied_products[profile]["sha256"]
            or item["md5"] != copied_products[profile]["md5"]
            for profile, item in h2o_products.items()
        )
    ):
        raise RuntimeError("H2O hypothesis copied NPZ inventory changed")

    expected_am_contracts = {
        "Big_Atmosphere/01_do_am_runs.sh": {
            "sha256": "02d64a26c85f615bb194abd6102206f5cef29267599c78d4318dc327b7ce12a3"
        },
        "Big_Atmosphere/generateAmModels.py": {
            "sha256": "29b5445f18463fee872cfa863e6c7799647980294ca2c85432aceb10ed8262a6"
        },
        "Big_Atmosphere/make_npz.py": {
            "sha256": "3a1c7b5283f03230a0d572620b4eca1a4859d61ca8c2b9786a67f4026e2717b5"
        },
        "am-12.2/src/config.c": {
            "sha256": "6e57faf4e58a536c8fdb66291c9a186f0f3c01356ee6b00a9677eea6c7fbce79"
        },
        "am-12.2/src/nscale.c": {
            "sha256": "c00a333583988c241fc80a2648378914361e0eaf8fdd8f1fc112b7d2ff913d06"
        },
    }
    expected_legacy_raw_sources = {
        "am_q25": {
            "filename": "amLMT25.npz",
            "sha256": FROZEN_RAW_SOURCE_FILES["amLMT25.npz"],
            "md5": "008d7fa69aff187a9edf419f3d961b4c",
            "tolteca_datafile_id": "454",
        },
        "am_q50": {
            "filename": "amLMT50.npz",
            "sha256": FROZEN_RAW_SOURCE_FILES["amLMT50.npz"],
            "md5": "6ec393672be8af4dfa06a3f4cf9aa32e",
            "tolteca_datafile_id": "455",
        },
        "am_q75": {
            "filename": "amLMT75.npz",
            "sha256": FROZEN_RAW_SOURCE_FILES["amLMT75.npz"],
            "md5": "d6cf4bb27008179ec491864388deac58",
            "tolteca_datafile_id": "456",
        },
    }
    expected_repair_inputs = {
        "include/citlali/core/timestream/rtc/calibrate.h": {
            "sha256": (
                "d70a55278227b43cdd7de19bc67e4ddb332524d40e1455c5fa7a80ae5e2d11ee"
            )
        },
        "validation/sci_cal_001_phase0_2026-07-31/generate_q_model_continuity.py": {
            "sha256": (
                "a46211c007bdc1fa11d1408c6db4c4a68264ca00cd383806fd421ba978fffe78"
            )
        },
    }
    if (
        Path(input_provenance["am_root"]).resolve() != AM_ROOT.resolve()
        or Path(input_provenance["legacy_source_dir"]).resolve()
        != RAW_SOURCE_DIR.resolve()
        or input_provenance["am_source_and_historical_workflow_contracts"]
        != expected_am_contracts
        or input_provenance["legacy_raw_sources"] != expected_legacy_raw_sources
        or input_provenance["missing_q95_raw_source"]
        != {
            "tolteca_datafile_id": "461",
            "expected_md5": "0ca7b331823237767d26016d19bffb3d",
            "status": "registered_raw_grid_absent_not_retrieved",
        }
        or input_provenance["repair_base_inputs"] != expected_repair_inputs
    ):
        raise RuntimeError("H2O hypothesis source/workflow/legacy lineage changed")

    context_builds = context.get("builds")
    native_reference_build = native_manifest["builds"]["copied_linux_reference"]
    context_compiler = context_builds["regeneration"]["compiler"]
    native_compiler = native_build["compiler"]
    if (
        context_builds != input_provenance["builds"]
        or context_builds["copied_linux_reference"]["sha256"]
        != native_reference_build["sha256"]
        or context_builds["copied_linux_reference"]["size_bytes"]
        != native_reference_build["size_bytes"]
        or context_builds["copied_linux_reference"]["binary_format"]
        != native_reference_build["binary_format"]
        or context_builds["regeneration"]["classification"]
        != native_build["classification"]
        or context_builds["regeneration"]["native_build_command"]
        != native_build["build_command_supplied_by_operator"]
        or any(
            context_compiler[field] != native_compiler[field]
            for field in native_compiler
        )
        or context_compiler.get("binary_format") != "mach-o"
        or any(
            context_builds["regeneration"][field] != native_build[field]
            for field in (
                "supplied_path",
                "resolved_path",
                "sha256",
                "size_bytes",
                "binary_format",
            )
        )
        or context_builds["regeneration"]["native_build_command"]
        != "make -j8 gcc-omp COMPILER_GCC=gcc-15"
        or context_builds["regeneration"]["compiler"]["supplied_path"]
        != "/opt/homebrew/Cellar/gcc/15.2.0_1/bin/gcc-15"
    ):
        raise RuntimeError("H2O build provenance is not native/context bound")

    expected_context_workflow = {
        path: {"path_relative_to_am_root": path, **metadata}
        for path, metadata in expected_am_contracts.items()
        if path.startswith("Big_Atmosphere/")
    }
    context_inputs = context.get("inputs")
    expected_context_legacy = [
        {
            "target": target,
            **metadata,
            "size_bytes": 37201984,
        }
        for target, metadata in expected_legacy_raw_sources.items()
    ]
    expected_context_repairs = [
        {
            "path_relative_to_repository": (
                "include/citlali/core/timestream/rtc/calibrate.h"
            ),
            "size_bytes": 7507,
            "sha256": (
                "d70a55278227b43cdd7de19bc67e4ddb332524d40e1455c5fa7a80ae5e2d11ee"
            ),
        },
        {
            "path_relative_to_repository": (
                "validation/sci_cal_001_phase0_2026-07-31/"
                "generate_q_model_continuity.py"
            ),
            "size_bytes": 26421,
            "sha256": (
                "a46211c007bdc1fa11d1408c6db4c4a68264ca00cd383806fd421ba978fffe78"
            ),
        },
    ]
    expected_protocol_files = [
        {
            "path": "FOLLOWUP_STUDY_DEVIATION_LOG.md",
            "size_bytes": 2066,
            "sha256": (
                "a3df86366c7869579b3255d9ea8f95cf6827e78018e0a2a83a1640360be1b036"
            ),
        },
        {
            "path": "FOLLOWUP_STUDY_PREREGISTRATION.md",
            "size_bytes": 8528,
            "sha256": (
                "65935dbc906317e984cf2ae8b35c5868a3f216eca2ec6290f2887976892d8457"
            ),
        },
        {
            "path": "FOLLOWUP_STUDY_PROTOCOL_ADDENDUM.md",
            "size_bytes": 5236,
            "sha256": (
                "0d47c11479a1ba0176babd3ea285e2871edbb1341406b6b044cbc53114c51a1d"
            ),
        },
    ]
    protocol_aggregate = hashlib.sha256()
    for entry in expected_protocol_files:
        protocol_aggregate.update(entry["path"].encode("utf-8"))
        protocol_aggregate.update(b"\0")
        protocol_aggregate.update(bytes.fromhex(entry["sha256"]))
        protocol_aggregate.update(b"\0")
    expected_protocol_inventory = {
        "file_count": 3,
        "total_bytes": sum(entry["size_bytes"] for entry in expected_protocol_files),
        "aggregate_sha256": protocol_aggregate.hexdigest(),
        "aggregate_algorithm": "sha256(relative_path NUL file_sha256_bytes NUL)",
        "files": expected_protocol_files,
    }
    if (
        context.get("historical_workflow") != expected_context_workflow
        or not isinstance(context_inputs, dict)
        or context_inputs["am_source_inventory"]
        != native_manifest["am_source_inventory"]
        or context_inputs["am_source_contracts"]
        != {
            path: metadata
            for path, metadata in expected_am_contracts.items()
            if path.startswith("am-12.2/src/")
        }
        or context_inputs["immutable_amc_profile_inventory"] != h2o_amcs
        or context_inputs["copied_scale1_npz_products"]
        != input_provenance["copied_scale1_npz_products"]
        or context_inputs["copied_suite_manifest"]
        != {
            "filename": "copied_am_manifest.json",
            "size_bytes": 43771,
            "sha256": sha256_path(copied_manifest_path),
            "canonical_product_manifest_sha256": (
                "18dfd96f4438151197d3b6be5201476f7a71710363d81ec49c801101fa12b3ac"
            ),
        }
        or context_inputs["legacy_raw_sources"] != expected_context_legacy
        or context_inputs["missing_q95_raw_source"]
        != input_provenance["missing_q95_raw_source"]
        or context_inputs["repair_base_inputs"] != expected_context_repairs
        or context_inputs["frozen_protocol_artifact_inventory"]
        != expected_protocol_inventory
    ):
        raise RuntimeError("H2O cache context input inventory changed")

    execution = manifest["execution"]
    expected_digest_algorithms = {
        "direct_run_numeric_text_aggregate_sha256": (
            "sha256(for each run sorted by "
            "(profile,target,f_min_centi_ghz,f_max_centi_ghz,"
            "zenith_angle_deg,scale_decimal): UTF-8 "
            "json.dumps(scientific_identity,sort_keys=True,separators=(',',':')) "
            "+ NUL + bytes.fromhex(numeric_text_sha256) + NUL)"
        ),
        "scale_trace_sha256": (
            "sha256(UTF-8 json.dumps(trace,indent=2,sort_keys=True) + newline); "
            "trace binds execution-context SHA-256 and each evaluation's "
            "numeric-text and normalized warning-bearing output SHA-256; it "
            "excludes raw combined-output SHA/cache ID"
        ),
        "all_referenced_normalized_output_aggregate_sha256": (
            "for each unique referenced run sorted by "
            "(stage,profile,target,f_min_centi_ghz,f_max_centi_ghz,"
            "zenith_angle_deg,scale_decimal): UTF-8 canonical JSON RunSpec "
            "request + NUL + digest bytes + NUL"
        ),
    }
    if (
        execution["raw_outputs_and_sidecars_committed"] is not False
        or execution["cache_dir"] != "external_caller_supplied_not_artifact_identity"
        or execution["working_directory_role"] != "Big_Atmosphere"
        or execution["argv_template"]
        != [
            "<am-executable>",
            "LMT_am_inputs/<immutable-profile>.amc",
            "<fmin>",
            "GHz",
            "<fmax>",
            "GHz",
            "10",
            "MHz",
            "<zenith-angle>",
            "deg",
            "<frozen-h2o-scale-decimal>",
        ]
        or execution["jobs"] != 8
        or execution["omp_threads_per_process"] != 1
        or execution["environment_overrides"]
        != {
            "OMP_NUM_THREADS": "1",
            "LANG": "C",
            "LC_ALL": "C",
            "AM_CACHE_PATH": (
                "external_cache/am_spectral_cache/shard_<deterministic-index>"
            ),
        }
        or execution["am_cache_sharding"]
        != {
            "shard_count": 8,
            "assignment": (
                "big-endian first 64 bits of sha256(cache_id) modulo shard_count"
            ),
            "locking": "one in-process lock per shard around each AM subprocess",
            "purpose": "prevent concurrent AM insert_as_mru rename races",
        }
        or execution["whole_cache_lock"]
        != {
            "filename": ".h2o_scale_hypothesis.lock",
            "run_mode": "nonblocking POSIX exclusive lock",
            "cache_only_mode": "nonblocking POSIX shared lock",
        }
        or execution["host"] != context_host
        or execution["execution_context_sha256"] != context_record["sha256"]
        or execution["check_mode"] != "cache_only_no_am_subprocess"
        or execution["normalized_numeric_digest_algorithms"]
        != expected_digest_algorithms
    ):
        raise RuntimeError("H2O hypothesis execution/digest contract changed")
    rejected_attempts = execution["rejected_attempt_inventory"]
    if (
        rejected_attempts["count"] != 0
        or rejected_attempts["classification_counts"] != {}
        or rejected_attempts["combined_output_sha256_location"]
        != "external failure sidecars only, never committed manifest"
    ):
        raise RuntimeError("H2O rejected-attempt inventory is inconsistent")

    scale_inference = manifest["scale_inference"]
    construction = manifest["full_grid_construction"]
    results = manifest["results"]
    ranking = manifest["ranking"]
    run_summary = results["execution_run_digest_summary"]
    run_summary_identity = {
        key: run_summary.get(key) for key in H2O_FINAL_RUN_SUMMARY_IDENTITY
    }
    if run_summary_identity != H2O_FINAL_RUN_SUMMARY_IDENTITY:
        raise RuntimeError("H2O final execution-run digest identity changed")
    run_count = run_summary["unique_referenced_run_count"]
    if (
        run_count != 13667
        or not is_sha256_hex(run_summary["normalized_numeric_text_aggregate_sha256"])
        or not is_sha256_hex(
            run_summary["normalized_warning_bearing_output_aggregate_sha256"]
        )
        or run_summary["normalized_numeric_text_aggregate_algorithm"]
        != expected_digest_algorithms[
            "all_referenced_normalized_output_aggregate_sha256"
        ]
        or run_summary["normalized_warning_bearing_output_aggregate_algorithm"]
        != expected_digest_algorithms[
            "all_referenced_normalized_output_aggregate_sha256"
        ]
        or sum(run_summary["return_code_counts"].values()) != run_count
        or not set(run_summary["return_code_counts"]).issubset({"0", "1"})
        or sum(run_summary["am_version_identity_counts"].values()) != run_count
        or len(run_summary["am_version_identity_counts"]) != 1
        or not next(iter(run_summary["am_version_identity_counts"])).startswith(
            "am version 12.2"
        )
        or run_summary["diagnostic_totals"]["other_warning_line_count"] != 0
        or run_summary["diagnostic_totals"]["error_line_count"] != 0
        or any(
            not isinstance(value, int) or value < 0
            for value in run_summary["diagnostic_totals"].values()
        )
        or run_summary["per_run_raw_and_normalized_digests"]
        != ("bound to execution-context SHA-256 in external execution sidecars")
    ):
        raise RuntimeError("H2O context-bound execution-run summary changed")
    if (
        scale_inference["hypothesis_count"] != 100
        or scale_inference["exact_parsed_anchor_count"] != 100
        or scale_inference["direct_full_grid_exact_anchor_count"] != 100
        or scale_inference["no_other_atmospheric_degrees"] is not True
        or construction["frozen_p1_fulfilled_by_all_direct_lane"] is not True
        or construction["direct_hypothesis_count"] != 100
        or construction["ancillary_affine_used_for_p1_completion_or_ranking"]
        is not False
        or results["direct_hypothesis_count"] != 100
        or results["direct_full_grid_exact_anchor_count"] != 100
        or execution["execution_context_sha256"] != context_record["sha256"]
        or ranking["unregistered_composite_score"] is not False
        or set(ranking["final_rank1_identities"]) != expected_targets
        or ranking["final_rank1_identities"] != results["direct_rank1_results"]
    ):
        raise RuntimeError("H2O hypothesis does not fulfill frozen all-direct P1")
    if manifest["interrupted_predecessor_attempt"] != {
        "status": "noncanonical_interrupted_excluded_from_v3_evidence",
        "external_cache_basename": "sci_cal_001_h2o_scale_p1_20260801_root_v2",
        "termination": (
            "stopped after cache-provenance review; no related probe or AM "
            "process remained"
        ),
        "reason": (
            "the process had no whole-cache cross-process lock and its sidecars "
            "did not bind an immutable execution context"
        ),
        "observed_partial_cache_inventory": {
            "raw_output_file_count": 12455,
            "execution_sidecar_file_count": 12455,
            "scale_trace_file_count": 100,
            "execution_sidecar_stage_counts": {
                "anchor_225ghz_el80": 9792,
                "direct_full_grid_all_hypotheses": 1764,
                "direct_full_grid_selected_transmission_rank1": 124,
                "full_grid_scale0_construction_endpoint": 775,
            },
            "direct_fitted_scale_expected_stage_counts": {
                "direct_full_grid_all_hypotheses": 2976,
                "direct_full_grid_selected_transmission_rank1": 124,
            },
            "total_direct_fitted_scale_expected_count": 3100,
            "total_direct_fitted_scale_observed_count": 1888,
            "targeted_sigint_failure_inventory": {
                "failure_sidecar_count": 3,
                "empty_combined_output_file_count": 3,
                "return_code": -2,
                "profile": "LMT_JJA_5",
                "target": "am_q25",
                "zenith_angles_deg": [10, 50, 54],
                "disposition": "termination records excluded from v3 evidence",
            },
        },
        "preservation": "external cache retained read-only; not deleted",
        "canonical_v3_artifacts_or_rankings_use_this_attempt": False,
    }:
        raise RuntimeError("H2O interrupted-v2 predecessor disposition changed")
    if manifest["interrupted_first_v3_development_attempt"] != {
        "status": "noncanonical_development_attempt_excluded",
        "external_cache_basename": (
            "sci_cal_001_h2o_scale_p1_context_v3_final_20260801_root"
        ),
        "execution_context_sha256": (
            "b6f7f88175983b49d2113bdbe626f115e7ced1da6922d7c5dea9636b64217fdd"
        ),
        "runner_sha256": (
            "dae40f4484dead989d4cc559ea7cc52f9af844651c52949a6399214373d82625"
        ),
        "termination": (
            "stopped during anchor inference after pre-full-grid "
            "memory-retention review; no related probe or AM process remained"
        ),
        "reason": (
            "the in-process digest inventory retained complete ParsedOutput "
            "sample arrays and would have added approximately 7.75 GB across "
            "the final full-grid runs"
        ),
        "observed_partial_cache_inventory": {
            "successful_raw_output_file_count": 1811,
            "successful_execution_sidecar_file_count": 1811,
            "scale_trace_file_count": 16,
            "targeted_sigint_failure_inventory": {
                "empty_combined_output_file_count": 6,
                "complete_failure_sidecar_count": 3,
                "empty_atomic_failure_sidecar_temporary_file_count": 3,
                "return_code_where_complete": -2,
                "profiles": [
                    "LMT_JJA_25",
                    "LMT_JJA_50",
                    "LMT_JJA_75",
                    "LMT_SON_5",
                    "LMT_SON_25",
                    "LMT_SON_75",
                ],
                "target": "am_q25",
                "zenith_angle_deg": 10,
                "disposition": "termination records excluded from evidence",
            },
        },
        "preservation": "external cache retained untouched; never reused",
        "canonical_artifacts_or_rankings_use_this_attempt": False,
    }:
        raise RuntimeError("H2O interrupted-first-v3 disposition changed")
    for path, value in walk_json(manifest):
        key = path.rsplit(".", 1)[-1].lower()
        normalized = str(value).strip().lower()
        if is_volatile_output_digest_name(key):
            raise RuntimeError(
                f"H2O manifest committed a volatile raw-output digest at {path}"
            )
        if "authoriz" in key and (
            value is True
            or normalized in {"true", "authorized", "approved", "selected"}
        ):
            raise RuntimeError(f"H2O hypothesis asserts authorization at {path}")
        if "custody" in key and (
            value is True
            or normalized in {"true", "established", "proven", "confirmed"}
        ):
            raise RuntimeError(f"H2O hypothesis asserts historical custody at {path}")
    rendered = json.dumps(manifest, sort_keys=True).lower()
    if "custody" not in rendered or "authoriz" not in rendered:
        raise RuntimeError("H2O hypothesis lacks explicit custody/authorization limits")
    if manifest["security"] != {
        "am_tree_modified": False,
        "legacy_repository_modified": False,
        "uploader_logs_read": False,
        "uploader_logs_or_credentials_copied": False,
        "network_access": False,
        "unity_access": False,
        "citlali_application_code_modified": False,
    }:
        raise RuntimeError("H2O hypothesis security/scope boundary changed")
    row_count_keys = {
        "h2o_scale_hypothesis_scales.csv": "scale_row_count",
        "h2o_scale_hypothesis_metrics.csv": "metric_row_count",
        "h2o_scale_hypothesis_coefficients.csv": "coefficient_row_count",
    }
    expected_row_counts = {
        "h2o_scale_hypothesis_scales.csv": 100,
        "h2o_scale_hypothesis_metrics.csv": 1200,
        "h2o_scale_hypothesis_coefficients.csv": 1050,
    }
    if set(manifest["artifacts"]) != {
        *row_count_keys,
        "H2O_SCALE_HYPOTHESIS_REPORT.md",
        "generator",
    }:
        raise RuntimeError("H2O-hypothesis artifact inventory changed")
    generator = manifest["artifacts"]["generator"]
    generator_path = package_basename_path(
        generator["filename"], label="H2O-hypothesis generator filename"
    )
    if (
        generator["filename"] != "probe_am12_h2o_scale_hypotheses.py"
        or sha256_path(generator_path) != generator["sha256"]
        or generator["sha256"]
        != "caa41ca105eec6df99f31d982ca69910ef2d7e1ebcbad86c96faa7d0e4cd3c2c"
    ):
        raise RuntimeError("H2O-hypothesis generator digest mismatch")
    for name, result_key in row_count_keys.items():
        rows = read_csv_rows(name)
        artifact = manifest["artifacts"][name]
        path = package_basename_path(name, label="H2O-hypothesis artifact filename")
        if (
            len(rows) != expected_row_counts[name]
            or len(rows) != manifest["results"][result_key]
            or len(rows) != artifact["row_count"]
            or sha256_path(path) != artifact["sha256"]
        ):
            raise RuntimeError(f"inconsistent H2O-hypothesis rows/digest for {name}")

    scale_rows = read_csv_rows("h2o_scale_hypothesis_scales.csv")
    if any(is_volatile_output_digest_name(name) for name in scale_rows[0]):
        raise RuntimeError("H2O scale table committed volatile raw-output digests")
    expected_hypothesis_matrix = {
        (target, profile)
        for target in expected_targets
        for profile in expected_profiles
    }
    actual_hypothesis_matrix = {
        (row["target_model"], row["source_profile"]) for row in scale_rows
    }
    if (
        len(scale_rows) != 100
        or len(actual_hypothesis_matrix) != len(scale_rows)
        or actual_hypothesis_matrix != expected_hypothesis_matrix
        or any(
            row["exact_parsed_target_transmission_match"] != "true"
            or row["direct_full_grid_evaluated"] != "true"
            or row["direct_full_grid_exact_target_match"] != "true"
            or row["scale_trace_path_relative_to_cache"]
            != f"scale_traces/{row['target_model']}_{row['source_profile']}.json"
            or not is_sha256_hex(row["scale_trace_sha256"])
            or int(row["scale_trace_evaluation_count"]) <= 0
            for row in scale_rows
        )
        or sum(
            row["ancillary_screening_transmission_rank1"] == "true"
            for row in scale_rows
        )
        != 4
    ):
        raise RuntimeError("H2O scale table does not contain 100 direct hypotheses")
    scale_by_hypothesis = {
        (row["target_model"], row["source_profile"]): row for row in scale_rows
    }
    if len({row["scale_trace_sha256"] for row in scale_rows}) != 100:
        raise RuntimeError("H2O scale traces are not uniquely digest-bound")
    for (target, profile), row in scale_by_hypothesis.items():
        _prefix, season, percentile = profile.split("_", 2)
        expected_family = (
            "copied_annual_merra2_2007_2016_profile"
            if season == "annual"
            else f"copied_explicit_seasonal_{season}_merra2_2007_2016_profile"
        )
        target_tx = float(expected_target_literals[target])
        target_tau = -math.log(target_tx)
        finite_fields = (
            "target_los_tau_from_literal",
            "fitted_h2o_scale_decimal",
            "direct_scale0_tau_los",
            "direct_scale0_transmission",
            "copied_scale1_tau_los",
            "copied_scale1_transmission",
            "direct_scale1_tau_los",
            "direct_scale1_transmission",
            "direct_minus_copied_scale1_tau",
            "direct_minus_copied_scale1_transmission",
            "affine_initial_scale_from_direct0_copied1_tau",
            "direct_fitted_tau_los",
            "direct_fitted_transmission",
            "signed_anchor_fractional_correction_error",
            "absolute_anchor_fractional_correction_error",
            "direct_fitted_minus_affine_copied1_tau",
            "direct_fitted_minus_affine_direct1_tau",
            "direct_full_grid_t225_el80_transmission",
        )
        numeric = {field: float(row[field]) for field in finite_fields}
        if (
            any(not math.isfinite(value) for value in numeric.values())
            or row["target_registry_family"]
            != "legacy_generic_unprefixed_am_q_registry"
            or row["target_t225_source_literal"] != expected_target_literals[target]
            or row["source_profile_season"] != season
            or row["source_profile_percentile"] != percentile
            or row["source_profile_family"] != expected_family
            or row["fit_method"]
            != (
                "direct_parsed_tx_plateau_midpoint_seeded_by_direct_scale0_plus_"
                "copied_scale1_tau"
            )
            or row["interpretation"]
            != "post_hoc_candidate_input_recipe_not_custody_proof"
            or not math.isclose(
                numeric["target_los_tau_from_literal"],
                target_tau,
                rel_tol=1.0e-15,
                abs_tol=0.0,
            )
            or numeric["fitted_h2o_scale_decimal"] < 0.0
            or numeric["affine_initial_scale_from_direct0_copied1_tau"] < 0.0
            or any(
                numeric[field] < 0.0
                for field in (
                    "direct_scale0_tau_los",
                    "copied_scale1_tau_los",
                    "direct_scale1_tau_los",
                    "direct_fitted_tau_los",
                )
            )
            or any(
                not 0.0 <= numeric[field] <= 1.0
                for field in (
                    "direct_scale0_transmission",
                    "copied_scale1_transmission",
                    "direct_scale1_transmission",
                    "direct_fitted_transmission",
                    "direct_full_grid_t225_el80_transmission",
                )
            )
            or any(
                abs(math.exp(-numeric[tau_field]) - numeric[tx_field]) > 1.0e-6
                for tau_field, tx_field in (
                    ("direct_scale0_tau_los", "direct_scale0_transmission"),
                    ("copied_scale1_tau_los", "copied_scale1_transmission"),
                    ("direct_scale1_tau_los", "direct_scale1_transmission"),
                    ("direct_fitted_tau_los", "direct_fitted_transmission"),
                )
            )
            or numeric["direct_fitted_transmission"] != target_tx
            or numeric["direct_full_grid_t225_el80_transmission"] != target_tx
            or not math.isclose(
                numeric["signed_anchor_fractional_correction_error"],
                math.expm1(numeric["direct_fitted_tau_los"] - target_tau),
                rel_tol=2.0e-15,
                abs_tol=1.0e-18,
            )
            or numeric["absolute_anchor_fractional_correction_error"]
            != abs(numeric["signed_anchor_fractional_correction_error"])
            or row["exact_parsed_target_transmission_match"] != "true"
            or row["direct_full_grid_evaluated"] != "true"
            or row["direct_full_grid_exact_target_match"] != "true"
            or len(row["anchor_run_return_codes"].split(";")) != 3
            or not set(row["anchor_run_return_codes"].split(";")).issubset({"0", "1"})
            or len(row["anchor_unresolved_warning_counts"].split(";")) != 3
            or any(
                int(value) < 0
                for value in row["anchor_unresolved_warning_counts"].split(";")
            )
        ):
            raise RuntimeError(f"H2O scale-row invariant changed: {target}/{profile}")

    metric_rows = read_csv_rows("h2o_scale_hypothesis_metrics.csv")
    if any(is_volatile_output_digest_name(name) for name in metric_rows[0]):
        raise RuntimeError("H2O metric table committed volatile raw-output digests")
    direct_lane = "direct_am_fitted_scale_all_25_profiles"
    affine_lane = "affine_scale0_to_copied_scale1_all_profiles"
    direct_validation_lane = (
        "affine_construction_vs_direct_am_all_hypotheses_validation"
    )

    def expected_scopes(
        target: str, *, direct_validation: bool
    ) -> list[tuple[str, str]]:
        if target != "am_q95" or direct_validation:
            return [
                ("full_grid", "all"),
                ("nominal_a1100", "a1100"),
                ("nominal_a1400", "a1400"),
                ("nominal_a2000", "a2000"),
            ]
        return [
            ("all_nominal_ratio_surfaces", "all"),
            ("nominal_ratio_surface_a1100", "a1100"),
            ("nominal_ratio_surface_a1400", "a1400"),
            ("nominal_ratio_surface_a2000", "a2000"),
        ]

    expected_metric_matrix = {
        (lane, target, profile, scope_name, band)
        for lane in (affine_lane, direct_lane, direct_validation_lane)
        for target in expected_targets
        for profile in expected_profiles
        for scope_name, band in expected_scopes(
            target, direct_validation=lane == direct_validation_lane
        )
    }
    actual_metric_matrix = {
        (
            row["evaluation_lane"],
            row["target_model"],
            row["source_profile"],
            row["comparison_scope"],
            row["band"],
        )
        for row in metric_rows
    }
    if (
        len(metric_rows) != 1200
        or len(actual_metric_matrix) != len(metric_rows)
        or actual_metric_matrix != expected_metric_matrix
    ):
        raise RuntimeError("H2O metric lane/scope matrix is incomplete or unexpected")
    metric_value_groups = (
        (
            "transmission_or_ratio_min_signed_residual",
            "transmission_or_ratio_max_signed_residual",
            "transmission_or_ratio_max_abs_residual",
            "transmission_or_ratio_p95_abs_residual",
            "transmission_or_ratio_median_abs_residual",
            "transmission_or_ratio_rms_residual",
        ),
        (
            "fractional_correction_min_signed_error",
            "fractional_correction_max_signed_error",
            "fractional_correction_max_abs_error",
            "fractional_correction_p95_abs_error",
            "fractional_correction_median_abs_error",
            "fractional_correction_rms_error",
        ),
    )
    trj_value_group = (
        "trj_min_signed_residual_k",
        "trj_max_signed_residual_k",
        "trj_max_abs_residual_k",
        "trj_p95_abs_residual_k",
        "trj_median_abs_residual_k",
        "trj_rms_residual_k",
    )

    def verify_summary_group(
        row: dict[str, str], fields: tuple[str, ...], *, label: str
    ) -> None:
        minimum, maximum, maximum_abs, p95, median, rms = (
            float(row[field]) for field in fields
        )
        values = (minimum, maximum, maximum_abs, p95, median, rms)
        tolerance = 2.0e-15 * max(1.0, maximum_abs)
        if (
            any(not math.isfinite(value) for value in values)
            or minimum > maximum
            or maximum_abs < 0.0
            or p95 < 0.0
            or median < 0.0
            or rms < 0.0
            or abs(maximum_abs - max(abs(minimum), abs(maximum))) > tolerance
            or median > p95 + tolerance
            or p95 > maximum_abs + tolerance
            or rms > maximum_abs + tolerance
        ):
            raise RuntimeError(f"invalid H2O {label} summary")

    for row in metric_rows:
        target = row["target_model"]
        profile = row["source_profile"]
        scale_row = scale_by_hypothesis[target, profile]
        for fields, label in zip(
            metric_value_groups,
            ("transmission/ratio residual", "fractional-correction"),
            strict=True,
        ):
            verify_summary_group(
                row,
                fields,
                label=f"{label}: {target}/{profile}/{row['evaluation_lane']}",
            )
        trj_blank = [row[field] == "" for field in trj_value_group]
        if any(trj_blank) and not all(trj_blank):
            raise RuntimeError(f"partial H2O Trj metric group: {target}/{profile}")
        if not all(trj_blank):
            verify_summary_group(
                row,
                trj_value_group,
                label=f"Trj residual: {target}/{profile}/{row['evaluation_lane']}",
            )
        correction_max = float(row["fractional_correction_max_abs_error"])
        expected_pass = str(correction_max <= 0.01).lower()
        direct_expected = row["evaluation_lane"] != affine_lane
        if (
            row["target_registry_family"] != "legacy_generic_unprefixed_am_q_registry"
            or row["source_profile_season"] != scale_row["source_profile_season"]
            or row["source_profile_percentile"]
            != scale_row["source_profile_percentile"]
            or row["source_profile_family"] != scale_row["source_profile_family"]
            or row["h2o_scale_decimal"] != scale_row["fitted_h2o_scale_decimal"]
            or row["anchor_exact_parsed_tx_match"] != "true"
            or row["direct_full_grid_evaluated"] != str(direct_expected).lower()
            or not is_sha256_hex(row["truth_sha256"])
            or int(row["sample_count"]) <= 0
            or int(row["fractional_correction_overflow_count"]) < 0
            or row["passes_provisional_1pct_numerical_diagnostic"] != expected_pass
            or int(row["am_run_count"]) != 31
            or row["maximum_return_code"] not in {"0", "1"}
            or not 0 <= int(row["warning_status_run_count"]) <= 31
            or int(row["unresolved_line_warning_count_sum"]) < 0
            or row["interpretation"]
            != (
                "post_hoc_provenance_hypothesis_not_custody_proof_or_"
                "operator_authorization"
            )
        ):
            raise RuntimeError(
                "H2O metric-row identity/execution invariant changed: "
                f"{target}/{profile}/{row['evaluation_lane']}/"
                f"{row['comparison_scope']}"
            )

    affine_principal_rows = [
        row
        for row in metric_rows
        if row["evaluation_lane"] == affine_lane
        and (
            (row["target_model"] != "am_q95" and row["comparison_scope"] == "full_grid")
            or (
                row["target_model"] == "am_q95"
                and row["comparison_scope"] == "all_nominal_ratio_surfaces"
            )
        )
    ]
    affine_rank_one: dict[str, str] = {}
    for target in expected_targets:
        candidates = [
            row for row in affine_principal_rows if row["target_model"] == target
        ]
        transmission_order = sorted(
            candidates,
            key=lambda row: (
                row["anchor_exact_parsed_tx_match"] != "true",
                float(row["transmission_or_ratio_rms_residual"]),
                float(row["transmission_or_ratio_max_abs_residual"]),
                row["source_profile"],
            ),
        )
        transmission_ranks = {
            row["source_profile"]: str(rank)
            for rank, row in enumerate(transmission_order, start=1)
        }
        affine_rank_one[target] = transmission_order[0]["source_profile"]
        if target != "am_q95":
            trj_order = sorted(
                candidates,
                key=lambda row: (
                    row["anchor_exact_parsed_tx_match"] != "true",
                    float(row["trj_rms_residual_k"]),
                    float(row["trj_max_abs_residual_k"]),
                    row["source_profile"],
                ),
            )
            trj_ranks = {
                row["source_profile"]: str(rank)
                for rank, row in enumerate(trj_order, start=1)
            }
        else:
            trj_ranks = {}
        for row in metric_rows:
            if row["target_model"] != target:
                continue
            is_affine = row["evaluation_lane"] == affine_lane
            should_select = is_affine and (
                row["source_profile"] == affine_rank_one[target]
            )
            if (
                row["ancillary_screening_transmission_rank1"]
                != str(should_select).lower()
                or (
                    is_affine
                    and row["transmission_rms_rank"]
                    != transmission_ranks[row["source_profile"]]
                )
                or (
                    is_affine
                    and target != "am_q95"
                    and row["trj_rms_rank"] != trj_ranks[row["source_profile"]]
                )
                or (is_affine and target == "am_q95" and row["trj_rms_rank"] != "")
            ):
                raise RuntimeError(f"H2O affine rank propagation changed: {target}")
        for profile in expected_profiles:
            scale_selected = (
                scale_by_hypothesis[target, profile][
                    "ancillary_screening_transmission_rank1"
                ]
                == "true"
            )
            if scale_selected != (profile == affine_rank_one[target]):
                raise RuntimeError(f"H2O scale/affine rank-one mismatch: {target}")
    if ranking["ancillary_affine_screening_transmission_rank1"] != affine_rank_one:
        raise RuntimeError("H2O manifest/CSV ancillary rank-one identities changed")

    principal_direct_rows = [
        row
        for row in metric_rows
        if row["evaluation_lane"] == direct_lane
        and (
            (row["target_model"] != "am_q95" and row["comparison_scope"] == "full_grid")
            or (
                row["target_model"] == "am_q95"
                and row["comparison_scope"] == "all_nominal_ratio_surfaces"
            )
        )
    ]
    principal_matrix = {
        (row["target_model"], row["source_profile"]) for row in principal_direct_rows
    }
    if (
        len(principal_direct_rows) != 100
        or len(principal_matrix) != len(principal_direct_rows)
        or principal_matrix != expected_hypothesis_matrix
        or any(
            row["anchor_exact_parsed_tx_match"] != "true"
            or row["direct_full_grid_evaluated"] != "true"
            or int(row["am_run_count"]) != 31
            or not is_sha256_hex(row["truth_sha256"])
            for row in principal_direct_rows
        )
    ):
        raise RuntimeError("H2O direct principal metric matrix is incomplete")

    direct_legacy_nominal_rows = [
        row
        for row in metric_rows
        if row["evaluation_lane"] == direct_lane
        and row["target_model"] in {"am_q25", "am_q50", "am_q75"}
        and row["comparison_scope"]
        in {"nominal_a1100", "nominal_a1400", "nominal_a2000"}
    ]
    expected_direct_legacy_nominal_matrix = {
        (target, profile, band)
        for target in ("am_q25", "am_q50", "am_q75")
        for profile in expected_profiles
        for band in ("a1100", "a1400", "a2000")
    }
    actual_direct_legacy_nominal_matrix = {
        (row["target_model"], row["source_profile"], row["band"])
        for row in direct_legacy_nominal_rows
    }
    worst_direct_legacy_nominal = max(
        direct_legacy_nominal_rows,
        key=lambda row: (
            float(row["fractional_correction_max_abs_error"]),
            row["target_model"],
            row["source_profile"],
            row["band"],
        ),
    )
    if (
        len(direct_legacy_nominal_rows) != 225
        or actual_direct_legacy_nominal_matrix != expected_direct_legacy_nominal_matrix
        or any(
            row["passes_provisional_1pct_numerical_diagnostic"] != "true"
            for row in direct_legacy_nominal_rows
        )
        or (
            worst_direct_legacy_nominal["target_model"],
            worst_direct_legacy_nominal["source_profile"],
            worst_direct_legacy_nominal["band"],
            worst_direct_legacy_nominal["fractional_correction_max_abs_error"],
        )
        != (
            "am_q75",
            "LMT_JJA_5",
            "a1100",
            "6.65829283961727556e-03",
        )
    ):
        raise RuntimeError("H2O direct nominal q25/q50/q75 one-percent result changed")

    direct_legacy_full_grid_rows = [
        row
        for row in metric_rows
        if row["evaluation_lane"] == direct_lane
        and row["target_model"] in {"am_q25", "am_q50", "am_q75"}
        and row["comparison_scope"] == "full_grid"
    ]
    if len(direct_legacy_full_grid_rows) != 75 or any(
        row["passes_provisional_1pct_numerical_diagnostic"] != "false"
        for row in direct_legacy_full_grid_rows
    ):
        raise RuntimeError("H2O direct legacy full-grid one-percent result changed")

    direct_q95_combined_rows = [
        row
        for row in principal_direct_rows
        if row["target_model"] == "am_q95"
        and row["comparison_scope"] == "all_nominal_ratio_surfaces"
    ]
    best_q95_max_error = min(
        direct_q95_combined_rows,
        key=lambda row: (
            float(row["fractional_correction_max_abs_error"]),
            row["source_profile"],
        ),
    )
    q95_rms_winner = next(
        row
        for row in direct_q95_combined_rows
        if row["source_profile"]
        == H2O_FINAL_DIRECT_RANK1_PROFILES["am_q95"]["transmission"]
    )
    if (
        len(direct_q95_combined_rows) != 25
        or any(
            row["passes_provisional_1pct_numerical_diagnostic"] != "false"
            for row in direct_q95_combined_rows
        )
        or (
            best_q95_max_error["source_profile"],
            best_q95_max_error["fractional_correction_max_abs_error"],
        )
        != ("LMT_annual_25", "1.11745240975796856e-02")
        or q95_rms_winner["fractional_correction_max_abs_error"]
        != "1.19094929017647764e-02"
    ):
        raise RuntimeError("H2O direct q95 combined-ratio one-percent result changed")

    recomputed_rank_ones: dict[str, dict[str, str]] = {}
    for target in expected_targets:
        target_rows = [
            row for row in principal_direct_rows if row["target_model"] == target
        ]
        transmission_metrics = [
            (
                float(row["transmission_or_ratio_rms_residual"]),
                float(row["transmission_or_ratio_max_abs_residual"]),
                row["source_profile"],
                row,
            )
            for row in target_rows
        ]
        if any(
            not math.isfinite(rms)
            or not math.isfinite(maximum)
            or rms < 0.0
            or maximum < 0.0
            for rms, maximum, _profile, _row in transmission_metrics
        ):
            raise RuntimeError(f"H2O non-finite transmission rank input: {target}")
        transmission_order = sorted(
            transmission_metrics, key=lambda item: (item[0], item[1], item[2])
        )
        if any(
            int(item[3]["transmission_rms_rank"]) != rank
            for rank, item in enumerate(transmission_order, start=1)
        ):
            raise RuntimeError(f"H2O direct transmission ranks changed for {target}")
        recomputed_rank_ones[target] = {
            "transmission": transmission_order[0][3]["source_profile"]
        }
        trj_ranks = [row["trj_rms_rank"] for row in target_rows]
        if target == "am_q95":
            if any(trj_ranks):
                raise RuntimeError("H2O q95 invented a Rayleigh-Jeans rank")
        else:
            trj_metrics = [
                (
                    float(row["trj_rms_residual_k"]),
                    float(row["trj_max_abs_residual_k"]),
                    row["source_profile"],
                    row,
                )
                for row in target_rows
            ]
            if any(
                not math.isfinite(rms)
                or not math.isfinite(maximum)
                or rms < 0.0
                or maximum < 0.0
                for rms, maximum, _profile, _row in trj_metrics
            ):
                raise RuntimeError(f"H2O non-finite Trj rank input: {target}")
            trj_order = sorted(
                trj_metrics, key=lambda item: (item[0], item[1], item[2])
            )
            if any(
                int(item[3]["trj_rms_rank"]) != rank
                for rank, item in enumerate(trj_order, start=1)
            ):
                raise RuntimeError(
                    f"H2O direct Rayleigh-Jeans ranks changed for {target}"
                )
            recomputed_rank_ones[target]["trj"] = trj_order[0][3]["source_profile"]
        expected_transmission_ranks = {
            item[2]: str(rank) for rank, item in enumerate(transmission_order, start=1)
        }
        expected_trj_ranks = (
            {item[2]: str(rank) for rank, item in enumerate(trj_order, start=1)}
            if target != "am_q95"
            else {}
        )
        for row in metric_rows:
            if row["target_model"] != target:
                continue
            if row["evaluation_lane"] == direct_lane:
                if (
                    row["transmission_rms_rank"]
                    != expected_transmission_ranks[row["source_profile"]]
                    or (
                        target != "am_q95"
                        and row["trj_rms_rank"]
                        != expected_trj_ranks[row["source_profile"]]
                    )
                    or (target == "am_q95" and row["trj_rms_rank"] != "")
                ):
                    raise RuntimeError(f"H2O direct rank propagation changed: {target}")
            elif row["evaluation_lane"] == direct_validation_lane and (
                row["transmission_rms_rank"] != "" or row["trj_rms_rank"] != ""
            ):
                raise RuntimeError(
                    f"H2O validation lane unexpectedly carries ranks: {target}"
                )
    if recomputed_rank_ones != H2O_FINAL_DIRECT_RANK1_PROFILES:
        raise RuntimeError("H2O final direct rank-one profile identities changed")

    direct_digest_rows = [
        row
        for row in metric_rows
        if row["evaluation_lane"] == direct_validation_lane
        and row["comparison_scope"] == "full_grid"
    ]
    direct_digest_matrix = {
        (row["target_model"], row["source_profile"]) for row in direct_digest_rows
    }
    if (
        len(direct_digest_rows) != 100
        or len(direct_digest_matrix) != len(direct_digest_rows)
        or direct_digest_matrix != expected_hypothesis_matrix
        or any(
            row["truth_kind"] != "direct_am_same_profile_and_frozen_scale"
            or row["truth_artifact"] != "external_cache_31_direct_full_grid_runs"
            or not is_sha256_hex(row["truth_sha256"])
            or row["direct_full_grid_evaluated"] != "true"
            or int(row["am_run_count"]) != 31
            for row in direct_digest_rows
        )
    ):
        raise RuntimeError("H2O direct normalized-digest matrix is incomplete")

    principal_by_hypothesis = {
        (row["target_model"], row["source_profile"]): row
        for row in principal_direct_rows
    }
    digest_by_hypothesis = {
        (row["target_model"], row["source_profile"]): row for row in direct_digest_rows
    }
    for target, target_results in ranking["final_rank1_identities"].items():
        rank_entries = (
            {
                "transmission": target_results["direct_transmission_rms_rank1"],
                "trj": target_results["direct_rayleigh_jeans_rms_rank1"],
            }
            if target != "am_q95"
            else {
                "transmission": target_results["direct_nominal_ratio_surface_rms_rank1"]
            }
        )
        if (
            target != "am_q95"
            and target_results["rankings_are_separate_no_composite"] is not True
        ):
            raise RuntimeError(f"H2O direct ranks were composited for {target}")
        if target == "am_q95" and target_results["rayleigh_jeans_ranking"] != (
            "not_applicable_raw_q95_absent"
        ):
            raise RuntimeError("H2O q95 ranking limitation changed")
        for ranking_name, entry in rank_entries.items():
            profile = entry["profile"]
            principal = principal_by_hypothesis[target, profile]
            digest_row = digest_by_hypothesis[target, profile]
            expected_rank = (
                principal["transmission_rms_rank"]
                if ranking_name == "transmission"
                else principal["trj_rms_rank"]
            )
            if (
                expected_rank != "1"
                or profile != recomputed_rank_ones[target][ranking_name]
                or entry["direct_full_grid_exact_parsed_target_match"] is not True
                or entry["direct_run_count"] != 31
                or not is_sha256_hex(entry["direct_run_numeric_text_aggregate_sha256"])
                or entry["direct_run_numeric_text_aggregate_sha256"]
                != digest_row["truth_sha256"]
            ):
                raise RuntimeError(
                    f"H2O final rank is not direct/digest-bound: {target}/{ranking_name}"
                )

    coefficient_rows = read_csv_rows("h2o_scale_hypothesis_coefficients.csv")
    if any(is_volatile_output_digest_name(name) for name in coefficient_rows[0]):
        raise RuntimeError(
            "H2O coefficient table committed volatile raw-output digests"
        )
    direct_coefficient_rows = [
        row for row in coefficient_rows if row["evaluation_lane"] == direct_lane
    ]
    expected_direct_coefficient_matrix = {
        (profile, band, power)
        for profile in expected_profiles
        for band in ("a1100", "a1400", "a2000")
        for power in range(7)
    }
    actual_direct_coefficient_matrix = {
        (row["source_profile"], row["band"], int(row["degree_power"]))
        for row in direct_coefficient_rows
    }
    expected_all_coefficient_matrix = {
        (lane, "am_q95", profile, band, power)
        for lane in (affine_lane, direct_lane)
        for profile in expected_profiles
        for band in ("a1100", "a1400", "a2000")
        for power in range(7)
    }
    actual_all_coefficient_matrix = {
        (
            row["evaluation_lane"],
            row["target_model"],
            row["source_profile"],
            row["band"],
            int(row["degree_power"]),
        )
        for row in coefficient_rows
    }
    if (
        len(coefficient_rows) != 1050
        or len(actual_all_coefficient_matrix) != len(coefficient_rows)
        or actual_all_coefficient_matrix != expected_all_coefficient_matrix
        or len(direct_coefficient_rows) != 525
        or len(actual_direct_coefficient_matrix) != len(direct_coefficient_rows)
        or actual_direct_coefficient_matrix != expected_direct_coefficient_matrix
        or any(row["target_model"] != "am_q95" for row in direct_coefficient_rows)
    ):
        raise RuntimeError("H2O direct q95 coefficient matrix is incomplete")
    coefficient_literals: dict[tuple[str, int], str] = {}
    expected_band_frequencies = {
        "a1100": 272.73,
        "a1400": 214.29,
        "a2000": 150.00,
    }
    for row in coefficient_rows:
        profile = row["source_profile"]
        scale_row = scale_by_hypothesis["am_q95", profile]
        band = row["band"]
        power = int(row["degree_power"])
        source_literal = float(row["repair_base_source_literal"])
        candidate = float(row["candidate_unrounded_binary64"])
        rounded = float(row["candidate_rounded_8_decimals"])
        difference = float(row["absolute_unrounded_to_source_difference"])
        literal_key = (band, power)
        previous_literal = coefficient_literals.setdefault(
            literal_key, row["repair_base_source_literal"]
        )
        expected_exact = str(rounded == source_literal).lower()
        if (
            row["target_model"] != "am_q95"
            or row["target_registry_family"]
            != "legacy_generic_unprefixed_am_q_registry"
            or row["source_profile_season"] != scale_row["source_profile_season"]
            or row["source_profile_percentile"]
            != scale_row["source_profile_percentile"]
            or row["source_profile_family"] != scale_row["source_profile_family"]
            or row["h2o_scale_decimal"] != scale_row["fitted_h2o_scale_decimal"]
            or row["evaluation_lane"] not in {affine_lane, direct_lane}
            or not math.isclose(
                float(row["frequency_ghz"]),
                expected_band_frequencies[band],
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or previous_literal != row["repair_base_source_literal"]
            or any(
                not math.isfinite(value)
                for value in (source_literal, candidate, rounded, difference)
            )
            or not math.isclose(
                rounded,
                round(candidate, 8),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            or not math.isclose(
                difference,
                abs(candidate - source_literal),
                rel_tol=2.0e-15,
                abs_tol=1.0e-15,
            )
            or row["exact_after_8_decimal_rounding"] != expected_exact
            or row["interpretation"]
            != ("weaker_q95_ratio_surface_provenance_hypothesis_raw_grid_absent")
        ):
            raise RuntimeError(
                "H2O coefficient-row invariant changed: "
                f"{row['evaluation_lane']}/{profile}/{band}/{power}"
            )
    if set(coefficient_literals) != {
        (band, power) for band in expected_band_frequencies for power in range(7)
    }:
        raise RuntimeError("H2O repair-base coefficient literals are incomplete")

    report_artifact = manifest["artifacts"]["H2O_SCALE_HYPOTHESIS_REPORT.md"]
    report_path = package_basename_path(
        "H2O_SCALE_HYPOTHESIS_REPORT.md",
        label="H2O-hypothesis report filename",
    )
    if sha256_path(report_path) != report_artifact["sha256"]:
        raise RuntimeError("H2O-hypothesis report digest mismatch")
    report = report_path.read_text(encoding="utf-8")
    required_report_statements = (
        "run directly for all 100 target/profile hypotheses",
        "Frozen P1 is fulfilled by the all-direct fitted-scale lane",
        "retained only as ancillary screening",
        "cannot establish generic-q custody",
        "do not establish 5--10% absolute flux accuracy",
        "binds every sidecar and scale trace to immutable execution-context",
        "whole-cache exclusive POSIX lock",
        "does not call the software execution clean or warning-free",
        "is noncanonical and excluded",
        "never used for v3 artifacts or rankings",
        "first context-bound v3 development cache",
        "cache remains untouched and is never reused",
        "canonical process retains only frozen lightweight run identity",
    )
    if any(statement not in report for statement in required_report_statements):
        raise RuntimeError("H2O-hypothesis report lost its all-direct/P1 limits")


def verify_followup_evidence() -> None:
    for name in FOLLOWUP_REQUIRED_FILES:
        if not (PACKAGE_DIR / name).is_file():
            raise RuntimeError(f"missing follow-up evidence file: {PACKAGE_DIR / name}")
    verify_copied_am_evidence()
    verify_frequency_resolution_evidence()
    verify_native_regeneration_evidence()
    verify_h2o_hypothesis_evidence()


def run_generated_checks(
    raw_source_dir: Path,
    *,
    include_external: bool,
    check_raw_source: bool,
    h2o_cache_dir: Path | None,
    native_cache_dir: Path | None,
) -> None:
    commands = [
        [
            sys.executable,
            str(PACKAGE_DIR / "generate_operator_analysis.py"),
            "--check",
        ],
    ]
    if check_raw_source:
        commands.append(
            [
                sys.executable,
                str(PACKAGE_DIR / "recover_legacy_raw_grids.py"),
                "--source-dir",
                str(raw_source_dir),
                "--check",
            ]
        )
    if include_external:
        commands.append(
            [
                sys.executable,
                str(PACKAGE_DIR / "analyze_copied_am_suite.py"),
                "--repo-root",
                str(REPO_ROOT),
                "--legacy-source-dir",
                str(raw_source_dir),
                "--tolteca-root",
                str(TOLTECA_REPOSITORY),
                "--check",
            ]
        )
        if h2o_cache_dir is not None:
            h2o_manifest = json.loads(
                (PACKAGE_DIR / "h2o_scale_hypothesis_manifest.json").read_text()
            )
            h2o_inputs = h2o_manifest["input_provenance"]
            h2o_execution = h2o_manifest["execution"]
            h2o_command = [
                sys.executable,
                str(PACKAGE_DIR / "probe_am12_h2o_scale_hypotheses.py"),
                "--am-executable",
                h2o_inputs["am_executable"]["supplied_path"],
                "--am-root",
                h2o_inputs["am_root"],
                "--legacy-source-dir",
                h2o_inputs["legacy_source_dir"],
                "--cache-dir",
                str(h2o_cache_dir.resolve()),
                "--jobs",
                str(h2o_execution["jobs"]),
                "--omp-threads",
                str(h2o_execution["omp_threads_per_process"]),
                "--check",
            ]
            h2o_build = h2o_inputs["builds"]["regeneration"]
            h2o_compiler = h2o_build["compiler"]
            if h2o_compiler["status"] == "supplied_by_operator_as_build_compiler":
                h2o_command.extend(
                    ["--compiler-executable", h2o_compiler["supplied_path"]]
                )
            if h2o_build["native_build_command"] is not None:
                h2o_command.extend(
                    ["--native-build-command", h2o_build["native_build_command"]]
                )
            commands.append(h2o_command)
        if native_cache_dir is not None:
            native_manifest = json.loads(
                (PACKAGE_DIR / "native_regeneration_manifest.json").read_text()
            )
            native_execution = native_manifest["execution"]
            native_build = native_manifest["builds"]["regeneration"]
            native_command = [
                sys.executable,
                str(PACKAGE_DIR / "run_am12_native_regeneration_check.py"),
                "--am-root",
                str(AM_ROOT),
                "--am-executable",
                native_build["supplied_path"],
                "--cache-dir",
                str(native_cache_dir.resolve()),
                "--jobs",
                str(native_execution["jobs"]),
                "--omp-threads",
                str(native_execution["omp_threads_per_process"]),
                "--check",
            ]
            compiler = native_build["compiler"]
            if compiler["status"] == "supplied_by_operator_as_build_compiler":
                native_command.extend(
                    ["--compiler-executable", compiler["supplied_path"]]
                )
            build_command = native_build["build_command_supplied_by_operator"]
            if build_command is not None:
                native_command.extend(["--native-build-command", build_command])
            commands.append(native_command)
        frequency_manifest = json.loads(
            (PACKAGE_DIR / "frequency_resolution_manifest.json").read_text()
        )
        first_run = frequency_manifest["runs"][0]
        first_amc_path = Path(first_run["argv"][1]).resolve()
        am_root = first_amc_path.parents[2]
        r1_cache_dir = (
            Path(first_run["environment_overrides"]["AM_CACHE_PATH"]).resolve().parent
        )
        commands.append(
            [
                sys.executable,
                str(PACKAGE_DIR / "run_am12_resolution_convergence.py"),
                "--am-executable",
                frequency_manifest["inputs"]["native_executable"]["path"],
                "--am-root",
                str(am_root),
                "--cache-dir",
                str(r1_cache_dir),
                "--omp-threads",
                str(frequency_manifest["execution"]["omp_threads"]),
                "--check",
            ]
        )
    commands.append(
        [
            sys.executable,
            str(PACKAGE_DIR / "generate_package_digests.py"),
            "--check",
        ]
    )
    for command in commands:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-external",
        action="store_true",
        help="skip machine-local coordination and provenance inputs",
    )
    parser.add_argument(
        "--raw-source-dir",
        type=Path,
        help=(
            "directory containing digest-identical q25/q50/q75 NPZ and "
            "supporting inputs; an explicitly supplied path remains checked "
            "with --skip-external"
        ),
    )
    parser.add_argument(
        "--h2o-cache-dir",
        type=Path,
        help=(
            "optional external P1 cache to validate in cache-only mode; "
            "never launches AM"
        ),
    )
    parser.add_argument(
        "--native-cache-dir",
        type=Path,
        help=(
            "optional external native-matrix cache to validate in cache-only "
            "mode; never launches AM"
        ),
    )
    args = parser.parse_args()
    if args.skip_external and (
        args.h2o_cache_dir is not None or args.native_cache_dir is not None
    ):
        parser.error("cache directories cannot be checked with --skip-external")
    raw_source_was_supplied = args.raw_source_dir is not None
    raw_source_dir = (
        (args.raw_source_dir if raw_source_was_supplied else RAW_SOURCE_DIR)
        .expanduser()
        .resolve()
    )
    check_raw_source = not args.skip_external or raw_source_was_supplied

    if not args.skip_external:
        verify_frozen_provenance_paths()
    if check_raw_source:
        verify_frozen_raw_sources(raw_source_dir)
    verify_json_schema()
    verify_manifest_artifact_files(include_external=not args.skip_external)
    verify_csvs()
    verify_followup_evidence()
    run_generated_checks(
        raw_source_dir,
        include_external=not args.skip_external,
        check_raw_source=check_raw_source,
        h2o_cache_dir=args.h2o_cache_dir,
        native_cache_dir=args.native_cache_dir,
    )
    print("SCI-CAL-001 atmosphere-operator package verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
