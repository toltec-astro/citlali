#!/usr/bin/env python3
"""Verify schemas, frozen provenance, generated tables, and package digests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import jsonschema


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
RAW_SOURCE_DIR = Path("/Users/gwilson/GitHub/toltec_beammap/src/toltec_sensitivity")
TOLTECA_REPOSITORY = Path("/Users/gwilson/GitHub/tolteca")
TOLTECA_REVISION = "2791e6a1e6349ad1d3ac549a648f41cbc51b98c7"

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
    RAW_SOURCE_DIR
    / "amLMT25.npz": "6ddffcd2c68bbc0f6d8f6470eba0d1aa81457dcc2f348fd2d7e44c9dfe48c87b",
    RAW_SOURCE_DIR
    / "amLMT50.npz": "1fe6dd2ab7a4d65f445e20c5a8f438eb42884836e7932d86f80c30e235710f81",
    RAW_SOURCE_DIR
    / "amLMT75.npz": "adbb8eb974c4e2744c3efb0f627708565f954c4029d9345e4f434699e8843f8e",
    RAW_SOURCE_DIR
    / "LMTAtmosphere.py": "66f580b85ccbfff9152519ec644df363e4571b9263fe06849dc89aa1858e52d0",
    RAW_SOURCE_DIR
    / "Detector.py": "82105317865ae1182d88d0874ed96c36a2b8c79c56d7fc6bb1990f008bd81d1a",
    RAW_SOURCE_DIR
    / "model_passbands.npz": "861e6ce7af55b18c14a800defaf0b9a11099a16c307da08e391e1d8f79a39765",
    Path(
        "/Users/gwilson/work_toltec/local_data/doc/mmccrackan_dissertation.pdf"
    ): "2aa4373aaa0394f1a79e6668047a7aecd07d4914ce162c931f495d5502a49be0",
}

CSV_ARTIFACTS = (
    "candidate_disagreement_metrics.csv",
    "candidate_surface_metrics.csv",
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


def verify_frozen_paths() -> None:
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


def verify_csvs() -> None:
    for name in CSV_ARTIFACTS:
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


def run_generated_checks(raw_source_dir: Path) -> None:
    commands = [
        [
            sys.executable,
            str(PACKAGE_DIR / "generate_operator_analysis.py"),
            "--check",
        ],
        [
            sys.executable,
            str(PACKAGE_DIR / "recover_legacy_raw_grids.py"),
            "--source-dir",
            str(raw_source_dir),
            "--check",
        ],
        [
            sys.executable,
            str(PACKAGE_DIR / "generate_package_digests.py"),
            "--check",
        ],
    ]
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
        default=RAW_SOURCE_DIR,
        help="directory containing digest-identical q25/q50/q75 NPZ inputs",
    )
    args = parser.parse_args()

    if not args.skip_external:
        verify_frozen_paths()
    verify_json_schema()
    verify_csvs()
    run_generated_checks(args.raw_source_dir.resolve())
    print("SCI-CAL-001 atmosphere-operator package verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
