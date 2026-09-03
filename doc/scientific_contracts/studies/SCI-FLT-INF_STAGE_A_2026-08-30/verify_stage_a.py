#!/usr/bin/env python3
"""Verify the SCI-FLT-INF holding study and protected authority bytes."""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path


BASE = "cd55752e716051383da54356833ef0fac20b083a"
NOI = "f28d7a2617160febca85c1c40e6f7ba7494e266e"
ROOT = Path(__file__).resolve().parents[4]
STUDY = Path(__file__).resolve().parent
FLT = ROOT / "doc/scientific_contracts/packages/SCI-FLT/v0.1"

REQUIRED_STUDY_OBJECTS = (
    "README.md",
    "SCOPE_BRIEF.md",
    "PRIOR_WORK.md",
    "IMPLEMENTATION_INFORMED_DOSSIER.md",
    "FAMILY_SPLIT_MATRIX.md",
    "OPERATOR_STATE_PRODUCT_TAXONOMY.md",
    "CROSS_PACKAGE_AND_NOI_BOUNDARIES.md",
    "CONTRADICTIONS_AMBIGUITIES_UNAVAILABLE_STATES.md",
    "PROPOSED_SANITIZED_AUTHOR_INPUTS.md",
    "SCIENTIFIC_OWNER_DECISION_LEDGER.md",
    "SCIENTIFIC_OWNER_ODQ_001_APPROVAL_2026-08-30.md",
    "SCIENTIFIC_OWNER_ODQ_002_APPROVAL_2026-08-30.md",
    "SCIENTIFIC_OWNER_ODQ_003_APPROVAL_2026-08-31.md",
    "SCIENTIFIC_OWNER_ODQ_004_AUTHOR_DELEGATION_2026-08-31.md",
    "SCIENTIFIC_OWNER_ODQ_005_APPROVAL_2026-08-31.md",
    "SCIENTIFIC_OWNER_ODQ_006_APPROVAL_2026-08-31.md",
    "SCIENTIFIC_OWNER_ODQ_007_APPROVAL_2026-08-31.md",
    "SCIENTIFIC_OWNER_ODQ_008_APPROVAL_2026-08-31.md",
    "SCIENTIFIC_OWNER_ODQ_009_APPROVAL_2026-08-31.md",
    "SCIENTIFIC_OWNER_ODQ_010_APPROVAL_2026-08-31.md",
    "SCIENTIFIC_OWNER_ODQ_011_APPROVAL_2026-08-31.md",
    "SCIENTIFIC_OWNER_ODQ_012_APPROVAL_2026-08-31.md",
    "SCIENTIFIC_OWNER_ODQ_013_APPROVAL_2026-08-31.md",
    "SCIENTIFIC_OWNER_PACKAGE_IDENTITY_APPROVAL_2026-08-31.md",
    "FROZEN_AUTHORITY_AND_SOURCE_BINDING.md",
    "STAGE_A_SOURCE_MANIFEST.md",
    "STAGE_A_SOURCE_MANIFEST.sha256",
    "verify_stage_a.py",
)

PROTECTED = {
    "SCOPE_BRIEF.md": "b66dbca45edc758e1fc29f9f14313deb52473527acec8ed4d8ce93e725e32468",
    "AUTHOR_SUPERSESSION_COVER.md": "68bb884513754375eba67d19881b092564e2c49a0345ca1b0ee2cbc9f0d55ef0",
    "AUTHOR_CONVENTIONS_AND_OWNERSHIP.md": "3daa40ca90fc290a91452d0493e92366ca784372daeeea96b92d2ddd602dc30f",
    "SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md": "118baee3555d7fb498c7f9a479a74a98b619e4eb6147cbe9c252bbd0f54831a2",
    "AUTHOR_OPERATOR_AND_PRODUCT_TAXONOMY.md": "214dad66b93d021a0c67014756dcf746e3bba46c624142298542330b5be9a659",
    "AUTHOR_DETERMINISTIC_TRANSFORMATION_EXTRACT.md": "8ec6bc2bf1c64a136203cbdc9db7c0ad2531bf1098022b7b813403d47ae9279f",
    "SCI-MAP_TO_SCI-FLT-FIXED_BOUNDARY.md": "2c04689734359a6fa8139b502a691238a118002ae27d8cd58fe82c3d0dddfbca",
    "SCI-JINC_TO_SCI-FLT-FIXED_BOUNDARY.md": "8c9cffe3641311ece334827136eafd47752a13750df1dcf2f55107ecc115892f",
    "SCI-FLT-FIXED_TO_SCI-NOI_BOUNDARY.md": "a349064e5bd0711eec54cd4f63ab02f934a60c2b1b6d5eccae0b64c02b47acd8",
    "FIXED_LINEAR_OPERATOR_AND_CONVOLUTION_SPECIFICATION.md": "91f9dd40d6784ff8544f88062645e022435cecac0319332a9a89f61365398ca6",
    "WCS_KERNEL_DISCRETIZATION_DECISION_TABLE.md": "d81b022f7a7feae8d465de3c09904222247a2b40debf6bb484e9825585cc9275",
    "EDGE_MISSING_NONFINITE_METHOD_DECISION_TABLE.md": "fa910834db9a3e9ee068f49d779cdc4a46ae4e7f67ef30e5a21bdb2f497f5ed4",
    "NORMALIZATION_UNIT_BEAM_DECISION_TABLE.md": "7b0090c4782abfe8d86dd15c286015aa9d8d9fd4e75a02cde97d0d99c017e828",
    "RESPONSE_NULLSPACE_COVARIANCE_PRODUCT_TABLE.md": "468d04343dfd715e97df001ca59db996a75f68c644506724d7220c6b86b3b991",
    "OBSERVATION_COADD_NONCOMMUTATION_TABLE.md": "4483fd81d690caf90f16953cd21fb1199cc68c562f42e354e95876e1521b816a",
    "SCI_FLT_VAL_PROFILE_DRAFTS.md": "9546082e4defcc3d83ce65969b44e35926fed7fe0185d306b9679701c0ce7976",
    "FIXED_PRODUCT_ROLE_AND_LIFECYCLE_TABLE.md": "0c10c8e7cf46b80c94bc2454090ab918f7f440bcd0d4bc8b7e94cf53df66efed",
    "AUTHOR_PACKET_MANIFEST.md": "7f2d03f182258ac9770f7dba869e9ae0b5018efdcdb93b18b299a9b9c6df1e4d",
}

NOI_OBJECTS = {
    "doc/scientific_contracts/packages/SCI-NOI/v0.1/FILTER_AND_FRUIT_SCOPE.md": (
        "71013fd53c2399aaa3cce33ead20a39b51a360f8",
        "08eba55f840e8f8aa265e1d2f1a981e16351a1c2460e74907cb4beb5ccb7df77",
    ),
    "doc/scientific_contracts/packages/SCI-NOI/v0.1/NOI_GEN_PARENT_OPERATOR_GRAPH.md": (
        "56d12f2ae736c951da6e6a679774a71f518ec82a",
        "d7cacc667f479965ab7d1a7c3acb453f0195224bf0df1051b319d424ac04e5ac",
    ),
    "doc/scientific_contracts/packages/SCI-NOI/v0.1/PRODUCT_ROLE_AND_LIFECYCLE_TABLE.md": (
        "cdb3562032a20b013ef047853c4e546290f447a1",
        "3d138d769c9629f93b0a493955d5a76f537d2d832c2b58de33fae595653e6212",
    ),
    "doc/scientific_contracts/packages/SCI-NOI/v0.1/SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md": (
        "5c3f34407468430a8d299311a9fd20d65d195e6b",
        "272ac939b8a7109a123073b1a39fcdd7ac4129c603683ee81257b94ab2f55a0b",
    ),
    "doc/scientific_contracts/packages/SCI-NOI/v0.1/stage_b/r0.3/SCIENTIFIC_OWNER_DECISION_SUPPLEMENT.md": (
        "e1fe3cd7a31a9680e69bbe130c93d82c09c5867a",
        "4b622de96e72860e544bf3699e3a131bc9cca5d2be0b0fe5bb41096327e977e5",
    ),
    "doc/scientific_contracts/packages/SCI-NOI/v0.1/stage_b/r0.4/SCIENTIFIC_OWNER_DIRECTIVE.md": (
        "a8452a107982b2cfbf2db2eb3496c2cf19de9268",
        "f155c1e70c1a8431bf4baf68f111960456ef24e9e77544837ff2efbecb86f423",
    ),
    "doc/scientific_contracts/packages/SCI-NOI/v0.1/stage_b/r0.5/SCIENTIFIC_OWNER_DIRECTIVE.md": (
        "bcff09e9abd63d9922cc785796c648a647b68051",
        "4f04e25a32b44b729bd291d4919c9acb4bc197f79d94f7f213f4d9505539114d",
    ),
    "doc/scientific_contracts/packages/SCI-NOI/v0.1/stage_b/r0.5/FINAL_AUTHORITY_SOURCE_PACKET.md": (
        "8187adee007805590e8bb1ffbbe70432fe137f1b",
        "bd1e0057723cccfcb96ef545437efd8fbec975b4fe6c4f54306f3b2a08f99ec9",
    ),
    "doc/scientific_contracts/packages/SCI-NOI/v0.1/stage_b/r0.5/NORMATIVE_MODULE_BINDING.json": (
        "671ca57bdac633b641519d0ef3e795f53b1a5d34",
        "aa59ecaaaa149e2990d07623563d90af76c7b3084ee37c497a06e17ebf0fe213",
    ),
    "doc/scientific_contracts/packages/SCI-NOI/v0.1/stage_b/r0.5/PROPOSED_FREEZE_MANIFEST.json": (
        "cd3a7f49f006498ca402f833ee39572f256af3c9",
        "b6915186424dd52d7c94fb0df47db91654d3c20cf4b3fa6ab98c3554626d8bfc",
    ),
    "doc/scientific_contracts/packages/SCI-NOI/v0.1/stage_b/r0.5/SOURCE_BYTE_AND_BYTE_EQUALITY_REPORT.md": (
        "0a6006dd7adf7f634b3d80586510b79fd02bd488",
        "ba5054f92891e9d9912cb0b3bb83bde7ca3c11734b8a9f854e83db848438abb8",
    ),
    "doc/scientific_contracts/packages/SCI-NOI/v0.1/stage_b/r0.5/POST_SNAPSHOT_SCIENTIFIC_OWNER_FREEZE_APPROVAL.md": (
        "169f560f4a58aac717b0c9d0c58a238699fd6500",
        "dba66966ed7082ea55b756c23f4cc9de6205022f1362ff0a69da63bc85190d2c",
    ),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=check, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)


def main() -> int:
    failures: list[str] = []

    for name in REQUIRED_STUDY_OBJECTS:
        if not (STUDY / name).is_file():
            fail(f"missing study object: {name}", failures)

    for name, expected in PROTECTED.items():
        path = FLT / name
        if not path.is_file():
            fail(f"missing protected object: {name}", failures)
            continue
        actual = sha256(path.read_bytes())
        if actual != expected:
            fail(f"protected hash mismatch: {name}: {actual}", failures)

    protected_paths = [
        str((FLT / name).relative_to(ROOT)) for name in PROTECTED
    ]
    protected_paths.append(
        str((FLT / "AUTHOR_PACKET_MANIFEST.sha256").relative_to(ROOT))
    )
    if git("diff", "--quiet", BASE, "--", *protected_paths, check=False).returncode:
        fail("protected SCI-FLT-FIXED bytes differ from the base commit", failures)

    manifest_pointer = (FLT / "AUTHOR_PACKET_MANIFEST.sha256").read_text()
    expected_pointer = (
        "7f2d03f182258ac9770f7dba869e9ae0b5018efdcdb93b18b299a9b9c6df1e4d  "
        "AUTHOR_PACKET_MANIFEST.md\n"
    )
    if manifest_pointer != expected_pointer:
        fail("protected author-manifest pointer changed", failures)

    for path, (expected_oid, expected_sha) in NOI_OBJECTS.items():
        spec = f"{NOI}:{path}"
        oid = git("rev-parse", spec).stdout.decode().strip()
        if oid != expected_oid:
            fail(f"frozen NOI OID mismatch: {path}: {oid}", failures)
        data = git("show", spec).stdout
        actual_sha = sha256(data)
        if actual_sha != expected_sha:
            fail(f"frozen NOI SHA-256 mismatch: {path}: {actual_sha}", failures)

    if (STUDY / "STAGE_A_SOURCE_MANIFEST.md").is_file():
        text = (STUDY / "STAGE_A_SOURCE_MANIFEST.md").read_text()
        rows = dict(
            re.findall(
                r"^\| `([^`]+)` \| `([0-9a-f]{64})` \|$", text,
                flags=re.MULTILINE,
            )
        )
        expected_manifest_names = {
            name for name in REQUIRED_STUDY_OBJECTS
            if name not in {"STAGE_A_SOURCE_MANIFEST.md", "STAGE_A_SOURCE_MANIFEST.sha256"}
        }
        if set(rows) != expected_manifest_names:
            fail("Stage A source-manifest object set is not exact", failures)
        for name in expected_manifest_names:
            path = STUDY / name
            if path.is_file() and rows.get(name) != sha256(path.read_bytes()):
                fail(f"Stage A source-manifest hash mismatch: {name}", failures)

        pointer = (STUDY / "STAGE_A_SOURCE_MANIFEST.sha256")
        if pointer.is_file():
            expected = (
                f"{sha256((STUDY / 'STAGE_A_SOURCE_MANIFEST.md').read_bytes())}  "
                "STAGE_A_SOURCE_MANIFEST.md\n"
            )
            if pointer.read_text() != expected:
                fail("Stage A source-manifest pointer mismatch", failures)

    required_phrases = {
        "README.md": ("not an approved package", "next gate"),
        "SCOPE_BRIEF.md": ("Program adherence and prior-work recovery", "Exclusions"),
        "PRIOR_WORK.md": ("Genuinely new scientific work remaining", "Unavailable"),
        "FAMILY_SPLIT_MATRIX.md": ("INF-A", "INF-J"),
        "SCIENTIFIC_OWNER_DECISION_LEDGER.md": (
            "SCI-FLT-INF-ODQ-013", "SCI-FLT-MATCHED",
            "No holding-study owner question remains",
        ),
        "SCIENTIFIC_OWNER_ODQ_001_APPROVAL_2026-08-30.md": (
            "optimal matched-template amplitude estimator",
            "not a posterior or Wiener reconstruction",
        ),
        "SCIENTIFIC_OWNER_ODQ_002_APPROVAL_2026-08-30.md": (
            "matched-filtered map",
            "source-estimation package or SRC ownership",
        ),
        "SCIENTIFIC_OWNER_ODQ_003_APPROVAL_2026-08-31.md": (
            "ordinary-MAP **observation bundle**",
            "No equivalence, commutation, or cross-observation combination",
        ),
        "SCIENTIFIC_OWNER_ODQ_004_AUTHOR_DELEGATION_2026-08-31.md": (
            "radially symmetrized average map noise PSD",
            "no noise/covariance option selected",
        ),
        "SCIENTIFIC_OWNER_ODQ_005_APPROVAL_2026-08-31.md": (
            "template-response product",
            "historical high-pass/delta case is not admitted",
        ),
        "SCIENTIFIC_OWNER_ODQ_006_APPROVAL_2026-08-31.md": (
            "authoritative reference estimator",
            "never establishes the scientific amplitude",
        ),
        "SCIENTIFIC_OWNER_ODQ_007_APPROVAL_2026-08-31.md": (
            "complete-support",
            "Adaptive edge/background conditioning is not part of base v0.1",
        ),
        "SCIENTIFIC_OWNER_ODQ_008_APPROVAL_2026-08-31.md": (
            "template-amplitude unit",
            "not a universal response",
        ),
        "SCIENTIFIC_OWNER_ODQ_009_APPROVAL_2026-08-31.md": (
            "C_cond = L C_parent L^T",
            "marginal conditional variance",
        ),
        "SCIENTIFIC_OWNER_ODQ_010_APPROVAL_2026-08-31.md": (
            "one exact immutable realized application state",
            "Fixed-state and relearned members cannot mix",
        ),
        "SCIENTIFIC_OWNER_ODQ_011_APPROVAL_2026-08-31.md": (
            "no automatic method selector and no fallback",
            "Data-thresholded spectral selection",
        ),
        "SCIENTIFIC_OWNER_ODQ_012_APPROVAL_2026-08-31.md": (
            "first-class FLT→FRUIT",
            "FRUIT receives its own future",
        ),
        "SCIENTIFIC_OWNER_ODQ_013_APPROVAL_2026-08-31.md": (
            "role-complete atomic signal bundle",
            "Atomicity is role-scoped",
        ),
        "SCIENTIFIC_OWNER_PACKAGE_IDENTITY_APPROVAL_2026-08-31.md": (
            "SCI-FLT-MATCHED",
            "Optimal matched-template map filtering",
        ),
        "CROSS_PACKAGE_AND_NOI_BOUNDARIES.md": (
            "fixed-state and relearned members cannot be mixed", "FRUIT boundary",
        ),
    }
    for name, phrases in required_phrases.items():
        text = (STUDY / name).read_text() if (STUDY / name).is_file() else ""
        for phrase in phrases:
            if phrase not in text:
                fail(f"missing required phrase in {name}: {phrase}", failures)

    if failures:
        for item in failures:
            print(f"FAIL: {item}")
        return 1

    print("SCI-FLT-INF Stage A verification passed")
    print(f"study objects: {len(REQUIRED_STUDY_OBJECTS)}")
    print(f"protected SCI-FLT-FIXED paths: {len(PROTECTED) + 1}")
    print(f"frozen SCI-NOI objects: {len(NOI_OBJECTS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
