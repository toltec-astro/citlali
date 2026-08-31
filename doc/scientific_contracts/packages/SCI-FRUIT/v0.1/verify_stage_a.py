#!/usr/bin/env python3
"""Verify the recovery-first SCI-FRUIT Stage A owner-review packet."""

from __future__ import annotations

import hashlib
import pathlib
import re
import subprocess


ROOT = pathlib.Path(__file__).resolve().parent
REPO = ROOT.parents[4]
LAUNCH = "7f9307ff4e1cda0f112f2398bb72f52a3f4f01d5"
LAUNCH_TREE = "03b77c9187eb5421488641d2ea1fe4dcb572a9a9"
PROVISIONAL = "faff97565ee27e375e1337febe5a0a6681507c3b"
PROVISIONAL_TREE = "0dfa3cdfa8a261bd00878cafd593aafb87394163"
AUDIT = "8c581bfb26f01b187f4f1e0565f4457bcc25f099"

SOURCE_HASHES = {
    (LAUNCH, "doc/scientific_contracts/README.md"):
        "351e9b7775b0bf78cba01bf4cd2fafd9591c4b43931b0dc23d82d97f0dfe82d2",
    (LAUNCH, "doc/scientific_contracts/DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md"):
        "0c0a7551689523ac16c72569834a687fd598a647d3f3c7dca3cd81cf5609a691",
    (LAUNCH, "doc/scientific_contracts/PRIOR_WORK_REGISTRY.md"):
        "8cd9868879a3fe7d5caaa2e6468d886d4b769242e65cc19968c6e6bb28cf8897",
    (LAUNCH, "doc/SCIENTIFIC_CONVENTIONS.md"):
        "24c8397b130de0fb1c0dcfcd87c057c06e4f095ee6a54472759a6ef276bb5add",
    (LAUNCH, "doc/adr/0006-fruit-loop-restart-checkpoint.md"):
        "c6abf5d5c0f9edd1e68cf080ed6d76d2cac93dc59e3bc398162ce92d6bb8ca2b",
    (LAUNCH, "doc/scientific_contracts/packages/SCI-PTC/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.5.md"):
        "8357961a49272adc40e27a8aa9e760e0d01ff2419ae2c88a62c0f93c9f959e66",
    (LAUNCH, "doc/scientific_contracts/packages/SCI-MAP/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.7.1.md"):
        "91801005ba2f2bce6471a9f6f4ed0b79806c893f498b4f3cca9e81e26df39ce1",
    (LAUNCH, "doc/scientific_contracts/packages/SCI-JINC/v0.1/FREEZE_AUTHORITY_MANIFEST_R0.3.md"):
        "ff4b79e7cca3950831eda95a16ec6a535597f543c4676378d2fc2f01d50faed2",
    (LAUNCH, "doc/scientific_contracts/packages/SCI-NOI/v0.1/SCIENTIFIC_OWNER_STAGE_A_FINAL_APPROVAL_2026-08-30.md"):
        "49377d1596c9e47a6e2328e890ebcd6b25f42af3781b533b5bd8c2cded08fa6b",
    (LAUNCH, "doc/scientific_contracts/packages/SCI-NOI/v0.1/FILTER_AND_FRUIT_SCOPE.md"):
        "08eba55f840e8f8aa265e1d2f1a981e16351a1c2460e74907cb4beb5ccb7df77",
    (LAUNCH, "doc/scientific_contracts/packages/SCI-FLT/v0.1/stage_b/SCIENTIFIC_OWNER_FREEZE_RECORD.md"):
        "ad00d2895982c2d26000fffb33a2dc73c716ea425f03a2a96b684563b2dfaf39",
    (LAUNCH, "doc/scientific_contracts/packages/SCI-FLT/v0.1/stage_b/SCIENTIFIC_OWNER_FREEZE_BINDING.json"):
        "17298df3328bf812de8f896a1f931c41754bdee73b8dfddf40d48b03652e89cd",
    (LAUNCH, "doc/FRUIT_LOOP_FEEDBACK_INVESTIGATION_2026-07-24.md"):
        "8ec6b4949e259eb1c2a07f45fae175955ffdc356297cdd616a2689feae150326",
    (LAUNCH, "doc/FRUIT_LOOP_CONVERGENCE_STUDY_2026-07-23.md"):
        "ab350193c78047bda5fe49cc8ead02b7e8d05ca1d73ec1cd9b4a40fd40d81178",
    (LAUNCH, "doc/FRUIT_LOOP_CONVERGENCE_CRITERIA_DISCUSSION_2026-07-27.md"):
        "e1d1a20988d4c5a8340b6b4c0519b2d353a3f79fcf363e885ddcd7617db86a55",
    (LAUNCH, "doc/FRUIT_LOOP_CALIBRATION_REFERENCE_INVESTIGATION_2026-07-26.md"):
        "26fe065c6f4b0fc32cff05349cbf00f8cc8656705d6cb52a55dcc5296f0501f8",
    (LAUNCH, "doc/FRUIT_LOOP_POPULATION_EXTENSION_PLAN_2026-07-26.md"):
        "cd726b6141a479db6fe6ae9c5656f4249fcd17186627ae224da183cf33584c7d",
    (PROVISIONAL, "doc/scientific_contracts/studies/SCI-FLT-INF_STAGE_A_2026-08-30/README.md"):
        "d70873561f5e6c408fd64dc0ddc6e92827e29a5b023b324efbf64c7b9a7dcd34",
    (PROVISIONAL, "doc/scientific_contracts/studies/SCI-FLT-INF_STAGE_A_2026-08-30/CROSS_PACKAGE_AND_NOI_BOUNDARIES.md"):
        "af862ffb29f690a94945fa6122e6858492a99aca5c8caae66e9963f5740a6929",
    (PROVISIONAL, "doc/scientific_contracts/studies/SCI-FLT-INF_STAGE_A_2026-08-30/SCIENTIFIC_OWNER_DECISION_LEDGER.md"):
        "bbcae3582eb7db058d1681a6be85e895aff251a1f544d49cfacab1f33e70dc16",
    (PROVISIONAL, "doc/scientific_contracts/studies/SCI-FLT-INF_STAGE_A_2026-08-30/STAGE_A_SOURCE_MANIFEST.md"):
        "b12e01a7dbb25ed4351d8bdc902d742be8ecc42b71e93afa060a8086109161e1",
    (AUDIT, "doc/audits/audit-ledger.yaml"):
        "91636d50ebea9f4502ed2dbccde22e981850bef9a79b3cf7301d8b90c616c906",
    (AUDIT, "doc/audits/handoffs/SCI-FRUIT-001/SCI-FRUIT-001-XAUD-001.yaml"):
        "8b0919fcfda18e338dbf3f1a8538d86dd29660e9fb0d605e0620373d93b1dd18",
    (AUDIT, "doc/audits/handoffs/SCI-MAP-003/SCI-MAP-003-XAUD-007.yaml"):
        "801928c460745b748163027191d20f5873110db937aa508e95542206e5623498",
}

REQUIRED_FILES = {
    "README.md",
    "PRIOR_WORK.md",
    "INTERNAL_DOSSIER.md",
    "SCOPE_BRIEF.md",
    "OWNERSHIP_AND_BOUNDARY_CLASSIFICATION.md",
    "CANDIDATE_PARENT_ADMISSION_MATRIX.md",
    "ITERATIVE_DAG_AND_STATE_OWNERSHIP.md",
    "RESTART_CHECKPOINT_AND_LIFECYCLE_TAXONOMY.md",
    "RESPONSE_UNCERTAINTY_CONVERGENCE_CLAIM_MATRIX.md",
    "CONTRADICTIONS_AMBIGUITIES_UNAVAILABLE_STATES.md",
    "OPERATOR_AND_PRODUCT_TAXONOMY.md",
    "PROPOSED_SANITIZED_AUTHOR_INPUTS.md",
    "SCIENTIFIC_OWNER_DECISION_LEDGER.md",
    "SOURCE_IDENTITY_MANIFEST.md",
    "DECISION_LOG.md",
    "CROSSWALK.md",
    "AUTHOR_PACKET_MANIFEST.md",
    "AUTHOR_SUPERSESSION_COVER.md",
    "AUTHOR_CONVENTIONS_AND_OWNERSHIP.md",
    "pdf/README.md",
    "src/scientific-rationale.tex",
    "src/engineering-conformance.tex",
    "src/common/notation.tex",
    "src/common/definitions.tex",
    "src/common/equations.tex",
    "src/common/assumptions.tex",
    "src/common/requirements.tex",
    "src/common/edge_cases.tex",
}

ALLOWED_GLOBAL_EDITS = {
    "doc/REFACTOR_STATUS.md",
    "doc/scientific_contracts/DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md",
    "doc/scientific_contracts/INDEX.md",
    "doc/scientific_contracts/PRIOR_WORK_REGISTRY.md",
    "doc/scientific_contracts/verify_layout.py",
}


def run(*args: str) -> str:
    return subprocess.check_output(args, cwd=REPO, text=True).strip()


def git_bytes(ref: str, path: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{ref}:{path}"], cwd=REPO
    )


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def verify_sources() -> None:
    if run("git", "rev-parse", f"{LAUNCH}^{{tree}}") != LAUNCH_TREE:
        fail("launch commit tree does not match recorded identity")
    if run("git", "rev-parse", f"{PROVISIONAL}^{{tree}}") != PROVISIONAL_TREE:
        fail("provisional FLT-MATCHED tree does not match recorded identity")
    if subprocess.call(
        ["git", "merge-base", "--is-ancestor", LAUNCH, "HEAD"], cwd=REPO
    ):
        fail("current branch is not descended from the exact launch commit")
    for (ref, path), expected in SOURCE_HASHES.items():
        actual = digest(git_bytes(ref, path))
        if actual != expected:
            fail(f"source identity changed: {ref}:{path}: {actual}")


def changed_paths() -> set[str]:
    paths = set(filter(None, run("git", "diff", "--name-only", LAUNCH).splitlines()))
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=REPO, text=True
    )
    for line in status.splitlines():
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.add(path)
    return paths


def verify_scope_of_edits() -> None:
    package_prefix = "doc/scientific_contracts/packages/SCI-FRUIT/v0.1/"
    untracked_package_directory = "doc/scientific_contracts/packages/SCI-FRUIT/"
    unexpected = sorted(
        path for path in changed_paths()
        if not path.startswith(package_prefix)
        and path != untracked_package_directory
        and path not in ALLOWED_GLOBAL_EDITS
    )
    if unexpected:
        fail("unexpected or protected paths changed: " + ", ".join(unexpected))


def verify_packet() -> None:
    for rel in REQUIRED_FILES:
        if not (ROOT / rel).is_file():
            fail(f"missing Stage A object: {rel}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    scope = (ROOT / "SCOPE_BRIEF.md").read_text(encoding="utf-8")
    matrix = (ROOT / "CANDIDATE_PARENT_ADMISSION_MATRIX.md").read_text(
        encoding="utf-8"
    )
    ledger = (ROOT / "SCIENTIFIC_OWNER_DECISION_LEDGER.md").read_text(
        encoding="utf-8"
    )
    manifest = (ROOT / "SOURCE_IDENTITY_MANIFEST.md").read_text(encoding="utf-8")
    author = (ROOT / "AUTHOR_PACKET_MANIFEST.md").read_text(encoding="utf-8")

    required_tokens = {
        readme: [
            "recovery-first Stage A owner-review candidate; no Stage B launch",
            LAUNCH,
            PROVISIONAL,
            "every numerical route is",
        ],
        scope: [
            "not owner-approved",
            "Approved source identifier: **none**",
            "Confirm that this opening was reviewed before launching scientific authorship:\n**no**",
        ],
        matrix: [
            "**Ordinary MAP**",
            "**JINC**",
            "**FLT-FIXED**",
            "**FLT-MATCHED (provisional)**",
            "UNAVAILABLE_PARENT_ROUTE",
            "PROVISIONAL_AND_UNAVAILABLE",
        ],
        ledger: [
            "SCI-FRUIT-ODQ-001",
            "open — first review question",
            "SCI-FRUIT-ODQ-012",
            "blocked on prior decisions",
        ],
        manifest: [LAUNCH_TREE, PROVISIONAL_TREE, "not an author-packet"],
        author: ["not approved", "not dispatchable", "No implementation-blind author packet exists"],
    }
    for body, tokens in required_tokens.items():
        for token in tokens:
            if token not in body:
                fail(f"required Stage A token absent: {token}")

    for name in (
        "notation.tex",
        "definitions.tex",
        "equations.tex",
        "assumptions.tex",
        "requirements.tex",
        "edge_cases.tex",
    ):
        text = (ROOT / "src" / "common" / name).read_text(encoding="utf-8")
        if "Reserved for implementation-blind Stage B" not in text:
            fail(f"non-placeholder Stage B common source: {name}")
    for name in ("scientific-rationale.tex", "engineering-conformance.tex"):
        text = (ROOT / "src" / name).read_text(encoding="utf-8")
        if "contains no normative science" not in text:
            fail(f"non-placeholder Stage B view: {name}")


def verify_links() -> None:
    for path in ROOT.rglob("*.md"):
        body = path.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^\]]+\]\(([^)#]+)(?:#[^)]+)?\)", body):
            if "://" in target or target.startswith("mailto:"):
                continue
            if not (path.parent / target).resolve().exists():
                fail(f"broken local link in {path.relative_to(ROOT)}: {target}")


def main() -> None:
    verify_sources()
    verify_scope_of_edits()
    verify_packet()
    verify_links()
    subprocess.check_call(["git", "diff", "--check"], cwd=REPO)
    print(
        "PASS: exact launch/provisional/historical sources, bounded edits, "
        "four unavailable candidate routes, Stage A firewall, placeholders, "
        "and local links"
    )


if __name__ == "__main__":
    main()
