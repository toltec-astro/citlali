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
HISTORICAL = "f70701ad488444f3e2528c6bbe3e798863c9e301"
HISTORICAL_TREE = "2009a1397bd67d615a1d6e9a8419e18fc794a81e"

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
    (HISTORICAL, "include/citlali/core/pipeline/reduction_iteration_loop.h"):
        "553a33ca0ba0849cb7cb8ce99b7542a0951b76d4ff27ea2a7c78d2c11266d9ef",
    (HISTORICAL, "include/citlali/core/pipeline/reduction_observation.h"):
        "328d6c3d5fe2e2a58654869d727e8079b09ebb966d507c245e5ef1bfa0ed8cb0",
    (HISTORICAL, "include/citlali/core/pipeline/reduction_observation_pipeline.h"):
        "191d452abdcbee35614623bae996fe62d37cfb7f88d62b61d7bbeaac91671e0a",
    (HISTORICAL, "include/citlali/core/engine/detail/lali_run_impl.h"):
        "e76fa90cc5c8a4dae84fdd7a258ee834c62d389e65b323b3010d2f92b7f56341",
    (HISTORICAL, "include/citlali/core/engine/detail/lali_fruitloop_impl.h"):
        "fc480b6e64b6108954dfc7f9e6fe129d25db2fbf4d2209b2872492460da32fd8",
    (HISTORICAL, "include/citlali/core/engine/detail/pointing_run_impl.h"):
        "0200e17b209023d51e71e34506752d33f51082e4849bee1ab33732a301c85cab",
    (HISTORICAL, "include/citlali/core/engine/detail/pointing_fruitloop_impl.h"):
        "eb6ca404a75fa56e04a7e515d1c81c48240fb0050891218bdd165faa8c520945",
    (HISTORICAL, "include/citlali/core/pipeline/previous_fruit_loop_map_loading.h"):
        "91f7cd5f0b311693569423bbd6fd28682790f63f675ab4fb364b8d8c398851e4",
    (HISTORICAL, "include/citlali/core/pipeline/fruit_loop_paths.h"):
        "45b29b588b51c002c06539d52ee1cad3a2448e47cb9d12ac95957fe4010dde2f",
    (HISTORICAL, "include/citlali/core/pipeline/fruit_loop_map_io.h"):
        "3b83959cfbb7771e707a5fc9a3a6b166bd57a5de9fbc58a6b572c7903b41eb23",
    (HISTORICAL, "include/citlali/core/pipeline/filtered_observation_outputs.h"):
        "e80906d60e982cc6523d675b8ecda34b7b8372246dfb4b6f38593519c980135e",
    (HISTORICAL, "include/citlali/core/pipeline/iteration_coadd_outputs.h"):
        "bd91b1c7271523a492b72b78fb3753f9dd265f26f48a7294129ebfd986b705ec",
    (HISTORICAL, "include/citlali/core/engine/learning.h"):
        "7ae32ad060858987bd299f04f7315be215f9a55ad2a3cafe700fcd00c755be4c",
    (HISTORICAL, "include/citlali/core/pipeline/fruit_loop_restart_lifecycle.h"):
        "dbc2c46f7992d0b56d96de1d49278ae7c0bb36cfbb47b8184dc38f36227930a1",
    (HISTORICAL, "src/citlali/core/pipeline/reduction_restart_checkpoint.cpp"):
        "b1d5b4ba0115ea66e43ea1742967917980b6291dd2e998d02e93c8d26008ea61",
    (HISTORICAL, "tests/test_fruit_loop_recurrence.cpp"):
        "1dfaa0677f291187dd88cafd3e26d8a2ccd49f958c1fda1595d3657cb5ba0138",
    (HISTORICAL, "tests/test_learning_and_fruit_contracts.cpp"):
        "b7eba3b2fa443b5067a4c2e2532a13958c65d60dc62275f06971672c11912863",
}

HISTORICAL_LAUNCH_EQUAL_PATHS = {
    path for ref, path in SOURCE_HASHES if ref == HISTORICAL
} - {
    "include/citlali/core/pipeline/filtered_observation_outputs.h",
    "include/citlali/core/pipeline/iteration_coadd_outputs.h",
}

REQUIRED_FILES = {
    "README.md",
    "PRIOR_WORK.md",
    "INTERNAL_DOSSIER.md",
    "SCOPE_BRIEF.md",
    "OWNERSHIP_AND_BOUNDARY_CLASSIFICATION.md",
    "CANDIDATE_PARENT_ADMISSION_MATRIX.md",
    "HISTORICAL_RECURRENCE_BASELINE.md",
    "ADDITIVE_REFORMULATION_EQUIVALENCE_ANALYSIS.md",
    "ODQ_001_RECURRENCE_DECISION_FRAME.md",
    "ITERATIVE_DAG_AND_STATE_OWNERSHIP.md",
    "RESTART_CHECKPOINT_AND_LIFECYCLE_TAXONOMY.md",
    "RESPONSE_UNCERTAINTY_CONVERGENCE_CLAIM_MATRIX.md",
    "CONTRADICTIONS_AMBIGUITIES_UNAVAILABLE_STATES.md",
    "OPERATOR_AND_PRODUCT_TAXONOMY.md",
    "PROPOSED_SANITIZED_AUTHOR_INPUTS.md",
    "SCIENTIFIC_OWNER_DECISION_LEDGER.md",
    "SOURCE_IDENTITY_MANIFEST.md",
    "DECISION_LOG.md",
    "SCIENTIFIC_OWNER_RECURRENCE_REVIEW_DIRECTION_2026-08-31.md",
    "SCIENTIFIC_OWNER_PROVISIONAL_CHOICE_3_DIRECTION_2026-08-31.md",
    "SCIENTIFIC_OWNER_COMPARATIVE_QUALITY_DIRECTION_2026-08-31.md",
    "COMPARATIVE_QUALITY_OBJECTIVE_GATE.md",
    "SCIENTIFIC_OWNER_ODQ_001E_FRAMEWORK_APPROVAL_2026-08-31.md",
    "SCIENTIFIC_OWNER_EMPIRICAL_METHOD_DIRECTION_2026-08-31.md",
    "ODQ_001F_PROFILE_QUALIFIED_EMPIRICAL_DEVELOPMENT_FRAME.md",
    "SCIENTIFIC_OWNER_ODQ_001F_BASELINE_RELATIVE_REPAIR_DIRECTION_2026-09-01.md",
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
    if run("git", "rev-parse", f"{HISTORICAL}^{{tree}}") != HISTORICAL_TREE:
        fail("historical recurrence tree does not match recorded identity")
    if subprocess.call(
        ["git", "merge-base", "--is-ancestor", LAUNCH, "HEAD"], cwd=REPO
    ):
        fail("current branch is not descended from the exact launch commit")
    for (ref, path), expected in SOURCE_HASHES.items():
        actual = digest(git_bytes(ref, path))
        if actual != expected:
            fail(f"source identity changed: {ref}:{path}: {actual}")
    for path in HISTORICAL_LAUNCH_EQUAL_PATHS:
        if git_bytes(HISTORICAL, path) != git_bytes(LAUNCH, path):
            fail(f"historical recurrence path differs at launch: {path}")


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
    baseline = (ROOT / "HISTORICAL_RECURRENCE_BASELINE.md").read_text(
        encoding="utf-8"
    )
    equivalence = (ROOT / "ADDITIVE_REFORMULATION_EQUIVALENCE_ANALYSIS.md").read_text(
        encoding="utf-8"
    )
    odq001 = (ROOT / "ODQ_001_RECURRENCE_DECISION_FRAME.md").read_text(
        encoding="utf-8"
    )
    quality = (ROOT / "COMPARATIVE_QUALITY_OBJECTIVE_GATE.md").read_text(
        encoding="utf-8"
    )
    empirical_direction = (
        ROOT / "SCIENTIFIC_OWNER_EMPIRICAL_METHOD_DIRECTION_2026-08-31.md"
    ).read_text(encoding="utf-8")
    odq001f = (
        ROOT / "ODQ_001F_PROFILE_QUALIFIED_EMPIRICAL_DEVELOPMENT_FRAME.md"
    ).read_text(encoding="utf-8")
    baseline_repair = (
        ROOT
        / "SCIENTIFIC_OWNER_ODQ_001F_BASELINE_RELATIVE_REPAIR_DIRECTION_2026-09-01.md"
    ).read_text(encoding="utf-8")

    required_tokens = {
        readme: [
            "recovery-first Stage A owner-review candidate; no Stage B launch",
            "v0.1-stage-a-r0.7",
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
            "owner provisionally favors Choice 3",
            "SCI-FRUIT-ODQ-001A",
            "SCI-FRUIT-ODQ-001E",
            "SCI-FRUIT-ODQ-001F",
            "SCI-FRUIT-OD-001E-FRAMEWORK-2026-08-31",
            "SCI-FRUIT-ODQ-001D",
            "SCI-FRUIT-ODQ-012",
            "blocked on prior decisions",
        ],
        manifest: [
            LAUNCH_TREE,
            PROVISIONAL_TREE,
            HISTORICAL_TREE,
            "not an author-packet",
        ],
        author: ["not approved", "not dispatchable", "No implementation-blind author packet exists"],
        baseline: [
            "complete route map bundle",
            "original observation",
            "residual-only",
            "non-authoritative",
        ],
        equivalence: [
            "equivalence not established",
            "Projection/remapping consistency",
            "Identical iteration-dependent learning",
            "stable identity",
        ],
        odq001: [
            "Choice 1 — Preserve The Recovered Historical Recurrence",
            "Choice 2 — Adopt A Mathematically Equivalent Reformulation",
            "Choice 3 — Intentionally Adopt A New Recurrence",
            "ODQ-001A",
            "ODQ-001D",
            "Ordinary additive accumulation",
            "mandatory compatibility reference and scientific control",
        ],
        quality: [
            "exact historical benchmark profile",
            "Angular-scale recovery",
            "Per-mode flux recovery",
            "Residual leakage",
            "Flux convergence",
            "Computational And Operational Performance Vector",
            "Owner-Approved Comparison And Acceptance Logic",
            "No scientific-versus-resource trade is admissible unless separately approved",
        ],
        empirical_direction: [
            "hypothesis testing and tuning",
            "PSF recovery for an OOF observation",
            "faint extended SZE signal",
            "not approval of an empirical",
            "assistant-authored material",
        ],
        odq001f: [
            "Disposition A — Complete A Priori Universal Parameterization",
            "Disposition B — Baseline-Relative, Profile-Qualified Staged Development",
            "Disposition C — Historical-Compatibility v0.1 With Separate Successor R&D",
            "Baseline-Relative Qualification, Not Global Optimization",
            "no new method qualifying",
            "Candidate-Neutral Invariants",
            "Development population",
            "Qualification population",
            "Challenge population",
            "Bounded automatic adaptation",
            "Candidate Metric Skeleton",
            "Candidate Profile Priority Structure",
            "Small `d` is not evidence of small `e`",
            "Prospective Meaning Of “Better Under Most Conditions”",
            "catastrophic-regression",
            "Staged Threshold Freeze",
            "K=(M,P,S,Q,D,H,Pi,E)",
            "Repaired Candidate Owner Decision",
            "Remaining Owner Decisions Before The Empirical Lane Can Launch",
            "ODQ-001F remains open and no empirical",
            "lane, method study, or implementation is authorized",
        ],
        baseline_repair: [
            "approved in principle subject to the exact repair",
            "baseline-relative qualification",
            "paired control and scientific-performance",
            "paired candidate-minus-historical comparisons",
            "catastrophic-regression",
            "Pareto trade space",
            "no replacement method",
            "final ODQ-001F approval is not yet recorded",
        ],
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
        "PASS: exact launch/provisional/historical sources, byte-identical "
        "historical recurrence evidence, revised three-choice ODQ-001, "
        "owner-approved comparative-quality framework, conditionally preferred "
        "baseline-relative ODQ-001F frame, bounded edits, four unavailable "
        "candidate routes, Stage A firewall, placeholders, and local links"
    )


if __name__ == "__main__":
    main()
