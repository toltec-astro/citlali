#!/usr/bin/env python3
"""Verify the exact SCI-NOI Stage A author packet and closure bindings."""

from __future__ import annotations

import hashlib
import pathlib
import re
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parent
MANIFEST = ROOT / "AUTHOR_PACKET_MANIFEST.md"
VAL_ROOT = ROOT.parent.parent / "SCI-VAL" / "v0.1"

PROCESS_BINDINGS = {
    ROOT / "SCIENTIFIC_OWNER_STAGE_A_FINAL_APPROVAL_2026-08-30.md":
        "49377d1596c9e47a6e2328e890ebcd6b25f42af3781b533b5bd8c2cded08fa6b",
    ROOT / "SCI_VAL_REGISTRY_BINDING_2026-08-30.md":
        "739b5c7d7818a4292ae4b0beeab5a2d0356d77f0525bd0198e67181ae6d28a2e",
    VAL_ROOT / "SOURCE_BINDING_REGISTER_NOI_STAGE_A_R0_18_2026-08-30.md":
        "04eca2da9ce76afacf18ae90dc2dbcb702fedbf55e03acb28e14e7dbc459a7c3",
    VAL_ROOT / "PROFILE_REGISTRY_NOI_STAGE_A_R0_18_2026-08-30.md":
        "5994f4dff49dff3a9c9da6fbb494671b14a2f926f325f1c7c4a9603a6c2a38c1",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_bytes(spec: str) -> bytes:
    return subprocess.check_output(["git", "show", spec], cwd=ROOT)


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def verify_external(name: str) -> None:
    binding_name = f"{pathlib.Path(name).stem}.sha256"
    record = (ROOT / binding_name).read_text(encoding="utf-8").strip()
    expected, recorded_name = record.split(maxsplit=1)
    if recorded_name != name:
        fail(f"external binding names {recorded_name!r}, expected {name!r}")
    actual = digest((ROOT / name).read_bytes())
    if actual != expected:
        fail(f"{name}: expected {expected}, got {actual}")


def section_map(text: str) -> dict[str, str]:
    sections = re.findall(
        r"(^### `[^\n]+`\n.*?)(?=^### |^## |\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    return {section.splitlines()[0]: section for section in sections}


def verify_registry_successors() -> None:
    for path, expected in PROCESS_BINDINGS.items():
        if not path.is_file():
            fail(f"missing process-only binding: {path}")
        actual = digest(path.read_bytes())
        if actual != expected:
            fail(f"{path.name}: expected {expected}, got {actual}")

    base_source = (
        VAL_ROOT / "SOURCE_BINDING_REGISTER_JINC_STAGE_A_Q002_2026-08-28.md"
    ).read_text(encoding="utf-8")
    noi_source = (
        VAL_ROOT / "SOURCE_BINDING_REGISTER_NOI_STAGE_A_R0_18_2026-08-30.md"
    ).read_text(encoding="utf-8")
    inherited_rows = [
        line for line in base_source.splitlines()
        if line.startswith("| ") and not line.startswith("| ---")
    ]
    for row in inherited_rows:
        if row not in noi_source:
            fail("NOI source-binding successor changed an inherited row")

    base_registry = (
        VAL_ROOT / "PROFILE_REGISTRY_JINC_STAGE_A_Q002_2026-08-28.md"
    ).read_text(encoding="utf-8")
    noi_registry = (
        VAL_ROOT / "PROFILE_REGISTRY_NOI_STAGE_A_R0_18_2026-08-30.md"
    ).read_text(encoding="utf-8")
    inherited_sections = section_map(base_registry)
    successor_sections = section_map(noi_registry)
    for heading, section in inherited_sections.items():
        if successor_sections.get(heading) != section:
            fail(f"NOI profile successor changed inherited record: {heading}")

    new_keys = {
        "### `SCI-NOI:generation_input_admission@1`",
        "### `SCI-NOI:uncertainty_member_admission@1`",
        "### `SCI-NOI:uncertainty_ensemble_admission@1`",
        "### `SCI-NOI:standardization_admission@1`",
    }
    if not new_keys.issubset(successor_sections):
        fail("NOI profile successor does not contain all four approved records")


def main() -> None:
    text = MANIFEST.read_text(encoding="utf-8")
    rows = re.findall(
        r"^\| \d+ \| `([^`]+)`(?:, `[^`]+`)? \| `([0-9a-f]{64})` \|",
        text,
        flags=re.MULTILINE,
    )
    if len(rows) != 17:
        fail(f"manifest has {len(rows)} parsed objects, expected 17")

    for source, expected in rows:
        if ":" in source and not (ROOT / source).exists():
            data = git_bytes(source)
        else:
            path = ROOT / source
            if not path.is_file():
                fail(f"missing manifest object: {source}")
            data = path.read_bytes()
        actual = digest(data)
        if actual != expected:
            fail(f"{source}: expected {expected}, got {actual}")

    verify_external("AUTHOR_PACKET_MANIFEST.md")
    verify_external("BYTE_EQUALITY_AND_SOURCE_CLOSURE_REPORT.md")
    verify_registry_successors()

    package_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in ROOT.glob("*.md")
        if path.name != "STAGE_A_CHANGE_LOG.md"
    )
    required = [
        "NOI-GEN/PTC-TO-FROZEN-MAP-CONDITIONAL-SIGN@1",
        "NOI-GEN/PTC-TO-FROZEN-JINC-CONDITIONAL-SIGN@1",
        "NOI-GEN/REALIZED-MAP-CONDITIONAL-SIGN@1",
        "NOI-GEN/REALIZED-JINC-CONDITIONAL-SIGN@1",
        "NOI-GEN/OWNER-TRANSFORMED-CONDITIONAL-SIGN@1",
        "SCI-NOI:generation_input_admission@1",
        "SCI-NOI:uncertainty_member_admission@1",
        "SCI-NOI:uncertainty_ensemble_admission@1",
        "SCI-NOI:standardization_admission@1",
        "SCI-VAL authors neither producer facts",
        "nor NOI policy",
        "Only `requested + applicable + eligible + realized` projects",
        "No profile",
        "automatically realizes the next GEN, UNC, or STD operation",
        "unit(empirical_scale_standardized_signal) = 1",
        "SCI-NOI-ODQ-102A",
        "SCI-NOI-ODQ-102B",
        "SCI-NOI-ODQ-102C",
        "SCI-NOI-ODQ-102D",
        "materialized randomized timestream",
        "not an ordinary MAP science product",
        "throughout the observation",
        "network-stratified",
        "cross-network",
        "marginal sign probability `1/2`",
        "approved only as a suggestion",
        "implementation-blind scientific-contract author",
        "scientifically consequential adjacent reduction state",
        "does not require exhaustive implementation provenance",
        "randomization is intended to suppress source signal",
        "does not, by construction alone, establish",
        "nonbinding terminology suggestion",
        "Candidate assignments rejected during finite-design construction",
        "fails the complete GEN ensemble",
        "survivor or partial design",
        "V_hat_cond(p) = sum_b omega_b M_b(p)^2",
        "finite ensemble mean is not subtracted",
        "no `B-1` correction",
        "common all-member domain",
        "not covariance merely because it is pointwise or diagonal-like",
        "A retained ensemble is a representation",
        "Unreported off-diagonal entries",
        "not zero or independent",
        "No representation implies invertibility",
        "NOI-UNC/INVERSE-CONDITIONAL-SECOND-MOMENT-SCALE",
        "W_hat_cond(p) = 1 / V_hat_cond(p)",
        "It shall not substitute a numerical zero",
        "not inverse variance or precision",
        "cross-boundary use requires explicit scientific authority",
        "NOI-STD/MAP-CONDITIONAL-SECOND-MOMENT-SCALE@1",
        "sigma_cond(p) = sqrt(V_hat_cond(p))",
        "S_cond(p) = q_MAP(p) / sigma_cond(p)",
        "MAP signal standardized by the stated conditional randomization",
        "does not create a second implicit STD method identity",
        "JINC standardization remains a separate future method",
        "Plan-selected persisted ensemble, compact deterministic regeneration, or streaming sufficient statistics",
        "no universal default or silent fallback",
        "byte-identical or numerical reproducibility",
        "mathematically sufficient for every exact published product and claim",
        "partial streaming state carries no UNC authority",
        "Failure of plan-required persistence fails that product",
        "NOI does not choose or define a deterministic transformation",
        "appropriate upstream/downstream scientific process",
        "every admitted compatible randomization",
        "exact transformed scientific product",
        "may be data-derived and still be fixed-state for NOI",
        "begins a new scientific-product and NOI generation",
        "`UNC_k` is an input to the successor transformation",
        "not independent evidence validating",
        "Learning or selecting a distinct Wiener operator for each realization",
        "FRUIT owns its source model, subtraction and add-back",
        "conditional on the frozen FRUIT state",
        "FRUIT_OWNER:LearnIterate_(k+1)",
        "Partial or complete FRUIT replay for each realization",
        "Fixed-residual, fixed-terminal-transform, partial-replay, and complete-replay",
        "Stage B at high reasoning effort",
        "only `AUTHOR_PACKET_MANIFEST.md` and its 17 exact admitted objects",
    ]
    for token in required:
        if token not in package_text:
            fail(f"required closure token absent: {token}")
    if "SCI-NOI:realization_member_completion@1" in package_text:
        fail("superseded completion-profile identity remains outside change log")

    for path in ROOT.glob("*.md"):
        body = path.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^\]]+\]\(([^)#]+)(?:#[^)]+)?\)", body):
            if "://" in target or target.startswith("mailto:"):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                fail(f"broken local link in {path.name}: {target}")

    print(
        "PASS: 17/17 author objects, external/process bindings, inherited "
        "Registry equality, closure invariants, and local links"
    )


if __name__ == "__main__":
    main()
