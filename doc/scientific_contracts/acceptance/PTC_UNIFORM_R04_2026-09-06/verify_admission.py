#!/usr/bin/env python3
"""Read-only exact-byte gates for this bounded scientific admission, not VAL Core."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
BASE = "3cb01ef3672435c0cf9e3fec505abaa33c2f37a0"
CANONICAL = "080df2a1431487e8cabc255beb9ddc59c0721b59"
SCIENCE = "b38ce251207e4f445d5b421f42f774ae3e4f0477"
GOVERNANCE = "06a3ade51c1b3f38887295433d913811bf25cd14"
LIB = "doc/scientific_contracts/"
PKG = LIB + "packages/SCI-PTC-COEFFICIENT-UNIFORM/v0.1/"
STUDY = LIB + "studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/"
MUTABLE = {
    "doc/REFACTOR_STATUS.md", "doc/INTEGRATION_LEDGER.md",
    LIB + "INDEX.md", LIB + "verify_layout.py",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def blob(commit, path):
    return git("show", f"{commit}:{path}")


def verify(subject=None, integration=False):
    def read(path):
        return blob(subject, path) if subject else (ROOT / path).read_bytes()

    def record(rec):
        require(digest(read(rec["path"])) == rec["sha256"], "digest: " + rec["path"])

    manifest_path = str((HERE / "ADMISSION_MANIFEST.json").relative_to(ROOT))
    admission = json.loads(read(manifest_path))
    require(admission["scientific_base"] == BASE, "scientific base changed")
    require(admission["canonical_base"] == CANONICAL, "canonical base changed")
    record(admission["freeze_manifest"])
    freeze = json.loads(read(admission["freeze_manifest"]["path"]))
    require(freeze["scientific_commit"] == SCIENCE, "approved subject changed")
    require(freeze["approval_commit"] == BASE, "approval subject changed")
    require(len(freeze["approved_original_artifacts"]) == 48, "approved inventory count")
    require(len(freeze["promoted_copies"]) == 24, "promotion inventory count")
    require(len(admission["protected_originals"]) == 347, "protected inventory count")
    approved_paths = {r["path"] for r in freeze["approved_original_artifacts"]}
    exact_approved = set(git("ls-tree", "-r", "--name-only", SCIENCE, "--", STUDY + "stage_b/draft").decode().splitlines())
    require(approved_paths == exact_approved, "approved inventory membership")
    for rec in freeze["approved_original_artifacts"]:
        record(rec)
        require(read(rec["path"]) == blob(SCIENCE, rec["path"]), "approved byte changed")
    for rec in freeze["promoted_copies"]:
        record(rec)
        require(read(rec["path"]) == blob(rec["original_commit"], rec["original_path"]), "promotion changed bytes")
    for rec in freeze["approval_records"]:
        record(rec)
        require(read(rec["path"]) == blob(BASE, rec["path"]), "approval changed bytes")
    for rec in freeze["freeze_records"]:
        record(rec)
    common = digest(b"".join(read(PKG + path) for path in freeze["core_files"]))
    require(common == freeze["common_source_sha256"] == admission["common_source_sha256"], "common source binding")
    require(digest(read(PKG + "POST_OWNER_CONSISTENCY_REVIEW_R0.4.md")) ==
            "2bf38f7fb17ad7690e588f2fdea3059a275f265944f1f05f9dd21c2c0efb7207", "post-owner review changed")
    protected_areas = [LIB + "packages/SCI-" + p + "/v0.1" for p in ("PTC", "MAP", "JINC", "VAL")] + [STUDY.rstrip("/")]
    protected_paths = set(git("ls-tree", "-r", "--name-only", BASE, "--", *protected_areas).decode().splitlines())
    require(protected_paths == {r["path"] for r in admission["protected_originals"]}, "protected inventory membership")
    for rec in admission["protected_originals"]:
        record(rec)
        require(read(rec["path"]) == blob(BASE, rec["path"]), "predecessor changed: " + rec["path"])
    for rec in [admission["inherited_registry"], admission["inherited_source_register"], *admission["bindings"]]:
        record(rec)
    for left, right in admission["boundary_pairs"]:
        require(read(left) == read(right), "boundary copies differ")

    # Non-disposition scientific fields must match the exact approved proposal.
    proposal = read(STUDY + "stage_b/draft/PROPOSED_REGISTRY_AND_BOUNDARY_RECORDS.md").decode()
    def fields(text):
        return {parts[1].strip(): parts[2].strip() for line in text.splitlines()
                if line.startswith("| ") and len(parts := line.split("|")) == 4}
    profile_path = next(r["path"] for r in admission["bindings"] if "/PROFILE_REGISTRY_" in r["path"])
    registry_path = next(r["path"] for r in admission["bindings"] if "/COEFFICIENT_REGISTRY_" in r["path"])
    old_family = fields(proposal.split("## Proposed family record", 1)[1].split("## Proposed complete", 1)[0])
    old_profile = fields(proposal.split("## Proposed complete VAL profile record", 1)[1].split("## Proposed boundary", 1)[0])
    new_family, new_profile = fields(read(registry_path).decode()), fields(read(profile_path).decode())
    for key, value in old_family.items():
        if key not in {"Field", "Proposed actual owner", "Source identity", "Permissions"}:
            require(new_family.get(key) == value, "unapproved family rule: " + key)
    for key, value in old_profile.items():
        if key not in {"Registry component", "Actual owner", "Source", "Status"}:
            require(new_profile.get(key) == value, "unapproved profile rule: " + key)
    require(new_family["Actual owner"].startswith("Grant Wilson acting for SCI-PTC"), "actual owner")
    require("SCI-MAP permission GRANTED" in new_family["Permissions"] and
            "SCI-JINC permission separately GRANTED" in new_family["Permissions"], "separate grants")
    require("No constant override, family default, or publication-size default" in new_family["Numerical parameters/defaults"], "default inferred")

    authority_hashes = {
        "doc/governance/ENGINEERING_GOVERNANCE.md": "70769787ce2ef4b7323cd2a38e221ade4af3310e0ad6b7b682e08cb4e4d61e76",
        "doc/governance/REVIEW_AND_CONFORMANCE.md": "691e6d6250102ef2f4a504397581ee67c5707d898ab20fb8dd9e874c47f99bb1",
        LIB + "README.md": "351e9b7775b0bf78cba01bf4cd2fafd9591c4b43931b0dc23d82d97f0dfe82d2",
    }
    for path, expected in authority_hashes.items():
        require(digest(read(path)) == expected == digest(blob(CANONICAL, path)), "effective authority changed")
    git("merge-base", "--is-ancestor", GOVERNANCE, CANONICAL)
    git("merge-base", "--is-ancestor", BASE, subject or "HEAD")

    allowed = MUTABLE | {r["path"] for r in admission["bindings"]}
    allowed |= {r["path"] for r in freeze["promoted_copies"] + freeze["freeze_records"]}
    allowed |= {admission["freeze_manifest"]["path"], manifest_path}
    allowed |= {str((HERE / name).relative_to(ROOT)) for name in
                ("README.md", "NEXT_IMPLEMENTATION_DECISIONS.md", "verify_admission.py", "VERIFICATION.md")}
    if integration:
        require(subject is not None, "integration requires an exact committed subject")
        parents = git("show", "-s", "--format=%P", subject).decode().split()
        require(len(parents) == 2 and parents[0] == CANONICAL, "integration parent order/base")
        git("merge-base", "--is-ancestor", BASE, parents[1])
        topic_changes = set(git("diff", "--name-only", CANONICAL + "..." + parents[1]).decode().splitlines())
        require(all(p.startswith("doc/") for p in topic_changes), "topic includes application work")
        for path in topic_changes - MUTABLE:
            require(read(path) == blob(parents[1], path), "integration changed topic byte: " + path)
        allowed |= topic_changes
        allowed |= {str((HERE / name).relative_to(ROOT)) for name in
                    ("INTEGRATION_RECEIPT.md", "SCIENTIFIC_CANDIDATE_REVIEW.md")}
        diff_base = CANONICAL
    else:
        diff_base = BASE
    changed = set(git("diff", "--name-only", diff_base, *([subject] if subject else [])).decode().splitlines())
    if not subject:
        changed |= set(git("ls-files", "--others", "--exclude-standard").decode().splitlines())
    require(changed <= allowed, "unexpected changed path: " + str(sorted(changed - allowed)))
    require(all(p.startswith("doc/") for p in changed), "non-documentation change")

    # Byte-preserved historical evidence may retain original-context links.
    # Newly authored navigation and the promoted scope/recovery links must resolve.
    for path in sorted(changed):
        if path.endswith(".md") and path not in {r["path"] for r in freeze["promoted_copies"]} and path not in {
            PKG + "POST_OWNER_CONSISTENCY_REVIEW_R0.4.md",
            str((HERE / "SCIENTIFIC_CANDIDATE_REVIEW.md").relative_to(ROOT)),
        } and (path.startswith(PKG) or path.startswith(str(HERE.relative_to(ROOT)))):
            for target in re.findall(r"\]\(([^)]+)\)", read(path).decode()):
                if "://" in target or target.startswith("#"):
                    continue
                resolved = (ROOT / path).parent / target.split("#", 1)[0]
                require(resolved.exists(), f"broken link {path}: {target}")
    print(json.dumps({"status": "PASS", "subject": subject or "working tree", "integration": integration,
                      "approved_artifacts": 48, "promoted_copies": 24, "protected_originals": 347,
                      "common_source_sha256": common, "changed_paths": len(changed),
                      "limitations": "Exact documentary integrity/scope; no runtime VAL evaluation, application or Spack validation."}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", help="exact committed candidate to inspect; default is working tree")
    parser.add_argument("--integration", action="store_true", help="check canonical-first two-parent merge")
    args = parser.parse_args()
    verify(args.subject, args.integration)
