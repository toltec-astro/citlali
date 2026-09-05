#!/usr/bin/env python3
"""Check Stage A evidence and author-input integrity; never approve science."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
AUTHOR = HERE / "author_packet"
PARENT = "84974658c8179057fd26184110871f187c4bb793"
CANONICAL_BASE = "00b974c9039d4c3025dcce18f26bca69d36af9c3"
BRANCH = "codex/cti-od-002-coefficient-authority-2026-09-05"
STUDY = HERE.parent.relative_to(ROOT).as_posix()


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_file(base, relative):
    p = base / relative
    require(p.is_file() and not p.is_symlink(), f"Missing/unsafe file: {relative}")
    require(p.resolve().is_relative_to(base.resolve()), f"Escaping path: {relative}")
    return p


def verify_links(path, boundary):
    text = path.read_text()
    # Literal source blocks are quotations, not additional admitted references.
    text = re.sub(r"<!-- EXCERPT .*?<!-- END EXCERPT .*? -->", "", text, flags=re.S)
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
        if "://" in target or target.startswith("#"):
            continue
        target_path = (path.parent / target.split("#")[0]).resolve()
        require(target_path.exists(), f"Broken link in {path.name}: {target}")
        if boundary:
            require(target_path.is_relative_to(boundary), f"Author link leaves packet: {target}")


def verify(precommit=False):
    manifest = json.loads((AUTHOR / "MANIFEST.json").read_text())
    extraction = json.loads((HERE / "EXTRACTION_MANIFEST.json").read_text())
    require(extraction["parent_commit"] == PARENT, "Wrong preparation parent")
    require(extraction["canonical_base"] == CANONICAL_BASE, "Wrong canonical base")
    require(manifest["release_status"] == "candidate; owner release approval pending", "Release state drift")
    paths = [x["path"] for x in manifest["files"]]
    require(len(paths) == len(set(paths)) == 10, "Author file count/identity drift")
    actual = {p.relative_to(AUTHOR).as_posix() for p in AUTHOR.rglob("*") if p.is_file()}
    require(actual == set(paths) | {"MANIFEST.json"}, "Author inventory mismatch")
    for entry in manifest["files"]:
        data = safe_file(AUTHOR, entry["path"]).read_bytes()
        require(len(data) == entry["bytes"] and digest(data) == entry["sha256"], f"Author digest: {entry['path']}")

    sources = {s["id"]: s for s in extraction["sources"]}
    require(len(sources) == len(extraction["sources"]), "Duplicate source ID")
    cache = {}
    for sid, source in sources.items():
        data = git("show", f"{source['commit']}:{source['path']}")
        fields = git("ls-tree", source["commit"], "--", source["path"]).split(b"\t")[0].decode().split()
        require(fields == [source["git_mode"], "blob", source["git_blob"]], f"Git identity: {sid}")
        require(len(data) == source["bytes"] and digest(data) == source["sha256"], f"Source digest: {sid}")
        if "approved_core_candidate" in source:
            require(data == git("show", f"{source['approved_core_candidate']}:{source['path']}"), f"VAL freeze drift: {sid}")
        cache[sid] = data
    require(sources["V04"]["sha256"] in cache["V01"].decode(), "Common fragment lacks Registry binding")
    require(not set(extraction["excluded_source_ids"]) & set(sources), "Excluded recovery source admitted")

    excerpt_ids = set()
    for item in extraction["excerpts"]:
        require(item["id"] not in excerpt_ids, "Duplicate excerpt ID")
        excerpt_ids.add(item["id"])
        data = cache[item["source_id"]][item["start_byte"]:item["end_byte_exclusive"]]
        require(digest(data) == item["sha256"], f"Excerpt source span: {item['id']}")
        contents = safe_file(AUTHOR, item["destination"]).read_bytes()
        begin = f"<!-- EXCERPT {item['id']} -->\n".encode()
        end = f"<!-- END EXCERPT {item['id']} -->".encode()
        require(contents.count(begin) == contents.count(end) == 1, f"Excerpt markers: {item['id']}")
        actual = contents.split(begin)[1].split(end)[0]
        expected = data + (b"\n" if item["append_final_lf"] else b"")
        if item["code_fence"]:
            expected = f"```{item['code_fence']}\n".encode() + expected + b"```\n"
        require(actual == expected, f"Literal excerpt changed: {item['id']}")
    marker_count = sum(p.read_text().count("<!-- EXCERPT ") for p in (AUTHOR / "references").glob("*.md"))
    require(marker_count == len(excerpt_ids) == 61, "Unmanifested/missing excerpt")

    require(digest((HERE / extraction["owner_scope_record"]).read_bytes()) == extraction["owner_scope_record_sha256"], "Owner approval record drift")
    require(digest((HERE / "RECOVERY_REVIEW_84974658.md").read_bytes()) == "c4f52fe248adb7712e2fb1293a03ab8e7b261fe32fa5c861c45586b09e71a83f", "Recovery review drift")
    for name in ["README.md", "PRIOR_WORK.md", "DECISION_BRIEF.md", "SOURCE_MANIFEST.json"]:
        path = f"{STUDY}/{name}"
        require((ROOT / path).read_bytes() == git("show", f"{PARENT}:{path}"), f"Reviewed recovery changed: {name}")
    for path in HERE.rglob("*.md"):
        if path.name == "RECOVERY_REVIEW_84974658.md":
            continue  # immutable external review retains its original absolute citations
        verify_links(path, AUTHOR.resolve() if path.is_relative_to(AUTHOR) else None)
    for name in ["README.md", "SCOPE_BRIEF.md", "PRIOR_WORK.md"]:
        headings = re.findall(r"^## (.+)$", (AUTHOR / name).read_text(), re.M)
        require(headings[0] == "Program adherence and prior-work recovery", f"Required opening: {name}")
    all_author = "\n".join((AUTHOR / p).read_text() for p in paths)
    for forbidden in ["CTI-FM-", "MSP-E0", "SCI-PTC-001_INDEPENDENT_CORE", "ODQ-102D delegates exact numerical balance", "## MAP-Local Decisions"]:
        require(forbidden not in all_author, f"Excluded material marker: {forbidden}")

    require(git("branch", "--show-current").decode().strip() == BRANCH, "Wrong owned branch")
    head = git("rev-parse", "HEAD").decode().strip()
    if precommit:
        require(head == PARENT, "Precommit parent drift")
        changed = set(git("diff", "--name-only", PARENT).decode().splitlines())
        changed |= set(git("ls-files", "--others", "--exclude-standard").decode().splitlines())
    else:
        require(git("show", "-s", "--format=%P", "HEAD").decode().strip() == PARENT, "Wrong sole parent")
        require(not git("status", "--porcelain=v1", "--untracked-files=all"), "Dirty review subject")
        changed = set(git("diff", "--name-only", PARENT, "HEAD").decode().splitlines())
    require(changed == set(extraction["allowed_changed_paths"]), "Changed-path scope mismatch")
    subprocess.run(["git", "-C", str(ROOT), "diff", "--check", PARENT], check=True)
    return dict(status="PASS",mode="precommit" if precommit else "postcommit",head=head,
                author_files=len(paths)+1,sources=len(sources),literal_excerpts=len(excerpt_ids),
                changed_paths=len(changed),manifest_sha256=digest((AUTHOR/"MANIFEST.json").read_bytes()),
                claim="input/evidence integrity only; no release or scientific approval")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precommit", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.precommit), indent=2))
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"author_packet_integrity=FAIL: {error}")
