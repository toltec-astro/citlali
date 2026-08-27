#!/usr/bin/env python3
"""Audit decentralized Spack recipes against accepted release source commits."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


EXPECTED_PACKAGES = {
    "citlali": ("citlali",),
    "tula_cmake": (
        "tula-cmake",
        "tula-logging",
        "tula-yaml-cpp",
        "tula-eigen3",
        "tula-ccfits",
        "tula-netcdf-cxx4",
        "tula-perflibs",
    ),
    "tula": ("tula",),
    "kidscpp": ("kidscpp",),
}
RECIPE_REPOSITORY_ROOTS = {
    "citlali": "spack/spack_repo/toltec/citlali",
    "tula_cmake": "spack_repo/toltec/tula_cmake",
    "tula": "spack_repo/toltec/tula",
    "kidscpp": "spack_repo/toltec/kidscpp",
}
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class VersionSource:
    version: str
    commit: str | None
    url: str | None
    sha256: str | None


@dataclass(frozen=True)
class PackageAudit:
    repository: str
    package: str
    source_commit: str
    recipe_commit: str
    recipe_path: str | None
    accepted: bool
    reason: str
    versions: tuple[VersionSource, ...]


def _run_git(checkout: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout


def _literal_string(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, str) else None


def parse_recipe(text: str, *, filename: str) -> tuple[str | None, list[VersionSource]]:
    """Return the package Git URL and declared immutable source versions."""
    tree = ast.parse(text, filename=filename)
    package_classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    if len(package_classes) != 1:
        raise ValueError(f"{filename} must define exactly one package class")

    git_url = None
    versions = []
    for statement in package_classes[0].body:
        if isinstance(statement, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == "git"
                for target in statement.targets
            ):
                git_url = _literal_string(statement.value)
        if not isinstance(statement, ast.Expr) or not isinstance(
            statement.value, ast.Call
        ):
            continue
        call = statement.value
        if not isinstance(call.func, ast.Name) or call.func.id != "version":
            continue
        version = _literal_string(call.args[0]) if call.args else None
        if version is None:
            raise ValueError(f"{filename} contains a non-literal package version")
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        versions.append(
            VersionSource(
                version=version,
                commit=_literal_string(keywords.get("commit")),
                url=_literal_string(keywords.get("url")),
                sha256=_literal_string(keywords.get("sha256")),
            )
        )
    return git_url, versions


def _normalized_repository_url(url: str | None) -> str | None:
    if url is None:
        return None
    return url.removesuffix(".git").rstrip("/")


def audit_recipe(
    *,
    repository: str,
    repository_url: str,
    expected_commit: str,
    recipe_commit: str,
    package: str,
    recipe_path: str,
    recipe_text: str,
) -> PackageAudit:
    """Require one recipe version to resolve the accepted repository commit."""
    git_url, versions = parse_recipe(recipe_text, filename=recipe_path)
    expected_repository = _normalized_repository_url(repository_url)
    for source in versions:
        if source.commit == expected_commit:
            if _normalized_repository_url(git_url) == expected_repository:
                return PackageAudit(
                    repository,
                    package,
                    expected_commit,
                    recipe_commit,
                    recipe_path,
                    True,
                    "Git source matches accepted repository and commit",
                    tuple(versions),
                )
        if source.url is not None and expected_commit in source.url:
            if (
                source.url.startswith(f"{expected_repository}/")
                and source.sha256 is not None
                and _SHA256.fullmatch(source.sha256) is not None
            ):
                return PackageAudit(
                    repository,
                    package,
                    expected_commit,
                    recipe_commit,
                    recipe_path,
                    True,
                    "archive source matches accepted repository and commit",
                    tuple(versions),
                )

    declared = []
    for source in versions:
        if source.commit is not None:
            declared.append(f"{source.version}:commit={source.commit}")
        elif source.url is not None:
            declared.append(f"{source.version}:url={source.url}")
        else:
            declared.append(f"{source.version}:source=unbound")
    reason = "no package version resolves the accepted commit"
    if declared:
        reason += f"; declared {', '.join(declared)}"
    else:
        reason += "; no version directives found"
    return PackageAudit(
        repository,
        package,
        expected_commit,
        recipe_commit,
        recipe_path,
        False,
        reason,
        tuple(versions),
    )


def _recipe_path_at_commit(
    tree_paths: Sequence[str],
    repository: str,
    package: str,
) -> str | None:
    package_directory = package.replace("-", "_")
    expected = (
        f"{RECIPE_REPOSITORY_ROOTS[repository]}/packages/"
        f"{package_directory}/package.py"
    )
    return expected if expected in tree_paths else None


def audit_repository(
    *,
    name: str,
    record: dict[str, Any],
    checkout: Path,
) -> list[PackageAudit]:
    """Audit all release packages owned by one pinned source repository."""
    repository_url = record.get("repository_url")
    source_commit = record.get("source_commit")
    recipe_commit = record.get("recipe_commit")
    if not isinstance(repository_url, str) or not repository_url:
        raise ValueError(f"sources.{name}.repository_url is invalid")
    if (
        not isinstance(source_commit, str)
        or _GIT_COMMIT.fullmatch(source_commit) is None
    ):
        raise ValueError(f"sources.{name}.source_commit is invalid")
    if (
        not isinstance(recipe_commit, str)
        or _GIT_COMMIT.fullmatch(recipe_commit) is None
    ):
        raise ValueError(f"sources.{name}.recipe_commit is invalid")
    if not checkout.is_dir():
        raise FileNotFoundError(checkout)

    _run_git(checkout, "cat-file", "-e", f"{source_commit}^{{commit}}")
    _run_git(checkout, "cat-file", "-e", f"{recipe_commit}^{{commit}}")
    tree_paths = _run_git(
        checkout,
        "ls-tree",
        "-r",
        "--name-only",
        recipe_commit,
    ).splitlines()
    results = []
    for package in EXPECTED_PACKAGES[name]:
        recipe_path = _recipe_path_at_commit(tree_paths, name, package)
        if recipe_path is None:
            results.append(
                PackageAudit(
                    name,
                    package,
                    source_commit,
                    recipe_commit,
                    None,
                    False,
                    "owned package recipe is missing at the accepted commit",
                    (),
                )
            )
            continue
        recipe_text = _run_git(
            checkout,
            "show",
            f"{recipe_commit}:{recipe_path}",
        )
        results.append(
            audit_recipe(
                repository=name,
                repository_url=repository_url,
                expected_commit=source_commit,
                recipe_commit=recipe_commit,
                package=package,
                recipe_path=recipe_path,
                recipe_text=recipe_text,
            )
        )
    return results


def audit_manifest_sources(
    manifest: dict[str, Any],
    source_roots: dict[str, Path],
) -> dict[str, Any]:
    """Build a deterministic recipe-source audit report for one manifest."""
    sources = manifest.get("sources")
    if not isinstance(sources, dict) or set(sources) != set(EXPECTED_PACKAGES):
        raise ValueError(f"manifest sources must be exactly {tuple(EXPECTED_PACKAGES)}")
    if set(source_roots) != set(EXPECTED_PACKAGES):
        raise ValueError("source roots do not cover the required repositories")

    audits = []
    for name in EXPECTED_PACKAGES:
        record = sources[name]
        if not isinstance(record, dict):
            raise ValueError(f"sources.{name} is invalid")
        audits.extend(
            audit_repository(name=name, record=record, checkout=source_roots[name])
        )
    failures = [audit for audit in audits if not audit.accepted]
    return {
        "schema_version": "citlali-release-recipe-audit-v1",
        "release_id": manifest.get("release_id"),
        "manifest_state": manifest.get("manifest_state"),
        "status": "accepted" if not failures else "blocked",
        "package_count": len(audits),
        "failure_count": len(failures),
        "packages": [asdict(audit) for audit in audits],
    }


def _parse_source_root(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", maxsplit=1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected NAME=PATH") from error
    if name not in EXPECTED_PACKAGES:
        raise argparse.ArgumentTypeError(f"unsupported source name: {name}")
    return name, Path(path).expanduser().resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repository_root / "spack/release/development-candidate.json",
    )
    parser.add_argument(
        "--source-root",
        action="append",
        type=_parse_source_root,
        default=[],
        metavar="NAME=PATH",
        help="override a default Git checkout used to inspect a pinned commit",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repository_root = Path(__file__).resolve().parents[2]
    source_roots = {
        "citlali": repository_root,
        "tula_cmake": repository_root / "build/spack-sources/tula_cmake",
        "tula": repository_root / "build/spack-sources/tula",
        "kidscpp": repository_root / "build/spack-sources/kidscpp",
    }
    source_roots.update(dict(args.source_root))
    manifest = json.loads(args.manifest.expanduser().resolve().read_text())
    report = audit_manifest_sources(manifest, source_roots)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    else:
        print(rendered, end="")

    if report["status"] != "accepted":
        failures = [item for item in report["packages"] if not item["accepted"]]
        print("release recipe audit blocked:", file=sys.stderr)
        for item in failures:
            print(
                f"- {item['repository']}/{item['package']}: {item['reason']}",
                file=sys.stderr,
            )
        return 1
    print(f"release recipe audit accepted: packages={report['package_count']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"release recipe audit invalid: {error}", file=sys.stderr)
        raise SystemExit(1) from None
