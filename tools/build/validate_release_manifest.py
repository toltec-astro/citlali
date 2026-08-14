#!/usr/bin/env python3
"""Validate Citlali's source, profile-lock, and release-evidence contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "citlali-release-manifest-v1"
REQUIRED_SOURCES = ("citlali", "tula_cmake", "tula", "kidscpp")
REQUIRED_PROFILES = ("macos-llvm20", "unity-gcc13")
MANIFEST_STATES = ("development-candidate", "release")
BUILDCACHE_MODES = (
    "source-build-default",
    "signed-buildcache-optional",
    "signed-buildcache-required",
)
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SPACK_HASH = re.compile(r"[a-z0-9]{32}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_exact_keys(
    value: dict[str, Any],
    required: Sequence[str],
    label: str,
) -> None:
    missing = sorted(set(required) - set(value))
    extra = sorted(set(value) - set(required))
    if missing or extra:
        raise ValueError(f"{label} keys differ: missing={missing} extra={extra}")


def _repository_path(repository_root: Path, value: Any, label: str) -> Path:
    relative = Path(_require_string(value, label))
    if relative.is_absolute():
        raise ValueError(f"{label} must be repository-relative")
    path = (repository_root / relative).resolve()
    try:
        path.relative_to(repository_root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository: {relative}") from error
    return path


def _validate_checked_file(
    repository_root: Path,
    record: Any,
    label: str,
) -> Path:
    identity = _require_mapping(record, label)
    path = _repository_path(repository_root, identity.get("path"), f"{label}.path")
    expected_sha256 = _require_string(
        identity.get("sha256"),
        f"{label}.sha256",
    )
    if _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError(f"{label}.sha256 must be lowercase SHA-256")
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{label} checksum mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    return path


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def _validate_source(
    repository_root: Path,
    name: str,
    record: Any,
    *,
    require_release: bool,
) -> None:
    source = _require_mapping(record, f"sources.{name}")
    _require_exact_keys(
        source,
        ("repository_url", "review_branch", "commit", "archive"),
        f"sources.{name}",
    )
    repository_url = _require_string(
        source.get("repository_url"),
        f"sources.{name}.repository_url",
    )
    if not repository_url.startswith("https://github.com/toltec-astro/"):
        raise ValueError(f"sources.{name}.repository_url is not an accepted URL")
    _require_string(source.get("review_branch"), f"sources.{name}.review_branch")
    commit = _require_string(source.get("commit"), f"sources.{name}.commit")
    if _GIT_COMMIT.fullmatch(commit) is None:
        raise ValueError(f"sources.{name}.commit must be a full lowercase Git SHA")

    archive = source.get("archive")
    if archive is None:
        if require_release:
            raise ValueError(f"sources.{name}.archive is required for release")
        return
    archive = _require_mapping(archive, f"sources.{name}.archive")
    _require_exact_keys(
        archive,
        ("path", "url", "sha256", "immutable"),
        f"sources.{name}.archive",
    )
    _validate_checked_file(
        repository_root,
        archive,
        f"sources.{name}.archive",
    )
    url = _require_string(archive.get("url"), f"sources.{name}.archive.url")
    digest = _require_string(
        archive.get("sha256"),
        f"sources.{name}.archive.sha256",
    )
    if not url.startswith("https://"):
        raise ValueError(f"sources.{name}.archive.url must use HTTPS")
    if commit not in url:
        raise ValueError(
            f"sources.{name}.archive.url must identify the declared commit"
        )
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"sources.{name}.archive.sha256 must be lowercase SHA-256")
    if archive.get("immutable") is not True:
        raise ValueError(f"sources.{name}.archive must be marked immutable")


def _validate_release_lock(
    repository_root: Path,
    record: Any,
    label: str,
) -> None:
    lock_path = _validate_checked_file(repository_root, record, label)
    lock_record = _require_mapping(record, label)
    _require_exact_keys(
        lock_record,
        ("path", "sha256", "root_dag_hash"),
        label,
    )
    expected_root = _require_string(
        lock_record.get("root_dag_hash"),
        f"{label}.root_dag_hash",
    )
    if _SPACK_HASH.fullmatch(expected_root) is None:
        raise ValueError(f"{label}.root_dag_hash must be a Spack DAG hash")
    lock = json.loads(lock_path.read_text())
    roots = lock.get("roots", [])
    if len(roots) != 1 or roots[0].get("hash") != expected_root:
        raise ValueError(f"{label} does not contain the declared single root")
    if _contains_key(lock, "dev_path"):
        raise ValueError(f"{label} contains non-portable develop paths")


def _validate_profile(
    repository_root: Path,
    name: str,
    record: Any,
    *,
    require_release: bool,
) -> None:
    profile = _require_mapping(record, f"profiles.{name}")
    _require_exact_keys(
        profile,
        (
            "platform",
            "compiler",
            "development_environment",
            "observed_development_lock",
            "release_environment",
            "release_lock",
        ),
        f"profiles.{name}",
    )
    _require_string(profile.get("platform"), f"profiles.{name}.platform")
    _require_string(profile.get("compiler"), f"profiles.{name}.compiler")
    development_environment = _require_mapping(
        profile.get("development_environment"),
        f"profiles.{name}.development_environment",
    )
    _require_exact_keys(
        development_environment,
        ("path", "sha256"),
        f"profiles.{name}.development_environment",
    )
    _validate_checked_file(
        repository_root,
        development_environment,
        f"profiles.{name}.development_environment",
    )
    observed_lock = _require_mapping(
        profile.get("observed_development_lock"),
        f"profiles.{name}.observed_development_lock",
    )
    _require_exact_keys(
        observed_lock,
        ("sha256", "root_dag_hash", "portable"),
        f"profiles.{name}.observed_development_lock",
    )
    observed_digest = _require_string(
        observed_lock.get("sha256"),
        f"profiles.{name}.observed_development_lock.sha256",
    )
    observed_root = _require_string(
        observed_lock.get("root_dag_hash"),
        f"profiles.{name}.observed_development_lock.root_dag_hash",
    )
    if _SHA256.fullmatch(observed_digest) is None:
        raise ValueError(f"profiles.{name} has invalid observed lock SHA-256")
    if _SPACK_HASH.fullmatch(observed_root) is None:
        raise ValueError(f"profiles.{name} has invalid observed root DAG hash")
    if observed_lock.get("portable") is not False:
        raise ValueError(f"profiles.{name} development lock must be non-portable")

    release_environment = profile.get("release_environment")
    release_lock = profile.get("release_lock")
    if (release_environment is None) != (release_lock is None):
        raise ValueError(
            f"profiles.{name} release environment and lock must be paired"
        )
    if release_environment is None or release_lock is None:
        if require_release:
            raise ValueError(f"profiles.{name} release environment and lock required")
        return
    release_environment = _require_mapping(
        release_environment,
        f"profiles.{name}.release_environment",
    )
    _require_exact_keys(
        release_environment,
        ("path", "sha256"),
        f"profiles.{name}.release_environment",
    )
    environment_path = _validate_checked_file(
        repository_root,
        release_environment,
        f"profiles.{name}.release_environment",
    )
    if re.search(r"^\s*develop\s*:", environment_path.read_text(), re.MULTILINE):
        raise ValueError(f"profiles.{name}.release_environment contains develop paths")
    _validate_release_lock(
        repository_root,
        release_lock,
        f"profiles.{name}.release_lock",
    )


def _validate_buildcache(record: Any) -> None:
    buildcache = _require_mapping(record, "buildcache")
    _require_exact_keys(
        buildcache,
        (
            "mode",
            "source_build_fallback",
            "mirrors",
            "trusted_signing_key_fingerprints",
        ),
        "buildcache",
    )
    mode = _require_string(buildcache.get("mode"), "buildcache.mode")
    if mode not in BUILDCACHE_MODES:
        raise ValueError(f"unsupported buildcache mode: {mode}")
    mirrors = buildcache.get("mirrors")
    fingerprints = buildcache.get("trusted_signing_key_fingerprints")
    if not isinstance(mirrors, list) or not isinstance(fingerprints, list):
        raise ValueError("buildcache mirrors and fingerprints must be arrays")
    if buildcache.get("source_build_fallback") is not True:
        raise ValueError("release must retain a source-build fallback")
    if mode == "source-build-default" and (mirrors or fingerprints):
        raise ValueError("source-build-default cannot declare unused trust roots")
    if mode != "source-build-default":
        if not mirrors or not fingerprints:
            raise ValueError("signed buildcache mode requires mirrors and trust roots")
        if any(not isinstance(url, str) or not url.startswith("https://") for url in mirrors):
            raise ValueError("buildcache mirrors must use HTTPS")
        if any(
            not isinstance(value, str) or re.fullmatch(r"[0-9A-F]{40,64}", value) is None
            for value in fingerprints
        ):
            raise ValueError("invalid buildcache signing-key fingerprint")


def validate_manifest(
    manifest_path: Path,
    *,
    repository_root: Path,
    require_release: bool = False,
) -> dict[str, Any]:
    """Validate a candidate or release manifest and checked repository inputs."""
    manifest = json.loads(manifest_path.read_text())
    _require_exact_keys(
        manifest,
        (
            "schema_version",
            "manifest_state",
            "release_id",
            "created_at",
            "sources",
            "spack",
            "profiles",
            "buildcache",
            "acceptance",
        ),
        "manifest",
    )
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported release manifest schema")
    state = manifest.get("manifest_state")
    if state not in MANIFEST_STATES:
        raise ValueError(f"unsupported manifest state: {state}")
    release_required = require_release or state == "release"
    if require_release and state != "release":
        raise ValueError("release-ready validation requires manifest_state=release")
    _require_string(manifest.get("release_id"), "release_id")
    created_at = _require_string(manifest.get("created_at"), "created_at")
    try:
        timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("created_at must be an ISO-8601 timestamp") from error
    if timestamp.tzinfo is None:
        raise ValueError("created_at must include a timezone")

    sources = _require_mapping(manifest.get("sources"), "sources")
    if set(sources) != set(REQUIRED_SOURCES):
        raise ValueError(f"sources must contain exactly {REQUIRED_SOURCES}")
    for name in REQUIRED_SOURCES:
        _validate_source(
            repository_root,
            name,
            sources[name],
            require_release=release_required,
        )

    spack = _require_mapping(manifest.get("spack"), "spack")
    _require_exact_keys(spack, ("version", "commit"), "spack")
    _require_string(spack.get("version"), "spack.version")
    spack_commit = _require_string(spack.get("commit"), "spack.commit")
    if _GIT_COMMIT.fullmatch(spack_commit) is None:
        raise ValueError("spack.commit must be a full lowercase Git SHA")

    profiles = _require_mapping(manifest.get("profiles"), "profiles")
    if set(profiles) != set(REQUIRED_PROFILES):
        raise ValueError(f"profiles must contain exactly {REQUIRED_PROFILES}")
    for name in REQUIRED_PROFILES:
        _validate_profile(
            repository_root,
            name,
            profiles[name],
            require_release=release_required,
        )

    _validate_buildcache(manifest.get("buildcache"))
    acceptance = _require_mapping(manifest.get("acceptance"), "acceptance")
    _require_exact_keys(
        acceptance,
        ("status", "evidence", "remaining"),
        "acceptance",
    )
    status = acceptance.get("status")
    if status not in ("incomplete", "accepted"):
        raise ValueError("acceptance.status must be incomplete or accepted")
    evidence = acceptance.get("evidence")
    remaining = acceptance.get("remaining")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("acceptance.evidence must be a non-empty array")
    if not isinstance(remaining, list):
        raise ValueError("acceptance.remaining must be an array")
    if any(not isinstance(item, str) or not item for item in remaining):
        raise ValueError("acceptance.remaining entries must be non-empty strings")
    for index, value in enumerate(evidence):
        path = _repository_path(
            repository_root,
            value,
            f"acceptance.evidence[{index}]",
        )
        if not path.is_file():
            raise FileNotFoundError(path)
    if release_required and (status != "accepted" or remaining):
        raise ValueError("release manifest requires accepted evidence with no remainder")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    source_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=source_root / "spack/release/development-candidate.json",
    )
    parser.add_argument("--repository-root", type=Path, default=source_root)
    parser.add_argument(
        "--require-release",
        action="store_true",
        help="reject a valid but incomplete development candidate",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = validate_manifest(
        args.manifest.expanduser().resolve(),
        repository_root=args.repository_root.expanduser().resolve(),
        require_release=args.require_release,
    )
    print(
        "release manifest valid: "
        f"state={manifest['manifest_state']} "
        f"sources={len(manifest['sources'])} "
        f"profiles={len(manifest['profiles'])} "
        f"release_ready={manifest['manifest_state'] == 'release'}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"release manifest invalid: {error}", file=sys.stderr)
        raise SystemExit(1) from None
