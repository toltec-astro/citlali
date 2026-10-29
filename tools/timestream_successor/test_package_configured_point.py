"""Real launcher subprocess checks with a small committed fake source tree."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest

from package_configured_point import DATA_FILES, REQUIRED_TOOLS, TOOL_DIR, digest, package


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    git(root, "init", "-q")
    for relative in [*(TOOL_DIR / name for name in REQUIRED_TOOLS), *DATA_FILES, TOOL_DIR / "transitive_import.py"]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n")
    (root / TOOL_DIR / "configured_point_cli.py").write_text(
        'import json,sys\nfrom transitive_import import MARKER\n'
        'print(json.dumps({"args":sys.argv[1:],"marker":MARKER}))\n')
    (root / TOOL_DIR / "transitive_import.py").write_text('MARKER="bundled-source"\n')
    git(root, "add", ".")
    git(root, "-c", "user.name=Runtime test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
    revision = git(root, "rev-parse", "HEAD")
    native = tmp_path / "fake native"
    native.write_text("#!/bin/sh\nprintf '%s\\n' " + shlex.quote("citlali " + revision[:9]) + "\n")
    native.chmod(0o755)
    # Exercise the absolute interpreter path with spaces through a small
    # transparent launcher; its actual Python executable is pinned separately.
    python = tmp_path / "python environment" / "python"
    python.parent.mkdir()
    python.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' "$@"\n')
    python.chmod(0o755)
    return root, native, python


def invoke(launcher, *args):
    return subprocess.run([str(launcher), *args], capture_output=True, text=True)


def test_copied_launcher_finds_exact_bundle_and_preserves_arguments(source, tmp_path):
    root, native, python = source
    destination = tmp_path / "runtime"
    launcher = package(root, native, python, destination)
    copied = tmp_path / "project" / "copied executable"
    copied.parent.mkdir()
    shutil.copy2(launcher, copied)
    result = invoke(copied, "--version", "argument with spaces")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"args": ["--native", str(destination / "bin/citlali-native"),
                                                  "--version", "argument with spaces"], "marker": "bundled-source"}
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["source"]["revision"] == git(root, "rev-parse", "HEAD")
    assert manifest["python"]["path"] == str(python)
    assert manifest["python"]["sha256"] == digest(python)
    assert manifest["python"]["interpreter_sha256"] == digest(sys.executable)
    assert manifest["python"]["dependency_roots"]
    assert manifest["python"]["package_contents_pinned"] is False
    copied_members = {entry["path"] for entry in manifest["files"]}
    assert "tools/timestream_successor/transitive_import.py" in copied_members
    assert "tools/timestream_successor/configured_point_weighted.py" in copied_members
    assert "tools/timestream_successor/configured_point_feedback.py" in copied_members
    assert "data/configured_point_v2.yaml" in copied_members
    assert {str(path) for path in DATA_FILES} <= copied_members
    assert "bin/citlali-native" in copied_members
    assert not list(destination.rglob("__pycache__"))


@pytest.mark.parametrize("relative", ["manifest.json", "bin/citlali-native", "tools/timestream_successor/transitive_import.py",
                                      "data/config.yaml", "data/configured_point_v2.yaml",
                                      "tools/timestream_successor/configured_point_weighted.py",
                                      "tools/timestream_successor/configured_point_feedback.py",
                                      "data/development/bright_point_base_v1/a1400-rtc.json"])
def test_snapshot_launcher_rejects_changed_manifest_or_payload(source, tmp_path, relative):
    root, native, python = source
    destination = tmp_path / "runtime"
    launcher = package(root, native, python, destination)
    target = destination / relative
    target.write_bytes(target.read_bytes() + b"\n# changed\n")
    result = invoke(launcher, "--version")
    assert result.returncode == 2
    assert "POINT runtime verification failed" in result.stderr
    assert result.stdout == ""


def test_snapshot_launcher_rejects_missing_payload(source, tmp_path):
    root, native, python = source
    destination = tmp_path / "runtime"
    launcher = package(root, native, python, destination)
    (destination / "data/config.yaml").unlink()
    result = invoke(launcher)
    assert result.returncode == 2 and "runtime member changed" in result.stderr


def test_changed_external_python_wrapper_is_rejected(source, tmp_path):
    root, native, python = source
    launcher = package(root, native, python, tmp_path / "runtime")
    python.write_text(python.read_text() + "\n# changed dependency\n")
    result = invoke(launcher)
    assert result.returncode == 2 and "external Python executable changed" in result.stderr


def test_source_can_change_after_packaging_without_changing_frozen_payload(source, tmp_path):
    root, native, python = source
    launcher = package(root, native, python, tmp_path / "runtime")
    (root / TOOL_DIR / "transitive_import.py").write_text('MARKER="changed source"\n')
    native.write_text("changed source binary")
    result = invoke(launcher)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["marker"] == "bundled-source"


def test_dirty_tracked_source_fails_without_creating_destination(source, tmp_path):
    root, native, python = source
    (root / TOOL_DIR / "configured_point_cli.py").write_text("# uncommitted change\n")
    destination = tmp_path / "runtime"
    with pytest.raises(ValueError, match="clean tracked"):
        package(root, native, python, destination)
    assert not destination.exists()


def test_revision_mismatch_fails_without_creating_destination(source, tmp_path):
    root, native, python = source
    native.write_text("#!/bin/sh\nprintf '%s\\n' 'citlali 000000000'\n")
    destination = tmp_path / "runtime"
    with pytest.raises(ValueError, match="does not match"):
        package(root, native, python, destination)
    assert not destination.exists()


def test_existing_destination_is_preserved(source, tmp_path):
    root, native, python = source
    destination = tmp_path / "runtime"
    destination.mkdir()
    marker = destination / "reference"
    marker.write_text("preserve me")
    with pytest.raises(ValueError, match="already exists"):
        package(root, native, python, destination)
    assert marker.read_text() == "preserve me" and list(destination.iterdir()) == [marker]


def test_untracked_helpers_are_not_accidentally_bundled(source, tmp_path):
    root, native, python = source
    (root / TOOL_DIR / "uncommitted.py").write_text("raise RuntimeError('uncommitted')")
    destination = tmp_path / "runtime"
    package(root, native, python, destination)
    assert not (destination / TOOL_DIR / "uncommitted.py").exists()


def test_uncommitted_required_helper_is_rejected(source, tmp_path):
    root, native, python = source
    git(root, "rm", "--cached", str(TOOL_DIR / "configured_point_inputs.py"))
    git(root, "-c", "user.name=Runtime test", "-c", "user.email=test@example.invalid", "commit", "-qm", "remove tracked helper")
    revision = git(root, "rev-parse", "HEAD")
    native.write_text("#!/bin/sh\nprintf '%s\\n' " + shlex.quote("citlali " + revision[:9]) + "\n")
    with pytest.raises(ValueError, match="not committed"):
        package(root, native, python, tmp_path / "runtime")
