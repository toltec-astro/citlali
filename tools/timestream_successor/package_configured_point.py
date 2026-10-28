#!/usr/bin/env python3
"""Freeze a verified local POINT runtime and a TolProj-copyable launcher.

The bundle retains an absolute dependency on the selected existing Python
environment and on the native executable's shared libraries. It is not a
portable installation. No source checkout or shared environment is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess


SCHEMA = "citlali-configured-point-runtime-v1"
TOOL_DIR = Path("tools/timestream_successor")
REQUIRED_TOOLS = ("configured_point_cli.py", "configured_point_config.py",
                  "configured_point_inputs.py", "configured_point_products.py", "configured_point_drops.py")
DATA_FILES = (Path("data/config.yaml"), Path("data/configured_point_v1.yaml"),
              *(Path("data/development/bright_point_base_v1") / f"{a}-rtc.json"
                for a in ("a1100", "a1400", "a2000")),
              Path("data/development/bright_point_base_v1/map-owner.json"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _git(repository, *args):
    return subprocess.check_output(["git", "-C", str(repository), *args], text=True).strip()


def _revision(repository):
    revision = _git(repository, "rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("runtime packaging requires an exact committed source revision")
    if _git(repository, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("runtime packaging requires a clean tracked source tree")
    return revision


def _python_identity(python):
    # Query metadata without importing scientific packages or changing them.
    script = """
import importlib.metadata,json,os,sys
packages={}
for name in ('numpy','scipy','astropy','netCDF4','psutil','PyYAML'):
    try: packages[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: packages[name]=None
print(json.dumps(dict(version=sys.version,executable=sys.executable,
    prefix=sys.prefix,base_prefix=sys.base_prefix,
    dependency_roots=[os.path.abspath(p) for p in sys.path if p],packages=packages)))
"""
    result = subprocess.run([str(python), "-B", "-c", script], capture_output=True, text=True, check=True)
    identity = json.loads(result.stdout)
    if not isinstance(identity, dict) or not isinstance(identity.get("dependency_roots"), list) or not isinstance(identity.get("executable"), str):
        raise ValueError("Python environment identity query did not return the required record")
    identity.update(path=str(python), realpath=str(python.resolve()), sha256=digest(python),
                    interpreter_sha256=digest(identity["executable"]),
                    environment_is_external=True,
                    package_contents_pinned=False)
    return identity


def _launcher(root, manifest_sha256, python):
    # The launcher can be copied alone by TolProj: neither __file__ nor cwd is
    # used to locate its immutable runtime. Verification uses only stdlib.
    program = f'''import hashlib,json,os,pathlib,sys
ROOT=pathlib.Path({str(root)!r})
EXPECTED={manifest_sha256!r}
PYTHON={str(python)!r}
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()
try:
    manifest=ROOT/"manifest.json"
    if sha(manifest)!=EXPECTED: raise ValueError("runtime manifest changed")
    data=json.loads(manifest.read_text())
    if data.get("schema")!={SCHEMA!r}: raise ValueError("unsupported runtime manifest")
    if data["python"]["path"]!=PYTHON or sha(PYTHON)!=data["python"]["sha256"]:
        raise ValueError("external Python executable changed")
    if sha(sys.executable)!=data["python"]["interpreter_sha256"]:
        raise ValueError("external Python interpreter changed")
    for entry in data["files"]:
        relative=pathlib.PurePosixPath(entry["path"])
        if relative.is_absolute() or ".." in relative.parts: raise ValueError("unsafe runtime member")
        path=ROOT/pathlib.Path(relative)
        if not path.resolve().is_relative_to(ROOT.resolve()): raise ValueError("runtime member escapes bundle")
        if not path.is_file() or path.stat().st_size!=entry["bytes"] or sha(path)!=entry["sha256"]:
            raise ValueError("runtime member changed: "+entry["path"])
except (OSError,ValueError,KeyError,TypeError) as exc:
    print("POINT runtime verification failed: "+str(exc),file=sys.stderr)
    raise SystemExit(2)
os.execv(PYTHON,[PYTHON,"-B",str(ROOT/"tools/timestream_successor/configured_point_cli.py"),
    "--native",str(ROOT/"bin/citlali-native"),*sys.argv[1:]])
'''
    # A shell front door permits an absolute Python path containing spaces.
    # shlex.quote protects literal paths/code; user arguments stay separate.
    return "#!/bin/sh\nexec " + shlex.quote(str(python)) + " -B -c " + shlex.quote(program) + ' "$@"\n'


def package(repository: Path, native: Path, python: Path, destination: Path) -> Path:
    """Create one fresh bundle and return its executable launcher path."""
    repository = Path(repository).resolve()
    native = Path(native).resolve()
    # Keep the venv's invocation path: resolving its Python symlink would lose
    # the environment's prefix and installed scientific dependencies.
    python = Path(os.path.abspath(Path(python).expanduser()))
    destination = Path(destination).expanduser().resolve()
    if destination.exists():
        raise ValueError("runtime destination already exists; choose a fresh directory")
    for label, executable in (("native", native), ("Python", python)):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError(f"{label} executable is unavailable: {executable}")
    revision = _revision(repository)
    version = subprocess.check_output([str(native), "--version"], text=True, stderr=subprocess.STDOUT).strip()
    if not re.search(r"(?<![0-9a-f])" + revision[:9] + r"[0-9a-f]*(?![0-9a-f])", version):
        raise ValueError("native --version does not match the committed source revision")
    tracked = set(_git(repository, "ls-files", "-z").split("\0"))
    sources = sorted(Path(name) for name in tracked
                     if name.startswith(TOOL_DIR.as_posix() + "/") and name.endswith(".py"))
    required = {TOOL_DIR / name for name in REQUIRED_TOOLS} | set(DATA_FILES)
    missing = sorted(str(path) for path in required if path.as_posix() not in tracked)
    if missing:
        raise ValueError("required runtime sources are not committed: " + ", ".join(missing))
    sources = sorted(set(sources) | set(DATA_FILES))
    for relative in sources:
        path = repository / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(repository):
            raise ValueError(f"runtime source must be a regular file in the checkout: {relative}")
    python_identity = _python_identity(python)
    native_sha256 = digest(native)
    destination.mkdir(parents=True, exist_ok=False)
    members = []
    for relative in sources:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repository / relative, target)
        members.append(dict(path=relative.as_posix(), sha256=digest(target), bytes=target.stat().st_size))
    native_target = destination / "bin/citlali-native"
    native_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(native, native_target)
    if digest(native_target) != native_sha256:
        raise ValueError("native executable changed while packaging; incomplete bundle retained for inspection")
    members.append(dict(path="bin/citlali-native", sha256=native_sha256, bytes=native_target.stat().st_size))
    if _revision(repository) != revision:
        raise ValueError("source revision changed while packaging; incomplete bundle retained for inspection")
    manifest = dict(schema=SCHEMA, source=dict(repository=str(repository), revision=revision, tracked_clean=True),
                    native=dict(version=version, source_path=str(native), bundled_path="bin/citlali-native"),
                    python=python_identity, files=members,
                    limitations=["Absolute bundle location is embedded in its launcher.",
                                 "External Python package contents and native shared libraries are not bundled or pinned.",
                                 "Use on the recorded local host; no Unity/Spack or portable deployment claim."])
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    launcher = destination / "bin/citlali"
    launcher.write_text(_launcher(destination, digest(manifest_path), python))
    launcher.chmod(0o755)
    return launcher


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        launcher = package(args.repository, args.native, args.python, args.destination)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"POINT runtime packaging failed: {exc}\n")
    print(json.dumps(dict(launcher=str(launcher), launcher_sha256=digest(launcher),
                          manifest=str(launcher.parent.parent / "manifest.json")), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
