"""Curated installation contract; no observation processing."""
import ast
import importlib
from pathlib import Path
import tomllib

import pytest

from configured_point_runtime import RUNTIME_MODULES, data_root


def test_curated_modules_cover_source_imports():
    root = Path(__file__).parent
    available = {p.stem for p in root.glob("*.py")}
    assert not any(name.startswith("test_") for name in RUNTIME_MODULES)
    for name in RUNTIME_MODULES:
        tree = ast.parse((root / (name + ".py")).read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert imported & available <= set(RUNTIME_MODULES), name
        importlib.import_module(name)


def test_wheel_and_freeze_share_module_inventory():
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    assert set(project["tool"]["setuptools"]["py-modules"]) == set(RUNTIME_MODULES) | {"package_configured_point"}


def test_resource_lookup_uses_installed_prefix_and_rejects_missing(tmp_path):
    module = tmp_path / "lib/python3.13/site-packages/configured_point_runtime.py"
    module.parent.mkdir(parents=True)
    module.touch()
    with pytest.raises(FileNotFoundError):
        data_root(module, tmp_path)
    data = tmp_path / "share/citlali-point/data"
    data.mkdir(parents=True)
    (data / "configured_point_v1.yaml").write_text("{}")
    assert data_root(module, tmp_path) == data


@pytest.mark.parametrize("prefix,platform", [("macos-arm64", "macOS arm64"), ("linux-x86_64", "Linux x86_64")])
def test_lock_artifacts_match_declared_versions_and_platform(prefix, platform):
    import json
    import re

    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / f"requirements/{prefix}-cp313-artifacts.json").read_text())
    assert manifest["python"] == "CPython 3.13"
    assert manifest["platform"] == platform
    artifacts = {row["name"].lower(): row for row in manifest["artifacts"]}
    locked = set()
    for path in (root / "requirements").glob(f"{prefix}-cp313-*.lock"):
        for line in path.read_text().splitlines():
            if not line or line.startswith(("#", "-r ")):
                continue
            match = re.fullmatch(r"([^=]+)==([^ ]+) --hash=sha256:([a-f0-9]{64})", line)
            assert match is not None, line
            name, version, sha = match.groups()
            row = artifacts[name.lower()]
            assert (row["version"], row["sha256"]) == (version, sha)
            locked.add(name.lower())
    assert locked == set(artifacts)
