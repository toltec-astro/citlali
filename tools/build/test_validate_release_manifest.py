from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from validate_release_manifest import SCHEMA_VERSION, validate_manifest


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReleaseManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.environment_paths = {}
        for profile in ("macos-llvm20", "unity-gcc13"):
            path = self.root / f"{profile}.yaml"
            path.write_text("spack:\n  specs: [citlali]\n")
            self.environment_paths[profile] = path
        self.evidence = self.root / "evidence.txt"
        self.evidence.write_text("accepted evidence\n")
        self.recipe_audit = self.root / "recipe-audit.json"
        self.manifest_path = self.root / "manifest.json"
        self.manifest = self.make_candidate()
        self.write_recipe_audit(accepted=False)
        self.write_manifest()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def make_candidate(self) -> dict:
        sources = {}
        for index, name in enumerate(
            ("citlali", "tula_cmake", "tula", "kidscpp"),
            start=1,
        ):
            sources[name] = {
                "repository_url": f"https://github.com/toltec-astro/{name}.git",
                "review_branch": "release-test",
                "source_commit": f"{index:x}" * 40,
                "source_archive": None,
                "recipe_commit": f"{index + 4:x}" * 40,
                "recipe_archive": None,
            }
        profiles = {}
        for profile, path in self.environment_paths.items():
            profiles[profile] = {
                "platform": profile,
                "compiler": "test compiler",
                "development_environment": {
                    "path": path.name,
                    "sha256": sha256(path),
                },
                "observed_development_lock": {
                    "sha256": "a" * 64,
                    "root_dag_hash": "b" * 32,
                    "portable": False,
                },
                "release_environment": None,
                "release_lock": None,
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "manifest_state": "development-candidate",
            "release_id": "test-candidate",
            "created_at": "2026-08-14T16:00:00Z",
            "sources": sources,
            "spack": {"version": "1.2.2", "commit": "a" * 40},
            "profiles": profiles,
            "recipe_source_audit": {},
            "buildcache": {
                "mode": "source-build-default",
                "source_build_fallback": True,
                "mirrors": [],
                "trusted_signing_key_fingerprints": [],
            },
            "acceptance": {
                "status": "incomplete",
                "evidence": [self.evidence.name],
                "remaining": ["release inputs are not frozen"],
            },
        }

    def write_manifest(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest))

    def write_recipe_audit(self, *, accepted: bool) -> None:
        packages = []
        for name, source in self.manifest["sources"].items():
            packages.append(
                {
                    "repository": name,
                    "package": name,
                    "source_commit": source["source_commit"],
                    "recipe_commit": source["recipe_commit"],
                    "recipe_path": f"{name}/package.py",
                    "accepted": accepted,
                    "reason": "test audit",
                    "versions": [],
                }
            )
        failure_count = 0 if accepted else len(packages)
        self.recipe_audit.write_text(
            json.dumps(
                {
                    "schema_version": "citlali-release-recipe-audit-v1",
                    "release_id": self.manifest["release_id"],
                    "manifest_state": self.manifest["manifest_state"],
                    "status": "accepted" if accepted else "blocked",
                    "package_count": len(packages),
                    "failure_count": failure_count,
                    "packages": packages,
                }
            )
        )
        self.manifest["recipe_source_audit"] = {
            "path": self.recipe_audit.name,
            "sha256": sha256(self.recipe_audit),
        }

    def make_release(self, *, include_develop_path: bool = False) -> None:
        self.manifest["manifest_state"] = "release"
        self.manifest["release_id"] = "citlali-test-release"
        for name, source in self.manifest["sources"].items():
            for identity in ("source", "recipe"):
                commit = source[f"{identity}_commit"]
                archive = self.root / f"{name}-{identity}-{commit}.tar.gz"
                archive.write_bytes(f"{identity} archive for {name}\n".encode())
                source[f"{identity}_archive"] = {
                    "path": archive.name,
                    "url": (
                        f"https://github.com/toltec-astro/{name}/archive/"
                        f"{commit}.tar.gz"
                    ),
                    "sha256": sha256(archive),
                    "immutable": True,
                }
        for profile, record in self.manifest["profiles"].items():
            environment = self.root / f"{profile}-release.yaml"
            environment.write_text("spack:\n  specs: [citlali]\n")
            root_hash = "d" * 32
            concrete_spec = {"name": "citlali", "hash": root_hash}
            if include_develop_path:
                concrete_spec["parameters"] = {"dev_path": "/tmp/source"}
            lock = self.root / f"{profile}.lock"
            lock.write_text(
                json.dumps(
                    {
                        "roots": [{"hash": root_hash, "spec": "citlali@4.0.0"}],
                        "concrete_specs": {root_hash: concrete_spec},
                    }
                )
            )
            record["release_environment"] = {
                "path": environment.name,
                "sha256": sha256(environment),
            }
            record["release_lock"] = {
                "path": lock.name,
                "sha256": sha256(lock),
                "root_dag_hash": root_hash,
            }
        self.manifest["acceptance"] = {
            "status": "accepted",
            "evidence": [self.evidence.name],
            "remaining": [],
        }
        self.write_recipe_audit(accepted=True)
        self.write_manifest()

    def test_accepts_development_candidate(self) -> None:
        result = validate_manifest(
            self.manifest_path,
            repository_root=self.root,
        )
        self.assertEqual(result["manifest_state"], "development-candidate")

    def test_release_gate_rejects_candidate(self) -> None:
        with self.assertRaisesRegex(ValueError, "manifest_state=release"):
            validate_manifest(
                self.manifest_path,
                repository_root=self.root,
                require_release=True,
            )

    def test_rejects_checked_file_drift(self) -> None:
        self.environment_paths["macos-llvm20"].write_text("changed\n")
        with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            validate_manifest(self.manifest_path, repository_root=self.root)

    def test_accepts_complete_release(self) -> None:
        self.make_release()
        result = validate_manifest(
            self.manifest_path,
            repository_root=self.root,
            require_release=True,
        )
        self.assertEqual(result["acceptance"]["status"], "accepted")

    def test_release_rejects_develop_paths(self) -> None:
        self.make_release(include_develop_path=True)
        with self.assertRaisesRegex(ValueError, "develop paths"):
            validate_manifest(
                self.manifest_path,
                repository_root=self.root,
                require_release=True,
            )

    def test_release_rejects_archive_not_bound_to_commit(self) -> None:
        self.make_release()
        self.manifest["sources"]["citlali"]["source_archive"]["url"] = (
            "https://github.com/toltec-astro/citlali/archive/main.tar.gz"
        )
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "declared commit"):
            validate_manifest(
                self.manifest_path,
                repository_root=self.root,
                require_release=True,
            )

    def test_release_rejects_recipe_archive_not_bound_to_recipe_commit(self) -> None:
        self.make_release()
        self.manifest["sources"]["citlali"]["recipe_archive"]["url"] = (
            "https://github.com/toltec-astro/citlali/archive/main.tar.gz"
        )
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "declared commit"):
            validate_manifest(
                self.manifest_path,
                repository_root=self.root,
                require_release=True,
            )

    def test_release_rejects_archive_from_wrong_repository(self) -> None:
        self.make_release()
        source = self.manifest["sources"]["citlali"]
        source["source_archive"]["url"] = (
            "https://github.com/other/citlali/archive/"
            f"{source['source_commit']}.tar.gz"
        )
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "declared repository"):
            validate_manifest(
                self.manifest_path,
                repository_root=self.root,
                require_release=True,
            )

    def test_release_rejects_blocked_recipe_source_audit(self) -> None:
        self.make_release()
        self.write_recipe_audit(accepted=False)
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "accepted recipe source audit"):
            validate_manifest(
                self.manifest_path,
                repository_root=self.root,
                require_release=True,
            )

    def test_rejects_unpaired_release_inputs(self) -> None:
        record = self.manifest["profiles"]["macos-llvm20"]
        record["release_environment"] = record["development_environment"]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "must be paired"):
            validate_manifest(self.manifest_path, repository_root=self.root)


if __name__ == "__main__":
    unittest.main()
