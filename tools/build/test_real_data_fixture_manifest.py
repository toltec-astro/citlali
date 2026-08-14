import json
import tempfile
import unittest
from pathlib import Path

from test_spack_kidscpp import (
    FIXTURE_MANIFEST_SCHEMA,
    _sha256,
    validate_fixture_manifest,
)


class RealDataFixtureManifestTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.fixture = self.root / "toltec0_fixture.nc"
        self.fixture.write_bytes(b"portable raw timestream fixture")
        self.manifest = self.root / "manifest.json"
        self.write_manifest()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_manifest(self, **file_overrides):
        file_identity = {
            "basename": self.fixture.name,
            "size_bytes": self.fixture.stat().st_size,
            "sha256": _sha256(self.fixture),
        }
        file_identity.update(file_overrides)
        self.manifest.write_text(
            json.dumps(
                {
                    "schema_version": FIXTURE_MANIFEST_SCHEMA,
                    "fixture_id": "test-fixture-v1",
                    "file": file_identity,
                }
            )
        )

    def test_accepts_matching_content_identity(self):
        result = validate_fixture_manifest(self.fixture, self.manifest)
        self.assertEqual(result["fixture_id"], "test-fixture-v1")

    def test_rejects_changed_payload(self):
        self.fixture.write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            validate_fixture_manifest(self.fixture, self.manifest)

    def test_rejects_wrong_filename(self):
        self.write_manifest(basename="another-file.nc")
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            validate_fixture_manifest(self.fixture, self.manifest)

    def test_rejects_incomplete_manifest(self):
        self.manifest.write_text(
            json.dumps(
                {
                    "schema_version": FIXTURE_MANIFEST_SCHEMA,
                    "fixture_id": "test-fixture-v1",
                }
            )
        )
        with self.assertRaisesRegex(RuntimeError, "missing file identity"):
            validate_fixture_manifest(self.fixture, self.manifest)


if __name__ == "__main__":
    unittest.main()
