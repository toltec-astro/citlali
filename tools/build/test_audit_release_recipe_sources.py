from __future__ import annotations

import unittest

from audit_release_recipe_sources import audit_recipe, parse_recipe


COMMIT = "a" * 40
REPOSITORY_URL = "https://github.com/toltec-astro/example.git"


def recipe(source: str) -> str:
    return f'''\
from spack.package import version

class Example:
    homepage = "https://github.com/toltec-astro/example"
    {source}
'''


class ReleaseRecipeSourceAuditTest(unittest.TestCase):
    def audit(self, source: str):
        return audit_recipe(
            repository="example",
            repository_url=REPOSITORY_URL,
            expected_commit=COMMIT,
            recipe_commit="c" * 40,
            package="example",
            recipe_path="spack_repo/example/package.py",
            recipe_text=recipe(source),
        )

    def test_parses_literal_git_and_version_sources(self) -> None:
        git_url, versions = parse_recipe(
            recipe(
                f'''git = "{REPOSITORY_URL}"
    version("1.0", commit="{COMMIT}")'''
            ),
            filename="package.py",
        )
        self.assertEqual(git_url, REPOSITORY_URL)
        self.assertEqual(versions[0].commit, COMMIT)

    def test_accepts_exact_git_commit(self) -> None:
        result = self.audit(
            f'''git = "{REPOSITORY_URL}"
    version("1.0", commit="{COMMIT}")'''
        )
        self.assertTrue(result.accepted)

    def test_rejects_commit_from_wrong_repository(self) -> None:
        result = self.audit(
            f'''git = "https://github.com/other/example.git"
    version("1.0", commit="{COMMIT}")'''
        )
        self.assertFalse(result.accepted)

    def test_accepts_checked_commit_archive(self) -> None:
        result = self.audit(
            f'''version(
        "1.0",
        url="https://github.com/toltec-astro/example/archive/{COMMIT}.tar.gz",
        sha256="{'b' * 64}",
    )'''
        )
        self.assertTrue(result.accepted)

    def test_rejects_unchecked_archive(self) -> None:
        result = self.audit(
            f'''version(
        "1.0",
        url="https://github.com/toltec-astro/example/archive/{COMMIT}.tar.gz",
    )'''
        )
        self.assertFalse(result.accepted)

    def test_rejects_unbound_version(self) -> None:
        result = self.audit('version("1.0")')
        self.assertFalse(result.accepted)
        self.assertIn("source=unbound", result.reason)


if __name__ == "__main__":
    unittest.main()
