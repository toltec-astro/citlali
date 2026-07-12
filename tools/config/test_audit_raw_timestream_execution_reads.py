from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.config import audit_raw_timestream_execution_reads as audit


class RawTimestreamExecutionReadAuditTest(unittest.TestCase):
    def test_strips_includes_comments_and_literals(self) -> None:
        text = '''
        #include <citlali/core/timestream/rtc/rtcproc.h>
        // rtcproc.run_kernel
        const char *value = "rtcproc.run_kernel";
        bool enabled = rtcproc.run_kernel;
        '''
        stripped = audit.strip_non_code(text)
        self.assertEqual(stripped.count("rtcproc.run_kernel"), 1)
        self.assertNotIn("rtcproc.h", stripped)

    def test_classifies_migration_categories(self) -> None:
        self.assertEqual(
            audit.classify_access("run_kernel"), "raw_policy_read"
        )
        self.assertEqual(
            audit.classify_access("downsampler.factor"),
            "observation_state",
        )
        self.assertEqual(
            audit.classify_access("append_to_netcdf"),
            "executor_operation",
        )
        self.assertEqual(
            audit.classify_access("run_polarization"),
            "separate_polarimetry_domain",
        )
        self.assertEqual(
            audit.classify_access("new_policy"), "review_required"
        )

    def test_scans_code_and_excludes_compatibility_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir)
            source_root = repo_root / audit.SOURCE_ROOTS[0]
            source_root.mkdir(parents=True)
            (source_root / "stage.h").write_text(
                "bool enabled = rtcproc.run_kernel;\n"
            )
            excluded = repo_root / next(iter(audit.EXCLUDED_FILES))
            excluded.parent.mkdir(parents=True, exist_ok=True)
            excluded.write_text("bool enabled = rtcproc.untracked;\n")

            records = audit.scan_accesses(repo_root)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["access"], "run_kernel")
        self.assertEqual(records[0]["classification"], "raw_policy_read")


if __name__ == "__main__":
    unittest.main()
