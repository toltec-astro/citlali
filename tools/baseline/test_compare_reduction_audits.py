from __future__ import annotations

import unittest
from unittest import mock

from tools.baseline import compare_reduction_audits as compare


class ReductionAuditComparisonTest(unittest.TestCase):
    def test_parser_accepts_candidate_provenance_requirements(self) -> None:
        args = compare.parse_args([
            "baseline",
            "candidate",
            "--require-candidate-coadd-provenance",
            "--require-candidate-noise-products-provenance",
            "--require-candidate-pointing-provenance",
        ])

        self.assertTrue(args.require_candidate_coadd_provenance)
        self.assertTrue(args.require_candidate_noise_products_provenance)
        self.assertTrue(args.require_candidate_pointing_provenance)

    def test_audit_for_forwards_candidate_provenance_requirements(self) -> None:
        with mock.patch.object(
            compare.audit_reduction_run, "build_audit", return_value={}
        ) as build_audit:
            compare.audit_for(
                "candidate",
                "science",
                "refactor",
                12,
                require_coadd_provenance=True,
                require_noise_products_provenance=True,
                require_pointing_provenance=True,
            )

        args = build_audit.call_args.args[0]
        self.assertTrue(args.require_coadd_provenance)
        self.assertTrue(args.require_noise_products_provenance)
        self.assertTrue(args.require_pointing_provenance)

    def test_compare_forwards_candidate_only_requirements(self) -> None:
        args = compare.parse_args([
            "baseline",
            "candidate",
            "--require-candidate-coadd-provenance",
            "--require-candidate-noise-products-provenance",
            "--require-candidate-pointing-provenance",
        ])
        empty_audit = {
            "log": {},
            "products": {},
            "provenance": {},
        }
        with mock.patch.object(
            compare, "audit_for", side_effect=[empty_audit, empty_audit]
        ) as audit_for:
            compare.compare_audits(args)

        candidate_call = audit_for.call_args_list[1]
        self.assertTrue(candidate_call.kwargs["require_coadd_provenance"])
        self.assertTrue(
            candidate_call.kwargs["require_noise_products_provenance"]
        )
        self.assertTrue(
            candidate_call.kwargs["require_pointing_provenance"]
        )


if __name__ == "__main__":
    unittest.main()
