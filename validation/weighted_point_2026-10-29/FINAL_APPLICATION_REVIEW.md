# Independent exact-SHA application review: repaired candidate

Candidate: `331ff29ea1720b1454e671cbbab93814d8df8321`
Tree: `c512f8e68b7dba3255467a9b06d06903aad43185`
Parent: `df550415789c5777d31fa37e59989021f4600f50`
Original bounded work-order base: `c6ea76247213cdb8ce51c9ed935a9744b15beb9a`

Verdict: **pass with recorded limitations in the application review scope**. No actionable application findings.

The reviewer verified HEAD, parent, tree, the full changed-path inventory and the repair diff. The working tree is clean. The final child differs from repaired candidate `df550415789c5777d31fa37e59989021f4600f50` only by removing trailing spaces from four lines of `validation/weighted_point_2026-10-29/INITIAL_APPLICATION_REVIEW.md`; `git diff --ignore-space-at-eol --exit-code` passes. No application, native or test bytes changed in this final child.

The repaired parent was separately inspected against `4f7d31957d2135601145c63387e310622fe16d64`. All previously reviewed Python application/configuration/feedback/publication/packaging code, v2 YAML and native application/FITS boundary code are unchanged. That repair delta is confined to the optical bootstrap owner, its method/limitation identity, two numerical regressions and review/evidence documentation.

The new `point-optical-heldout-network-block-panel-bootstrap-v2` identity propagates through native health publication using `PointOpticalEvidence::method`. Native restoration requires that exact current method string, so predecessor v1 health evidence cannot be silently restored. The Python health summary copies the producer method and uncertainty-limitations strings without a stale v1 expectation. The clarified meaning of `informative_blocks` is included in the required producer uncertainty text and does not alter the application schema or joins.

The prior application review's scope and limitations remain applicable. Its 105 synthetic Python tests passed at the unchanged application source; no redundant tests were rerun. Numerical correctness of the bootstrap repair is delegated to the separate independent native reviewer and is not approved by this application-only verdict. Neither review establishes observational behavior or statistical coverage.

Three separate dispositions:

1. **Scientific and behavioral conformance: pass with recorded limitations in the application scope.** The repaired method identity/limitation text is transported and checked consistently. No application contract regression found; numerical and owner-trial acceptance remain separate.
2. **Architectural conformity and ownership: pass in the application scope.** The repair remains in the existing optical scientific owner; no application ownership, lifecycle, feedback or Engine change.
3. **Repository, branch and evidence hygiene: pass with recorded limitations.** Exact parentage and clean state verified. The failed predecessor review and reproducer are preserved and the candidate remains unqualified for integration/production. No source edits, build, observational processing, reductions, Unity access, push or cleanup occurred in this re-review.

Reference: `/tmp/weighted-point-app-review.md` and `/tmp/weighted-point-app-review-pytest.log`.
