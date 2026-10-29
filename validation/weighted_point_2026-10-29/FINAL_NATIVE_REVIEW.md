# Independent exact-SHA Tier 2 native follow-up review

Verdict: **pass with recorded limitations**. Initial scientific finding R1 is closed. No remaining actionable scientific/behavioral, architectural/ownership, or repository/evidence finding was identified.

- Candidate: `331ff29ea1720b1454e671cbbab93814d8df8321`.
- Tree: `c512f8e68b7dba3255467a9b06d06903aad43185`.
- Parent / numerical repair: `df550415789c5777d31fa37e59989021f4600f50` (tree `d54d1e2d4313d91d04899fd78eab178e7a4c7cb2`).
- Initial rejected implementation parent: `4f7d31957d2135601145c63387e310622fe16d64`.
- Original local work-order base: `c6ea76247213cdb8ce51c9ed935a9744b15beb9a`.
- Branch/worktree: `codex/configurable-point-20261028`, `/Users/gwilson/.codex/worktrees/configurable-point-20261028/citlali-refactor`.
- Initial and final staged, unstaged and untracked inventories were empty. Review was read-only; no build, source edit, observation processing, Unity access, push, integration or cleanup occurred.

## Authority and scope

This exact follow-up covers the correction to native optical bootstrap support, method identity, public Learn/Consider regressions and accompanying evidence records. It supplements the initial fresh-context native review preserved at `validation/weighted_point_2026-10-29/INITIAL_NATIVE_REVIEW.md`. All native weighted PTC, empirical MAP, FRUIT and adapter sources reviewed there remain unchanged in this repair commit. Governance, work-order scope, owner-selected 0.10 default and conditional-resampling authorization remain those read and recorded in that review.

The numerical repair delta consists of nine files. The final candidate changes only four trailing-whitespace pairs in the preserved application review relative to the numerical repair commit; its application/test sources are identical. Application source changes are limited to the optical implementation/header; numerical tests and validation records are the other changes. It neither widens the scientific policy nor alters an unrelated owner.

## Scientific and behavioral conformance: pass; R1 closed

At `src/citlali/core/pipeline/point_optical_response.cpp:104-116`, every block represented by fitted rows now contributes its frozen-weight moment record to the resampling population. Singleton and constant-reference blocks are no longer silently discarded. The existing minimum two contributing blocks and explicit `resample_rank_unavailable` failure remain. Thus the interval and full fit use the same paired-row population, and a non-identifiable required draw withholds the detector instead of producing a confident weak decision.

The public method is now `point-optical-heldout-network-block-panel-bootstrap-v2` (`include/citlali/core/pipeline/point_optical_response.h:82-89`). The adapter already compares persisted `method` against this owner constant before restoration (`src/citlali/cli/point_weighted_execution.h:48-54`), so predecessor v1 health evidence fails restoration. The uncertainty description and implementation record explicitly define contributing-block count and the rank-failure outcome. The numeric threshold, confidence level and bootstrap budget were not changed.

The two added public-native regressions substantively cover the failure:

- `BootstrapRetainsSparseAndConstantBlocksThatIdentifyTheFullResponse` exercises both singleton and constant-within-block support. It verifies near-unit fitted response, all 12 full-domain/six half-domain contributing blocks in every peer-panel check, and no false weak classification.
- `RankDeficientBlockResampleIsUnavailableAndDetectorUnresolved` verifies that two constant contributing blocks remain represented but a singular draw explicitly makes the interval unavailable and the final detector state unresolved.

The saved isolated old-filter mutant log confirms that restoring only the removed filtering behavior makes both new tests fail. In the first regression both support variants become `weak_response`, with old interval upper bounds approximately 0.03999 despite near-unit full response. This demonstrates that the tests catch the scientific fault rather than merely mirror the repaired code.

No additional scientific/behavioral finding arose from the repair review. The initial low-severity observation that the typed threshold API accepts values above one remains nonblocking because the configured application parser enforces the documented domain.

## Architectural conformity and ownership: pass

The bounded correction remains inside the optical evidence owner. Learn/Consider separation, immutable observation health, exact source bindings, PTC/FRUIT/MAP ownership, and the frozen `Engine` boundary are preserved. The method revision prevents ambiguous reuse of evidence produced by the old estimator. No cross-stage authority or new mutable synchronization was introduced.

## Repository, branch and evidence hygiene: pass

The supplied candidate, tree and parent match the clean worktree. The initial rejected review and synthetic reproducer are durably preserved, and documentation records the failed gate and bounded repair without claiming observational success.

The intermediate repair commit had four trailing-whitespace pairs in `validation/weighted_point_2026-10-29/INITIAL_APPLICATION_REVIEW.md:3-6`. The final candidate removes only those pairs. Reviewer verified the exact documentation-only diff and reran `git diff --check c6ea76247213cdb8ce51c9ed935a9744b15beb9a HEAD`, which passed. No additional numerical test run was needed for that whitespace-only descendant.

## Reproduced gates and limitations

Reviewer directly ran the existing `build/bin/citlali_point_optical_response_test`: **10/10 passed**, including both new regressions; no unexpected error-level output. Binary SHA-256: `bb565b7bd87591c7119a7b696a259c00d7d4bcbd7ff4098a160c053d34002368`.

Inspected, without rerunning, `/tmp/weighted-point-repaired-ctest.log`: 1,366 enabled tests passed, zero failures, one inherited disabled `MapFitterLifecycle.ExactProductSequence` test, 7.36 seconds. Inspected `/tmp/citlali-point-optical-old-block-filter-regression.log`: both new tests fail with the original filter reinstated. The initial review independently ran 55 native tests covering the unchanged weighted PTC, empirical MAP, FRUIT, AST and native-binding surfaces.

All results are supplemental Mac AppleClang/Homebrew fallback evidence. The test executable contains the repaired numerics but was built before the new commit identity; this review does not establish a matching CLI source/executable revision. The writer must finish the configured CLI rebuild and verify its revision against the final source before freezing runtimes. No Spack/Unity campaign or observational processing was performed.

Owner-run 152389/152418 trials remain the affected-mode gate. Actual detector decisions, contribution/coverage, weights, source response, trajectory and end-to-end runtime remain unvalidated by these engineering tests. The authorized conditional interval still makes no demonstrated repeated-sampling-coverage or observation-wide false-rejection claim. This review permits proceeding to source/executable verification and owner trial preparation within the work order; it does not authorize a push, canonical integration or production activation.
