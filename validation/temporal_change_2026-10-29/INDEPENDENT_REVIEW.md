# Independent exact-SHA review — temporal-change synthetic study

Date: 2026-10-29. Reviewer: independent fresh-context, read-only review agent.

**Verdict: pass with recorded limitations for preservation as experimental evidence after a failed scientific calibration gate. This is not approval of the temporal method for admission, detector restoration, map-safety bounds, integration, or production. No candidate repair is required for that preservation disposition.**

## Identity, authority, and scope

- Candidate: `a8b720e17ef5b72afc2824d55570ca0ec1e159b7`.
- Tree: `336402e30ec747a50a6bb1f10564351163f59677`.
- Exact parent/base: `9bcbfcf539355a36faf18c5114492fd56c600fb0`.
- Branch: `codex/configurable-point-20261028`.
- Worktree: `/Users/gwilson/.codex/worktrees/configurable-point-20261028/citlali-refactor`.
- Initial and final staged, unstaged, and untracked states: clean. Parent and tree independently verified. No repository files were modified.
- Authority sources read: repository AGENTS; Engineering Governance; Timestream Successor Governance, including the 2026-09-30 standing amendment; Review and Conformance; current status and its candidate delta; integration-ledger governance-effectiveness record; optical-response owner policy; scientific-contract charter opening and recovery boundary; candidate WORK_ORDER, METHOD, and REPORT; earlier synthetic fixture used by the counterexample replay; TolTEC context skill and authority/repository maps.
- The ledger records accepted governance `06a3ade51c1b3f38887295433d913811bf25cd14` and effectiveness record `77507836325eff9f469062d5884481ea37599594`. This review does not infer authority from the documents' presence alone.
- User-authorized purpose: develop/test a fixed-reference temporal method with synthetic cases and illustrative map tradeoffs, preserve configurable 0.10 optical policy, choose no harm threshold, run no reductions.
- Tier 2 evidence review. This candidate creates no application implementation slot or scientific-contract package. Its opening recovery dispositions do not claim an implementation-blind scientific derivation.

Changed paths: `doc/REFACTOR_STATUS.md` and twelve files under `validation/temporal_change_2026-10-29/`: `WORK_ORDER.md`, `METHOD.md`, `REPORT.md`, `temporal.py`, `experiment.py`, `oracle_control.py`, `map_tradeoffs.py`, `plot_calibration.py`, `test_temporal.py`, `RESULTS.json`, `ORACLE_RESULTS.json`, and `MAP_RESULTS.json`.

## Scientific and behavioral conformance

**Disposition: pass with recorded limitations for truthful experimental evidence; scientific readiness remains failed. Findings requiring candidate repair: none.**

The code matches the stated estimand. The residualized one-column contrasts are the coefficients from separate nested regressions with the same six block offsets and constant-response column. They are projection coefficients, not a reconstructed gain curve. The test searches the five boundary steps and drift jointly across both panels. Shared wild signs preserve the intended panel coupling in the prototype, and the duplicate-panel regression checks that duplication is not counted as independent confirmation. The full encompassing model is used for interval residuals; the null model is used for test calibration. Neither claim is silently conflated.

Numerical rank failure, insufficient excitation, dependence failure, and zero contrast variance produce unavailable results. No scientific excitation cutoff or optical retention rule was introduced. The optical cut remains a policy identity in the prototype, not an applied temporal threshold; native configuration and optical implementation are unchanged.

The Monte Carlo report faithfully distinguishes implementation validity from calibration failure. Independent reproduction gives 975/1000 simultaneous interval coverage for iid, 909/1000 for AR(0.6), 918/1000 heteroskedastic AR, 902/1000 offsets, 901/1000 limited excitation, 912/1000 gaps, 911/1000 heavy tails, 979/1000 quiet, and 155/250 for AR(0.95). These do not support the nominal 99% learned intervals. Counts, denominators, Wilson Monte Carlo intervals, and unavailable cases correspond to the code. The unavailable cases are not quietly assigned coverage or called stable.

Power is honestly limited: the 5% step flags 2/250, 10% step 29/250, 30% drift 46/250, and temporary 30% excursion 13/250. Stationary nonlinear coupling and an impulse can flag, so causal optical-gain attribution is unsupported. The report preserves that distinction. Earlier seeds 7 and 15 are imported from the original synthetic fixture without changing their generated input; p values 0.172 and 0.115 establish only those two examples.

The oracle covariance factor implements the fixture's actual Gaussian recursion, with dependence reset at science-block boundaries and gaps. Independent calibration draws and target seeds yield 988/1000, 990/1000, and 993/1000 coverage. The report reasonably identifies noise-law estimation as a major limitation without claiming an exclusive diagnosis or deployable repair. This control does not establish uncertain-reference, nonlinear, heavy-tail, or general time-series guarantees.

The map calculation also matches its stated scope: four hits per pixel, coordinates in beam FWHM, arbitrary brightness, a fixed source-masked offset-plus-atmosphere regression, coherent fixed weight shares, and separate atmospheric-only versus all-signal perturbations. The source is assumed preserved before adding perturbation; the prototype does not relearn an ALS basis or simulate FRUIT. Noise draws pass through the same fixed cleaner and projector. Source fits report their bounded-fit success. All 160 saved cases reproduce, including the 30% midpoint, 20% weight example: background/noise about 0.3673 and atmospheric/all-signal amplitude biases about 1.675%/4.675%. The 16-observation factor of four and dropped-share inverse-variance cost are clearly conditional analytic illustrations. No conversion to arcseconds, array-specific performance, real-observation predictions, or map harm acceptance is claimed.

Remaining scientific blockers apply to future activation, not preservation of this failed candidate: calibrated learned uncertainty; justified treatment of reference uncertainty and disturbances; sufficient tested change alternatives/power; propagation into map-impact uncertainty; owner harm budgets and realistic beam/scan/noise context; and separately authorized application/observational validation. The minimal current disposition is to retain the failed-gate evidence and its explicit non-readiness. A future repair must be a tested statistical design, not an empirically chosen multiplier that retrofits these seeds.

## Architecture and ownership

**Disposition: pass. Findings: none.**

The study is standalone NumPy/SciPy evidence tooling. No application source, Engine state, typed stage boundary, CAL/RTC ownership, configuration parser, frozen reduction, APT, or runtime is changed. The earlier import is synthetic-only. Trial outputs are diagnostics and never admission products. This supports omitting native build, full CTest, baseline, and config gates for this exact candidate; it supplies no accepted Spack/Unity validation evidence.

## Repository and evidence hygiene

**Disposition: pass with recorded limitations. Findings requiring repair: none.**

The 13-path delta is bounded to the study and status entry. The status accurately records uncertainty failure and absence of activation. `git diff --check` passes. The exact source and tree bind all scripts and compact results, including deterministic seeds and the imported predecessor fixture. The main result's wall clock is correctly excluded from reproducibility equality; every other parsed field matches exactly. Oracle and map JSON match completely. SciPy version is recorded below because map fitting depends on it. Compact summaries cannot recover individual draw logs directly; rerunning the exact scripts recovers the study, as demonstrated here.

No build, observation processing, application executable, Unity access, Git mutation, push, integration, or cleanup was performed. Only review outputs were written under `/tmp`. No additional agents were spawned.

## Reproduced gates and environment

Local `/Users/gwilson/tolteca/bin/python -B -W error`; Python 3.13.2, NumPy 2.2.3, SciPy 1.17.1, macOS-27.0.1-arm64-arm-64bit-Mach-O. Numerical libraries requested single-thread operation. Pytest plugin autoload and cache disabled. Figures used Agg with review-specific `/tmp` configuration/cache directories.

- Focused tests: **20 passed in 1.29 s**.
- Full temporal study: **12,000 realizations**, 999 resamples each, **37.626 s**; all saved fields identical except `elapsed_seconds` (original 36.351 s).
- Oracle: **3,000 target realizations**, 50,000 calibration draws per case; exact saved JSON equality.
- Map sweep: **160 cases** and **256 noise realizations**; exact saved JSON equality; all source fits interior/successful.
- Both calibration and map figures rendered and visually inspected. Labels and units agree with the documented limitations; no material clipping or unreadable content found.
- No unexpected warnings or error-level output. Matplotlib's first-run font-cache progress message was informational.
- Final worktree state: clean, still bound to the reviewed candidate.

Review outputs: `/tmp/independent-temporal-results.json`, `/tmp/independent-temporal-oracle-results.json`, `/tmp/independent-temporal-map-results.json`, `/tmp/independent-temporal-map.png`, `/tmp/independent-temporal-calibration.png`.

Evidence SHA-256 identities at the reviewed candidate:

- `RESULTS.json`: `5b239372f2b9851f224834c5a8ce6477c3b26438d510e70d6b43445ff0a9bb2b`.
- `ORACLE_RESULTS.json`: `c658b981ffcfee8a03b7f6e2cf0e3a2b90a4bb6ac4803b3f2ccaee3c23e9c5b4`.
- `MAP_RESULTS.json`: `d46cf58c7a2a00f91fbc9896a1ca929504a55c423dad03ac0e1c7023a76bd2c6`.
- `REPORT.md`: `6e39f7acdd75c48b71a0825b9cea797858db54247ceb534baa8498cec64cbae0`.

No integration, activation, or production disposition follows from this review. Any repaired implementation candidate requires a new exact-SHA review; adding this review as documentary closure does not extend its scientific claim.
