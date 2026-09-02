# SCI-FRUIT EL-F2 r0.3 independent-pointing early-stop result

Result: **valid primary screen; fixed alpha 1.25 stopped at iteration 5 does
not reproduce the protected compact-source result**

The owner-authorized test was
`SCI-FRUIT-EL-F2-INDEPENDENT-POINTING-EARLY-STOP-R0.3`, bound by the exact
`EL_F2_BUNDLE_MANIFEST_R0.3.md` with SHA-256
`440fbb18a3190563061b0dad9c4a156a5a7e7c6699b7df0bec48fec5ae5bc579`.
The r0.3 approval changed only the fit-report path and allowed one final
input-related attempt. The original r0.1 scientific question, recurrence,
matrix, terminal iterations, metrics, thresholds, run order, and conditional
restart rule remained unchanged.

This is development evidence only. It is not method or APT qualification and
is not a comparison with the unavailable exact historical executable.

## Frozen execution and completed matrix

The exact executable SHA-256 was
`a49082dde8f71d6f50edd8c378ad94195496b5eb0e0855b746e189f3442acbcc`.
Its full identity, all configuration and analyzer hashes, and the final text
fit-report checks are recorded in `FROZEN_INPUTS_R0.3.md`.

The final replacement and the other three primary trajectories completed in
the frozen BAAB order. Each used `grppiex: seq` and one configured runtime
thread. The candidate trajectories produced exactly iterations 0--5; the
reference trajectories produced exactly iterations 0--6. All four logs end in
normal Citlali completion and contain zero error- or critical-level messages.

| Order | Alpha | Variant | Iterations | Wall (s) | User (s) | System (s) | Maximum RSS (bytes) | Retained (KiB) |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.25 | control | 0--5 | 175.71 | 167.52 | 5.31 | 953892864 | 352656 |
| 2 | 1.00 | control | 0--6 | 205.38 | 196.33 | 6.14 | 930562048 | 370308 |
| 3 | 1.00 | injected | 0--6 | 197.84 | 191.98 | 3.17 | 924712960 | 370308 |
| 4 | 1.25 | injected | 0--5 | 168.64 | 163.60 | 2.60 | 931053568 | 352664 |

The four valid runs took 747.57 seconds in aggregate and retained 1,445,936
KiB at inspection time. They remained well below the trajectory and aggregate
resource limits. `PRIMARY_EXECUTION_R0.3.csv` was checked against the complete
logs, timings, memory values, iteration directories, sequential policy, and
normal-completion markers.

The complete log identities are:

| Trajectory | Bytes | SHA-256 |
| --- | ---: | --- |
| alpha-1.25 control | 2674836 | `1b7b5f22e2efa4e94fbb4d37608cbfda123bae3017c28d5c58c0eff0a4400c0d` |
| alpha-1.00 control | 3144322 | `1926f378f3402fd0f23d8473f33f5f680d510766daaba5ce442be0cffcd2be2d` |
| alpha-1.00 injected | 3200143 | `5e425ebee9b376273b27f4d6f9681042dfb473b820bc4852e8d07f6d749bc726` |
| alpha-1.25 injected | 2721336 | `3d52201d0c44d33c5b159a779bba87d79c312bf6a911a73fcd1a1f065445ab98` |

## Performance result

The reference pair-mean wall time was 201.610 seconds. The candidate pair mean
was 172.175 seconds, a 14.600 percent reduction. This passes the prospective
10-percent performance target.

The two earlier invalid starts lasted only 1.22 and 0.82 seconds and did not
reach mapmaking, but they may have warmed file metadata. That limitation was
approved and is retained here. The valid reference pair followed a complete
candidate control and the injected candidate remained last, so the frozen
pair-mean calculation was applied without adjustment. No exact cache-equality
claim is made.

Passing the timing target cannot override a protected scientific failure.

## Frozen scientific result

The frozen analyzer accepted the exact expected iteration sets and realized
alpha/injection settings. It verified bitwise control/injected identity before
injection, stable paired and cross-iteration shape/WCS, and identical finite
support for signal, kernel, and weight. Every analyzed image had all 126,735
pixels in common finite support. It produced the 33 prospective rows in
`ITERATION_METRICS_R0.3.csv`.

The decisive comparison is alpha 1.00 at iteration 6 against alpha 1.25 at
iteration 5. Width ratios had to lie from 0.97 through 1.03, centroid error had
to be at most 0.1 arcsec, and each residual ratio to the reference had to be at
most 1.10. Candidate recovery error could be no more than 0.01 worse than the
reference error.

| Array | Reference recovery | Candidate recovery | Candidate major/minor over kernel | Centroid (arcsec) | Annular residual ratio | Kernel-residual ratio | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| a1100 | 0.867388 | 0.869745 | 0.966040 / 0.926893 | 0.034933 | 1.177722 | 0.990613 | fail: widths and annular residual |
| a1400 | 0.896659 | 0.822828 | 0.951671 / 0.895653 | 0.299145 | 7.252958 | 2.398996 | fail: recovery, widths, centroid, and both residuals |
| a2000 | 0.731317 | 0.734453 | 0.919201 / 0.865560 | 0.360848 | 1.184923 | 1.029911 | fail: widths, centroid, and annular residual |

The a1400 result is not a marginal threshold crossing. Candidate central
recovery rose to 0.890451 at iteration 4, then fell to 0.822828 at iteration
5. Its terminal successive-transfer change was 1.211585 times the preceding
transfer-map RMS. This is diagnostic evidence of an unstable final update on
this pair, not a post-hoc alternative stopping rule.

The exact prospective classification is therefore `does_not_replicate`.
Scientific protections fail even though the timing target passes.

## Restart and disposition

The frozen protocol requires a restart replay only for
`promising_early_stop_result`. This result is `does_not_replicate`, so no
restart run is authorized or required.

Do not promote fixed alpha 1.25 with an iteration-5 stop, and do not rerun this
same observation/candidate merely to seek a different result. Together with
EL-F1 r1, this independent result argues against treating fixed global alpha
1.25 as a generally safe compact-pointing acceleration: the earlier case
failed an a1100 residual protection, and this one fails multiple protections,
most strongly in a1400.

Alpha 1.00 remains the same-build compatibility control, not truth, a newly
qualified production method, or proof of historical superiority. The weak
absolute a2000 recovery and non-unit reference widths also remain visible;
the paired control is not represented as a perfect sky answer.

A later empirical proposal may test a genuinely different, prospectively
defined recurrence or causal damping/stop guard that responds to residual
growth. It must use new reviewed bounds and cannot relabel an iteration chosen
after inspecting this outcome as a successful EL-F2 result.

This result does not launch Gate D or Stage B, qualify a recurrence, change a
production default, select a fallback or stopping rule, establish historical
superiority, or authorize Unity work.

## Repository evidence identities

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `PRIMARY_EXECUTION_R0.3.csv` | 470 | `fd62465c0099d19946367703d452bad98da8d0627b553a817b46eb73348dab68` |
| `ITERATION_METRICS_R0.3.csv` | 7133 | `3728e5de118c0799ce797a6074b376b22d76a45faf8a9f20b7bbff4cd213ca58` |
| `SCREEN_RESULT_R0.3.json` | 1543 | `4914fb8fc6ca177291f36ed5dab24023d35543b059a4e7740cc75dfe19dc6422` |
| `SCREEN_RESULT_R0.3.md` | 1263 | `415128a872589ceb04fe38eaaf6648dbb80251aa3d40242774ad837e27d94032` |
| frozen `analyze_early_stop_screen.py` | 10329 | `6ec845afb77da71cc1033c26a49b4ba44168adee1981b6976b723958cd182aa4` |

The original data, legacy APT, processed tune files, text fit reports, and all
pre-existing reduction products remained unchanged.

## Repository verification

The complete baseline and FRUIT-loop Python suite passed with 194 tests. The
compiled CTest suite passed all 610 enabled tests, with one pre-existing test
disabled. The full configuration preflight passed all 127 tests and its mode-kit
and audit checks. The validation ledger passed with 60 records, and the
science-change ledger passed with three changes and five integration commits.
`git diff --check` also passed. No production code changed in this result commit.
