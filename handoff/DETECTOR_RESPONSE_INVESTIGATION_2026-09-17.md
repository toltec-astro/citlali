# Detector-response investigation handoff — 2026-09-17

The common-mode diagnostic and repeatability census are accepted and complete
for their present RTC purpose. Further detector-response science is separate
from RTC implementation and does not gate it. Retain the existing Learn/Consider
capability; no diagnostic-derived rejection, calibration, or weighting rule.

**New hypothesis, untested:** between-observation response changes may depend
on the tunes. The two observations have different Tune/KMP realizations, but
also different elevation, recorded tau, and contamination. Neither causation
nor a detector correction follows from this comparison. A separate investigation
may inspect those operating-point differences with the exact identities below.

Both observations are NGC4449 SCIENCE, network 0/a1100, 122.0703125 Hz:
152390 has 151,537 rows, 152392 has 151,943. All 630 targets per observation were
retained. Only one suitable local repeat was available. The same exact baseline
APT bundle establishes these accepted unique correspondences:

| 152390 channel | 152392 channel | Baseline seed UID | Existing finding |
|---:|---:|---:|---|
| 221 | 218 | 219 | q 0.101→4.192; weak coupling becomes larger, noisy response |
| 445 | 445 | 437 | q 6.206→4.394; recurring calibration-relative discrepancy |
| 426 | 426 | 418 | Negative fitted duration 618.08→2.38 s; prior flags remain |
| 236 | 233 | 233 | Strong original spectral contamination; baseline gain reference-dependent |
| 301 | 298 | 299 | Comparatively stable control, q 0.812→0.878 |
| 492 | 492 | 485 | Comparatively stable control, q 0.619→0.616 |

q is the existing initial-reference diagnostic, not a calibration-error factor.
426 has flag=1, flag2=2, flxscale=0; its `is_good_match=false` means flagged
selected seed, **not ambiguous identity**. The producer's unique matching
relation is accepted. Negative response is not proven to be a stable physical
inversion. Exact-support and two disjoint-reference comparisons preserve the
main changes in 221/445; 236's original gain remains inconclusive. Formal
uncertainties and net incremental rejection cost are unavailable. Shared
calibration errors cannot be diagnosed by peer normalization.

Evidence root (sealed; retain unchanged):
`/private/tmp/citlali-rtc-common-mode-repeatability-2026-09-16`.
Read `REPORT.md`, `observation-selection.json`, `analysis/followed-targets.csv`,
`analysis/matched-comparisons.csv`, `analysis/matched-support-pieces.csv`,
`analysis/self-excluded-interval-history.csv`, and
`prepared/identity-152390.csv` / `identity-152392.csv`.
The full census is `analysis/full-network-census.csv` (1,260 rows).
Plots are `analysis/response-history.png`, `reference-adequacy.png`, and
`population-comparison.png`. Source, gates, parity and review are bound by
`SOURCE.json`, `CLOSURE.json`, `GATES.json`, `verification.json`, and
`INDEPENDENT_REVIEW.md`. Baseline evidence:
`/private/tmp/citlali-rtc-common-mode-health-2026-09-16`.

Selection SHA256:
`1b6fdcc25793c5dc9201f23beacc5c71f3366b90c1c9cd349118b0c4435d3665`.
Report SHA256: `898928311a976bd22fa6a95a96d02bc5676fef558e889ae90def19737ff7b150`.
The manifest records exact raw/Tune/matched-APT paths and digests. Raw and Tune
files are under `/Users/gwilson/work_toltec/local_data/citlali-validation/v1/science/data`;
matched APT bundles under `.../citlali-validation/v2/science/apts/v2`.
Baseline APT occurrence:
`apt-occurrence:entropy/2f407f35b2334ac7804cdd8356293054631bb551ae7cfc6d501e4b4bfbd14fca`;
semantic SHA256 `944a82d92231408d416012701d41a670b02b4e18c9fb59a4decb64d867641f51`.
Never substitute equal channel numbers for this recorded correspondence.

Accepted diagnostic runtime `9a47dd8d30c844ac2838827c438486f3a53a58d7`;
census baseline `4162c2b18bd4c7d39608e25205fbeabf859106fa`, repeat
`14b67cccf7c638d91d7e7a3aa13b5be4bcf0d44a`, closure
`e9ff1fb0ce7acd34d0093d9b90e37244eb43ad00`. Core diagnostic was unchanged.
Local supplemental tests and independent review passed; no production
qualification is implied. RTC now resumes matched-support treatment outcomes.
