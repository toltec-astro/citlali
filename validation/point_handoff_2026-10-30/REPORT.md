# POINT correction compatibility and table handoff

The combined development candidate now accepts the demonstrated OG pointing
inputs and publishes a complete all-array astrometry table through the existing
TolTECA interface. This is an implementation candidate, not an observational or
flux-calibration qualification. No reductions or saved-map refits were run.

## Preserved behavior and repaired boundaries

| Boundary | Result |
|---|---|
| Physical coordinates | Existing AST equations, detector rotation, tangent-azimuth sign and FITS orientation unchanged; no extra cosine or negation |
| User/Paddle corrections | Header values retained in radians; existing TolTECA composes POINT minus SCIENCE header differences and negates measured displacement once |
| Single pointing | One positive MJD and one az/alt value are accepted as an observation-wide constant; source date retained |
| Undated bracket | Two values without actual epochs use the already prepared common observation-grid endpoints, as OG did; raw TEL endpoints are not substituted |
| Dated bracket | Native ingress calls OG's exact MJD-to-whole-Unix-second conversion; invalid, reversed, overflowed and integer-collapsed support is rejected |
| Saved CAL/MAP | Two support scalars and a conversion marker preserve the chosen model through the saved-product boundary; mismatches fail explicitly |
| Table publication | One conventional `ppt_successor_pointing_<obs>_citlali.ecsv`, after all three final fit measurements qualify; no second fit is performed |

Native evaluation still occurs on the successor's assigned sample times. Formula
and same-time numerical compatibility do not imply identical sample scheduling
or identical fitted centroids to OG. The intent is to preserve correction behavior
while developing improved map estimates, not force old map amplitudes.

Older constant-offset CAL receipts remain usable. Older variable-offset CAL
receipts lack a declaration distinguishing the former fractional-second
conversion from OG's whole-second conversion. This candidate refuses to reinterpret
those saved products silently. Replaying such inputs with this candidate requires
fresh CAL preparation by the owner; old evidence and frozen runtimes are intact.
No fresh CAL preparation was performed here.

## Published interface and unavailable measurements

The POINT input adapter captures metadata while opening the same bound telescope
file already used for acquisition validation. It carries observation identity,
source, mean telescope MJD, corresponding UTC date, the four User/Paddle headers,
applied az/alt offsets, and the existing telescope path/hash. There is no new
heavy tracing or detector-loop work.

| Product or column | Meaning / availability |
|---|---|
| Three existing FITS maps and per-array JSON records | Retained; same map bytes and fit behavior |
| `point-table.json` | Always states publication/withholding, cause, table hash and per-array source association |
| `array` | Integer 0/1/2 = a1100/a1400/a2000, explicitly mapped in metadata |
| `x_t`, `y_t` | Signed measured tangent azimuth / elevation displacement, arcsec; not corrections |
| `amp` | Existing fitted peak, mJy per nominal beam; not measured-beam or full-response calibration |
| `amp_err`, `x_t_err`, `y_t_err` | NaN: uncertainty unavailable; never zero and never substituted conditional `sigma_A` |
| Eligibility | Pointing handoff eligible only after the publication checks; flux-calibration eligibility remains false |

Missing metadata or any incomplete/unqualified array fit withholds the entire
PPT while preserving valid maps. Software/I/O failures propagate and staged
publication stays hidden. Nonzero applied POINT corrections also withhold the
PPT: publishing a corrected residual as an uncorrected displacement would allow
it to be applied again. Such maps remain supported; residual-to-measurement table
conversion is a separately identified compatibility limitation, not a silent
change to the OG formula.

This is the concrete table interface needed by TolTECA's pointing resolver.
It does not claim complete compatibility with every legacy plotting/table
consumer; legacy map-analysis compatibility remains false.

## Verification

[Consumer results](consumer_results.json) bind actual read-only TolTECA source
2791e6a1e6349ad1d3ac549a648f41cbc51b98c7 and TolProj
6309a5a852a0412cee867992aff01341a56b0ebf. The checker extracts their real functions
from exact Git objects, rather than importing the unrelated local TolTECA v3
working tree or approximating the correction formula. It exercises synthetic
published ECSV through ordinary observation-folder discovery, the actual resolver,
and the configured POINT adapter for bracket/reversed/single-before/single-after/
exact-observation cases. Nonzero unequal User/Paddle values verify composition.
All three array IDs, amplitudes, signed centroids, units and discovery pass.
The actual TolProj flux reader rejects NaN `amp_err`; eligibility metadata alone
would not enforce that hold. No sibling source edit is needed for this boundary.

Focused native fixtures exercise the actual CAL parser, fractional-epoch parity,
constant and observation-span interpolation, invalid input, and saved-receipt
validation. Existing geometry/orientation and native timing tests cover unchanged
physical conventions. Python fixtures cover metadata, all-array publication,
failed/missing fits, explicit unavailable errors, no extra fit calls, preservation,
CLI staging, package inventory and consumer behavior.

Final gate counts and exact source/environment identities are recorded in
`LOCAL_VALIDATION.json` at closure. Local native compilation uses the existing
AppleClang/Homebrew supplemental dependency realization in a separate build;
it is not accepted Spack/Unity operational evidence. Python uses Grant's existing
`/Users/gwilson/tolteca/bin/python`. No observation processing, input restoration,
Unity access, production activation, canonical movement or push is part of these
checks. Original reductions, frozen trials and the other combined worktree are
preserved. Noise weighting, the 10% optical-response rule, FRUIT and grid settings
are unchanged; the wing study remains paused.

## Remaining connected-pilot blockers

1. Decide and qualify the flux uncertainty/eligibility policy before releasing
   this table to pointing-derived flux calibration. The current guard's
   conditional uncertainty is not a substitute.
2. Implement the SCIENCE/coadd connection on the accepted candidate and follow
   the resulting calibration into the exact SCIENCE APT; this task does not
   establish those products.
3. Qualify the exact candidate in the supported environment and owner-run
   POINT–SCIENCE–POINT pilot, including all three arrays and real coadds.
4. Before using corrected POINT runs as calibration pointings, qualify how their
   residuals should be represented to the established resolver. Current zero-
   correction pilot pointings do not require this extension.

Integration, release and production acceptance remain Grant's decisions.

## Completion disposition

Independent exact-source review of `8fa064d1169cb7d37ccfea4a046c66a78fe675e1`
passed with recorded limitations and no actionable findings; see [review](REVIEW.md).
The [local validation receipt](LOCAL_VALIDATION.json) records six new native tests,
1,394 enabled CTests (one inherited disabled), 207 baseline tests, 320 application
tests and the final 14-test CLI suite, configuration preflight and static checks.
The source remains an isolated development candidate. Documentation closure does
not change the reviewed application files; the refreshed closure executable's
exact identity is recorded under `build/point-handoff-checks/closure-runtime.json`.
