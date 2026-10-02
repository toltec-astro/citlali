# OOF a2000 saved-sample JINC assessment — 2026-10-02

All three owner-run map-only applications completed, exactly once, at source
524845cd7c730a1f9bf7dfb705f8d3bae1ba94a5. No RTC/CAL/PTC reexecution, coordinate
change, feedback, source fit, noise inference or optical solver. No unexpected
native output. All358 prepared bindings and native output digests were verified;
FITS planes equal their native arrays; every published signal passes the approved
arithmetic bound and every missing pixel is explicit. Correct scan1 identities
appear in the native receipt and FITS.

## Central-map findings

Fixed native-centered square windows are [-16,+16) and [-50,+50) arcsec per
axis, without recentering, rotation, interpolation or background subtraction.
These are consistent angular windows, not an assertion of aligned FK5/AltAz
sky coordinates. Raw parents are preserved. Normalized figures divide each
panel by its own central32″ peak and preserve those divisors in measurements.

| Observation | Historical1″ JINC | Same samples2″ ordinary | New1″ JINC |
| --- | ---: | ---: | ---: |
| 152385 | 126.995 | 148.107 | 137.314 |
| 152386 | 559.846 | 564.852 | 554.334 |
| 152387 | 214.697 | 245.771 | 227.045 |

Values are central32″ **pixel peaks**, mJy/nominal beam, not fitted amplitudes.
New JINC versus historical peaks differ by about+8.1%,−1.0%,+5.8%. Uniform
successor weights, upstream processing and frame still differ from historical
R. Matching gridding parameters does not remove those differences; shared JINC
also retains exact-root/ceil-square conventions, with29×29 versus historical27×27
support.

JINC greatly reduces visible pixel grain in these same processed samples.
Central100″ minima change from−90.97/−78.53/−88.92 in ordinary maps to
−24.24/−32.73/−22.40; historical minima in the same defined native windows are
−20.92/−29.45/−16.83. All new central100″ windows have10000/10000 numerically
supported pixels. This is a gridding effect, not improved PTC recovery or a
measured sensitivity gain. No region is certified source-free.

Normalized images preserve the broad–compact–broader focus sequence. Visible
elongation, wing placement and positional differences remain between historical
and new panels, most conspicuously in the defocused maps. Their physical
significance remains unresolved until a correct common source-offset frame is
used. The results do **not establish equal recovered morphologies**.

## Outer-map limitation

Full-map JINC has extreme signed-normalization artifacts: maxima about
133175/985029/185485 and minima−332748/−704035/−186866mJy/beam. These occur
outside the central100″ windows, where signed C nearly cancels (extreme-pixel
|C|/sum|coefficients| about4.1e-5 to2.8e-4). Detailed positions, membership,
C and arithmetic error bounds are preserved. Numerical certification can
accurately evaluate a poorly supported signed quotient; it does not make it
credible astronomical brightness. No new rho/coverage cutoff is invented here.
The full numerical mask is not a blanket scientific-quality mask, and the full
map cannot be declared quality-equivalent to historical R. Central display
limits are intentional and the full-context figure is retained.

## Cost, evidence and disposition

Wall time127.14s; native applications total about115s (exact native receipts),
role wrappers117.75s. Recorded maximum child-process RSS294,912,000bytes
(0.295GB; not whole-process-tree RSS). Campaign retention about0.424GB before
final documentation; shared actual derived storage about22.135GB, below30GB.
The1GB campaign reservation and2GB free reserve remain satisfied.

Saved results and figures:
`/Users/gwilson/work_toltec/local_data/citlali-validation/successor-oof-a2000-jinc-20261002/assessment/`.
MEASUREMENTS.json binds exact executed/assessment sources, all input/output
checks and PNGs. The first exploratory report remains preserved in report/.
No products were altered or deleted. The three-map allowance is consumed;
READINESS.md's launch command is historical, not a retry command.

This closure is saved-product analysis/documentation only. Native1324 enabled,
baseline207, POINT214, OOF24, native contract9 and required config gates are
inherited exact execution prerequisites, not newly rerun for reporting. One
pre-existing disabled lifecycle fixture remains disclosed. Local build evidence
is AppleClang/Homebrew, not representative Spack/Unity qualification. Analysis
is validated through authenticated product checks and visual inspection; focused
independent closure review binds its final source. No integration or push.

Owner has authorized delta-source(offset) AltAz as mandatory for both POINT and
OOF after this comparison. That separate next implementation is recorded in
NEXT_OWNER_DIRECTIVE_ALTAZ_2026-10-02.json and remains deferred while these
results are discussed. Prior POINT scientific campaigns stay closed.
