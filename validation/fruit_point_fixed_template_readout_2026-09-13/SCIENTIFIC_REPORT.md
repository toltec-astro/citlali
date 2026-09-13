# Fixed-template readout result: a mixed map and measurement problem

2026-09-13 · SCI-FRUIT v0.1 development · r0.1

## Program adherence and prior-work recovery

Follow the [charter](../../doc/scientific_contracts/README.md),
[reviewed prior work](../../doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.4/PRIOR_WORK.md)
and the [authorized protocol](PROTOCOL.md). The owner approved the exact
48-fit saved-map comparison proposed by the earlier peak diagnosis. Starlet
remains **parked for POINT**; all registered outcomes, availability labels and
the failed image-stability qualification remain unchanged.

**The candidate maps contain substantially better recovered amplitude along
the known source shape in these cases, while the common free-Gaussian readout
introduces substantial errors. Some crossed-noise error nevertheless remains
in the maps.** This is a mixed result, with direct evidence of inadequate
numerical solution selection in six of the saved source fits. It does not
qualify a deployable fixed-template method or reverse the parking decision.

## What changed when only the measurement changed

The same 24 terminal H/D array maps were measured with a signed amplitude for
the true injected shape plus a free plane, on both existing domains. The
injected peak was not imposed: amplitudes were solved freely. No PTC, feedback
or free-Gaussian fit was rerun. The old fit is compared on identical pixels.
The following counts describe raw errors, including originally withheld peaks;
they are not new usable-result or qualification counts.

| Radius | Readout | P absolute peaks within 5% | C absolute peaks within 5% | P crossed H/D within 5% | C crossed H/D within 5% |
| --- | --- | ---: | ---: | ---: | ---: |
| 60 arcsec | Saved free Gaussian | 6/12 | 4/12 | 3/6 | 2/6 |
| 60 arcsec | Fixed true shape | 0/12 | 11/12 | 4/6 | 4/6 |
| 52 arcsec | Saved free Gaussian | 6/12 | 5/12 | 2/6 | 3/6 |
| 52 arcsec | Fixed true shape | 0/12 | 11/12 | 3/6 | 4/6 |

At radius 60, the fixed-template C peak errors span **-3.87% to +5.49%**;
only first-seed H a1400 misses 5%. The saved free-Gaussian C errors span
**-5.23% to +16.69%**. The strong absolute-amplitude improvement survives the
52-arcsec comparison (11/12 within 5%, maximum error 6.27%). It therefore does
not depend on choosing the more favorable of the two domains.

For P, all twelve fixed-template amplitudes fall below truth: **-5.08% to
-22.67%** at radius 60. All its saved free-Gaussian H/D widths are narrower
than the injected widths. Allowing a narrower profile raises the fitted peak
and can make peak agreement look better despite less recovered amplitude
along the known source shape. This supports a profile/response discrepancy in
the reference maps, with nuisance still present; it is not an independent
measurement of the PTC transfer function.

Matched-noise gain remains favorable for C: all six fixed-template H/D ratios
are within 5%, maximum error **2.56%**, on both domains. That is weaker numerical
agreement than the original 0.80% matched result, but still inside the same
5% descriptive comparison. P remains 5/6 matched, maximum 5.45% at radius 60.
All ratios and directions are retained in [the tables](READOUT_TABLES.md).

## What remains in the maps

The fixed template does not eliminate between-noise variation. At radius 60,
C's H1/D2 crossed errors remain **+7.27% in a1400** and **+6.72% in a2000**;
the reverse directions are -3.78% and -4.82%. At radius 52 the corresponding
positive errors are **+8.31%** and **+6.17%**. Both domains therefore retain two
crossed errors above 5%. These depend on source-aligned content in the saved
maps, not solely on the nonlinear free-shape fit.

For C, fixed-template H seed changes are +0.056%, -6.180%, -4.884% across
1100/1400/2000; D changes are +0.125%, -4.393%, -6.229%. P retains still larger
changes in the two longer-wavelength arrays. Matched ratios partly cancel
these seed effects. Two reused realizations cannot establish an uncertainty,
false-alarm rate or population-level bias. This experiment does not separate
residual nuisance from cleaning/feedback transfer effects.

## A concrete numerical problem in the common readout

The fixed-template solution, including its fitted plane, is a feasible member
of the free Gaussian-plus-plane model family: its center, widths, angle and
amplitude satisfy that family's bounds. Yet **six saved free fits have a
higher residual sum of squares than this known feasible solution**. Their
saved costs were independently reproduced on exactly the same pixels.

| Saved fit with a known better feasible solution | Domain | Reduction in saved fit SSE |
| --- | ---: | ---: |
| P H_20260912 a1100 | 52 arcsec | 2.6207% |
| C H_20260912 a1400 | 60 arcsec | 0.4168% |
| C D_20260911 a1100 | 52 arcsec | 1.8397% |
| C D_20260911 a1400 | 60 arcsec | 0.2218% |
| C D_20260911 a2000 | 52 arcsec | 0.6689% |
| C D_20260912 a2000 | 52 arcsec | 1.3878% |

The four inner-domain cases are exactly the four distinct H/D states rejected
by the registered domain-sensitivity check. This establishes numerical
solution-selection failures in the common measurement. It is stronger evidence
than simply observing that two different fit domains give different peaks.
It does **not** prove a repaired fitter would make these measurements usable:
that would require actual authorized refitting and qualification, and the
low-score rejections remain a separate issue.

Across all 24 fixed-template domain pairs, the largest amplitude change between
52 and 60 arcsec is **0.954%**, versus **3.240%** for the saved free fits. That
smaller sensitivity uses the known source shape and therefore cannot be used
to relabel the original measurements. Central fitted-plane coefficients and
values at the known source center are recorded for both readouts; the separately
saved outer-annulus plane remains distinct. The remaining cases where the free
fit has lower map residual cost but worse peak truth agreement also show why
least-squares completion alone is not an astronomical accuracy certificate.

## Consequence and limits

The earlier diagnosis is now narrower: the reference loses substantial
known-shape amplitude in these saved cases; the candidate restores it much
better; free-shape readout errors obscure that difference; and some noise
dependence remains even with an exact source template. There is concrete
reason to address the common source fitter before using its domain-sensitivity
failures to judge another feedback change. No fitter repair or new experiment
has been started here.

The fixed template deliberately knows the injected centroid, shape and
orientation. It is an oracle diagnostic ruler, not a POINT recipe or proof of
its usability. Original peak labels, pointing outcomes, null safeguards,
noise-related failures and runtime limits all stand. The original candidate
cost misses are untouched, and no independent pointing or reserved observation
was evaluated. Starlet stays parked; no OOF or production claim follows.

## Verification and provenance

All 48 linear fits are finite and rank four, with design condition numbers
12.14–14.61. Maximum normalized normal-equation residual is 1.87e-16. The
48 saved free-fit costs were reproduced, and each fixed solution was independently
checked as a feasible Gaussian-family model without another fit. Execution took
1.09 seconds overall, including read/preservation work; the 48 linear solves
and their residual checks took 6.86 milliseconds and process peak RSS was
57.75 MiB. This diagnostic timing is not a new FRUIT latency result.

The [source/input freeze](FREEZE.json) was committed at
`e8ea8b74cfe1dbead2d38604f7f2225fe4b186ca` before fitting, from clean parent
`5d0ab7e7d95786fcc02822938452c79910b7e725`. Exactly 48 fits were executed, with
zero verification refits, nonlinear fits, feedback fits, PTC calls, pre-PTC
reads or reserved observations. All 256 prior packet payloads, 1804 prior
external products, 57 frozen authority payloads and opaque archive statuses
remain unchanged. [Verification](VERIFICATION.json),
[objective witnesses](FIT_OBJECTIVE_WITNESSES.json),
[complete numerical comparison](COMPARISON.json),
[figure](READOUT_COMPARISON.png) and [result manifest](RESULT_MANIFEST.json)
preserve every case and original label.
