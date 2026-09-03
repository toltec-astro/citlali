# SCI-RTC v0.1/r0.10 Scientific-Owner Reopening Directive

Date: 2026-08-21

Owner: Grant Wilson

Status: Binding scientific-owner authority for a bounded formal reopening of
frozen SCI-RTC v0.1/r0.9 as candidate v0.1/r0.10.

Source pin:
`2ad12caeabc4a1f84b6748cd7a4cf5683202c51d`, a descendant of required
scientific-contract source pin
`9564bcca0323dacb8bea13a5ec4bbbf3b908de8f` on governing line
`codex/scientific-contract-library`.

## Reopening authority and retained baseline

The owner authorizes surgical contract edits implementing the six decisions
below. Frozen v0.1/r0.9 remains the unchanged baseline at the source pin and
retains its scientific authority until the owner explicitly freezes a reviewed
v0.1/r0.10 successor. This directive does not itself freeze r0.10.

The reopening is limited to conditioned-r production through RTC's existing
paired-product extension point. It shall not change conditioned-x numerical
behavior, import calibration into RTC, define a PTC x-from-r estimator, decide
SCI-VAL policy, reopen unrelated RTC algorithms, or make an implementation,
validation, performance, science-qualification, or production claim.

## Decision D01 - Scientific role, optionality, and paired lifecycle

SCI-RTC v0.1/r0.10 admits conditioned r as the optional paired RTC companion to
conditioned x while preserving immutable raw-r parentage. Conditioned r retains
diagnostically useful information from the quadrature coordinate for RTC
diagnostics and downstream consumers, including a future explicitly selected
PTC joint-r mode.

Conditioned r remains in its corresponding raw detector-coordinate convention.
It is uncalibrated, is not a Stokes observable, does not enter SCI-CAL, does not
replace raw r, cannot alter conditioned x within RTC, and does not authorize an
RTC r-to-x correction. Any later use of r to modify calibrated x, including an
x-from-r nuisance prediction, remains PTC-owned.

Conditioned r is optional scientific product content. This contract does not
prescribe whether routine production configuration enables or disables it.
RTC uses one application context, one resolved plan, and one realized record
for the paired coordinates; there is no independent r-processing lifecycle or
independently optimized r plan. Any future departure from pair-coherent RTC
treatment requires explicit scientific justification and owner authorization.

## Decision D02 - Pair-coherent pathology support without invented r repair

RTC artifact detection and support decisions are pair-coherent across x and r.
Detection evidence and fitted coordinate amplitudes may remain
coordinate-specific, but a pathology affecting either coordinate conservatively
affects the validity, causes, and support of the paired occurrence.

Existing authorized RTC repair or replacement of x does not require numerical
replacement of r. Where no scientifically justified r-domain correction exists,
conditioned r is flagged invalid or unavailable over the affected support
rather than reconstructed from donors, copied from x, filled, or otherwise
invented. No x-to-r or r-to-x numerical mixing is permitted. The governing
principle is: protect x first, preserve honest information about r, and do not
invent r-repair machinery without a concrete scientific need and new owner
authority.

## Decision D03 - Exact selected grid and honest local unavailability

When conditioned r is requested, RTC produces it on exactly the conditioned-x
grid. It retains the same detector-occurrence ordering, timestamps,
cardinality, representative raw-pair occurrence, segmentation, and applicable
temporal support.

An r location that cannot be scientifically computed remains at its paired
grid position with typed invalid or unavailable state and exact cause. RTC
shall not drop, independently reindex, interpolate, donor-fill, or substitute
that location. Local or global failure to produce valid conditioned r shall
never corrupt or invalidate otherwise valid conditioned x.

## Decision D04 - Coordinate-diagonal response and joint statistics

Conditioned x and conditioned r each retain complete realized response, or a
typed unavailable response state, relative to their corresponding admitted raw
coordinate. RTC has zero cross-coordinate numerical response:

\[
\frac{\partial x_{\rm RTC}}{\partial r_{\rm raw}}=0,
\qquad
\frac{\partial r_{\rm RTC}}{\partial x_{\rm raw}}=0.
\]

Where an RTC operation is mathematically applicable to both coordinates, the
r-domain response uses the same filter, mask, state, sampling, phase, grid, and
support operator as x. An x-specific repair that is not applied to r does not
appear in the r response; r and its response are unavailable over the affected
support.

The conditioned-r response retains its raw-r mapping, units, sign, reference,
source-response and optical-leakage lineage, source protection, support,
exceptions, and provenance. Shared filtering propagates genuine optical
leakage through the declared response rather than silently removing or
renormalizing it.

Absence of numerical mixing does not imply statistical independence. Any
paired covariance or uncertainty claim preserves admitted x/r cross-covariance
and discloses included and excluded selection, artifact, learned-parameter,
mapping, leakage, and model terms. At fixed state and on valid support, an
available joint conditional covariance follows

\[
\begin{bmatrix}
L_x\Sigma_{xx}L_x^{\mathsf T} & L_x\Sigma_{xr}L_r^{\mathsf T}\\
L_r\Sigma_{rx}L_x^{\mathsf T} & L_r\Sigma_{rr}L_r^{\mathsf T}
\end{bmatrix}.
\]

RTC is not required to publish a numerical covariance when the input model or
required components are unavailable.

## Decision D05 - Optical leakage and pair-coherent source protection

Nonzero astronomical or atmospheric response in r is diagnostically meaningful
and is not by itself an artifact, invalidity, or failure of conditioned r. RTC
shall not suppress, subtract, rotate away, or renormalize that response merely
to make r appear optically orthogonal.

Coordinate-specific optical-leakage evidence retains source class, estimator,
raw or conditioned stage, mapping revision, units or scales, normalization,
response, support, masks, uncertainty, validity, and provenance. Raw-r and
conditioned-r leakage diagnostics are distinct. A pre/post comparison is
available only on declared common support with the applicable RTC responses
accounted for.

Source protection used by artifact detection, parameter learning, replacement
selection, plateau estimation, or other contamination models is pair-coherent:
an astronomical interval protected in x is also protected in r, and
coordinate-specific evidence does not silently remove that protection. Missing
or invalid required source-protection authority makes the affected learned or
corrective operation unavailable rather than allowing source signal to be
learned as contamination. Source protection does not prevent application of an
already resolved shared linear filter or selected output-grid operator.

RTC records realized source-protection and leakage-characterization state but
does not decide cross-package eligibility, promote leakage evidence into a
correction, or authorize PTC's future x-from-r nuisance model.

## Decision D06 - Failure isolation and consumer-owned disposition

When requested, conditioned r is produced on exactly the conditioned-x grid and
invalid locations are preserved honestly rather than assigned invented values.
Failure to produce valid r locally or globally shall never corrupt or invalidate
otherwise valid conditioned x.

Downstream consumers receive the paired conditioned-x/conditioned-r product,
immutable raw-r parentage, and enough RTC-owned identity, mapping, response,
support, cause, validity, availability, leakage, source-protection, uncertainty,
and provenance facts to know what happened to the detector data. Each consumer,
together with SCI-VAL where applicable, owns how those facts affect its
admission, weighting, fallback, eligibility, or scientific use. RTC shall not
silently decide those downstream policies.

## Required r0.10 change discipline

The r0.10 candidate shall:

- implement these decisions surgically in the shared normative core and both
  audience views;
- supersede r0.9 language that makes conditioned r unavailable pending an
  independent channel-specific operator;
- preserve every unaffected r0.9 identifier, equation, requirement, owner-ledger
  state, algorithm boundary, and claim limitation;
- update the exact authority crosswalk and mechanical inventory coherently;
- pass implementation-blind consistency and PDF build checks; and
- return the candidate to the owner for review without claiming it frozen.

