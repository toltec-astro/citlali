# WP-7 RTC Scan/Array Planning Scientific-Owner Authority

Date: 2026-08-29

Scientific owner: Grant Wilson

Status: approved bounded successor authority; numerical policy corrected to v2
and scan-wide upper-speed planning corrected to occurrence-level admission by
scientific-owner dispositions 2026-08-30; AST authority closed by
`wp7-ast-scan-motion-v1`; bounded AST implementation passes local,
representative-data, and fresh exact-SHA conformance gates; certified filter
bank and nonidentity RTC implementation remain pending

This decision supersedes only the fixed or observation-common planning
interpretations identified in the
[authority crosswalk](WP7_RTC_SCAN_ARRAY_PLANNING_AUTHORITY_CROSSWALK_2026-08-29.md).
It does not reopen unrelated SCI-RTC decisions or the accepted network-timed
identity RTC route.

## Governing science requirement

RTC shall preserve the beam-convolved astronomical signal for every occurrence
scientifically admitted by a candidate mode while avoiding retention of sky
modes shifted into the detector `1/f` regime by inadequately slow scanning.
Candidate low-pass and integer-downsampling evidence is derived
deterministically for each science scan, TolTEC array, exact input cadence, and
mode from authoritative telescope motion, the array beam model, and universal
scientific tolerances. A historical cutoff, convenient factor, velocity
percentile, detector timestream, chunk partition, or cross-network container
shape is not planning authority.

## Occurrence admission

The exact minimum independently admissible on-sky speed is

```text
v_min = 1 arcsec / s
```

For scan `s`, define

```text
S_s = { q : q is in the science scan,
              its realized telescope state and derivative are valid, and
              v(q) >= v_min }

v(q) = norm(d theta(q) / dt)
v_max,s = max over q in S_s of v(q)
```

Equality at `1 arcsec/s` is admitted. The threshold is an occurrence-admission
boundary, not a low-pass-cutoff floor:

- an occurrence with valid realized speed below the threshold is unavailable
  as an independent astronomical measurement with typed RTC cause
  `below_minimum_science_scan_speed`;
- this pair-wide astronomical disposition does not erase producer `x/r`
  availability, member-local numerical validity, or member-local causes;
- invalid derivatives, invalid telescope state, telemetry defects, slews, and
  non-science motion retain their own typed causes and are not relabelled as
  merely slow;
- a scan with no nonempty admitted run has a truthful
  no-admitted-astronomical-product disposition and does not publish an ordinary
  astronomical timestream product for that scan; and
- RTC does not inspect detector values to make or override this decision.

AST shall retain the actual maximum over `S_s`. A percentile or clipped maximum
is not an authorized substitute for that AST diagnostic. Under the later
approved
[`wp7-rtc-occurrence-speed-admission-v1`](WP7_RTC_OCCURRENCE_SPEED_ADMISSION_OWNER_AUTHORITY_2026-08-30.md),
the raw maximum no longer sets one whole-scan RTC plan. Each array, exact
cadence, and certified mode declares an inclusive physical upper-speed ceiling.
An AST-valid occurrence above it is unavailable for that mode with typed cause
`scan_speed_above_mode_support`. AST owns the realized trajectory and validity;
RTC must not repair or relabel those facts.

Admitted occurrences form bounded runs. Filter state and numerical support may
not cross a lower-speed, upper-speed, invalid, non-science, or producer/AST gap
boundary. An output whose required filter support intersects such a boundary
is unavailable unless an approved boundary rule provides complete valid
support. Chunk boundaries are engineering boundaries and neither split these
runs nor reset the scientific operator.

## Authoritative array beam

Planning uses one circular diffraction-limited reference beam per TolTEC array
at its approved nominal center frequency:

```text
theta_DL,a = C_beam * c / (D * nu_0,a)
```

The authoritative array-model artifact shall bind, for every array:

- stable array identity;
- nominal center frequency `nu_0,a` and unit;
- aperture value and convention `D`;
- beam-width coefficient and convention `C_beam`;
- the normalized circular diffraction-limited profile used to construct the
  temporal point-source response; and
- artifact identity, version, precision, and change authority.

RTC planning shall not use scan-direction projections, detector beam fits,
empirical ellipticity, observation-local effective PSFs, or real-beam
broadening to claim a smaller required bandwidth. Existing wavelength,
frequency, or empirical-FWHM constants in implementation code are evidence,
not this approved artifact.

## Array/scan science bandwidth

For every array `a`, exact cadence family `c`, and certified mode `m`, derive
the inclusive realized-speed ceiling `v_limit(a,c,m)` for which the approved
margins, full science band, and complete certified product behavior hold. Scan
the authoritative circular beam at the margin-adjusted ceiling when certifying
the entry. The limits must cover the applicable point-source peak,
integrated flux, beam width/shape, centroid, and calibration-transfer errors.
An arbitrary beam-crossing, half-power, `3 dB`, or historical filter frequency
is not authoritative unless it is proven to meet those product-level limits.

The realized response shall obey

```text
1 - delta_p,a,c,m <= |H_a,c,m(f)| <= 1 + delta_p,a,c,m
for 0 <= |f| <= f_sci,a,c,m
```

and the approved phase/centroid constraint over the same band. Because motion
below `v_min` is inadmissible, the planner shall not narrow the science band to
preserve slower occurrences. The lowest planned astronomical bandwidth is the
one produced by the relevant array beam at `v_min`.

## Automatic factor and filter selection

For each scan and array, and for each exact input cadence, inspect the
applicable entries in the approved finite, pre-certified filter bank. For
candidate factor `M`,

```text
f_Nyq,out = f_sample,in / (2 * M)
```

An immutable candidate entry is structurally eligible only when it
simultaneously provides:

1. the complete science passband through `f_sci,a,c,m`;
2. the approved passband amplitude and phase behavior;
3. a realizable transition band;
4. broadband behavior meeting the noise-weighted alias budget;
5. the approved minimum sampling of the diffraction-limited beam;
6. bounded support and acceptable edge loss; and
7. one identical numerical operator, occurrence selection, and support
   relation for paired `x` and requested conditioned `r`.

Filter design, response evaluation, noise-envelope integration, and end-to-end
map/OOF comparison occur offline. Ordinary Citlali performs no filter synthesis,
order estimation, response optimization, or detector-PSD estimation. Narrow
lines below input Nyquist are handled by the separate line-detection/mitigation
strategy and do not set the generic filter-bank attenuation or factor. The
prior largest-factor and scan-wide `M=1` failure rules are superseded. `M=1`
applies the physical ceiling occurrence by occurrence; a nonidentity mode
retains only outputs with complete admitted support. The certification program
reports raw and support-eroded duration, weighted exposure, spatial coverage,
response, noise, and performance for every candidate. Automatic factor
selection and the final no-product cause require a later bounded owner
decision. Never reduce the physical science band or substitute a percentile to
make a preferred factor pass.

Different arrays in one scan may have different filters, factors, and output
cadences. Each realization remains bound to its source network axis and creates
a new network-scoped occurrence/time/support relation when `M > 1`. Networks
in the same array may share equal resolved numerical plan values only when
their exact planning inputs, including cadence, make them equal; this does not
create a common time grid. A cross-network common analysis grid remains a
separate ALIGN-owned relation requested only by named synchronous mathematics
under ADR 0015.

## Immutable lifecycle and support

The planner consumes bounded typed views or stable handles to immutable
network timing, AST trajectory/validity, array model, and universal policy. It
does not copy telescope axes, pointing planes, or full support merely to place
them in `RtcEvidence`.

`RtcPlan` records the complete deterministic scan/array decision before Apply:
admission-policy identity, scan and array identity, exact input cadence,
diagnostic `v_max,s`, mode upper-speed ceiling, occurrence-admission summary,
beam-model identity, certified science band, factor, certified bank-entry and
coefficient identities, response and noise-weighted alias bounds, phase,
output-occurrence rule, run/boundary policy, support, precision, and all policy
identities. The plan cannot change by chunk or by detector data.

Apply uses the same occurrence action and ordinary operator for paired `x/r`
while retaining member-local availability, validity, and causes. The compact
realization records what the immutable plan actually realized; it does not
duplicate the product, support planes, input axes, or provenance history.

## Corrected numerical closure

The scientific owner approved v1, narrowly superseded its filter/error clauses
with
[`wp7-rtc-scan-array-numerical-policy-v2`](WP7_RTC_SCAN_ARRAY_FILTER_BANK_OWNER_AUTHORITY_2026-08-30.md)
on 2026-08-30, and later superseded scan-wide upper-speed admission and
selection with
[`wp7-rtc-occurrence-speed-admission-v1`](WP7_RTC_OCCURRENCE_SPEED_ADMISSION_OWNER_AUTHORITY_2026-08-30.md).
The retained numerical rules include the nominal array frequencies, aperture
and Airy profile, full optical temporal support, output beam sampling,
integer-factor universe, support/edge behavior, arithmetic, and uncertainty
margins. V2 replaces the v1 product/ripple and
spectrum-independent alias bounds with independent `1%` mapped-response limits
and a noise-weighted `1%` retained-variance alias limit, and it replaces runtime
Kaiser synthesis with lookup of a pre-certified filter-bank entry.

The v1 synthetic factor and tap-count sweep is historical feasibility evidence,
not a v2 plan, approved coefficient artifact, or observation-152390 result.
Representative occurrence/support loss, PSD, naive/JINC, OOF/fruitloops, and
coefficient certification remain acceptance gates. Automatic factor selection
remains pending owner closure after that evidence.

## Scope and claims

This authority does not authorize despiking, level shifts, detector-informed
sampling learning, CAL, VAL, PTC, runtime MAP/JINC planning, persistent TOD
publication, production activation, or legacy-route retirement. Offline
naive/JINC and OOF/fruitloops certification is an acceptance requirement, not
authorization to change those algorithms. This authority does not claim
astronomical transfer, observational performance, or implementation
conformance. The accepted `M=1` network-timed terminal route remains unchanged
and available while the approved
[`wp7-ast-scan-motion-v1`](WP7_AST_SCAN_MOTION_OWNER_DECISION_PACKET_2026-08-30.md)
role and nonidentity RTC implementation are pending conformance.
