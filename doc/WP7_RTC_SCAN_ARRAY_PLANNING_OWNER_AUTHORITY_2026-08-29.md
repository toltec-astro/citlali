# WP-7 RTC Scan/Array Planning Scientific-Owner Authority

Date: 2026-08-29

Scientific owner: Grant Wilson

Status: approved bounded successor authority; numerical policy closed by
scientific-owner disposition 2026-08-30; AST prerequisite and nonidentity
implementation remain pending

This decision supersedes only the fixed or observation-common planning
interpretations identified in the
[authority crosswalk](WP7_RTC_SCAN_ARRAY_PLANNING_AUTHORITY_CROSSWALK_2026-08-29.md).
It does not reopen unrelated SCI-RTC decisions or the accepted network-timed
identity RTC route.

## Governing science requirement

RTC shall preserve the beam-convolved astronomical signal for every
scientifically admitted telescope motion while avoiding retention of sky modes
shifted into the detector `1/f` regime by inadequately slow scanning. The
low-pass and integer downsampling plan is derived deterministically for each
science scan and TolTEC array from authoritative telescope motion, the array
beam model, the exact input cadence, and universal scientific tolerances. A
historical cutoff, convenient factor, detector timestream, chunk partition, or
cross-network container shape is not planning authority.

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

Use the actual maximum over `S_s`. A percentile or clipped maximum is not an
authorized substitute. AST owns the realized on-sky trajectory, derivative,
science-scan membership, validity, and telemetry-defect facts needed by this
calculation. AST must reject or flag an invalid velocity spike before it can
enter `S_s`; RTC must not silently repair one or allow it to set the plan.

Admitted occurrences form bounded runs. Filter state and numerical support may
not cross an inadmissible occurrence or a producer/AST gap. An output whose
required filter support intersects such a boundary is unavailable unless an
approved boundary rule provides complete valid support. Chunk boundaries are
engineering boundaries and neither split these runs nor reset the scientific
operator.

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

For every array `a` and scan `s`, scan the authoritative circular beam at
`v_max,s` and derive `f_sci,a,s` from the approved limits on the resulting
astronomical product. The limits must cover the applicable point-source peak,
integrated flux, beam width/shape, centroid, and calibration-transfer errors.
An arbitrary beam-crossing, half-power, `3 dB`, or historical filter frequency
is not authoritative unless it is proven to meet those product-level limits.

The realized response shall obey

```text
1 - delta_p,a,s <= |H_a,s(f)| <= 1 + delta_p,a,s
for 0 <= |f| <= f_sci,a,s
```

and the approved phase/centroid constraint over the same band. Because motion
below `v_min` is inadmissible, the planner shall not narrow the science band to
preserve slower occurrences. The lowest planned astronomical bandwidth is the
one produced by the relevant array beam at `v_min`.

## Automatic factor and filter selection

For each scan and array, and for each exact input cadence to which the plan is
bound, evaluate the approved finite set of integer factors. For candidate
`M`,

```text
f_Nyq,out = f_sample,in / (2 * M)
```

Select the largest allowed `M` for which the simplest approved low-pass
realization simultaneously provides:

1. the complete science passband through `f_sci,a,s`;
2. the approved passband amplitude and phase behavior;
3. a realizable transition band;
4. stopband behavior meeting the alias-error budget before output Nyquist;
5. the approved minimum sampling of the diffraction-limited beam;
6. bounded support and acceptable edge loss; and
7. one identical numerical operator, occurrence selection, and support
   relation for paired `x` and requested conditioned `r`.

The required stopband attenuation is derived from the alias-error budget; it
is not a conventional decibel default. The transition and support are selected
by a deterministic tie rule as the simplest permitted realization meeting all
constraints. If no `M > 1` passes, select `M=1` with no sampling change only
when the input cadence itself still meets the approved science-band and
beam-sampling requirements. Otherwise publish no admitted ordinary
astronomical product with typed cause
`input_cadence_inadequate_for_science_band`. The new planner's
occurrence-admission dispositions still apply; this fallback does not rewrite
the separate accepted identity-RTC conformance context. Never reduce
`f_sci,a,s` to make a preferred factor pass.

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
`v_max,s`, beam-model identity, `f_sci,a,s`, factor, filter/coefficient identity,
response and alias bounds, phase, output-occurrence rule, run/boundary policy,
support, precision, and all policy identities. The plan cannot change by chunk
or by detector data.

Apply uses the same occurrence action and ordinary operator for paired `x/r`
while retaining member-local availability, validity, and causes. The compact
realization records what the immutable plan actually realized; it does not
duplicate the product, support planes, input axes, or provenance history.

## Approved numerical closure

The scientific owner approved
[`wp7-rtc-scan-array-numerical-policy-v1`](WP7_RTC_SCAN_ARRAY_NUMERICAL_CLOSURE_PACKET_2026-08-29.md)
on 2026-08-30. That authority fixes the nominal array frequencies, aperture
and Airy profile, full optical temporal support, independent astronomical
product tolerances, passband/phase/DC rules, folded-alias norm, output beam
sampling, integer-factor set, deterministic Kaiser FIR construction and tie
rule, support/edge behavior, arithmetic, uncertainty margins, and the
inadequate-input `M=1` disposition.

The packet's synthetic factor and tap-count sweep remains feasibility evidence,
not approved coefficient artifacts or an observation-152390 plan. Exact
coefficient construction and certification remain implementation gates.

## Scope and claims

This authority does not authorize despiking, level shifts, detector-informed
sampling learning, CAL, VAL, PTC, MAP/JINC, persistent TOD publication,
production activation, or legacy-route retirement. It does not claim
astronomical transfer, observational performance, or implementation
conformance. The accepted `M=1` network-timed terminal route remains unchanged
and available while the AST prerequisite and nonidentity implementation are
pending.
