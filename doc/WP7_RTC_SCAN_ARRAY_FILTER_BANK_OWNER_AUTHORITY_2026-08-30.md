# WP-7 RTC Scan/Array Filter-Bank Scientific-Owner Authority

Date: 2026-08-30

Scientific owner: Grant Wilson

Authority identity: `wp7-rtc-scan-array-numerical-policy-v2`

Status: **approved bounded correction; AST authority closed by
`wp7-ast-scan-motion-v1`; bounded AST implementation passes local,
representative-data, and fresh exact-SHA conformance gates; certified filter-
bank artifacts, RTC implementation, and acceptance evidence pending**

Supersedes only the response-budget, alias-budget, and runtime-filter-design
clauses of
[`wp7-rtc-scan-array-numerical-policy-v1`](WP7_RTC_SCAN_ARRAY_NUMERICAL_CLOSURE_PACKET_2026-08-29.md).
All v1 clauses not named below remain authoritative.

## Governing scientific claim

For every admitted scan speed, array, native sample-rate family, and supported
decimation factor, RTC shall:

1. preserve the beam-convolved astronomical response to within `1%` under the
   offline end-to-end acceptance comparisons defined below; and
2. add no more than `1%` to the variance of the broadband noise expected to
   survive timestream cleaning.

Production Citlali selects a versioned precomputed and pre-certified filter-bank
entry. It does not synthesize a filter, estimate its order, optimize its
frequency response, or estimate a detector PSD during an ordinary reduction.

## Astronomical-response budget

Retain the v1 beam-derived science-band definition through the full ideal
aperture temporal support. Replace the v1 independent `1e-3` product limits and
`1e-4` passband-ripple allocation with the following limits. Relative to the
native-rate reference processed through the same mapmaking and analysis route,
each applicable filtered/decimated result shall independently satisfy:

| Metric | Maximum filter/decimation change |
| --- | ---: |
| point-source map peak | `1e-2` relative |
| integrated point-source response | `1e-2` relative |
| normalized map-profile residual | `1e-2` of the reference peak |
| fitted beam FWHM | `1e-2` relative |
| fitted centroid | `1e-2` reference-beam FWHM |
| calibration-transfer magnitude | `1e-2` relative |

An at-most-`1e-2` passband magnitude error throughout the adopted science band
is an approved conservative engineering surrogate. A filter-bank construction
may target a smaller value, such as `5e-3`, when inexpensive, but that smaller
target is not scientific authority and shall not be promoted without a new
owner decision.

The v1 centered-zero-phase requirement and `1e-12` DC-gain-error bound remain
unchanged. Exact unit DC response preserves a constant signal and, apart from
declared unavailable edge support, the time-integrated response.

## Offline map and OOF certification

Each proposed bank entry is compared with the native-rate reference at the
same admitted scan-speed ceiling and source phase through both:

- naive MAP; and
- JINC MAP.

The comparison covers a diffraction-limited point source and a representative
OOF template. For the mapmaking route used by OOF, the native-rate and
filtered/decimated products are also passed through fruitloops. Added map
structure must satisfy the independent `1%` limits above, and recovered OOF
outputs must satisfy the separately governed OOF acceptance criteria without a
meaningful new coherent bias. This authority does not invent a new OOF-solution
tolerance or make MAP/JINC part of runtime RTC planning.

Initially one bank entry is accepted only when it passes both naive and JINC.
Mapmaker-specific RTC filter policies require a separate measured benefit and
owner decision.

## Broadband alias-noise budget

Replace the v1 pointwise, spectrum-independent folded-alias power-gain bound
with a noise-weighted certification. For output rate
`f_out = f_sample,in/M`, define over the retained output Nyquist interval

```text
P_alias = integral sum over nonzero images k
          |H(f + k f_out)|^2 N(f + k f_out) df

P_retained = integral |H(f)|^2 N(f) df
```

using the exact signed and mirrored image mapping for real sampled data. Every
certified representative broadband-noise envelope shall satisfy

```text
P_alias / P_retained <= 1e-2.
```

This permits at most a `1%` increase in retained noise variance, or about a
`0.5%` increase in RMS noise. The denominator and PSD envelope shall represent
the noise expected to survive into the cleaned timestream or map: photon,
detector, and readout noise plus representative residual atmosphere. Raw
low-frequency atmospheric variance that is expected to be removed by PTC must
not dominate the denominator and conceal an alias contribution that survives
cleaning.

The certification artifact binds the PSD-envelope identity, input cadence,
integration convention, filter response, image set, numerator, denominator,
and result. Alias-induced map-noise increments through naive and JINC must also
remain within the same `1%` variance budget relative to their native-rate
references.

## Narrow-line ownership

Narrow lines below the native input Nyquist are not part of the representative
broadband PSD envelope and do not set the generic low-pass length, attenuation,
or decimation factor. The established line-detection/mitigation strategy owns
their detection and treatment.

When such a line lies above a selected output Nyquist and could fold into the
retained band, its effective mitigation must occur before the information-losing
decimation. This is an ordering consequence of the existing line strategy, not
a new filter-bank line budget or a new line algorithm. This correction neither
redesigns nor independently approves line detection, notch construction,
source protection, validity, causes, or support behavior.

## Pre-certified filter bank

Offline certification produces a small versioned bank. Each entry binds at
least:

- array identity;
- native sample-rate family and admitted cadence interval;
- integer factor `M`;
- maximum admitted scan velocity, including the approved velocity margin;
- coefficient bit patterns and numerical operator identity;
- phase, DC response, support, edge, and arithmetic policy;
- point-source, naive, JINC, OOF, and fruitloops certification results;
- broadband PSD-envelope identities and alias-noise results; and
- artifact version, immutable identity, and change authority.

Filter family, design procedure, and tap count are offline engineering choices.
The v1 Kaiser-windowed-sinc construction and prototype tap counts remain useful
historical design evidence but are no longer scientific authority. A Kaiser
entry may be retained if it is the simplest well-performing certified choice;
it is not retained merely for v1 conformance.

At scan setup, for each array and exact native cadence, Citlali obtains the
authoritative `v_max`, applies the approved margins, and selects the largest
permitted factor having a certified bank entry that admits the resulting
velocity and cadence. This is a bounded table lookup, not runtime filter
design. If no larger entry applies, the unchanged v1 `M=1` and
`input_cadence_inadequate_for_science_band` disposition governs.

The permitted factor universe remains every integer in `[1, 256]`; the bank
need contain only the combinations that have actually passed certification.
Absence of a certified entry never authorizes synthesis or science-band
relaxation during a reduction.

## Preserved v1 authority

The correction does not change:

- exact 272/214/150 GHz array center frequencies;
- the exact 50.0 m unobscured circular Airy planning model and coefficient;
- the full ideal-aperture temporal science-band definition;
- at least four output samples per Airy FWHM;
- the permitted integer-factor universe `[1, 256]`;
- the five-second maximum filter half-support and run-boundary behavior;
- centered zero phase, unit-DC normalization, and the `1e-12` DC error bound;
- binary64 coefficient/data/arithmetic policy;
- the 5% velocity and 100 ppm cadence margins;
- network-specific timing, paired `x/r` operator/support behavior, and chunk
  invariance; or
- the inadequate-input `M=1` disposition and exact typed cause.

## Evidence and implementation boundary

The v1 feasibility calculator and its predicted factors/tap counts do not
select a v2 plan. They may be retained as historical design evidence only.

Before a nonidentity RTC route can be accepted, the project still needs:

1. a conforming implementation of the approved
   [`wp7-ast-scan-motion-v1`](WP7_AST_SCAN_MOTION_OWNER_DECISION_PACKET_2026-08-30.md)
   science-scan membership, velocity, validity, and cause authority;
2. representative broadband PSD envelopes appropriate to cleaned products;
3. versioned filter-bank entries and independent numerical certification;
4. native-rate versus filtered naive, JINC, and OOF/fruitloops evidence;
5. bounded learn-consider-apply implementation and repository gates;
6. representative paired-data acceptance evidence; and
7. fresh independent exact-SHA conformance review.

No common analysis grid, persistent RTC TOD schema, production activation,
CAL, VAL, PTC/PCA expansion, or new MAP/JINC algorithm is authorized here.

## Exact owner correction

The scientific owner supplied and agreed with the scientific assessment that
motivated this correction, then added:

```text
The only point of disagreement I have is in the concern about lines. In-band
(sub nyquist) lines will be dealt with by our line detection/mitigation
strategy.
```

This authority incorporates that clarification by separating the broadband
alias certification from line ownership and by requiring any anti-alias-
relevant line mitigation to precede information-losing decimation.
