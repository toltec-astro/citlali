# WP-7 RTC Scan/Array Decimation Authority Packet

Status: **historical planning packet; numerical budgets retained, scan-wide
upper-speed planning and automatic factor selection superseded; occurrence-
support evidence, selection closure, certified filter bank, and implementation
pending**

Prepared: 2026-08-29

Revised by owner decisions: 2026-08-29 and 2026-08-30, including the v2
filter-bank correction

Accepted identity-RTC base:
`0574d9a50fe6df6f7ded07c1d229bcb8ca04309d`

Historical preparation commit:
`6f59f0a13d3fa3090fc55155ec1e4d30d6e2b815`

## Disposition of the original proposal

The historical preparation proposed one fixed, network-local `M=2` conformance
witness. Its inspection of representative observation 152390, legacy `32 Hz`
configuration, incumbent decimator, paired support, edge behavior, compact
lifecycle, and implementation candidates remains design evidence.

The fixed-factor recommendation is superseded. The scientific owner selected a
deterministic scan/array planner because the required temporal bandwidth is set
by realized science-scan velocity and each array's authoritative beam. The
bounded successor authority is recorded in:

- [scan/array planning owner authority](WP7_RTC_SCAN_ARRAY_PLANNING_OWNER_AUTHORITY_2026-08-29.md);
- [frozen-entry crosswalk](WP7_RTC_SCAN_ARRAY_PLANNING_AUTHORITY_CROSSWALK_2026-08-29.md);
- [current v2 numerical policy](WP7_RTC_SCAN_ARRAY_FILTER_BANK_OWNER_AUTHORITY_2026-08-30.md);
- [occurrence-level upper-speed authority](WP7_RTC_OCCURRENCE_SPEED_ADMISSION_OWNER_AUTHORITY_2026-08-30.md);
- [historical v1 numerical packet](WP7_RTC_SCAN_ARRAY_NUMERICAL_CLOSURE_PACKET_2026-08-29.md);
- [ADR 0019](adr/0019-scan-array-rtc-bandwidth-planning.md); and
- [ADR 0018](adr/0018-network-specific-timing-and-common-analysis-grid.md)
  for network timing and explicit common-analysis grids.

The corrected v2 numerical policy closes the science budgets formerly left
open by this packet. The v1 prototype factor/tap sweep remains historical
evidence only; no nonidentity code should be constructed from those estimated
factors or tap counts.

## Retained implementation and data evidence

### Representative observation 152390

The 11 locally retained TolTEC network files for observation 152390 all
declare:

- `Header.Toltec.FpgaFreq = 256000000 Hz`;
- `Header.Toltec.AccumLen = 2097152`; and
- `Header.Toltec.SampleFreq = 122.0703125 Hz`.

At that exact cadence, an `M=2` output would have rate `61.03515625 Hz` and
Nyquist `30.517578125 Hz`. The retained `redu04` configuration disables
downsampling with factor `1`, while its enabled legacy RTC filter uses
`freq_high_Hz: 32.0`, `n_terms: 32`, and `a_gibbs: 50.0`. That `32 Hz` edge is
above the `M=2` output Nyquist and cannot be adopted unchanged as the
anti-alias authority.

These are exact input and configuration facts for one representative workload.
They do not set an array's astronomical band, an alias budget, or a universal
factor.

### Incumbent implementation

The legacy `timestream::Downsampler` selects rows `0, M, 2M, ...` and ORs flags
over the following block of at most `M` rows. The surrounding path separately
downsamples telescope, pointing, polarization, and kernel containers. The
legacy FIR leaves edge cells outside its convolution body unchanged and does
not normalize the generated coefficient sum.

That code is not a conforming successor planner or operator. It lacks the
complete scan/array scientific decision, network-scoped successor identity,
transitive filter support, typed coordinate availability, canonical paired
operator, response/alias statement, and chunk-independent scientific run
semantics. None of its factor, filter, flag, or edge choices is promoted by
this packet.

### Existing array constants

The repository contains legacy wavelength-derived frequency and expected-FWHM
maps. They are implementation evidence only. They do not state the approved
nominal-frequency convention, telescope aperture convention, diffraction
coefficient/profile, precision, artifact identity, or change authority needed
by the new planner.

## Approved planning rule

### Science occurrence admission

Use the exact inclusive boundary

```text
v_min = 1 arcsec/s
admitted when valid science-scan speed v >= v_min
```

This is not a filter cutoff floor. Valid occurrences below the boundary are
excluded as independent astronomical measurements with typed cause
`below_minimum_science_scan_speed`. Invalid telescope state, derivative, or
telemetry and non-science motion keep their distinct causes. A scan with no
nonempty admitted run produces no admitted ordinary astronomical timestream
product.

The exclusion is a conservative pair-wide astronomical action because `x` and
`r` share the same network occurrence, while member-local producer validity,
availability, and causes remain distinct. RTC neither erases those facts nor
uses detector values to make the motion decision.

Admitted runs stop at slow, invalid, gap, slew, or non-science occurrences.
Filter support may not cross such a boundary. A candidate output lacking
complete approved input support is unavailable with a typed support cause;
engineering chunks do not create or remove these boundaries.

### Velocity authority

AST supplies the immutable science-scan membership, reconstructed on-sky
trajectory, scalar velocity, and their validity/cause facts. For scan `s`, form
the lower-speed-admitted set `S_s` and retain the actual diagnostic maximum

```text
v_max,s = max(v(q) for q in S_s)
```

with no percentile substitution. The later occurrence-speed authority
supersedes using this maximum to bind the whole RTC plan. Each
array/cadence/mode instead declares an inclusive physical ceiling and excludes
only occurrences above it. An invalid velocity spike remains an AST cause and
cannot be relabelled by RTC.

### Array beam and science band

Use one circular diffraction-limited reference beam per TolTEC array:

```text
theta_DL,a = C_beam * c / (D * nu_0,a)
```

The future authoritative array artifact binds exact array identity, nominal
center frequency, aperture value/convention, beam coefficient/width convention,
normalized circular profile, precision, version, and artifact identity. Do not
use scan-direction projections, per-detector fits, empirical ellipticity, or
observation-local effective PSFs for RTC planning.

For every array `a`, cadence `c`, and certified mode `m`, derive an inclusive
`v_limit(a,c,m)` after the approved margins and certify the complete beam band
at that ceiling. The astronomical tolerances
must constrain the applicable point-source peak, integrated flux, beam shape or
broadening, centroid, and calibration transfer. A conventional crossing,
half-power, `3 dB`, or historical cutoff is not the criterion.

The realized response must satisfy

```text
1 - delta_p,a,s <= |H_a,s(f)| <= 1 + delta_p,a,s
for 0 <= |f| <= f_sci,a,s
```

plus the approved phase/centroid behavior. Since occurrences below `v_min` are
not admitted, the planner never narrows the science band to retain them.

### Factor and filter selection

For every scan, array, and exact input cadence, evaluate every member of the
approved finite integer-factor set. Candidate `M` has

```text
f_Nyq,out = f_sample,in / (2 * M)
```

An immutable pre-certified candidate bank entry must satisfy all of:

1. complete passband through `f_sci,a,s`;
2. approved amplitude and phase behavior;
3. realizable transition;
4. the noise-weighted broadband alias budget;
5. approved minimum beam sampling;
6. support and edge-loss limits; and
7. identical numerical transformation, output selection, and support for
   paired `x/r`.

Filter design, PSD integration, response analysis, and native-rate versus
filtered naive/JINC and OOF/fruitloops comparisons occur offline. Runtime
selection is a bounded table lookup; it does not synthesize or optimize a
filter or estimate a detector PSD. Narrow sub-input-Nyquist lines remain owned
by the established line-detection/mitigation strategy and do not set the
generic broadband filter length or factor. `M=1` and nonidentity candidates
apply upper-speed admission occurrence by occurrence; filtered outputs require
complete admitted support. The scan-wide fallback/failure and largest-factor
rules are superseded. Automatic selection and the final no-product cause await
representative retained-support evidence and a later owner decision. Never
reduce the science band or use a percentile to admit a desired factor.

Every sampling-changing product remains a network-keyed timed stream with a new
network occurrence/time/support relation. The planner neither requests nor
constructs a common analysis grid. Equal plan values across networks do not
synchronize their axes.

## Lifecycle and product consequences

Extend the accepted immutable context/evidence/plan/apply/result/realization
lifecycle. `RtcEvidence` references rather than duplicates network timing, AST,
array-model, producer validity, and support facts. It owns only derived compact
evidence such as admitted-run summaries and velocity extrema.

`RtcPlan` is bound to the scan, array, exact input cadence, source networks,
admission policy, AST facts, beam artifact, universal policy, selected factor,
coefficient/realization artifact, response, phase, alias, run/boundary,
occurrence, support, and arithmetic identities. Selection completes before
Apply.

The numerical product owns its new network-scoped axes and contiguous paired
data where required. `RtcRealization` remains a compact account of the
immutable plan actually realized; it does not copy full products, axes,
support planes, or provenance history. No persistent RTC TOD schema is added
without an approved immediate consumer.

## Numerical authority correction

The scientific owner approved v1 and then narrowly superseded its response,
alias, and runtime-filter-design clauses with
[`wp7-rtc-scan-array-numerical-policy-v2`](WP7_RTC_SCAN_ARRAY_FILTER_BANK_OWNER_AUTHORITY_2026-08-30.md)
on 2026-08-30. V2 binds the `1%` mapped astronomical-response budget, the
noise-weighted `1%` retained-variance alias budget, offline naive/JINC and
OOF/fruitloops certification, pre-certified bank lookup, and separate line
ownership. The later occurrence-speed authority supersedes the retained
scan-wide `M=1` disposition and largest-factor lookup while preserving the
numerical budgets and margins.

The approved
[`wp7-ast-scan-motion-v1`](WP7_AST_SCAN_MOTION_OWNER_DECISION_PACKET_2026-08-30.md)
velocity/validity role is implemented and conforming. The scan/array planner
remains unavailable pending occurrence/support evidence, automatic-selection
owner closure, filter-bank certification, and bounded implementation gates.
The separate accepted identity RTC remains available unchanged.

## Evidence study under the corrected numerical policy

Generate the exact array-model, representative cleaned-noise PSD envelopes,
and candidate filter-bank artifacts first. Then, for representative scans
including observation 152390:

- verify AST-valid admitted runs and the exact `1 arcsec/s` equality boundary;
- report diagnostic `v_max,s`, every candidate ceiling, raw upper-speed
  exclusion, complete-support erosion, retained weighted exposure, spatial
  coverage, and certified-entry disposition per scan/array/network;
- compare native-rate and filtered point-source and OOF products through naive,
  JINC, and the applicable OOF/fruitloops route;
- compute peak, flux, width/shape, centroid, calibration, passband, phase, and
  broadband noise-weighted alias results independently of runtime selection;
- compare direct symmetric FIR/polyphase and FFT overlap-save only where
  realized support and segment sizes make them plausible;
- measure complete paired-route time, allocation, memory movement, and RSS on
  representative network/detector/occurrence sizes; and
- retain distinct network times and verify that a gap or slow run on one
  network manufactures no ordinary-RTC state on another.

Eigen, FFTW, OpenMP, a reusable workspace, and any particular C++23 view remain
implementation candidates. Choose among them from end-to-end evidence, not
baseline conformance.

## Required implementation gates

Under the corrected numerical policy, the bounded increment must prove at least:

- accepted `M=1` behavior remains bitwise and semantically unchanged;
- below/exact/above `1 arcsec/s` admission and typed-cause behavior;
- no admitted scan product when no valid run reaches the threshold;
- invalid spikes, slow turnarounds, slews, and telemetry defects retain exact
  causes and cannot enter candidate support;
- upper-bound below/exact/above admission is inclusive at equality and uses
  exact cause `scan_speed_above_mode_support` above it;
- filter support cannot cross an inadmissible run boundary;
- `M=1` consequences are occurrence-local and nonidentity outputs require
  complete admitted footprints;
- raw and support-eroded count, time, exposure, and coverage consequences are
  reported without selecting a factor;
- exact per-array differences are preserved without a common grid;
- new network occurrence, time, representative-source, and transitive-support
  relations are exact;
- paired `x/r` use an identical operator with member-local availability;
- every accepted entry passes the independent `1%` mapped-response metrics and
  the `1%` retained broadband-noise-variance alias limit through its required
  naive, JINC, and OOF/fruitloops certification matrix;
- sub-input-Nyquist lines remain routed to line detection/mitigation, with any
  anti-alias-relevant mitigation effective before decimation;
- one-segment and supported multi-chunk results agree under the declared
  arithmetic behavior;
- stale AST, beam, policy, plan, coefficient, context, or run identity fails
  before publication;
- failure publishes no false completion or persistent TOD schema;
- focused dependency, public-header, and scientific tests pass;
- all repository and unchanged-legacy gates pass;
- representative real-data evidence uses the accepted exact dataset/SHA
  discipline; and
- a fresh independent exact-SHA review returns PASS.

## Current stop condition

The owner has approved the numerical budgets and occurrence-level upper-speed
correction, superseding fixed `M=2`, runtime Kaiser design, scan-wide maximum
admission, and automatic largest-factor selection. AST conformance is closed.
The next bounded action is candidate occurrence/support census repair, followed
by PSD/filter-bank and end-to-end evidence. Automatic factor selection returns
for owner closure after that evidence; nonidentity RTC implementation does not
begin before the prerequisites conform.
Production activation, CAL, VAL, PTC/PCA expansion, runtime MAP/JINC planning,
and legacy-route retirement remain outside scope.
