# WP-7 RTC Filtering and Downsampling Certification Test Plan

Date: 2026-08-30

Status: **accepted bounded execution plan, corrected for occurrence-level
upper-speed admission; no nonidentity RTC implementation or production
activation authorized by this document**

Governing authority:

- [`wp7-rtc-scan-array-numerical-policy-v2`](WP7_RTC_SCAN_ARRAY_FILTER_BANK_OWNER_AUTHORITY_2026-08-30.md)
- [`wp7-rtc-scan-array-planning-v1`](WP7_RTC_SCAN_ARRAY_PLANNING_OWNER_AUTHORITY_2026-08-29.md)
- [`wp7-rtc-occurrence-speed-admission-v1`](WP7_RTC_OCCURRENCE_SPEED_ADMISSION_OWNER_AUTHORITY_2026-08-30.md)
- [`wp7-ast-scan-motion-v1`](WP7_AST_SCAN_MOTION_OWNER_DECISION_PACKET_2026-08-30.md)
- [ADR 0016](adr/0016-scan-array-rtc-bandwidth-planning.md)
- [ADR 0017](adr/0017-precertified-rtc-filter-bank-and-science-error-budgets.md)
- [ADR 0018](adr/0018-ast-scan-motion-velocity-and-validity.md)
- [ADR 0019](adr/0019-occurrence-level-rtc-upper-speed-admission.md)

## Purpose and bounded outcome

This plan determines the engineering facts still needed to construct and
certify the first nonidentity, network-timed RTC filter/downsampling bank. It
does not revisit the approved science budgets. It converts the remaining
unknowns into measurements, names the exact primary observations, and defines
the evidence needed before the scientific owner can close automatic bank-entry
selection and production Citlali may use it.

The two primary end-to-end cases are the project owner's proposed cases:

1. Beammap observation `148670`, which supplies a bright compact source,
   detector-resolved beam products, source crossings, calibration transfer,
   and the iterative Beammap route.
2. The standard science sequence `152390`--`152392`, whose science scans are
   `152390` and `152392` and whose associated pointing observation is `152391`.
   It supplies realistic atmospheric and detector noise, source and extended-
   structure response, coaddition, and established naive/JINC products.

Those two cases are the main discovery and regression corpus. They do not by
themselves exercise the already-required OOF solution and fruitloops route.
The established `152385`--`152387` OOF sequence is therefore retained as a
narrow route-completeness witness. This is not an expansion of the scientific
scope: OOF/fruitloops evidence is already an explicit prerequisite of the
approved filter-bank authority.

The bounded outcome is:

- a small set of versioned, certified bank entries for the exact cadence,
  array, factor, and inclusive physical upper-speed domains demonstrated by
  the evidence;
- candidate-specific raw and support-eroded occurrence, duration, weighted-
  exposure, spatial-coverage, response, noise, and performance evidence; and
- an owner packet closing automatic factor selection and the final no-product
  disposition before any production bank is frozen.

Passing these cases does not silently certify unmeasured cadence families or
velocity domains.

## Authority that tests shall not tune

The following are fixed inputs, not free parameters in the experiment:

- array center frequencies `272`, `214`, and `150 GHz`;
- the exact 50 m unobscured circular Airy planning model and approved
  coefficient;
- the full ideal-aperture temporal support used to define the science band;
- at least four output samples per Airy FWHM;
- integer factors `M` in `[1, 256]`;
- per-scan, per-array planning on network-specific native timing axes;
- the inclusive `1 arcsec/s` lower threshold, occurrence-level inclusive
  physical upper-speed ceilings, exact cause
  `scan_speed_above_mode_support`, and unchanged AST validity, derivative,
  cause, and raw-maximum rules;
- a 5% velocity margin and 100 ppm cadence margin;
- centered zero phase, unit DC, and DC-gain error no larger than `1e-12`;
- no filter support across invalid, slow, upper-speed-excluded, gap, or
  physical-run boundaries;
- no more than five seconds of half-support;
- binary64 coefficients, samples, and declared ordered arithmetic;
- identical filter and support for paired `x` and `r`;
- chunk-partition invariance of scientific identities, values, support,
  decisions, causes, pointing correspondence, and declared arithmetic;
- each applicable mapped-response metric independently within `1%` of its
  native-rate reference;
- broadband folded-alias power no larger than `1%` of retained power and no
  more than a `1%` map-noise variance increment through either naive or JINC;
- narrow-line detection and mitigation remaining owned by the established line
  strategy; and
- immutable pre-certified bank entries, with no runtime filter synthesis,
  order search, or PSD estimation. Automatic entry selection is explicitly
  not fixed until the measured owner-closure gate.

A design target tighter than these bounds may be used when inexpensive, but a
test result shall not promote it to scientific authority.

## Questions the program must answer

| ID | Unknown or uncertainty | Evidence that resolves it | Consequence |
| --- | --- | --- | --- |
| U1 | Exact network cadence families, jitter, gaps, and run lengths in the representative observations | Native occurrence/time census for every participating network | Defines candidate bank domains and detects unsupported cadence families |
| U2 | Valid velocity distribution and native-occurrence mapping by scan/network | Approved AST product and exact ALIGN mapped views | Preserves the truthful maximum and supplies occurrence-level candidate admission without using a percentile |
| U3 | Physical upper-speed and retained-support consequences for each array/cadence/factor | Analytic sweep of `M=1..256`, then exact support erosion once a filter exists | Restricts design/replay candidates and supplies the later selection trade rather than choosing a factor |
| U4 | Appropriate broadband PSDs for noise expected to survive cleaning | Native-rate residual PSD measurements on Beammap and science, with source masks, established cleaning, line masks, and detector/network summaries | Produces versioned PSD-envelope candidates rather than using raw removable atmosphere |
| U5 | Sensitivity to the PSD aggregation rule | Compare per-detector worst cases, robust array envelopes, and observation-to-observation variation without selecting an unapproved rule implicitly | Identifies whether an owner choice is still needed before freezing the envelope |
| U6 | Whether a detected line can fold for a candidate factor and whether effective mitigation precedes decimation | Line inventory, selected output Nyquist, and execution-order trace/replay | Withhold that factor unless the established strategy demonstrably protects it before sample removal |
| U7 | Simplest filter family and tap count that meet all budgets | Offline comparison of a small set of mature symmetric linear-phase FIR designs | Engineering selection for each certified entry |
| U8 | Response extrema between a convenient frequency or source-phase grid | Adaptive frequency extremum search plus full sub-output-sample point-source phase sweep | Prevents certification from depending on a lucky grid |
| U9 | Admission/support versus filter versus sample-removal effects | Complete-native R0, mode-matched native Rm, filter-only F, and filtered/decimated D arms | Keeps support cost outside the transfer budget and localizes numerical failures |
| U10 | Paired `x/r`, validity, cause, support, and native-axis correctness | Exact timestream contract comparisons, including gaps and member-local failures | Proves the numerical method preserves the accepted RTC semantics |
| U11 | Map and calibration effects in real reductions | Native versus candidate Beammap and standard-science reductions through naive and JINC | Closes point, beam, profile, centroid, integrated response, transfer, and noise gates |
| U12 | OOF and fruitloops effects | Controlled OOF template plus the `152385`--`152387` route witness | Closes the separately required iterative OOF route without inventing a new OOF tolerance |
| U13 | Edge loss and short-run behavior at real scan boundaries | Support accounting by cause and duration on all three cases, plus synthetic exact-boundary cases | Quantifies exposure loss and rejects filters that pass only on long uninterrupted streams |
| U14 | Runtime, allocation, and memory tradeoffs | Representative direct and, where useful, polyphase benchmarks after numerical survival | Chooses execution architecture from measured workload rather than inheritance |
| U15 | How far the evidence generalizes | Compare cadence, velocity, PSD, and line domains across Beammap, science, and OOF | Bounds each bank entry; identifies rather than guesses missing production domains |

## Test corpus and data gate

### Local inventory on 2026-08-30

The raw inputs needed to begin this program are already on this Mac:

| Case | Local raw-input evidence | Role |
| --- | --- | --- |
| Beammap `148670`, subobservation/scan `(0,2)` | Eleven detector-network files plus telescope and housekeeping files under `/Users/gwilson/work_toltec/local_data/citlali-validation/v1/beammaps/data` (directory total approximately 16 GB) | Primary point/beam/calibration case |
| Standard sequence `152390`--`152392`, `(0,2)` data | Eleven detector-network files for each observation plus telescope, housekeeping, and HWPR inputs under `/Users/gwilson/work_toltec/local_data/citlali-validation/v1/science/data` (directory total approximately 15 GB) | Primary real-science, noise, map, and coadd case |
| OOF `152385`--`152387`, `(0,1)` data | Eleven detector-network files for each observation plus telescope/HWPR inputs under `/Users/gwilson/work_toltec/local_data/citlali-validation/v1/oof/data` (directory total approximately 1.2 GB) | Minimal OOF/fruitloops route gate |

Existing reductions and diagnostics are useful historical context, but they
are not substitutes for fresh native and candidate arms from the same raw
bytes and executable.

Before any scientific comparison, Gate D0 shall write a machine-readable
fixture manifest containing:

- every raw detector, telescope, housekeeping, HWPR, tune, and APT input;
- byte size and SHA-256 for every file;
- observation, subobservation, scan, network, array, and detector inventory;
- exact requested and effective configurations;
- executable Git SHA and build identity;
- external numerical-library and compiler identities;
- thread count, chunking, and arithmetic controls; and
- the immutable relationship between the reference and candidate arms.

These file digests are offline fixture-custody evidence. They are not runtime
`RtcEvidence`, `RtcPlan`, `RtcTimestream`, or `RtcRealization` identity fields,
and the implementation shall not content-hash full timestream planes. Runtime
binding remains lightweight and uses the accepted bounded identities/handles.
The manifest neither copies raw planes nor creates per-sample provenance.

Missing required networks, mismatched APT identity, duplicate inputs,
unexplained cadence domains, or an unhashable input is a D0 failure, not a
skipped test.

### Synthetic fixtures

Small deterministic fixtures remain necessary even with excellent real data.
They cover facts that a finite observation cannot prove:

- constant, impulse, step, ramp, and isolated point-source signals;
- sinusoids at DC, science-band extrema, output Nyquist, image boundaries, and
  native Nyquist;
- diffraction-limited point-source and OOF-template scans at every proposed
  bank entry's maximum admitted velocity, not only at the velocities happened
  to be present in the real observations;
- a dense and adaptively refined sub-output-sample phase domain;
- white, colored, `1/f`, residual-atmosphere, and mixed broadband noise with
  known spectra;
- narrow lines on both sides of a candidate output Nyquist, including an exact
  fold relation;
- independent network axes with distinct cadences and times;
- member-local `x` and `r` invalidity and causes;
- gaps and runs exactly below, at, and above the support boundary; and
- alternate engineering chunk partitions over one identical scientific run.

Synthetic noise seeds and source phases are part of the evidence identity.

## Experimental arms

Every candidate comparison starts from identical immutable input bytes and all
non-RTC settings are held fixed.

| Arm | Purpose | Scientific status |
| --- | --- | --- |
| R0: complete native reference | Accepted identity RTC over all AST-valid lower-speed-admitted native occurrences, with no candidate upper-speed exclusion, filtering, or sample removal | Reference for admission/support cost |
| Rm: mode-matched native reference | Native-rate evaluation with the candidate's upper-speed admission and equivalent complete-support domain, but no candidate filtering or decimation | Reference for numerical filter/downsampling change |
| F: filter-only diagnostic | Apply the exact candidate centered operator but evaluate it at every supported native occurrence | Diagnostic decomposition only; not a production bank product |
| D: filtered/decimated candidate | Apply the candidate operator and retain its declared network-native output occurrences | Proposed production behavior |
| E: execution-equivalence diagnostic | Compare the straightforward ordered reference evaluator with any optimized direct/polyphase realization | Engineering evidence only; each differing operator identity must be declared |

R0 versus Rm measures the scientific support removed by mode admission and
filter-footprint erosion. Rm versus D measures the candidate numerical change
without hiding admission loss inside the `1%` transfer budgets. Arm F separates
distortion introduced by the low-pass response from distortion introduced by
reduced sampling. Arm E is run only after a coefficient set passes the
scientific screens. It does not authorize a numerically different fast path
merely because its aggregate map looks similar.

The legacy fixed-rate filter/downsampler may be measured as historical context,
but it is not an acceptance reference and cannot select a bank entry.

## Candidate-design funnel

The program deliberately reduces the candidate space before running expensive
full reductions.

### F0: cadence, motion, and eligibility census

For every scan and network:

1. reconstruct the accepted native occurrence/time axis;
2. classify cadence intervals and jitter without projecting networks to a
   common analysis grid;
3. map the approved AST velocity/validity product to each native axis;
4. form the AST-valid, lower-speed-admitted base runs;
5. derive each factor's structural inclusive upper-speed ceiling using the
   approved cadence/velocity margins, output sampling, Airy beam, full optical
   band, and native Nyquist constraints;
6. classify every mapped occurrence below, exactly at, or above that ceiling
   with exact typed cause accounting; and
7. report raw excluded count/duration and retained run lengths for all factors
   `1..256` without selecting one.

Before a filter is certified, the structural ceiling is an upper bound on a
future entry's admitted domain; passband, response, alias, and support evidence
may lower it. F0 can report exact M=1 occurrence-local consequences. Filter-
footprint erosion for `M>1` is added only after F1 supplies exact coefficients
and half-support. The compact table contains no filter coefficients and makes
no runtime decision.

### F1: bounded filter-family comparison

Start with three mature, centered, symmetric FIR construction families:

- Kaiser-windowed sinc, retained as the historical and implementation-simple
  baseline;
- Parks--McClellan/equiripple linear-phase FIR; and
- weighted least-squares linear-phase FIR.

For each candidate array/cadence/factor/upper-speed domain, find the shortest
credible designs near the pass/fail boundary, then verify rather than trust the
designer's sampled response. The comparison records:

- exact coefficient bit patterns and normalization;
- tap count and half-support in samples and seconds;
- stable extrema of passband magnitude and folded image response;
- DC error, symmetry, phase, group-center convention, and arithmetic order;
- response sensitivity to coefficient rounding; and
- estimated and measured operations, memory traffic, and workspace needs.

IIR and forward/backward filtering are excluded from the initial comparison:
their state, edge, support, and centered-phase behavior add complexity that is
not justified before symmetric FIR candidates are shown inadequate. This is an
engineering search boundary, not a permanent scientific prohibition.

### F2: independent numerical certification

An independent verifier that does not call the design routine shall:

- recompute exact DC response and symmetry from stored coefficients;
- search frequency-response extrema with dense bracketing and local refinement;
- integrate the exact real-data alias images using the bound PSD artifact;
- test constant, impulse, sinusoid, and source-phase fixtures;
- test point-source and OOF response at the exact inclusive upper-speed ceiling
  claimed by each proposed entry;
- verify no unavailable source occurrence contributes to an output;
- verify the stable native-axis ordinal and output-center convention;
- verify exact support, cause, and boundary accounting; and
- compare multiple engineering chunk partitions.

The design program and verifier shall produce byte-stable machine-readable
artifacts from checked inputs.

### F3: timestream replay

Replay R0, Rm, F, and D over selected whole-network scans from all three cases.
Measure before committing to full reductions:

- native and output occurrence/time identities by network;
- paired `x/r` values, validity, causes, and support;
- raw upper-speed-excluded and support-eroded counts/durations by cause;
- retained weighted exposure, run lengths, edge loss, and spatial coverage;
- science-band transfer on natural and injected sources;
- input, filtered, aliased, and cleaned residual PSDs;
- line inventory and fold status; and
- direct reference runtime, allocations, peak resident memory, and data volume.

A single-network pilot from each array may be used to size the full run, but a
pilot cannot certify the bank.

### F4: end-to-end reductions

Only candidates passing F0--F3 enter full A/B reductions. The required matrix
is:

| Route | References | Candidate | Required comparisons |
| --- | --- | --- | --- |
| Beammap `148670`, naive | R0 and Rm | D | admission/support cost from R0/Rm; detector identities/flags; peak, integral, profile, FWHM, centroid, calibration transfer; signal, weight, kernel, sensitivity; source-crossing TOD |
| Beammap `148670`, JINC or controlled JINC point-source replay | R0 and Rm | D | admission/support cost plus the approved RTC response metrics and map-noise variance; if full detector-group Beammap JINC is not an authorized route, use the common controlled template rather than inventing a product |
| Science `152390` and `152392`, naive | R0 and Rm | D | admission/support cost; observation and coadd maps; injected-source response; extended structure; coverage/weight; residual/noise-map variance; spatial residuals and power |
| Science `152390` and `152392`, JINC | R0 and Rm | D | admission/support cost plus the same response/noise gates through the established JINC route, including kernel/support products |
| Controlled OOF template, naive and JINC | R0 and Rm | D | admission/support cost, mapped response, phase dependence, and recovered template behavior |
| OOF `152385`--`152387` with its established mapmaker and fruitloops | R0 and Rm | D | admission/support cost, existing OOF acceptance criteria, iteration trajectory, final solution, and coherent residual-bias audit |

The existing product-contract and scientific-equivalence machinery remains a
structural and regression gate. Its historical OG/refactor tolerances are not
substituted for the WP-7 filter/downsampling budgets. In particular:

- `beammap-scientific-equivalence-v1` already supplies useful detector-product
  and beam-fit measurements;
- `science-scientific-equivalence-v2` already supplies raw/filtered map and
  diagnostic comparisons; and
- the SCI-MAP product contracts cover naive/JINC product structure.

WP-7-specific comparators shall add the approved `1%` response and noise
questions directly.

The controlled point-source and OOF-template cases run at the exact cadence and
inclusive upper-speed ceiling claimed by each entry. A real observation whose
retained occurrences do not reach that ceiling cannot by itself certify the
higher-velocity bank domain.

## PSD and alias protocol

### Measurement products

For each detector and admitted run, retain enough compact evidence to
reproduce:

- the native sample-rate estimate and cadence interval;
- source/off-source and validity masks;
- the established line mask;
- the native raw broadband PSD;
- the native-rate post-cleaning residual PSD used to represent noise expected
  to survive;
- photon/detector/readout and residual-atmosphere interpretation where the
  existing products support it; and
- estimator, window, segment length, overlap, detrending, units, one/two-sided
  convention, and equivalent-noise bandwidth.

The raw atmospheric PSD is reported but cannot dominate the retained-power
denominator. Source masking and established PTC cleaning must be identical
between candidate-envelope comparisons; no hand-tuned subtraction is allowed.

### Envelope selection study

Before freezing an envelope, report at least:

- all valid per-detector ratios by array/network/observation;
- array-level median and upper-tail behavior;
- the worst credible detector and network cases;
- sensitivity to segmenting and source masks;
- Beammap-versus-science-versus-OOF differences; and
- the resulting candidate-factor/tap changes under plausible conservative
  envelope constructions.

This study is intended to reveal whether one compact conservative envelope is
adequate or whether multiple measured domains are needed. If the aggregation
choice changes bank eligibility materially and is not already scientific
authority, it is returned to the owner with evidence instead of being hidden
inside a tool default.

### Alias computation and map check

For each PSD identity and candidate entry, the artifact binds the signed and
mirrored image mapping, integration grid/refinement, numerator, denominator,
ratio, and independent verifier result. The numerical gate is

```text
P_alias / P_retained <= 1e-2.
```

The separate end-to-end check compares native and candidate map-noise variance
through naive and JINC. For finite-data estimates, report a paired confidence
interval. A conservative classification is:

- pass when the upper uncertainty bound is at or below the `1%` increment;
- fail when the lower uncertainty bound is above it; and
- indeterminate when the interval straddles it, requiring more independent
  samples or realizations rather than a relaxed threshold.

The exact confidence method and independence unit are recorded; scans or
detectors are not treated as independent when they share the same noise mode.

## Narrow-line ordering gate

Lines are not folded into the broadband envelope. They receive a separate
eligibility audit:

1. run the established detector/network line inventory on the native input;
2. compare every protected line and its uncertainty/support with the proposed
   output Nyquist and image mapping;
3. identify whether the established mitigation is effective before the first
   information-losing decimation; and
4. replay a line-containing fixture to prove the realized order.

The current legacy call path contains a model-protected line audit on PTC
residuals after RTC processing. That is useful line-strategy evidence, but it
does not by itself prove protection before RTC sample removal. Therefore:

- a line that cannot fold does not block the broadband candidate;
- a line that can fold blocks that factor unless effective established
  mitigation demonstrably precedes decimation; and
- this plan does not redesign, retune, or independently approve the line
  detector or notch algorithm.

If the required pre-decimation ordering is absent, the smallest follow-up is an
explicit integration/ordering task owned jointly by the existing line strategy
and RTC, not a longer generic low-pass filter chosen to chase narrow lines.

## Response and product metrics

### Point, Beammap, and calibration

For every array and source phase, measure independently:

- peak response relative to R;
- aperture/model integrated response relative to R;
- normalized radial and two-dimensional profile residual relative to the R
  peak;
- fitted major/minor/effective FWHM and ellipticity;
- centroid displacement normalized by the R beam FWHM;
- calibration-transfer magnitude;
- detector classification and fit-success population;
- scan-direction and turn-around residual structure; and
- natural-source and deterministic injected-source agreement.

The six approved WP-7 quantities must each pass; an aggregate score cannot
hide one failing quantity.

### Science maps

Measure for each observation, array, and coadd:

- deterministic injected-source peak, integral, profile, FWHM, centroid, and
  transfer;
- raw and cleaned map difference fields;
- coverage, exposure, weight, and normalization differences;
- off-source variance and RMS;
- spatial noise power and directional residual structure;
- extended-emission transfer on the natural field and a bounded deterministic
  extended template; and
- naive/JINC agreement in the sign and location of any residual.

Extended-structure diagnostics are discovery tools unless already governed by
an approved metric. They may expose a defect or motivate a later owner
decision, but they cannot replace or silently add to the six fixed response
limits.

### OOF and fruitloops

Use the separately governed OOF criteria without altering them. Record:

- per-observation source maps entering the solution;
- fruitloop iteration count, stopping reason, and objective/summary trajectory;
- recovered OOF coefficients or solution fields;
- R-minus-D solution and residual structure;
- array-consistent coherent bias tests; and
- final product-contract and scientific checks.

Any persistent same-sign shift across the OOF sequence is reported even when a
scalar tolerance passes.

## Timing, support, and pairing gates

For every network:

- R0 and Rm occurrence/time equal their admitted native inputs exactly for
  `M=1`;
- D declares a new network-specific output occurrence, time, and complete
  primitive-source support relation;
- no common analysis grid is constructed or invoked;
- selected output centers follow the stable native-axis ordinal rule and do
  not restart at an engineering chunk boundary;
- no source from another network contributes to the output;
- all source samples required by both `x` and `r` are available before the pair
  is admitted;
- below/equal/above upper-speed admission is exact and inclusive at equality,
  with pair-wide cause `scan_speed_above_mode_support` only above it;
- the same coefficient/operator/support is applied to `x` and `r`;
- member-local causes remain inspectable and pair-wide consequences are
  conservative in both directions;
- gaps in one network manufacture neither a slot nor an absence in another;
- filter support crosses no invalid, slow, upper-speed-excluded, gap, or
  physical-scan boundary; and
- leading/trailing unavailable outputs and their causes are exact under at
  least three materially different engineering chunk partitions.

Chunk-local buffers and product-instance identities may differ, as already
allowed. Scientific occurrences, times, values, support, decisions, causes,
pointing correspondence, and declared arithmetic may not.

## Performance and implementation experiment

Performance cannot rescue a scientifically failing entry. Benchmark only
scientific survivors, using a warm-up and repeated whole-scan trials on the
same machine and build. Record:

- elapsed and CPU time by learn, consider, apply, PTC, and map stages;
- samples and detector-values processed per second;
- peak resident memory and owned RTC bytes;
- allocations and bytes moved in the hot path;
- direct ordered FIR versus any factor-aware/polyphase realization;
- one-thread reference and representative thread scaling; and
- output data-volume reduction and total end-to-end reduction time.

The initial implementation preference is one compact immutable coefficient
entry, contiguous network/detector sample planes, explicit support, and a
reusable bounded workspace only where measurement shows benefit. OpenMP,
Eigen, FFTW, or any replacement remains an engineering choice; none is chosen
by this plan.

Before launching the full A/B matrix, one pilot records actual output size and
free-space needs. Screening artifacts may be discarded after their compact
manifest and failure result are retained. Raw inputs, native references, final
candidate products, and certification artifacts remain immutable. Large data
products stay outside Git; their manifests, tools, tables, and compact plots
are committed.

## Gates and stop rules

| Gate | Required result | Stop condition |
| --- | --- | --- |
| D0: fixture identity | Complete hashed raw/config/APT/build manifest for all required cases | Missing, ambiguous, duplicate, or changed inputs |
| D1: native census | Network-local cadence, AST motion, structural mode ceilings, occurrence admission, gaps, validity, and run inventory | Common-grid dependency, unresolved cadence family, nonconforming AST mapping, percentile substitution, or implicit factor selection |
| D2: PSD/line evidence | Reproducible residual PSD candidates and separate line/fold inventory | Removable atmosphere hides alias; material envelope choice unresolved; foldable line lacks pre-decimation protection |
| D3: analytic/synthetic certification | Independent response, DC, phase, alias, support, pairing, and chunk gates pass | Any fixed authority bound fails or verifier disagrees |
| D4: representative replay | Whole-network R/F/D replay passes timing, support, cause, response, and memory checks | Pair semantics, native axes, boundaries, or declared arithmetic differ |
| D5: full products | Beammap and science native/candidate comparisons pass naive and JINC; OOF/fruitloops witness passes | Any independent response/noise limit or existing route contract fails |
| D6: selection evidence | Scientific survivors characterized by support loss, products, runtime, and memory; owner packet prepared | Any candidate hides admission loss or a faster realization changes the scientific operator |
| D7: owner selection closure and frozen package | Owner-approved selection policy; byte-stable bank/PSD artifacts, exact result record, full repository gates, representative paired-data run, and fresh exact-SHA review | Missing selection authority, artifact drift, incomplete gate, or unresolved finding |

Additional stop rules:

- Do not narrow the science band to make a factor pass.
- Do not average a failing array, phase, PSD domain, or metric into a passing
  aggregate.
- Do not treat a skipped required route or missing real-data product as pass.
- Do not use old reduction differences to excuse a new R-versus-D failure.
- Do not certify a velocity or cadence outside the artifact's admitted domain.
- Do not select a factor from raw sample fraction, a velocity percentile, or
  merely nonzero retained data.
- Do not proceed to production implementation while the PSD envelope or
  line-before-decimation ordering remains materially unresolved.

## Evidence package

The final review package contains:

1. the D0 input/build manifest and raw-input digests;
2. network cadence, gap, and AST velocity census;
3. source/validity/line masks and versioned PSD-envelope artifacts;
4. the `M=1..256` structural-ceiling and raw/support-eroded occurrence table;
5. rejected and surviving design summaries;
6. exact bank-entry coefficient and operator artifacts;
7. independent numerical-verifier results;
8. synthetic response, phase, line, boundary, pairing, and chunk results;
9. Beammap, science naive/JINC, and OOF/fruitloops A/B products and compact
   comparison reports;
10. performance, allocation, memory, and data-volume results;
11. an explicit domain-of-validity and missing-domain statement;
12. the exact repository and artifact SHA set;
13. focused, full-repository, and config-gate results; and
14. fresh independent exact-SHA conformance review.

The result record distinguishes `pass`, `fail`, `indeterminate`, and
`not_applicable`. It never converts pending owner-run evidence into readiness.

## Decision boundary during execution

The fixture manifest, native census, synthetic harness, bounded FIR comparison,
PSD measurements, and diagnostic replays can proceed without another
scientific decision. Evidence always returns to the owner before a selection
policy or bank is frozen. It also returns earlier when any of the following
occurs:

- plausible PSD-envelope aggregation rules materially change factor or filter
  eligibility;
- a foldable line requires behavior not already authorized by the established
  line strategy;
- a proposed new map or OOF diagnostic would be promoted from a discovery
  signal to an acceptance limit;
- a mapmaker-specific filter would be preferred;
- the desired production domain extends beyond measured cadence, velocity, or
  PSD evidence; or
- no candidate, including occurrence-local `M=1`, supports an ordinary
  astronomical product.

Filter family, tap count, direct/polyphase organization, and workspace use are
engineering selections when all fixed scientific and evidence gates pass.
Factor selection is not: it awaits the explicit owner closure above.

## Reviewable execution sequence

The recommended bounded sequence is:

1. **Fixture/census harness:** D0 manifests, native cadence/AST census,
   occurrence-level structural ceilings and raw admission, and synthetic
   timing/support fixtures, with no factor selection.
2. **PSD and line evidence:** native-rate residual PSD extraction, aggregation
   sensitivity study, line/fold inventory, and an explicit owner packet only if
   the evidence leaves a material scientific choice.
3. **Offline bank research:** limited FIR-family design, independent numerical
   verification, and synthetic phase/alias certification.
4. **Representative replay:** whole-network R/F/D timestream comparisons and
   bounded performance screening.
5. **End-to-end evidence:** Beammap `148670`, standard science
   `152390`--`152392` through naive/JINC, and the minimal OOF
   `152385`--`152387` fruitloops gate.
6. **Owner selection and bank freeze:** return support/product/performance
   tradeoffs for a bounded factor-selection decision, freeze only the approved
   survivors, then prepare the learn-consider-apply production increment and
   its exact acceptance gates.

The first actionable increment is item 1. It constructs measurement and
fixture tooling only. It does not yet add nonidentity RTC to an executable
science route.
