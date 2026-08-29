# WP-7 RTC Fixed-Decimation Owner-Decision Packet

Status: **proposed; not scientific authority; implementation not authorized**

Prepared: 2026-08-29

Implementation base:
`0574d9a50fe6df6f7ded07c1d229bcb8ca04309d`

## Purpose

This packet isolates the smallest scientific-owner decision needed before the
WP-7 successor can implement a nonidentity RTC operation. It proposes one
explicit, network-local, fixed-mode, phase-zero decimation witness. It neither
changes frozen SCI-RTC v0.1/r0.12 nor imports a numerical policy from the
legacy implementation.

Until the owner approves a complete numerical disposition, fixed
downsampling beyond `M=1` remains unavailable. The accepted network-timed
identity RTC route remains unchanged.

## Governing authority

The proposal is subordinate to:

- frozen SCI-RTC v0.1/r0.12, especially DEF-015, DEF-018, DEF-023--024,
  DEF-039--040, DEF-046, EQ-011, EQ-014, EQ-040--042, REQ-021--031, and
  REQ-037--041;
- resolved OWNER-024, OWNER-077--081, OWNER-084--089, and OWNER-090--103;
- open OWNER-002, OWNER-007--009, OWNER-031--034, OWNER-036, and OWNER-052;
- the network-specific timing correction in ADR 0015 and the WP-7 network
  timing owner correction; and
- the accepted identity-RTC implementation and its exact-SHA PASS package.

The frozen ledger's exact unavailable-state rule controls. A software default,
legacy coefficient set, or convenient sentinel cannot resolve an open owner
entry.

## Observed implementation and data facts

### Representative observation 152390

The 11 locally retained TolTEC network files for observation 152390 all
declare:

- `Header.Toltec.FpgaFreq = 256000000 Hz`;
- `Header.Toltec.AccumLen = 2097152`; and
- `Header.Toltec.SampleFreq = 122.0703125 Hz`.

Therefore a fixed `M=2` plan would have:

- output rate `61.03515625 Hz`; and
- output Nyquist frequency `30.517578125 Hz`.

The accepted `redu04` merged configuration has downsampling disabled with
`factor: 1`, while its legacy RTC filter is enabled with
`freq_high_Hz: 32.0`, `n_terms: 32`, and `a_gibbs: 50.0`. The configured
`32 Hz` edge is above the `M=2` output Nyquist. It therefore cannot be adopted
unchanged as an anti-alias plan for this witness.

These are observed configuration and input facts, not evidence that `32 Hz`
is the scientifically required retained band.

### Incumbent downsampler

The legacy `timestream::Downsampler` selects input rows `0, M, 2M, ...` and
ORs flags over the following block of at most `M` rows. The surrounding legacy
RTC path separately downsamples telescope, pointing, polarization, and kernel
containers.

That numerical point selection agrees with the phase-zero index sequence, but
the complete legacy behavior is not a conforming successor operator because
it does not itself provide:

- a network-scoped successor output occurrence identity;
- complete prefilter and transitive input support;
- coordinate-local availability with typed causes;
- the canonical identical paired `x/r` operator;
- an exact response and alias statement;
- chunk-independent phase anchored to the scientific segment; or
- a compact immutable resolved plan and realized record.

The incumbent FIR also leaves edge cells outside its valid convolution body
unchanged and does not normalize its generated coefficient sum. Neither
behavior is adopted by this packet.

## Proposed bounded context

The first fixed-decimation operation should be a new explicit context named
`wp7-rtc-fixed-decimation-conformance-v1` with these boundaries:

- it is requested explicitly and is never a default or fallback;
- the admitted factor set is `{1, 2}`;
- `M=1` is the already accepted identity plan;
- `M=2` is the only new numerical witness;
- it operates independently on each network's coherent native occurrence
  segments;
- required conditioned `x` and requested conditioned `r` use one identical
  ordinary filter-and-selection operator;
- it terminates through the existing inspectable in-memory RTC-only route;
- it does not enter CAL, AST, VAL, PTC, MAP/JINC, or a common analysis grid;
- it publishes no persistent TOD schema; and
- it claims algebraic and implementation conformity only, not astronomical
  transfer, observational performance, production readiness, or external
  consumer acceptance.

This context is proposed because it exercises the first sampling-changing RTC
relation without prematurely coupling the work to pointing, calibration, PCA,
or mapmaking.

## Proposed owner decisions

The following items are recommendations for owner disposition. They are not
effective merely because they appear in this file.

### D1. Factor and plan selection

Admit exactly `M in {1, 2}` for this context. Reject every other integer,
frequency-derived request, learned-factor request, and silent fallback before
Apply. The resolved plan records the exact input rate, output rate, factor,
zero phase, coefficient artifact, segment policy, response policy, and context
identity.

Rationale: `M=2` is the smallest nonidentity witness and avoids inventing a
general factor framework before another factor has a named workload.

### D2. Prefilter form

Use one immutable, finite, symmetric, real `float64` FIR coefficient artifact
applied as a centered zero-phase operation before phase-zero selection. Store
the exact coefficient sequence, coefficient-production procedure and version,
sum/DC gain, pass/stop metrics, rate, precision, impulse support, and SHA-256.

The coefficient artifact is scientific input. It is not generated or adjusted
during Apply. Direct symmetric convolution, a polyphase realization, and an
FFT overlap-save realization are permitted implementation candidates only if
they reproduce the declared arithmetic policy and support.

No coefficient sequence is proposed yet. Its passband, transition, stopband,
ripple, attenuation, and equality boundaries require the unresolved owner
choice in D8.

### D3. Network-local output identity and timing

For each network and each coherent segment of length `N`, define candidate
output indices by SCI-RTC-EQ-011:

```text
input_index(n) = M * n
output_count = 0                         when N = 0
output_count = 1 + floor((N - 1) / M)   when N > 0
```

Each output has a new stable RTC occurrence identity scoped by observation,
network, plan, segment, and `n`. Its representative source occurrence and time
are exactly the network input occurrence and time at `M*n`. A gap, segment
boundary, or missing occurrence on one network manufactures no occurrence or
absence state on another network.

No cross-network common analysis relation is requested or constructed.

### D4. Scientific support and edges

Retain every phase-zero candidate output occurrence. Its RTC-local support is
the exact transitive union of the admitted source occurrences in the realized
FIR footprint, not the selected center and not the block `[M*n, M*n+M)`.

Do not extend, reflect, repeat, wrap, or zero-pad beyond a coherent segment.
If the complete centered FIR footprint is unavailable, retain the output grid
location but mark the affected coordinate unavailable with a typed
`incomplete_filter_support` cause. A short nonempty segment therefore retains
its phase-zero grid even when every numerical output is unavailable. An empty
segment produces zero outputs.

Engineering chunks may supply overlap and workspace state, but they never
create resets, edges, identities, support, or phase.

### D5. Validity and non-finite behavior

Use the already admitted producer availability, producer validity, and finite
payload facts as inputs. Do not repair, interpolate, fill, or coerce a
non-finite or invalid cell in this increment.

If any required input for one coordinate is invalid, unavailable, or
non-finite within an output's exact FIR support, that coordinate's output and
response are unavailable with the accumulated typed causes over that output.
Pair-action and ordinary-operator support remain common, while numerical
availability remains coordinate-local. Unavailable conditioned `r` never
invalidates otherwise conforming conditioned `x`.

This is a bounded no-recovery policy for this context; it does not resolve a
general recovery policy for other RTC operations.

### D6. Paired arithmetic and response

Apply exactly the same masks, FIR coefficients, `float64` arithmetic policy,
support, boundaries, phase-zero selection, representative occurrences, and
output grid to `x` and requested `r`. Fixed-state cross-coordinate numerical
derivatives remain zero.

Record the complete realized local FIR-plus-selection response or a typed
unavailable response. State amplitude and power response, phase, group delay,
input/output rates, Fourier convention, impulse support, coefficient
precision, and every folded alias image. The centered symmetric FIR has zero
nominal temporal displacement under this proposed context; AST is not invoked
to manufacture or correct a shifted time.

Do not claim a bounded-alias, astronomical-transfer, covariance, uncertainty,
or total-response result until its separately required authority exists.

### D7. Compact result and diagnostics

Extend the existing context/evidence/plan/apply/result/realization lifecycle
without copying immutable input axes or full support planes into evidence,
plan, or realization. The scientific output necessarily owns its new
downsampled numerical product and compact per-network output axes.

Retain compact summaries sufficient to reconcile:

- input/output occurrences and detector-occurrences by network;
- available/unavailable `x` and requested `r` outputs by cause;
- full-support and edge-unavailable populations;
- factor, coefficient artifact, segment, and response identities;
- stage entry and completion counts;
- owned bytes, allocations where measurable, wall/CPU time, and peak RSS; and
- chunk-partition comparisons.

Do not add a full event manifest, generalized lineage system, per-cell
identity, or persistent RTC TOD product.

### D8. Required numerical owner selection

The owner must select the exact application-domain filter envelope before D2
can produce a coefficient artifact:

1. retained input passband and equality convention;
2. transition band;
3. stopband start, including its relation to the `30.517578125 Hz` `M=2`
   output Nyquist for observation 152390;
4. maximum passband ripple;
5. minimum stopband attenuation and alias metric;
6. coefficient-design authority and deterministic tie rule;
7. maximum FIR support/edge loss; and
8. whether this first context makes only an algebraic conformance claim or a
   named astronomical-transfer claim.

The observed legacy `32 Hz` setting cannot answer these questions because it
is above the `M=2` output Nyquist. Choosing a lower passband is a scientific
decision, not an implementation repair.

## Entries deliberately not resolved

This packet does not request learned sampling, so OWNER-011--020 and OWNER-029
remain open. It does not select despiking, donors, level shifts, notches,
high-pass filtering, atmosphere templates, covariance, source/beam response,
or external consumers. Their owner entries remain unchanged.

OWNER-032 and OWNER-036 remain open if an astronomical-transfer or
observational-performance claim is requested. The recommended first context
can close algebraic implementation conformance without making either claim.
OWNER-051 remains open because this increment neither changes native mapping
families nor makes a cross-revision or native-end-to-end response claim.

## Implementation choices left open

After owner approval and coefficient-artifact construction, compare at least:

1. a direct symmetric FIR with explicit phase-zero gathering;
2. a direct polyphase decimator that avoids computing discarded outputs; and
3. FFT overlap-save only if the realized filter length and segment sizes make
   it plausible.

Measure the complete paired route, including cause/support propagation and
output construction, on synthetic edge/pathology fixtures and the full
representative observation 152390 workload. Include allocation and memory
movement. Eigen, FFTW, OpenMP, a reusable workspace, and a new C++23 view are
implementation candidates, not scientific authority or required architecture.

Choose one reference implementation from end-to-end evidence. Do not retain
the legacy Eigen stride map or FIR body merely because it already exists.

## Required implementation gates after approval

The eventual code increment must prove:

- `M=1` remains bitwise and semantically identical to the accepted route;
- exact `M=2` cardinality for empty, singleton, even, and odd segments;
- exact per-network representative identity and time at `2*n`;
- distinct network time vectors remain distinct;
- no common-analysis-grid dependency or AST invocation;
- complete FIR support and typed edge/non-finite unavailability;
- identical paired ordinary operators with independent coordinate
  availability;
- a gap or reset in one network has no effect on another;
- one-segment and allowed multi-chunk execution agree under the declared
  arithmetic policy;
- stale plan/context/coefficient identities fail before publication;
- failure publishes no false completion;
- no persistent TOD schema or later scientific stage appears;
- focused public-header and dependency guards pass;
- all repository gates and legacy behavior pass unchanged;
- exact clean-SHA observation 152390 evidence is retained; and
- a fresh independent exact-SHA conformance review returns PASS.

Representative real-data acceptance must use all 11 retained networks and the
same exact dataset/slice binding discipline as the accepted v8 identity route.
The numerical record must add exact factor, coefficient-artifact, response,
support, edge, availability, performance, and output-cardinality facts.

## Owner disposition requested

No code implementation should begin until the scientific owner records one of
these outcomes:

1. approve D1--D7 and supply or approve the exact D8 numerical envelope;
2. approve the structure but request a named change before numerical
   selection;
3. authorize an evidence-only design study to compare explicitly bounded D8
   candidates without producing an accepted RTC scientific product; or
4. defer fixed decimation and keep `M=1` as the only available RTC sampling
   plan.

Approval of this packet would authorize only the bounded fixed-decimation
vertical increment and its evidence. It would not authorize despiking, level
shifts, learned sampling, CAL, AST, VAL, PTC, MAP/JINC, production activation,
or legacy-route retirement.
