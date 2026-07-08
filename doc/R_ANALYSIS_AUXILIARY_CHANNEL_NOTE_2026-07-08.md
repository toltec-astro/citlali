# R-Channel Analysis And Auxiliary Timestream Design Note - 2026-07-08

This note captures a design discussion from the `gw_dev` Citlali thread for
the structural refactor worktree in:

`/Users/gwilson/GitHub/citlali-refactor`

The goal is to give the refactor thread enough context to account for an
incoming `r`/quadrature-channel analysis feature before the refactor hardens
the timestream and pipeline interfaces.

No code changes were made as part of this note.

## Detector Signal Context

TolTEC detectors output two solved phase streams from the raw KIDs data:

- `x`: the signal phase used as the science timestream.
- `r`: the quadrature phase.

Both are available in the raw input data. In the current code, these appear as
`result.data_out.xs.data` and `result.data_out.rs.data` from the KIDs solver.

The idealized physics contract is:

```text
x = optical signal + photon/background noise + atmosphere/common modes + detector/readout noise
r = detector/readout noise
```

In practice, `r` is not perfectly quadrature. Leakage from optical signal into
`r` is typically at the few-percent to roughly ten-percent level. Large leakage
is itself scientifically useful because it indicates poor tuning or an
otherwise problematic detector state.

A perfect `r` stream should be sensitive to readout electronics noise and
detector noise, but not to background-limited photon noise or true optical sky
signal. That makes `r` valuable as a control channel, a tuning diagnostic, and
a way to distinguish electronics/readout pathology from optical/atmospheric
structure.

## Current Reduction-Learning Direction

Recent `gw_dev` work has moved toward staged reduction learning:

1. A first pass learns noise properties, glitch occurrence, detector/scan
   pathologies, and map artifacts without assuming a source model.
2. A second learning pass uses fruitloops source subtraction to reduce
   confusion from real sources.
3. Later fruitloops/apply iterations consume the learned state.

The current learning state already records:

- learned sample masks
- detector penalties
- high-weight detector diagnostics
- map-pixel outliers and contributor dominance
- busy-network summaries
- source-protection summaries
- learned-mask and detector-exclusion applications

The natural role for `r` is to add evidence to this learning state rather than
to become a second science map by default.

## Recommended Role For R

The recommended first use of `r` is as an auxiliary measured control channel:

```text
scans.data       = primary science stream, normally x
quadrature.data  = optional measured r stream
kernel.data      = synthetic model/transfer diagnostic
```

The important distinction is that `kernel` is a model sidecar, while `r` is
measured data. The two should not be conflated.

`r` should initially inform diagnostics and learning decisions:

- per-detector and per-scan `r` RMS, PSD, white-noise level, and low-frequency
  excess
- `x-r` correlation/coherence by frequency band
- `r` glitch/event rate
- shared readout-line/RFI strength in `r`
- network-common structure in `r`
- source leakage estimates after fruitloops/source subtraction
- high-weight detector validation
- detector/scan/network pathology scoring
- optional `r` null/control maps

The first implementation should avoid blind subtraction of scaled `r` from
`x`. Direct `r` subtraction can inject `r` noise or remove real optical signal
when leakage is nonzero. If active `r`-based cleaning of `x` is introduced
later, it should be gated by measured `x-r` coherence or regression, and by
source-subtracted leakage diagnostics.

## PTC PCA Discussion

The current PTC cleaner computes a detector-space covariance from the primary
timestream, solves detector eigenvectors, and subtracts the projection onto the
leading detector-space modes. The `kernel` sidecar is then cleaned using the
same x-derived modes.

For `r`, the agreed initial policy is:

1. The `r` stream should receive whatever cleaning operator is applied to `x`,
   so it experiences the same science transfer function.
2. It may also be useful to run PCA on `r` itself diagnostically, because `r`
   should be sensitive to the same correlated electronics/readout modes seen
   in `x`.

Useful `r` PCA products include:

- overlap between leading `x` and `r` detector eigenvectors
- `r` eigenspectrum shape by scan/network/array
- identification of electronics modes hidden under atmospheric modes in `x`
- validation of PCA depth in `x`
- detector/network tuning and readout-quality metrics

The safe hierarchy is:

```text
Phase 1: x PCA cleans x; the same x-derived cleaning operator is applied to r.
Phase 2: run r PCA diagnostically; compare x/r spectra, mode overlap, and coherence.
Phase 3: use r PCA to validate or adjust x PCA decisions.
Phase 4: only then allow r-derived modes to clean x, gated by x-r coherence and leakage tests.
```

Plain `r` eigenvectors should not blindly clean `x`, because projecting `x`
onto an `r`-derived detector basis removes whatever `x` has in that detector
direction, even if the time-domain behavior is not the same electronics mode.

## Calibration Policy

In both `gw_dev` and the current refactor branch, if `timestream.type: rs` is
selected as the primary stream, Citlali calibrates it identically to `xs`.
After raw loading, the RTC/PTC code no longer knows whether `scans.data` came
from `xs` or `rs`; flux calibration and extinction correction multiply
`scans.data` blindly.

That behavior is mechanically true but physically ambiguous.

For simultaneous `x/r` processing, the auxiliary `r` channel should have an
explicit calibration policy. Recommended defaults:

- Keep native `r` units for detector/readout diagnostics.
- Optionally produce an `x`-equivalent or sky-equivalent calibrated `r`
  product for leakage/null-map analysis.
- Do not apply optical extinction correction to pure `r` diagnostics unless the
  product is explicitly labeled as apparent sky-equivalent leakage.

## Refactor Architecture Findings

The refactor repository has already made useful structural progress:

- typed config enums exist for `TodType` (`xs`, `rs`, `is`, `qs`)
- raw KIDs loading is split into focused helper headers
- observation and iteration orchestration has been pulled into smaller
  `core/pipeline` helpers
- output writers and learning hooks are more isolated than in `gw_dev`

This is favorable for adding `r` analysis, but the core chunk data model still
assumes one primary measured timestream:

```text
TCData<RTC/PTC>.scans.data   primary measured matrix
TimeStream::kernel.data      optional synthetic/model matrix
```

The three main runtime paths still construct an `RTC` chunk and assign exactly
one `rtcdata.scans.data` from `tod_type`:

- lali/science
- pointing
- beammap

The raw loader currently chooses one channel from the KIDs solver result. A
paired loader could copy both `xs` and `rs` from the same solver result without
doubling raw reads.

The main structural risk is not raw access. The risk is allowing the refactor
to harden the `TCData`, TOD writer, PTC cleaner, and scan-generator interfaces
around a single measured matrix.

## What Is Straightforward

The following should be straightforward if the refactor introduces an explicit
auxiliary measured channel now:

- load `x` and `r` together from one raw KIDs solve
- carry `r` through scan alignment and gap interpolation
- carry `r` through the same RTC filters, notches, highpass, edge guards, and
  downsampling as `x`
- apply x-derived PTC mean subtraction and PCA cleaning to `r`
- add diagnostic-only `r` summaries to the existing learning state
- write optional `r` TOD products or sidecar variables
- create optional `r` null/control maps

## What Becomes Difficult Later

The feature becomes structurally difficult if the refactor lands more
interfaces that assume:

- one measured sample matrix per `TCData`
- all non-primary matrices are synthetic kernel-like model streams
- calibration always applies to the one primary matrix
- TOD output schema has only `signal`, `flags`, and optional `kernel`
- PTC cleaning sidecars are special-cased only for `kernel`
- learning records have no place to store channel-specific diagnostics

Under that shape, adding `r` later would require touching raw loading, every
scan generator, RTC filtering, PTC cleaning, output, diagnostics, learning CSVs,
and mapmaking in a scattered way.

## Recommended Refactor Request

The refactor thread should consider adding an explicit optional measured
sidecar-channel contract before finalizing the chunk and pipeline interfaces.

Suggested concepts:

```text
enum class TimestreamChannel {
    science_x,
    quadrature_r,
    synthetic_kernel
};

struct AuxiliaryMeasuredStream {
    Eigen::MatrixXd data;
    std::string name;          // "r" / "rs" / future names
    std::string native_unit;
    CalibrationPolicy calibration_policy;
    bool apply_primary_linear_transfer = true;
    bool use_for_science_map = false;
};
```

The exact API can differ, but the architecture should preserve these
invariants:

- `x` remains the default science signal.
- `r` is measured data, not a kernel.
- `kernel` remains synthetic/model transfer data.
- Auxiliary measured streams can be carried through the same linear operations
  as `x`.
- Calibration and extinction policy are explicit per stream.
- PTC cleaning can apply x-derived modes to auxiliary streams.
- Auxiliary streams can optionally run their own diagnostics/PCA without
  changing the science channel.
- Output and learning records can identify which channel a diagnostic came
  from.

## Suggested First Milestone

Before implementing full `r` science diagnostics, the refactor could add a
small behavior-neutral scaffold:

1. Add an optional `quadrature` or generic auxiliary measured matrix to
   `TCData`.
2. Add config gating that defaults off, for example:

   ```yaml
   timestream:
     auxiliary_channels:
       quadrature_r:
         enabled: false
         source_type: rs
         calibration_policy: native
         apply_primary_transfer: true
         diagnostics_enabled: false
   ```

3. Extend raw loaders to optionally populate `quadrature.data` from
   `result.data_out.rs.data`.
4. Thread that matrix through RTC linear operations and downsampling only when
   present.
5. In PTC, apply the x-derived cleaning operator to `quadrature.data` only when
   present.
6. Add minimal shape/unit/status logging and optional TOD output variable.
7. Leave all diagnostics and behavior off by default.

This would make later `r` learning work local and controlled, while preserving
current reduction behavior and performance when the auxiliary channel is off.

## Bottom Line

Introducing `r` analysis is not a blocker for the structural refactor. It is
medium difficulty now and likely high difficulty later if the one-measured-
stream assumption becomes more entrenched.

The architectural message to preserve is:

```text
Citlali needs a first-class optional measured sidecar channel, not just a
primary science matrix and a synthetic kernel matrix.
```

