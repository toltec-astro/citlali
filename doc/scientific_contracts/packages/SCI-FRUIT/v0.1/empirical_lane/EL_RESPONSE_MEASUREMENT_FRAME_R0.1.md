# SCI-FRUIT empirical-lane response measurement frame r0.1

Status: **owner-review proposal; no new experiment or scientific claim is
authorized**

## Why this distinction is needed

An injected-minus-uninjected FRUIT map is always a legitimate measurement of
the pipeline's total response to adding a source. It is not automatically the
transfer function of that source alone. FRUIT can allow the source to change
detector penalties, masks, weights, target selection, feedback state, and
later processing. When that happens, the subtraction also contains any real
sky, atmospheric, instrumental, or scan-synchronous material that no longer
cancels between the two branches.

EL-F6 demonstrated this distinction directly for observation 123424. The
off-source compact component remained well localized, while a branch-specific
UID 4460 exclusion caused real Neptune and scan-shaped structure to appear in
the total paired response.

## Three response identities

For a fixed observation, array, grid, support rule, and output iteration, let
`M(state; input)` denote the produced signal map. The following quantities are
different and must be named separately.

### 1. Total adaptive trajectory response

\[
T_k = M(S^{\rm injected}_{k-1};\,D+J)
      -M(S^{\rm control}_{k-1};\,D),
\]

where `D` is the observed data, `J` is the injected signal, and each branch
carries the state learned along its own preceding trajectory. `T_k` includes
both astronomical response and every causal state or operator change induced
by the injection. It is the correct end-to-end answer to “what did adding this
source make this adaptive pipeline produce?”

It is not, without further evidence, an isolated-source transfer map or an
independently calibrated sky product.

### 2. Shared-incoming-state one-step response

\[
S_k = M(S^{\rm common}_{k-1};\,D+J)
      -M(S^{\rm common}_{k-1};\,D).
\]

Both branches enter the measured transition with the same complete checkpoint
state. This removes differences inherited from earlier iterations and gives a
cleaner measurement of what the added source causes during one transition.

It still is not a fully matched-operator transfer function. The source may
change data-dependent operations learned or selected *within* the measured
iteration, including detector participation, cleaning, masks, weighting, or
diagnostics. The result must therefore retain the name
**shared-incoming-state one-step response**.

### 3. Fully matched-operator source transfer

\[
X_k = \mathcal{L}_k(D+J)-\mathcal{L}_k(D),
\]

where the same demonstrated operator `L_k` acts in both branches: detector
participation, penalties, masks, weights, cleaning modes, filtering,
normalization, response, WCS/grid, and support are fixed or proven equivalent.

This quantity is currently **unavailable** for the adaptive FRUIT transition.
It may be claimed only after the required operator equality is instrumented
and demonstrated, or after a separately validated frozen-operator replay is
defined. A common checkpoint alone is insufficient.

## Required reporting for all three quantities

Every response measurement must identify:

- observation, source profile, amplitude, sign, position, and injection point;
- initial checkpoint or trajectory history and output iteration;
- recurrence/method identity and every intentionally changed configuration;
- signal response and processed response/kernel on a common WCS, grid,
  normalization, finite-support rule, and unit;
- detector penalties, masks, weights, target selection, and other learned
  state that differ between paired branches;
- compact-source amplitude, flux, centroid, and width relative to the
  same-iteration processed kernel;
- residual structure after the best-fit kernel component is removed; and
- leakage measured separately near known real sources and in a declared
  background or annular region.

Complete difference maps must be retained for the bounded experiment. Scalar
metrics alone are not sufficient because scan-shaped positive and negative
structure can cancel in an aperture or flux sum.

## Interpretation rules

- `T_k` is the primary end-to-end adaptive-pipeline response.
- `S_k` is a controlled one-step response with common incoming state.
- Neither is renamed `X_k` by visual compactness or approximate scalar
  agreement.
- Per-iteration differences are identified diagnostic quantities, not
  automatically calibrated or scientifically independent sky products.
- Flux recovery and non-kernel leakage are reported separately; acceptable
  integrated flux does not excuse structured leakage.
- Real-field leakage is not attributed to the injected sky profile unless a
  matched-operator comparison or another identified causal test establishes
  that interpretation.
- A result from one pointing, detector, source position, or amplitude does not
  establish population-level generality.

This frame governs the proposed EL-F7 interpretation only if the owner
approves its exact bundle. It does not revise the accepted Stage A recurrence
baseline or qualify a method.
