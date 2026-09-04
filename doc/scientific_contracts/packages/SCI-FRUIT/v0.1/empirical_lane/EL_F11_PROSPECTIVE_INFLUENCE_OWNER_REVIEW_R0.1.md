# SCI-FRUIT v0.1 — EL-F11 Prospective Influence Persistence Review r0.1

Decision candidate:
`SCI-FRUIT-EL-F11-PROSPECTIVE-INFLUENCE-PERSISTENCE-R0.1`

Status: **owner-review proposal; no staging, replay, analysis execution, or
algorithm change is authorized**

## The short version

EL-F10 explains the large arcs after UID 4460 scan 5 is removed from the
iteration-5 map. Before inventing a better safeguard, we need to know whether
the consequence could have been anticipated using the preceding iteration.

EL-F11 proposes one short replay from the already preserved injected
iteration-3 checkpoint to iteration 4. The existing, neutral JINC accounting
diagnostic would measure UID 4460 scan 5's whole-map influence in iteration 4.
We would compare that map with the already measured iteration-5 deletion
response. No detector action changes in this test.

## The exact question

> Does UID 4460 scan 5's exact iteration-4 JINC deletion-response map retain
> enough spatial, signed, and mechanistic relationship to its iteration-5
> deletion-response map to justify designing a response-aware safeguard test?

The result will be descriptive, not a pass/fail qualification. It can support
or weaken the case for a later experiment, but cannot itself choose a
safeguard.

## Why this comes before three candidate actions

The existing repeat-count rule selects a scan-local hard exclusion from four
extreme pixels. EL-F10 showed that the resulting map consequence occurs
elsewhere and depends on signed local leverage times processed-signal
contrast. A response-aware alternative sounds attractive, but its useful
input has not yet been shown to persist from the decision iteration to the
affected iteration.

Running hard, soft, or map-local alternatives now would require choosing
quantities and thresholds after seeing this one known outcome. EL-F11 first
tests the premise on which such candidates would rest.

## One bounded replay

Use the frozen EL-F10-R4 executable and an exact copy of the EL-F5
off-source-injected completed iteration-3 checkpoint. Advance once to
completed iteration 4 with the existing source, processing, learning, and
mapmaking configuration unchanged. Enable only the already implemented,
disabled-by-default JINC accounting diagnostic for a1400, UID 4460, zero-based
scan 5, and use isolated development output paths.

The exact executable, checkpoint, configuration constraints, output root,
accounting equations, and compatibility references are bound in
`EL_F11_PROSPECTIVE_INFLUENCE_PERSISTENCE_DESIGN_R0.1.md` and the bundle
manifest.

Run locally with one configured thread and `--grppiex seq`. The replay is
bounded to 1 hour and 64 GiB, with at most 8 GiB of newly retained products.
At most one replacement is permitted for an environmental interruption or a
narrow diagnostic defect under the owner's standing routine-defect direction.
An unfavorable scientific result, compatibility failure, or changed method
does not permit a replacement.

## Gates before interpretation

1. Reverify every frozen executable, checkpoint, configuration, and
   comparison-product identity before staging.
2. Require all nine ordinary iteration-4 science planes to match the existing
   EL-F5 iteration-4 products bitwise.
3. Require learning output to match byte-for-byte. Require map-diagnostic and
   checkpoint scientific content to match value-for-value after only the
   already registered creator, omitted-default, and diagnostic-provenance
   normalization; compare structure and the exact observed difference set.
4. Re-run the exact EL-F10 accumulator, ledger, finalization, support, and
   forward-error gates for the iteration-4 receipt.
5. Require identical a1400 units, WCS/grid, pixel ordering, and registered
   common support before comparing iteration 4 with iteration 5.
6. Require complete products, finite registered values, and zero unexpected
   error- or critical-level messages.

Any failed compatibility, accounting, support, or identity gate stops the
scientific comparison. A tolerance may not be loosened after values are seen.

## What will be reported

Retain the complete iteration-4 response, iteration-5 response, and difference
maps. Report their signed and rank correlations, normalized inner product,
best descriptive scale and scaled residual, sign agreement, top-response
overlap, and captured iteration-5 response power at fixed 1%, 5%, and 10%
iteration-4 quantiles.

Report RMS, peak, signed sum, and cross terms over the complete footprint and
the already registered Neptune, injected-source, and annular regions. Compare
signed leverage, processed-signal contrast, cancellation, absolute and
quadratic support, occurrence counts, and unique-detector counts separately.

These are descriptive measurements. No cutoff will be invented to label the
result predictive or non-predictive, and spatially correlated pixels will not
be presented as independent statistical samples.

## Interpretation limits

The named target was selected using known later history. The resulting
iteration-4 quantity is temporally available before the hard iteration-5
action, but this is still an oracle-targeted feasibility test. It does not
provide a causal method for finding candidates in an ordinary run and it does
not estimate a false-action rate.

One pointing, detector, scan, and iteration transition cannot establish a
generic policy. Even a compelling result authorizes only preparation of a
later intervention packet. That packet would still need prospective candidate
selection, predeclared actions and thresholds, science/performance metrics,
and independent-pointing replication before any safeguard recommendation.

## Owner choices

### Choice A — Approve the bounded persistence study (recommended)

Approve
`SCI-FRUIT-EL-F11-PROSPECTIVE-INFLUENCE-PERSISTENCE-R0.1` exactly against its
bundle manifest. This authorizes only the isolated setup, frozen analysis
registration, one local iteration-3-to-4 replay, compatibility and accounting
checks, descriptive comparison with retained EL-F10 iteration 5, and result
record described above.

### Choice B — Proceed directly to intervention design

Do not run this replay. Return a revised owner-review packet that proposes
hard, soft, or map-local interventions despite the unresolved persistence
premise. No intervention is authorized by selecting this choice.

### Choice C — Stop after EL-F10

Retain EL-F10 as the final explanation of this case. Do not stage or run
EL-F11 and do not design a safeguard from the present evidence.

General agreement to prepare this packet is not exact approval of Choice A.
