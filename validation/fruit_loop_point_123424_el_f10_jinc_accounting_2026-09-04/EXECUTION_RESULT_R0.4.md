# SCI-FRUIT EL-F10-R3 execution result r0.4

Test ID: `SCI-FRUIT-EL-F10-R3-FITS-ORIENTATION-READER-REPAIR-R0.1`

Status: **stopped at the unchanged target-ledger gate; routine instrumentation defect identified**

The R0.4 registration validated all 26 bound files. The repaired analyzer then
passed ordinary-map neutrality, checkpoint compatibility, and exact
total-accumulator closure. All six closure checks were true.

The target-ledger gate stopped the analysis because the table contained 610
rows rather than 305. No result products were written.

## Read-only diagnosis

The 610 rows are two contiguous, complete sequences of sample indices 0–304:

- the noise-only JINC pass recorded 271 unflagged samples as
  `center_outside_map` and 34 as `final_flagged`; and
- the final observation-map pass recorded exactly the registered 271 admitted
  and 34 already-final-flagged occurrences.

Every sample index occurs exactly twice. The APT has exactly one a1400 row for
UID 4460, so the duplication is not a duplicate detector identity. Source
inspection confirms that the ledger record was created regardless of
`run_omb`, while target contribution accumulation already occurs only inside
the observation-map branch. The retained target `N_t`, `C_t`, and `Q_t` are
therefore not contaminated by the noise-only pass.

## Disposition

Under the standing routine-defect direction, R4 will restrict diagnostic
sample-ledger recording to the final observation-map pass, add a focused test,
run full local verification, and perform one isolated local replacement replay
from the same checkpoint and unchanged science configuration. The defective
replay remains retained. No science algorithm, gate, bound, target, input, or
interpretation changes.
