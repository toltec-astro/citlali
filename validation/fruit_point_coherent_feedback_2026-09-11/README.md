# POINT coherent-source feedback: bounded innovation experiment

2026-09-11. The owner authorized one substantive improvement screen, with a
keep/revise/reject outcome. **Completed: reject candidate r0.2 as tested after
one targeted revision.** Read the [decision report](SCIENTIFIC_REPORT.md).

## Program adherence and prior-work recovery

Follow the [charter](../../doc/scientific_contracts/README.md), the existing
[reviewed recovery](../../doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.4/PRIOR_WORK.md),
and [owner direction](OWNER_DIRECTION.md). Preserve frozen generic science and
prior results. This scoped empirical experiment replaces the paper-only next
step; OG environment reconstruction, comprehensive noise qualification and OOF
are not prerequisites. Production contracts and profiles remain unchanged.

[PROTOCOL.md](PROTOCOL.md) defines one free-centroid elliptical-Gaussian source
feedback candidate against matched pixelwise selection. Background is excluded
from feedback. Both use the same immutable pre-PTC parent, rank-5 network PTC
relearning, ordinary maps, uniform coefficients and fixed support. Seven passes
are retained and evaluated against cumulative timing. The one targeted revision
has been used; no parameter sweep or further revision follows.

The [input preflight](INPUT_PREFLIGHT.json) identifies a usable 123424 pre-PTC
export and corroborates its chunk labels. The [OG benchmark](OG_BENCHMARK.json)
binds an existing seven-pass historical-recurrence operational control, including
its diagnostic overhead. The small synthetic known-signal/null/mismatch set
reruns relevant learning and remains explicitly conditional on its stated noise
construction. Both candidate versions were frozen before execution. 129081
remains reserved and was not evaluated because the development candidate is rejected.

The isolated [harness](experiment.py) and [focused invariant checks](test_experiment.py)
change no production behavior. Checks cover flexible source recovery/Jacobian,
background exclusion, support formulas, original-parent reconstruction/centering,
and rank failure. A guarded matrix-product wrapper handles observed macOS BLAS
status warnings while rejecting every nonfinite numerical product; warning-as-error
checks pass. The [freeze](FREEZE.json) binds the protocol and implementation before
any candidate trajectory. The [single revision](REVISION_R0.2.md) and its
[freeze](FREEZE_R0.2.json) preserve the original results and identify the sole
change: coherent numerical starts for the same free Gaussian fit.

The revised candidate reaches joint known-Gaussian recovery targets in 2–3
passes, versus no qualifying result within seven reference passes. Real a2000
source identification remains unreliable, with a 46.43-arcsec separation from
the OG published pointing estimate. This is not a measured true error, but the
unresolved conflict prevents a keep decision. At seven real passes the candidate
costs 18.0% more matched method time. The positive controlled result does not
qualify a POINT successor.

Review the [r0.2 numerical evidence](NUMERICAL_EVIDENCE_R0.2.json),
[initial evidence](NUMERICAL_EVIDENCE_R0.1.json),
[OG/decision evidence](DECISION_EVIDENCE.json),
[integrity verification](VERIFICATION.json), and
[result manifest](RESULT_MANIFEST.json). Both 84-pass campaigns, including every
map/model/state and timing receipt, are retained in the external local paths
listed by that manifest. They are not included in Git.
