# POINT relative-position prior screen

2026-09-11. Completed owner-authorized experiment. **Decision: REVISE.**
Read the [decision report](SCIENTIFIC_REPORT.md).

## Program adherence and prior-work recovery

Follow the [charter](../../doc/scientific_contracts/README.md), existing
[reviewed prior-work record](../../doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.4/PRIOR_WORK.md),
and [new owner direction](OWNER_DIRECTION.md). Adopt the previous screen's
recurrence, input/control bindings and isolated PTC/MAP harness. Preserve its
[rejected candidate and positive conditional evidence](../fruit_point_coherent_feedback_2026-09-11/SCIENTIFIC_REPORT.md).
Only relative-position inference is new. Frozen scientific contracts, weighting
closure, production sources and the independent-author boundary are unchanged.

The [protocol](PROTOCOL.md) defines common a1100/a1400 positional inference and
a separate free a2000 offset within 2 arcsec, with independent amplitude and
shape parameters in each array. These are explicit approximations to the owner's corrected
alignment evidence. The [implementation](experiment.py) imports the unchanged
previous cleaner/mapmaker and feedback control. [Focused tests](test_prior.py)
cover analytic derivatives, free shape/position recovery, background exclusion
and own-array/prior-tension admission. Three tests passed with warnings as errors.

The [freeze](FREEZE.json) binds code, protocol and exact predecessor dependencies
before execution at `ae758cd2f`. Eleven cases × two arms × seven passes retain 154 passes.
The original unconstrained evaluator is used to judge resulting maps; constrained
agreement cannot qualify pointing.

The prior reduces a2000 peak error from −19.48% to −0.07% in the declared
source-plus-contaminant test, and creates no source feedback in either null or
absent-a2000 case. It fails the mismatch gate: a1100 direct error rises 75.0%
and exterior error 77.9%. Real a2000 displaced feedback is rejected, without
demonstrating weak-source recovery. Equal-pass real method time rises 19.7%.

The screen is closed with a revise recommendation; no scientific revision or
129081 evaluation follows automatically. The full [numerical record](NUMERICAL_EVIDENCE.json),
[gate assessment](DECISION_EVIDENCE.json), [integrity check](VERIFICATION.json)
and [result manifest](RESULT_MANIFEST.json) retain the evidence. Every iteration
product remains in the manifest-listed external local directory, outside Git.
