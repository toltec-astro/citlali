# POINT relative-position prior screen

2026-09-11. Owner-authorized experiment; candidate prepared, results pending.

## Program adherence and prior-work recovery

Follow the [charter](../../doc/scientific_contracts/README.md), existing
[reviewed prior-work record](../../doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.4/PRIOR_WORK.md),
and [new owner direction](OWNER_DIRECTION.md). Adopt the previous screen's
recurrence, input/control bindings and isolated PTC/MAP harness. Preserve its
[rejected candidate and positive conditional evidence](../fruit_point_coherent_feedback_2026-09-11/SCIENTIFIC_REPORT.md).
Only relative-position inference is new. Frozen scientific contracts, weighting
closure, production sources and the independent-author boundary are unchanged.

The [protocol](PROTOCOL.md) defines common a1100/a1400 positional inference and
a separate free a2000 offset within 2 arcsec, without transferring flux or shape
between arrays. These are explicit approximations to the owner's corrected
alignment evidence. The [implementation](experiment.py) imports the unchanged
previous cleaner/mapmaker and feedback control. [Focused tests](test_prior.py)
cover analytic derivatives, free shape/position recovery, background exclusion
and own-array/prior-tension admission. Three tests passed with warnings as errors.

The [freeze](FREEZE.json) binds code, protocol and exact predecessor dependencies
before execution. Eleven cases × two arms × seven passes retain 154 passes.
The original unconstrained evaluator is used to judge resulting maps; constrained
agreement cannot qualify pointing. Observation 129081 remains reserved unless
this frozen candidate first passes the development screen.
