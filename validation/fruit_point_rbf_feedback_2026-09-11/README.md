# Bounded RBF feedback experiment

## Program adherence and prior-work recovery

This isolated owner-authorized experiment follows the
[program charter](../../doc/scientific_contracts/README.md) and the adopted
[prior-work recovery](../../doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.4/PRIOR_WORK.md).
It changes only feedback inference; production contracts and previous results
are preserved. See the [owner directive](OWNER_DIRECTIVE.md), frozen
[protocol](PROTOCOL.md), [representation decision](REPRESENTATION_DECISION.md)
and [129081 exposure audit](EXPOSURE_AUDIT.md).

The first candidate is frozen before empirical execution. P is the matched
pixelwise control, G the independent Gaussian control, and R the nonnegative
regularized RBF candidate. All use the same rank-5 relearning recurrence.
