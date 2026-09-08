# SCI-MAP v0.2/r0.4 PRED-025 quantity-specific zero fixtures

Status: **candidate prospective scientific fixtures; no application evidence,
execution result, conformance verdict, profile activation, or product is
asserted**.

These four fixtures refine `SCI-MAP-PRED-025` without changing the estimator,
projection, coefficient family, support selector, product roles, or failure
scopes. Here `N_p` is the signal numerator for pixel `p`, whereas `N` is
the cardinality of the declared finite-positive support-policy population.
They are different quantities. An exact numerical zero is interpreted only
under the identity, unit, domain, and validity rule of the quantity that owns
it.

The use of the existing exact uniform value `gamma_i = 1` below is a fixture
instantiation of the already admitted family
`SCI-PTC:uniform_constant@draft-0.1`. It introduces no family, selection,
fallback, default, realized handoff, active MAP profile, or application route.

## Fixture A — cancelling admitted signals

Take one support-authorized pixel `p` and two otherwise admitted samples with
`G_p1 = G_p2 = 1`, exact dimensionless
`gamma_1 = gamma_2 = 1`, and finite signals
`z_1 = +3 mJy/beam` and
`z_2 = -3 mJy/beam`. Every structural, coordinate, product,
admission, and support gate is stipulated to pass for this prospective
calculation. Then

\[
a_{p1}=a_{p2}=1,\qquad
N_p=(+3)+(-3)=0\,\mathrm{mJy/beam},\qquad
Q_p=1+1=2,
\]

so `m_hat_p = N_p / Q_p = 0 mJy/beam`. The row and its normalized
zero are valid because the numerator is finite, normalization is finite and
positive, support is valid, and all other gates pass. Cancellation does not
fabricate a value for unavailable support.

Mapping: REQ-010--012, REQ-033--035; PRED-003, PRED-025.

## Fixture B — constant-zero input on nonempty support

Take three otherwise admitted samples placed in one support-authorized pixel,
each with finite `z_i = 0 mJy/beam` and exact dimensionless
`gamma_i = 1`. Then `N_p = 0`, `Q_p = 3 > 0`, and
`m_hat_p = 0 mJy/beam`. This is the zero-valued instance of
constant preservation on a nonempty valid row domain. The valid measured zero
must remain distinguishable from an unsupported row, a sentinel, and a dense
zero substituted for a product that was not produced.

Mapping: REQ-011--014, REQ-033; PRED-001, PRED-003, PRED-025.

## Fixture C — zero index and zero placement offset

For an addressable storage domain of length `L > 0`, zero-based index `0` is
in bounds. It is valid when its exact row identity, WCS generation, frame, and
index relation pass. Its numerical value alone is not an error.

For two otherwise compatible observation/coadd grids with equal shape in each
axis, each shape difference is `0`, which is nonnegative and even. The exact
centered-integer half-difference offset is therefore `0`. It is valid when
the reference pixels map to the same world coordinate and every other exact
embedding condition passes. These cases do not relax the existing failure
rules for negative or upper-out-of-bounds indices, odd or negative shape
differences, fractional offsets, incompatible WCS relations, overflow, or
unrepresentable conversion; each retains its declared scope and atomic
nonmutation behavior.

Mapping: REQ-038, REQ-043, REQ-045--047; PRED-018, PRED-025.

## Fixture D — zero normalization and empty applied output

Two outcomes remain distinct.

1. For an individual row with `Q_p = 0`, finite-positive normalization support
   fails. Division by `Q_p` is unauthorized and the row is absent from the
   normalized scientific output. This quantity-specific normalization rule is
   not a generic assertion that every zero value, index, or offset is invalid.
   A support-population count `N = 0` separately retains the exact convention
   `Q_star = 0`, selects no population element, and authorizes no row by
   itself. Likewise `coverage_cut=0` removes only the relative cut and does not
   restore a row lacking finite-positive `Q_p`. An exact zero coefficient
   retains its existing noncontributing treatment.
2. If a valid applied observation or coadd support policy yields an empty
   support-authorized output-row set, the application state is
   `applied_no_support_authorized_output_rows`, the attempt outcome is
   `not_produced`, and the exact cause is
   `no_support_authorized_output_rows`. The record retains the reached
   lifecycle prefix, exact plan, parents, identities, support proof,
   `Q_star`, thresholds, attempt identity, cause, and provenance. No MAP
   product, base-validity state, coadd member, response, covariance, completion
   marker, dense zero, or failure is created.

Mapping: REQ-009, REQ-012, REQ-031--033, REQ-035, REQ-039, REQ-043,
REQ-046--048; PRED-012, PRED-015, PRED-017, PRED-025.

## Prospective discriminator

A conforming result cannot classify these fixtures through a generic
“zero means invalid” rule. It must first identify whether zero is a signal
numerator or normalized signal, normalization, coefficient, policy scalar,
support-population boundary convention, zero-based index, placement offset, or
empty-set cardinality, and then apply that quantity's exact domain and outcome
rule. Non-finite, overflowed, out-of-domain, and unrepresentable quantities
continue to follow their exact declared failure scopes.
