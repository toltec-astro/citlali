# SCI-FRUIT — support and information origin r0.4 (amended)

Status: owner-directed Stage A candidate. Authority: repair direction
SCI-FRUIT-OD-STAGE-A-FINAL-AMENDMENT-2026-09-07, sections 1, 3, 6;
the eight inherited support/influence roles remain distinct.

## Separate support and influence roles

| Role | Required scientific identity |
| --- | --- |
| Measured-parent support | Exact original measured occurrences, producer availability, domain and causes |
| Model-definition support | Separately bind M_k^candidate, M_k^accepted and M_k^applied and where their model objects are defined, with parent/prior origin, all transformations and any exact equality; no common support/unit/normalization/sign/representation/response/uncertainty/parentage is assumed |
| Projection/removal support | Where Pi_k(m_k^applied) and compatible input subtraction are defined; exact mapping, boundary and missing-data behavior |
| Residual-processing support | Exact input and output support of every named F_k operation, with retained membership and causes |
| Model-rejoin support | Where B_k(m_k^applied) is defined at the declared rejoin point, independently of removal support |
| Final output support | Method-defined output domain and valid composition of the two paths |
| Learning influence | Exact transitive dependence of learned/selected state on its inputs; neither local membership nor acquired exposure by identity |
| Response-query support | Domain, perturbation/source basis, conditioning, output support and limitations of the exact requested response family |

Do not identify these roles by array shape, shared coordinates or a common
mask name. They must be materialized or exactly reconstructible with path
contributions, operation/application identities and their scientific queries.
The method specifies any mapping or restriction needed to add contributions
in a common result domain. This table selects no support threshold or rule.
Acceptance does not imply application on identical support. Any support
restriction or other acceptance-to-application transformation is a named
method/state binding. None of the eight roles can be waived by a numerical
method; a missing value retains a truthful typed status and cause.

For removal and rejoin, these support records also bind exact compatibility
of scientific quantity, unit and numerical scale, frame/domain, grid/sampling,
support, additive origin/reference or gauge, null-space state, calibration
convention and response convention. Y_k and Z_k are additive quantity spaces
or explicit affine/quotient spaces with declared reference, gauge and
well-defined arithmetic. Missing compatibility makes the affected composition
unavailable; equal support masks alone do not establish it.

## Output occurrence classification

| Exact class | Information origin and conditional behavior |
| --- | --- |
| `residual_and_model_supported` | Both declared contributions exist on compatible output support; combine only under the method's specified composition |
| `residual_only` | Only the residual path supplies supported output under an explicitly permitted method rule; absence of the model contribution is not an assumed numerical zero |
| `model_only` | Only the model path supplies supported output under an explicitly permitted rule; identify it as model/prior-supported, not a newly measured sky value |
| `unavailable` | The requested output lacks a defined/permitted composition or required input; retain cause and affected scope |

A method may prohibit any of these classes. Supported here means supported
by the exact path definition and its information origin, not simply a finite
payload. If unequal-support behavior is not defined, the affected output is
unavailable. Even the existence of a model at an unmeasured location does not
authorize a model-only output; the method must separately permit it.

No implicit zero fill, extrapolation, imputation, support renormalization,
nearest-value substitution or model completion of missing measured support
is permitted. The notation “residual plus model” does not implement any such
policy or establish equal supports.

The [universal completion gates](ITERATION_STATE_LIFECYCLE.md) govern every
class. A method may add requirements, but cannot waive path support or
information-origin records. Typed unavailable roles may coexist with a
complete iteration only for a narrower claim permitted by both core and
method. Missing science is never zero, identity, successful bypass or
permission to omit a required role.

## Measurement and dependence claims

An upstream unavailable absolute or other additive mode remains observationally
unavailable after model rejoin. Its rejoined contribution is
`model_or_prior_supported`, not `observationally_measured`,
`recovered_from_data`, `acquired_exposure` or `unit_response_truth`.
This is a mode-level information-origin status, not a fifth occurrence-support
class. Even a residual-and-model-supported occurrence may contain such a mode.
Declare its reference/gauge, null-space identity and response/uncertainty
consequences; a numerical gauge choice supplies no missing observation.

Iteration, projection and rejoin create no acquired exposure or independent
observation. Model-only output retains its model/prior origin. Original
measurement lineage and its producer-owned exposure remain distinct from
path contribution and transitive learning influence. Repeated use of a
parent, shared model, calibration, response or reference preserves dependence.
This packet supplies no numerical exposure product or exposure reconstruction
for a producer that does not authorize that role.
