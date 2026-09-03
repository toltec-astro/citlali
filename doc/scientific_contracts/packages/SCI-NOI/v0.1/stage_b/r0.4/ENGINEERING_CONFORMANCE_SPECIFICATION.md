# SCI-NOI v0.1 r0.4 Engineering Conformance Specification

Document identity: `SCI-NOI_ENGINEERING_CONFORMANCE v0.1/draft-r0.4`

Scientific owner: Grant Wilson

Date: 2026-08-30

Status: implementation-blind proposed final Stage B draft; scientifically
freezeable conditional; not frozen.

Normative authority: the six ordered modules in
`SCI-NOI_NORMATIVE_MODULE_BINDING v0.1/r0.4`, binding-file SHA-256
`5c6de3bd5180c9231c79cabe5f5918938571340a9f836f437962779e0410d55a`.
This ECS supplies prospective evidence procedures only and authors no science.

# 1. Evidence identity and result vocabulary

Every result shall identify exact product, method, parent, observation, TolTEC
array, plan, generation, lifecycle, and named use. Review compares serialized
scientific identities and typed states, never filenames, shapes, generic flags,
approximate WCS, or undocumented defaults.

Every gate shall report one of pass, fail, unavailable, or not applicable with
exact cause. Method identity, source closure, plan resolution, numerical
realization, response-bearing realization, empirical calibration,
implementation conformity, validation, and production authorization shall be
reported separately.

# 2. Ordinary operator and immutable-parent evidence

Evidence shall expose retained PTC occurrence, detector/network, coefficient
family/value/QC, canonical rational coefficient, projection contribution,
unsigned denominator, numerical `coverage_cut`, WCS/support, response,
assignment, and application count. It shall demonstrate one sign occurrence
only on `Z_i^PTC` and no changed MAP gate. A required response companion shall
use identical membership and sign exactly once.

The all-`+1` arithmetic shall be compared only to an independently governed,
scientifically realized ordinary SCI-MAP product. Missing coefficient,
`coverage_cut`, MAP admission, response, profile authority, or other parent fact
shall make the dependent route unavailable.

# 3. Cardinality and universal design evidence

A disabled fixture shall have `N_requested=0`, no design identity, no member,
no successful ensemble, no UNC, and no STD. An enabled-success fixture shall
have positive integer request and exact equality
`N_resolved=N_requested>0`. An enabled-failure fixture shall fail the complete
design when one required member exhausts its cap and shall publish no smaller
successful resolved ensemble. Rejected candidates shall appear only in design-
construction evidence.

Every design record shall include exact law or finite measure; stable
population/order; alphabet; probabilities/weights; equality, duplicate,
complement, and equivalence relations; all counts; applicable ranks/null
spaces; deterministic reconstruction; lifecycle, generation, completion, and
failure; and preregistration of every applicable field. Unused fields shall be
`not_applicable`.

# 4. Exact balance-arithmetic evidence

Every persisted `a_pi` and `tau_h` shall serialize as canonical ASCII `n/q`.
Evidence shall reject nonpositive denominators, unreduced pairs, noncanonical
zero, nonpositive active mass, and thresholds outside `[0,1)`. Exact rational
accumulation in stable identity order shall reproduce every `beta_d`, imbalance
numerator, total, and admitted-set membership. Admission shall be reproduced by
integer cross multiplication, with no floating comparison, epsilon, tolerance
relaxation, or best-failed selection.

The following boundary fixtures are mandatory:

| Fixture | Exact `L_h` | Exact `T_h` | Exact `tau_h` | Required decision |
| --- | --- | --- | --- | --- |
| Equality | `1/10` | `1/1` | `1/10` | admit, because equality is included |
| One rational step above | `10000000000000001/100000000000000000` | `1/1` | `1/10` | reject |
| One rational step below | `9999999999999999/100000000000000000` | `1/1` | `1/10` | admit |

The second fixture is deliberately close enough that a finite floating
conversion can collapse it onto the equality value; exact arithmetic must keep
the opposite decisions. A reordered-accumulation fixture with widely separated
rational magnitudes shall reproduce identical canonical totals and membership.

# 5. Selected rejection law and keying evidence

Evidence shall enumerate complete and active populations, zero-mass facts,
zero-total `not_applicable` strata, unavailable ambiguous positive-mass
membership, and unavailable singleton active strata. It shall reproduce exact
thresholds, independent symmetric base candidates, first-accepted search,
positive cap, fail-closed exhaustion, replacement, separate duplicate/orbit
relations, equal `1/N_resolved` weights, and uncentered `r_sign`.

The plan shall bind generator identity and version, opaque seed/key bytes,
namespace, canonical serialization, observation, array, stratum, member,
candidate-attempt counter, stable detector order, and counter-domain separation.
Replays varying scheduling, process and worker counts, traversal, container
layout, and persistence mode shall yield identical assignments.

An exact enumeration or analytic proof shall establish conditional uniformity:
for each admitted vector `x`, the probability that trial `k` is the first
accepted vector and equals `x` is `(1-q)^(k-1) p`, where common candidate
probability `p` is identical for every admitted vector and `q` is total
admission probability. Summing over the cap supplies the same factor for every
`x`; normalization is uniform. Separate member and stratum streams shall support
the complete conditional joint law `epsilon_b iid ~ Uniform(A)`.

# 6. Completion, UNC membership, and information evidence

Evidence shall distinguish candidate rejection, member failure, scientific
replacement, and idempotent execution retry. A retry retaining scientific
identity shall prove identical work and distinct `I_attempt`. Persistence
evidence shall prove exact reconstruction, regeneration, sufficiency, lost
questions, lifecycle, completion, and failure semantics for the selected mode.

Initial UNC shall prove set equality `A_UNC=B_resolved` and count equality
`N_admitted=N_completed=N_resolved=N_requested>0`. Every resolved member shall
have a positive exact member decision and belong to one common domain. A fixture
in which every member completes but one is UNC-ineligible shall produce an
unavailable complete estimator and no subset, renormalization, or changed
divisor.

Estimator reproduction shall use the exact known-zero-center formula with
divisor `N_resolved`, no finite-ensemble centering, and no `B-1`. The exact
primary product name and target-law variance consequence shall remain separate.
Counts, duplicates, complement orbits, `r_sign`, `r_map`, estimator uncertainty,
and effective information shall be independently reported. An effective-
information result shall state estimator, target, domain, dependence model, and
calculation or be unavailable.

# 7. Reciprocal, STD, and response evidence

No reciprocal ordinary-base evidence is presently evaluable. A future route
requires exact owner disposition, finite-positive parent domain, separate
method/profile/use, and no role promotion.

STD evidence shall bind independently realized `m_MAP` and compatible
`Vhat_cond`, reproduce `sigma_cond=sqrt(Vhat_cond)` and `zeta_cond`, verify unit
`1`, and emit only the bounded claim. It shall carry numerator/scale dependence,
support intersection, response state, and cause, with no interpolation,
extension, implicit reciprocal, JINC substitution, or significance language.

Response fixtures shall distinguish:

1. scale explicitly held fixed, reproducing only
   `delta zeta_fixed_scale=diag(1/sigma_cond) delta m_MAP`; and
2. scale responding, reproducing the additional
   `-m_MAP delta Vhat_cond/(2 sigma_cond^3)` term.

Without separately authorized `delta Vhat_cond`, the full STD procedure response
shall report unavailable. Dividing MAP response by `sigma_cond` shall not pass a
full-response gate.

# 8. Profile, source, and parity evidence

Evidence shall preserve request, applicability, eligibility, and realization
as separate fields. The four immutable r0.18 profiles shall reproduce their
exact block digests and Registry/source bindings. Each changed r0.4 use shall
bind the exact complete `@2` bytes, owner approval, and successor Registry/source
record before evaluation. A proposed name or old profile shall not pass the
changed-action gate; missing authority yields unavailable, not numerical
ineligibility or a successful empty product.

`REQUIREMENT_PREDICTION_OWNER_TRACEABILITY.csv` supplies one row per stable
requirement with exact owner source, predictions, dependencies, and review
kind. Future review shall report every row. This draft supplies no current
implementation conformity, validation, calibration, performance, readiness,
freeze, or production result.
