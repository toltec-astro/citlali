# SCI-NOI v0.1 — Scientific-Owner ODQ-111 Approval

Decision identity: `SCI-NOI-ODQ-111`

Scientific owner: Grant Wilson

Decision date: `2026-08-30`

Status: approved NOI-owned VAL profile identities and exact consumer actions

## Exact Owner Decision

The owner approved the four NOI-owned profile identities and their exact
consumer actions for later versioned SCI-VAL Registry/source binding:

1. `SCI-NOI:generation_input_admission@1` may admit one exact object as a
   candidate occurrence for
   `NOI-GEN/PTC-TO-FROZEN-MAP-CONDITIONAL-SIGN@1` only. Assignment, MAP
   application, GEN completion/QC, ensemble completion, and UNC admission
   remain separate.
2. `SCI-NOI:uncertainty_member_admission@1` may admit one exact GEN member as a
   candidate member for one named UNC method. It does not admit the ensemble.
3. `SCI-NOI:uncertainty_ensemble_admission@1` may admit one exact complete,
   all-members-successful GEN ensemble to one exact named UNC estimator and
   domain. Any method-specific estimator/representation action remains limited
   to that exact method, domain, and claim; no UNC product exists until its
   operator realizes and atomically publishes it.
4. `SCI-NOI:standardization_admission@1` may permit construction of
   `S_cond=q_MAP/sqrt(V_hat_cond)` as the unit-`1`
   `empirical_scale_standardized_signal` on the exact compatible finite-
   positive intersection, with only the approved conditional-randomization-
   scale claim.

## Ownership And Evaluation Boundary

Producer facts remain owned by their producers. In particular, GEN owns
realization-member completion, terminal state, duplicate/equivalence, support,
source-imprint, QC, persistence/reconstruction, lifecycle, cause, and failure
facts. NOI owns each named-use admission policy and exact consumer action.

SCI-VAL may bind approved immutable policy bytes in a versioned Registry/source
record and VAL Core may evaluate them. SCI-VAL authors neither producer facts
nor NOI policy and may not broaden, aggregate, combine, or replace an action.

Every profile retains distinct request, applicability, eligibility, and
realization fields. Only `requested + applicable + eligible + realized`
projects to that profile's exact action. No generic pass bit, producer flag,
finite payload, completed member, another-use decision, or aggregate state has
universal veto or rescue authority.

## Separation Of Actions

Generation-input admission does not create an assignment or realization.
Member admission does not admit an ensemble. Ensemble admission does not
realize or publish an estimator. Standardization admission does not construct
its uncertainty parent or create a stronger statistical claim. No profile
automatically realizes the next GEN, UNC, or STD operation or modifies an
immutable parent.

Changing any profile's object/domain, source binding, required facts,
restrictions, exceptions, source-imprint/response/uncertainty role,
missing/conflict behavior, propagation, lifecycle, or action requires a new
immutable profile version and evaluation generation.

## Registry And Claim Boundary

This approval authorizes the exact SCI-NOI profile policy bytes for later
binding. It does not create a SCI-VAL Registry entry or source-binding record,
make any profile evaluable, admit any numerical method, launch Stage B, or
approve the final packet hashes. Those remain separate gates.

Registration or evaluation would establish no implementation conformity,
validation, calibration, physical-noise meaning, covariance completeness,
Gaussian significance, performance, readiness, or production authorization.
