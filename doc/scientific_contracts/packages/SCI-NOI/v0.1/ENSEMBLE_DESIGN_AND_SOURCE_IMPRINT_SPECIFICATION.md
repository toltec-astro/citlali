# SCI-NOI v0.1 — Ensemble Design And Source-Imprint Specification

Artifact identity: `SCI-NOI_ENSEMBLE_DESIGN_AND_SOURCE_IMPRINT v0.1/r0.7`

Status: ordinary coherence, network-stratified coefficient-balance family, and
source-suppression claim boundary owner-approved; exact finite-design mechanics
remain unavailable, Stage-B terminology remains to be authored, and
plan-controlled persistence/regeneration modes are owner-approved

## Finite Assignment-Design Identity

Every GEN method binds

```text
S = {s_bg},  b = 1,...,B,  g in the exact ordered coherence-unit set.
```

The design identity includes:

- exact marginal assignment law and complete joint law;
- coherence-unit definition, owner, stable identity, population, and one stable
  total ordering;
- exact design probability or deterministic design weight for every admitted
  assignment/member;
- balance, complement pairing, replacement, duplicate, and cross-observation
  rules;
- scheduling-independent seed/key derivation and exact algorithm/version;
- exact key fields and canonical serialization;
- the assignment-equivalence relation used by `B_unique`;
- whether complement-related assignments are distinct, paired, or equivalent
  for every count and rank calculation;
- exact duplicate detection; and
- exact design-rank definition, matrix/operator, weights, and domain.

## Stable Ordering And Canonical Key

### Approved ordinary coherence partition

ODQ-102B defines one coherence unit as one stable realized detector/channel
within one exact observation. For member `b`, its assignment applies to every
admitted PTC occurrence of that detector throughout that observation. Scan,
subscan, chunk, sample/time, traversal, worker, container, and MAP accumulation
order do not split or change the unit. The same stable detector identity in a
different observation is a different coherence unit.

The ordinary unit identity is exactly the canonical observation UID joined to
the stable realized detector/channel UID. Units are ordered lexicographically
by those canonical serialized fields, never numerical container position or
encounter order. ODQ-102B does not select the sign law or finite design;
ODQ-102C separately selects the balance family below.

Each coherence unit `g` has an owner-supplied stable scientific identity. The
design order is the lexicographic order of its canonical serialized identity,
never traversal, worker, thread, process, container, row, or memory order.

The candidate canonical assignment-key field order is:

```text
SCI-NOI package/version
GEN route-specific method/version
assignment-design generation
earliest immutable parent product/application generation
observation UID when applicable
stable array/group when applicable
stable coherence-unit type and owner identity
stable coherence-unit identity
realization member identity b
seed/key namespace
assignment algorithm/version.
```

The proposed canonical serialization is a type-tagged, length-prefixed UTF-8
NFC sequence in that exact field order. Integers use canonical base-10 with no
leading zero except `0`; Boolean and unavailable states use fixed lowercase
tokens; no locale, whitespace normalization, platform path, floating-point
display, container index, or implicit absent value participates. The complete
serialized bytes and algorithm/version determine the scheduling-independent
assignment. This serialization remains an owner decision under ODQ-102D; no
implementation is selected by proposing it.

## Approved Ordinary Balance Family

ODQ-102C selects detector assignments `s_bd` in `{-1,+1}` under a
network-stratified, coefficient-balanced randomized design. For one exact
observation and stable readout network `h`, let `D_h` be its ordered admitted
detector population and define

```text
B_d = sum_p sum_{i in C_p, detector(i)=d} a_pi,
a_pi = G_pi gamma_i > 0.
```

`C_p`, `G_pi`, and `gamma_i` are the exact frozen MAP contribution population,
projection, and PTC-owner-selected MAP-facing coefficient used by the ordinary
host route. The design balances the signed totals `sum_{d in D_h} s_bd B_d`
separately for every network. Cross-network, cross-array, and
cross-observation cancellation cannot satisfy a network's balance rule.

The admissible assignment set and probabilities are complement-symmetric: if
`s_b` is admitted, `-s_b` is admitted with equal probability. Each detector
therefore retains marginal sign probability `1/2`, while detector signs are
not asserted independent after conditioning on network-local balance. Equal
numbers of positive and negative detectors are not required.

`B_d` is an NOI design coefficient derived from exact frozen MAP numerical
influence. It is not inverse variance, precision, empirical NOI weight,
exposure, support, validity, or a replacement for `gamma_i`. Changed parent,
observation, network, admitted population, coefficient family/generation, or
MAP plan changes the design identity.

ODQ-102C authorizes no exact imbalance norm, tolerance, feasibility rule,
candidate/search algorithm, retry cap, failure behavior, count, forced
complement pairing, replacement, equivalence, duplicate treatment, or rank
definition. ODQ-102D delegates those selections and their scientific rationale
to the implementation-blind contract author. Its tolerance-conditioned scheme
is a nonbinding suggestion, not approved normative mechanics. Until exact
Stage B mechanics are authored and later owner-accepted, the selected balance
family is numerically unavailable.

Observation-global, detector-count, pixel-vector, source-template,
complement-paired, and other balance families remain possible only as
separately named future methods.

## Counts, Equivalence, Duplicates, And Rank

The following are never aliases:

- `B_requested`;
- `B_resolved`;
- `B_completed`;
- `B_unique`;
- complement-pair count;
- `B_admitted_for_UNC`;
- exact design rank; and
- use-specific effective information.

Assignment equality means exact equality of the assignment vector on the same
ordered coherence domain with identical design generation and value semantics.
The method must separately declare whether global complements are distinct,
paired-but-distinct, or equivalent for `B_unique`, rank, and effective-
information purposes. No default complement equivalence is inferred.

Duplicate detection first compares a cryptographic digest of the canonical
assignment serialization and then requires byte equality of the serialized
assignment and identical ordered domain before declaring a duplicate. A digest
collision is not assignment equality.

Design rank is the exact mathematical rank of the method-declared weighted
design/contrast operator on its exact ordered coherence/member domain after
the declared centering, equivalence, completion, and admission rules. The
operator, field, tolerance or exact-arithmetic rule, weights, and null space are
part of the method identity. Member count alone is not rank.

A changed scheduling order, container order, worker count, traversal, or
parallel decomposition shall not change ordering, key bytes, assignments,
equivalence, duplicates, counts, or rank.

## Admission And Fail-Closed Completion

GEN owns each member's completion and terminal state. Enabled GEN requires a
positive resolved design. Disabled GEN is explicit zero-member/no-work.
Candidate assignments rejected during finite-design construction are search
outcomes, not failures or ensemble members. Failure to resolve the required
positive admitted design under the bounded construction is a design-resolution
failure, not an individual candidate failure.

Every admitted assignment is one requested member and must complete through
the full declared frozen operator. Any incomplete, failed, or unavailable
admitted member fails the complete GEN ensemble and makes it unavailable for
every UNC use. Completed members from that failed ensemble cannot be admitted
as a survivor or partial design. If retained diagnostically, they remain bound
to the failed ensemble and carry no UNC authority. A retry or replacement uses
a new exact generation/design identity.

GEN records sufficient failed ensemble/member identity, scientifically
meaningful stage, terminal state, cause category, and diagnostic context to
investigate the run without requiring exhaustive implementation provenance. A
VAL evaluation neither authors completion truth nor rescues a failed ensemble.

## Approved Persistence And Regeneration Modes

ODQ-109 admits three plan-selected modes with distinct requested, effective,
applied, and realized state: persisted ensemble, compact deterministic
regeneration, and streaming sufficient statistics. There is no universal
default and no silent fallback among them.

Persisted mode retains every required completed member with exact ensemble,
parent, method, operator, assignment/design, domain/support/response,
lifecycle, and provenance identity. Compact mode may omit member payloads only
when immutable parents, exact method/algorithm versions, frozen operator state,
canonical unit ordering, finite design, admitted membership, assignment
key/seed/counter, and full configuration reconstruct every assignment and
declared scientific product under an explicit byte-identical or numerical
reproducibility contract. Dense signs, per-sample sign provenance, randomized
timestreams, and realization maps are not universally required.

Streaming mode may consume members transiently only when the retained state is
mathematically sufficient for every exact published product and claim. The
initial second moment requires its exact design-weighted accumulation, common-
all-member availability/domain state, and every dependence/design quantity
needed for effective-information and estimator-uncertainty reporting. The
record lists every unsupported later reconstruction or reanalysis and cannot
claim a retained ensemble.

ODQ-105A remains absolute in every mode. Every admitted member completes;
non-persistence cannot create a survivor ensemble. Any admitted-member failure
fails the ensemble for UNC, and partial streaming accumulators carry no UNC
authority. Failure of plan-required persistence fails that product; planned
transience is not failure. Completion/membership/mode/audit truth remains
immutable even without payload persistence.

Persistence, regeneration, or retained sufficient statistics establish no
adequacy, covariance completeness, calibration, significance, conformity,
performance, readiness, or production authority.

## Source-Imprint Identity

Every GEN method declares:

1. signal and source content in the earliest parent;
2. exact cancellation or suppression target;
3. assumptions supporting expected cancellation;
4. finite-design balance residual;
5. variation of support, coefficient, projection, filtering, or selection;
6. scan-synchronous and other structured residuals;
7. source-model use and model error;
8. known or bounded leakage; and
9. permitted and prohibited claims.

The randomization is intended to suppress source signal. Global sign balance
does not, by construction alone, establish source-free maps or pixelwise source
cancellation, especially when support, coefficients, projection, filtering, or
membership varies. Unless stronger authority is separately approved, the
product is not a repeated physical-noise ensemble, source-free null, calibrated
null, variance, covariance, precision, or significance product.

The Stage-B scientific author shall select exact scientist-readable terminology
that preserves this meaning. `source_imprinted_conditional_randomization_ensemble`
is a nonbinding terminology suggestion only; it is not a required product name.
