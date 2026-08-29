# SCI-NOI v0.1 — Ensemble Design And Source-Imprint Specification

Artifact identity: `SCI-NOI_ENSEMBLE_DESIGN_AND_SOURCE_IMPRINT v0.1/r0.2`

Status: proposed sanitized Stage A scientific input; exact bytes await owner
approval

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
encounter order. ODQ-102B does not select the sign law, probabilities, balance,
complements, cross-observation dependence, or finite design; ODQ-102C owns
those choices.

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
assignment. This serialization remains an owner decision under ODQ-102C; no
implementation is selected by proposing it.

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

## Completion And Partial Designs

GEN owns each member's completion and terminal state. Enabled GEN requires a
positive resolved design. Disabled GEN is explicit zero-member/no-work.
Failed, incomplete, duplicate, or unavailable members remain exact facts.

An ensemble with failed members is eligible for a named UNC method only when
that exact method admits the resulting completed design and recomputes all
counts, probabilities/weights, equivalence, rank, null space, and effective
information on the admitted population. Otherwise the ensemble is unavailable
for that UNC use. A VAL evaluation does not author completion truth.

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

Global sign balance is not pixelwise source cancellation when support,
coefficients, projection, filtering, or membership varies. Unless stronger
authority is separately approved, the ordinary claim is
`source_imprinted_conditional_randomization_ensemble`. It is not a repeated
physical-noise ensemble, source-free null, calibrated null, variance,
covariance, precision, or significance product.
