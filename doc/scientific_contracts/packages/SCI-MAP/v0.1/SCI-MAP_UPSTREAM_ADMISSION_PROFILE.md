# SCI-MAP Upstream-Admission Profile

Profile identity: `SCI-MAP:map_upstream_admission@1`

Status: MAP-owned r0.5 profile registered in the continuing SCI-VAL Registry;
implementation evaluation and conformity not assessed

Scientific-policy owner: Grant Wilson

The authoritative registry record is in
`SCI-VAL/v0.1/PROFILE_REGISTRY.md`. This package copy is the scientist-readable
MAP view and shall remain semantically identical.

## Decision

The profile evaluates one exact PTC occurrence for the named use
`map_upstream_admission`. The object binds observation, detector
occurrence/UID, stable RTC output `n`, exact PTC product/application
generation, segment, array/network/group, and complete ancestry.

Applicability requires the requested ordinary positive-rank PTC-to-MAP route.
The exact PTC product must exist; `SCI-PTC:output_retention@1` must be eligible;
the transformed signal must be available; an exact PTC-owner-selected
MAP-facing coefficient and coefficient/QC permission must be available; and
the exact AST RTC-grid coordinate for the same `n` and parent chain must be
structurally bound.

PTC-disabled, no-product, direct CAL input, inferred no-op PTC, PTC retention
ineligibility, direct synthesized/replaced representative signal origin,
incompatible parents/generations, or coefficient/QC exclusion is decisive.
CAL `engineering-only` classification may remain a candidate only when PTC
retains it and the classification remains explicit; no science qualification
is created. Transitive influence and other causes are preserved but are not
universal vetoes unless a required predicate names them.

Response and uncertainty are advisory for base numerical signal admission.
Their exact class, state, domain, limitations, and causes remain carried. A
later claim may require them through its own exact binding. Missing or
conflicting applicability, identity, generation, parent, permission,
coefficient, or coordinate facts yields `applicability_unknown` and
`decision_unavailable`; a decisive false predicate yields `ineligible`; every
required predicate true yields `eligible`.

The result is occurrence-level and atomic-only. `eligible` creates a MAP-route
candidate, not a pixel contribution. MAP still owns `G_pi`, boundary,
finiteness, positivity, support, required companions, accumulation, exposure,
and MAP-local validity. VAL registers and evaluates this policy; it does not
author, broaden, aggregate, place, or execute it.
