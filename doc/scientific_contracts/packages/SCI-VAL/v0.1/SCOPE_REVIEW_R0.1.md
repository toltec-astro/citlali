# SCI-VAL v0.1 — Stage A Scope Review r0.1

Status: review absorbed; bounded Scope Brief revision prepared; owner approval
still required

Review supplied: `2026-08-20`

Review source SHA-256:
`df5fdb7cc80dc870993c8cbe8c24a09536e2fedb587d4fe2e2c76a426a126336`

## Overall Disposition

The review accepts the central scientific problem and producer–VAL–consumer
boundary, then identifies seven bounded corrections needed before Stage B.
All seven are adopted as Stage A revisions. No implementation, audit,
validation, or production work is authorized.

## Required Corrections

| Review issue | Disposition |
| --- | --- |
| Package name can overstate VAL authority | Retain the inventory name, but add an explicit early limitation and require every output to read “eligible under policy P for use U,” never unqualified “valid” or “eligible.” |
| Three dispositions alone conflate request, applicability, eligibility, and realization | Add four independent axes and a gating rule that distinguishes structural decision-domain failures, decisive false predicates, unknown required predicates, and all-true admission. |
| “Cause precedence” can imply one global ordering | Replace global precedence with an order-independent, idempotent cause set/graph plus use-specific policy evaluation; add open-world explicit-negative semantics. |
| Policy algebra risks being scientifically empty | Adopt the mandatory base gate and eight canonical named-use profiles recommended by the review. |
| Post-fit facts can create circular or retroactive decisions | Add immutable fact-set, VAL-decision, consumer-stage, successor-fact lifecycle with a new decision identity for every later fact set. |
| Aggregation is under-specified | Require population, time support, counts, denominator, missing semantics, operator/threshold, polarity, propagation authority, and data-dependence/uncertainty. |
| Exact and conservative influence can be conflated | Require exact/conservative and confirmed/possible distinctions, support/rule identity, and profile-specific treatment. |

## Owner-Question Dispositions Proposed By The Review

- `VAL-OWNER-Q001`: approve with narrowing to reusable evaluation of typed
  producer facts under named-use policies.
- `VAL-OWNER-Q002`: approve a mandatory base gate plus a minimal canonical
  profile set.
- `VAL-OWNER-Q003`: approve decision-unavailable gating semantics while
  allowing a known decisive disqualifier to establish ineligibility after the
  decision domain is identified.
- `VAL-OWNER-Q004`: approve direct representative synthesis/replacement as a
  disqualifier specifically for `independent_exposure`, with every other use
  governed by its named profile.

These are incorporated as recommended dispositions. Because the review was
provided as feedback rather than an explicit owner approval statement, the
ledger remains OPEN pending confirmation by Grant Wilson.

## Author-Packet Correction

SCI-VAL is intrinsically cross-package. The initial three-file packet is too
isolated. Add one sanitized, owner-approved
`AUTHOR_CROSS_PACKAGE_BOUNDARY_PROFILE.md` containing exact RTC, CAL, PTC, and
MAP boundary facts. Full adjacent drafts, implementation, schemas, audits,
repairs, tests, validation, and current flag encodings remain excluded.

## Stage B Rationale Direction

The revised brief requires the rationale to begin with one finite retained
but replaced occurrence under several named uses, then explain producer facts,
applicability, dispositions, direct versus transitive influence,
nonretroactivity, aggregation, response/uncertainty gating, missing facts,
replay, and validation. Formal properties include determinism, cause-order
invariance, idempotent cause composition, monotonicity under a declared
stricter-profile relation, nonretroactivity, and fail-closed missing-fact
behavior.

No Stage B author has been dispatched and Ultra has not been invoked.
