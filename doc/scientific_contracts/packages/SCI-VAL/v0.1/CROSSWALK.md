# SCI-VAL v0.1/r0.3 Crosswalk

Status: targeted second-pass crosswalk; scientific authority not frozen;
implementation conformity and validation not assessed

The scientist-facing sources below refer to numbered sections of the r0.3
Science-Team Rationale. The engineering interpretation is carried by the
canonical formal modules and rendered in the Engineering Conformance
Specification.

## Requirement coverage

| Requirement | Scientist-facing source | Engineering interpretation |
| --- | --- | --- |
| SCI-VAL-REQ-001 | §2, four-authority chain | Exact producer facts plus one owner-bound profile are the only scientific inputs. |
| SCI-VAL-REQ-002 | §2, authority table | Producer, profile owner, registry, Core, and consumer authority remain distinct. |
| SCI-VAL-REQ-003 | §2, producer facts | Bind stable occurrence/detector/parent/lifecycle identity, never row order. |
| SCI-VAL-REQ-004 | §3, open-world facts | Preserve true, explicit false, unknown, conflict, and applicability separately. |
| SCI-VAL-REQ-005 | §3, silence and absence | Silence is unknown; only an owning producer with a complete family can assert absence. |
| SCI-VAL-REQ-006 | §2, producer authority | Consume a producer-local composite as an opaque, versioned producer fact. |
| SCI-VAL-REQ-007 | §4, causal records | Cause/influence union is order-independent, idempotent, and non-erasing. |
| SCI-VAL-REQ-008 | §4 and §6, immutable replay | Bind use, registry record, owner/source, parent, lineage, and reasons. |
| SCI-VAL-REQ-009 | §6, package-qualified names | Use classes do not supply policy; distinct package questions receive distinct names. |
| SCI-VAL-REQ-010 | §2 and §3, registry/structural gate | Complete authority and registry checks before predicates or consumer mutation. |
| SCI-VAL-REQ-011 | §3, four-axis table | Request, applicability, eligibility, and realization are independent axes. |
| SCI-VAL-REQ-012 | §3, partial proposition | Not requested and known inapplicable assert no eligibility proposition. |
| SCI-VAL-REQ-013 | §3, unavailable versus failure | Realized decision-unavailable differs from failed or unproduced artifact. |
| SCI-VAL-REQ-014 | §3, conflict and no-rescue logic | False dominates unrelated non-gating unknown/conflict; without false, a required unknown/conflict blocks eligibility. |
| SCI-VAL-REQ-015 | §3 and §4, preserved reasons | Retain every material false, unknown, conflict, influence, exception, and binding reason. |
| SCI-VAL-REQ-016 | §3, no-rescue logic | Applicable restrictions compose conjunctively in permission. |
| SCI-VAL-REQ-017 | §3, permitted exceptions | Only resolved same-profile permitted exceptions apply; none erase causes or invariants, and exception conflict cannot neutralize a restriction. |
| SCI-VAL-REQ-018 | §1 and §3 | A permission is a disposition, not a producer cause. |
| SCI-VAL-REQ-019 | §1, worked occurrence | Direct representative synthesis/replacement is ineligible under the canonical profile. |
| SCI-VAL-REQ-020 | §1 and §4 | Direct replacement does not decide a different registered use. |
| SCI-VAL-REQ-021 | §4, influence diagram | Preserve exact/conservative and confirmed/possible influence. |
| SCI-VAL-REQ-022 | §4, possible influence | Owner profile selects consequence; review remains action metadata. |
| SCI-VAL-REQ-023 | §6, payload ordering | Exclude invalid payload before numerical evaluation; finiteness gives no admission. |
| SCI-VAL-REQ-024 | §6, payload ordering | An admitted required non-finite payload fails or becomes unavailable at declared scope. |
| SCI-VAL-REQ-025 | §6, response and uncertainty | Owner assigns exactly one of structural gate, required permission, decisive exclusion, or advisory; VAL applies deterministic role semantics. |
| SCI-VAL-REQ-026 | §2 and §4, PTC stage distinctions | Basis fit, loading fit, application, output, coefficient/QC, weight, and advice stay separate. |
| SCI-VAL-REQ-027 | §4, immutable generations | Later facts, profiles, sources, or propagation create a new generation. |
| SCI-VAL-REQ-028 | §3 and §4 | Equal immutable inputs produce equal results independent of order or ambient state. |
| SCI-VAL-REQ-029 | §4 and §6, source/profile replay | Replay binds recorded scientific content, never ambient current state. |
| SCI-VAL-REQ-030 | §5, aggregation declaration | Bind a distinct aggregate profile, exact atomic source profile, owner/source/domain, population, support, operator, threshold, propagation, and failure semantics. |
| SCI-VAL-REQ-031 | §5, aggregate lineage | Retain aggregate-profile lineage, every atomic identity/profile/source lineage, and missing/failed count. |
| SCI-VAL-REQ-032 | §5, denominator rule | Empty or unknown denominator is never a valid zero fraction. |
| SCI-VAL-REQ-033 | §5, propagation diagram | Reverse propagation requires authority and creates successor facts. |
| SCI-VAL-REQ-034 | §5, partition rule | Homogeneous permutation invariance is mandatory; partition equivalence is owner-declared. |
| SCI-VAL-REQ-035 | §1 and §6, MAP boundary | MAP upstream admission is not projection, contribution, support, or final validity. |
| SCI-VAL-REQ-036 | §2 and §6 | Consumers cannot erase causes or promote invalid parents from finite descendants. |
| SCI-VAL-REQ-037 | §3 and §6 | Structural conflict makes applicability unknown and the decision unavailable; established-domain non-gating conflict follows false-dominant precedence. |
| SCI-VAL-REQ-038 | §2, §3, and §6 | VAL supplies no missing fact, clean cause, identity response, zero uncertainty, alias, or predicate. |
| SCI-VAL-REQ-039 | §3, owner-declared profile relation | Monotonicity applies only to an owner-declared relation on one exact domain. |
| SCI-VAL-REQ-040 | §4, immutable generations | Requested-to-realized profile lineage flows only forward. |
| SCI-VAL-REQ-041 | §2 and §7, claim limits | No representation, numerical policy, adjacent policy, or production default is selected. |
| SCI-VAL-REQ-042 | §7, present status | No capability authorization, conformity, validation, freeze, or readiness claim. |
| SCI-VAL-REQ-043 | §2, Core/Registry split | Core evaluates; registry binds; neither authors or acquires policy. |
| SCI-VAL-REQ-044 | §2 and §6, registry binding | A usable record binds exact owner, source/digest, domain, restrictions, exceptions, compatibility, and missing behavior. |
| SCI-VAL-REQ-045 | §1, forbidden override | An attempted exception to the direct-origin invariant is policy-invalid and never eligible. |
| SCI-VAL-REQ-046 | §6, profile names | Reject the ambiguous legacy MAP label and require package-qualified PTC/MAP identities. |
| SCI-VAL-REQ-047 | §5, homogeneous aggregation | Exact atomic profile/version, stage, object type, and domain must match, and the distinct aggregate profile binds that source key. |
| SCI-VAL-REQ-048 | §5, anti-circular propagation | Successor propagation cannot rewrite or feed its own denominator generation. |
| SCI-VAL-REQ-049 | §6, source table | Bind exact adjacent source/version/digest; changed or unavailable meaning cannot be inferred. |

## Prediction coverage

| Prediction | Scientist-facing source | Engineering falsifier |
| --- | --- | --- |
| SCI-VAL-PRED-001 | §1, independent-exposure row | Any eligible direct-replacement result under the canonical profile falsifies the contract. |
| SCI-VAL-PRED-002 | §1, hypothetical non-independent rows | Another use is conditionally eligible only if its exact profile is actually registered, applicable, and satisfied. |
| SCI-VAL-PRED-003 | §1 and §6 | An unbound PTC/MAP profile name cannot yield eligibility. |
| SCI-VAL-PRED-004 | §4, cause preservation | Cause/influence permutation must preserve disposition and reason graph. |
| SCI-VAL-PRED-005 | §4, cause preservation | Duplicate cause/edge insertion must be idempotent. |
| SCI-VAL-PRED-006 | §3, no-rescue logic | False plus unrelated unknown/conflict must remain ineligible with every reason. |
| SCI-VAL-PRED-007 | §3, no-rescue logic | Required unknown/conflict with no false must remain decision unavailable until resolved. |
| SCI-VAL-PRED-008 | §2, §3, and §6 | Structural parent, digest, source, compatibility, or applicability conflict must produce applicability unknown and block mutation. |
| SCI-VAL-PRED-009 | §3, four axes | Not requested, inapplicable, and realized unavailable must remain distinguishable. |
| SCI-VAL-PRED-010 | §3, permitted exception | A resolved permitted exception affects only its named exceptionable restriction; unknown/conflicting applicability cannot neutralize it. |
| SCI-VAL-PRED-011 | §4, possible influence | Different profiles may map the same possible edge differently without relabeling it. |
| SCI-VAL-PRED-012 | §4, immutable generations | Generation k scientific content remains unchanged after generation k+1. |
| SCI-VAL-PRED-013 | §5, denominator rule | Empty/unknown denominator cannot produce an eligible zero fraction. |
| SCI-VAL-PRED-014 | §5, aggregation law | Homogeneous permutations match under a distinct aggregate profile; partitions match only under profile-declared equivalence. |
| SCI-VAL-PRED-015 | §6, payload ordering | Invalid-before-numeric and admitted-non-finite failure remain distinct. |
| SCI-VAL-PRED-016 | §6, response/uncertainty | The four owner-supplied roles have deterministic consequences; unavailable state cannot replay as zero or identity. |
| SCI-VAL-PRED-017 | §1 and §6, MAP boundary | MAP upstream eligibility alone cannot establish contribution or final validity. |
| SCI-VAL-PRED-018 | §3, declared strictness | A valid stricter relation cannot admit what the weaker profile excludes. |
| SCI-VAL-PRED-019 | §1, forbidden override | Attempting to except direct replacement must yield policy-invalid, never eligible. |
| SCI-VAL-PRED-020 | §4 and §6, source replay | Different source digests create distinct replay identities; a missing digest is not substituted. |
| SCI-VAL-PRED-021 | §5, homogeneity and aggregate identity | Mixed atomic keys, missing aggregate profile/source binding, or aggregate/atomic identity reuse must be unavailable before arithmetic. |
| SCI-VAL-PRED-022 | §5, propagation diagram | Propagation creates generation k+1 while denominator decisions remain unchanged. |
| SCI-VAL-PRED-023 | §6, reserved names | A reserved package-qualified name alone cannot produce eligibility. |
| SCI-VAL-PRED-024 | §6 and §7, source-change status | An incompatible source revision makes the dependent evaluation unavailable. |

## Mechanical coverage

- `src/engineering-conformance.tex` imports the six canonical formal modules
  exactly once; the science-team rationale is a distinct narrative view and
  declares no normative identifiers.
- `src/verify_contract.py` verifies the original content-bound Stage B packet,
  r0.2 and r0.3 revision authority, sequential and unique declarations, preservation of
  IDs `REQ-001--042` and `PRED-001--018`, exact crosswalk set coverage,
  scientist-facing section anchors, canonical PDF names, engineering-PDF ID
  coverage, and page-count expectations.
- The canonical profile and exact adjacent source bindings are recorded in
  `PROFILE_REGISTRY.md` and `SOURCE_BINDING_REGISTER.md`.
