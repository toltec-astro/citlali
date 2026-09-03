# SCI-MAP v0.1 Prior-Work Delta: PTC Handoff

Date: `2026-08-26`

Status: Stage A recovery record; formal contract unchanged

This addendum supplements [`PRIOR_WORK.md`](PRIOR_WORK.md). It does not repeat
the ordinary-map derivation recovered there. It records only the authoritative
changes that arose after the original MAP author packet was frozen.

## Content-Bound Sources Examined

| Source | Role | SHA-256 |
| --- | --- | --- |
| `SCI-PTC/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.5.md` | Exact frozen PTC authority and status | `8357961a49272adc40e27a8aa9e760e0d01ff2419ae2c88a62c0f93c9f959e66` |
| `SCI-PTC/v0.1/src/common/requirements.tex` | Normative PTC-to-MAP intermediate, disabled routing, and coefficient obligations | `f2047600cc06c234a78aa3ddf6a575abf2f9592b3e3da810491f6db0150fe21c` |
| `SCI-PTC/v0.1/README.md` | Frozen package boundary and handoff summary | `d8f723cef8cd1d575757ea0ecea5c2182c5f4b633f6897ffd584ab2367f26cd7` |
| `SCI-VAL/v0.1/SOURCE_BINDING_REGISTER.md` | Current producer binding and deferred MAP profile state | `ff5402b71c40f31daac1f7c820a705a5a23eb64688f70955fac76e10e2916430` |
| `SCI-MAP/v0.1/SCOPE_BRIEF.md` | Existing owner-approved MAP scope premise | `e2a9eb51edb5956191813b4cdbd23866e875d52cdf89cd8b6c272988b4f26674` |
| `SCI-MAP/v0.1/src/common/requirements.tex` | Existing MAP normative premise | `da49405f2702b9a658c63bb9a3ce33f801947ab532b05bff6edf76c9b792393b` |

The program-level owner-decision record is a new approved Stage A authority:
`ALIGN_TO_MAP_HORIZONTAL_OWNER_DECISIONS_2026-08-26.md`, SHA-256
`f5a7cbd0352751b4eecb02f6a6931bd174a08f68a1184b85f517e35c118573f8`.
Its sanitized scientific content enters the proposed author packet through the
bounded reference extract rather than as an additional author input.

## Already-Answered Science Retained

The existing MAP packet remains the recovered authority for ordinary
positive-coefficient normalized gridding, its raw-map estimator, conditional
uncertainty identity, support and validity vocabulary, response/kernel
description, provenance, and centered-integer compatible-grid coaddition.
The 52 existing requirements, 25 predictions, and nine-item owner-decision
ledger are not discarded or renumbered.

## Later Authority That Supersedes Existing Premises

1. **Input identity.** The existing direct SCI-CAL-to-MAP premise is stale.
   Frozen `SCI-PTC-001-D005`, `SCI-PTC-REQ-069`, and
   `SCI-PTC-REQ-077` make the PTC-transformed timestream the ordinary MAP
   input and forbid an inferred CAL-to-MAP fallback.
2. **Neutral versus disabled PTC.** Configuration may realize a neutral
   cleaning transformation while retaining the PTC handoff. Explicitly
   disabled or invalid PTC still yields no ordinary MAP result under the
   frozen PTC authority.
3. **Admission ownership.** PTC availability is necessary; MAP supplies the
   additional policy for the named map use. VAL evaluates the exact registered
   profile and does not author that policy.
4. **Source binding.** The current VAL register intentionally leaves MAP
   deferred and unbound. Stage A must define the MAP-owned profile before the
   register can bind it.
5. **Coordinates and projection.** ALIGN/AST own realized sample coordinate
   facts. MAP owns the target grid and the projection operation. This ownership
   boundary does not select a projection class or normalization.
6. **Response and covariance claims.** Missing or incomplete information must
   be disclosed with meaning, domain, and limitations. It limits supported
   claims but does not automatically invalidate the map or prohibit later
   analysis. Later response/covariance estimates or corrected maps are new
   versioned products and do not rewrite the original bundle's claims.

## Material Excluded From Fresh Derivation

No new author needs to re-derive the estimator, coaddition identity, support
threshold, WCS conventions, response mathematics, or conditional covariance
equation merely because their upstream input name changes. Implementation
behavior and all audit, repair, test, validation, Unity, and performance
evidence remain excluded. JINC and every later roadmap tranche remain separate.

## Existing Questions Preserved Or Narrowed

- `SCI-MAP-OD-001`, `002`, `005`, `006`, `007`, and `009` are unchanged.
- `SCI-MAP-OD-008` remains open for projection classes, normalization,
  boundary loss, and required metadata. Its question of upstream versus MAP
  ownership is answered only to the extent stated by `A2M-OWNER-D007`.
- `SCI-MAP-OD-003` and `SCI-MAP-OD-004` require precise narrowing against
  `A2M-OWNER-D004--D006`; Stage A must propose the wording and normative impact
  without silently marking either fully resolved.

## New Work Required

The revised Stage A packet must identify the exact PTC product and MAP-owned
admission profile, remove VAL-as-policy-author language, preserve causes and
lineage, and state honest response/covariance and later-derivative semantics.
Only that delta belongs in the next implementation-blind author round. The
packet requires scientific-owner approval before dispatch.
