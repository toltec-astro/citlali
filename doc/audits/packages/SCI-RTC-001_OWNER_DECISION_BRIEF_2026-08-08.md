# SCI-RTC-001 owner decision brief

Date: 2026-08-08<br>
Application SHA: `46ad23888a40f5102cdfd50c06e49a549bdf8a20`<br>
Audit branch: `codex/audit-sci-rtc-001`<br>
Disposition: documentation-only audit complete; stop for coordinator/owner review

## Decision in one paragraph

Adopt or supersede the frozen RTC contract before authorizing any repair. The
audit recommends accepting its four defaults: continuity replacements and
synthesized cells have zero direct independent science eligibility while all
downstream influence retains cause/support; the RTC kernel represents every
enabled conditioned signal stage or is marked unavailable; outer, inner, full,
mini, diagnostic, simulated, and processed RTC products receive distinct stage
identities; and exact normalization, state, edge, phase/time/support, and alias
semantics are approved and serialized. Until those decisions, upstream closure,
bounded repair, focused exact-successor evidence, and independent re-audit,
retain `existing_use_only` and fail closed on every new calibration, response,
kernel, covariance, eligibility, coordinate, or production claim.

## Audit outcome

| Axis | Proposal |
|---|---|
| Contract | `proposed` |
| Implementation | `nonconformant` |
| Validation | `in_progress` |
| Production | `existing_use_only` |
| Verdict | `amend` |

The frozen claimed operator is

```text
y = D F_K ... F_1 B_omega C_g A x + exact conditioned affine offsets
```

with matched response/kernel, causal support, typed validity, covariance
availability, eligibility, and four-stage product provenance.

The current inner signal path is

```text
y_impl = P_X N_post D_0 S N_det Q_HP N_cfg W N_pre R_hat C_impl A x
         + implemented offsets
```

while flags evolve separately. The kernel follows the filter/stride subset but
omits donor replacement and the enabled flag-conditioned alt-az projection
`P_X`. Outer RTC stops earlier, before inner crop/stride, post-filter masks, and
alt-az projection, so a generic `RTC` product label is not an equivalence claim.

## Findings that require disposition

| ID | Class / severity | Decision-relevant result | Falsifiable closure |
|---|---|---|---|
| `SCI-RTC-001-F001` | `implementation_defect` / P0 | Extinction omits sample airmass after deriving zenith-equivalent opacity; q0 suppresses low-opacity correction; factor/domain identity is incomplete. | Exact CAL factor table and q0/varying-airmass/invalid-domain fixtures at an accepted re-audited successor. |
| `SCI-RTC-001-F002` | `implementation_defect` / P1 | Calibrated donor replacement applies only the responsivity ratio, omitting the target/donor calibration-factor ratio and donor-link provenance. | Unequal-factor/responsivity, invalid-target, no-donor, endpoint, and repeat fixtures match RTC-07--09. |
| `SCI-RTC-001-F003` | `implementation_defect` / P0 | Flags do not cover donor/FIR/IIR causal influence or full decimation support; kernel omits donor mixing. | Impulse, flag, non-finite, synthesis, and donor fixtures prove exact causes, support, kernel/local response, and eligibility. |
| `SCI-RTC-001-F004` | `implementation_defect` / P0 | Current science and Beammap apply alt-az projection to signal but not kernel. This is the bounded Tier-A response reopen trigger. | Matched conditioned signal/kernel projection or explicit response unavailability, plus source-crossing/template fixtures. |
| `SCI-RTC-001-F005` | `implementation_defect` / P1 | Invalid/unavailable source geometry becomes outside or unprotected; mask validity is ignored. | Invalid coordinate/frame/shape/radius/detector cases fail closed with valid controls. |
| `SCI-RTC-001-F006` | `implementation_defect` / P1 | FIR leaves edge cells unchanged; current science enables FIR while omitting the RTC edge guard. | First/last/short-scan and missing-context fixtures enforce one approved response and persist support. |
| `SCI-RTC-001-F007` | `contract_gap` / P1 | Four-stage provenance cannot reconstruct coefficients/state, factors, donors, masks, support, response, or distinct output bundles. | Replay the exact operator solely from serialized state; product round trips/digests match. |
| `SCI-RTC-001-F008` | `dependency_gap` / P0 | Complete ALIGN mapping/origin/validity/synthesis/scan/timing/support is unavailable. | Accepted re-audited ALIGN successor and RTC admission/influence fixtures. |
| `SCI-RTC-001-F009` | `dependency_gap` / P0 | Complete CAL factor/opacity/binding/responsivity/validity/uncertainty successor is unavailable. | Accepted integrated CAL successor and RTC boundary re-audit at one SHA. |
| `SCI-RTC-001-F010` | `dependency_gap` / P0 | Accepted AST coordinate/frame/validity/detector binding is unavailable and waits behind ALIGN. | Accepted dependencies and permutation/invalid-coordinate/frame/topology/kernel-placement fixtures. |
| `SCI-RTC-001-F011` | `evidence_gap` / P1 | No exact-governing compiled RTC scientific or sequential/OpenMP suite exists in this worktree. | Every applicable preregistered case passes at the selected repair SHA with no required-data skip, then independent re-audit. |
| `SCI-RTC-001-F012` | `scientific_policy_decision` / P1 | Eligibility/influence, complete response, stage identity, and filter/multirate semantics require owner authority. | Approve or supersede D001--D004 before repair. |

## Owner decisions

1. `SCI-RTC-001-D001` — replacement/synthesis eligibility and influence
   - Recommended: zero direct independent eligibility; preserve every influence
     cause/support and declared local response/covariance.
2. `SCI-RTC-001-D002` — complete RTC kernel/response
   - Recommended: include every enabled conditioned linear/local-Jacobian signal
     stage, including donor mixing and alt-az projection, or mark response unavailable.
3. `SCI-RTC-001-D003` — output-stage identities
   - Recommended: distinct immutable outer, inner, full, mini, diagnostic,
     simulated, and processed identities with explicit links.
4. `SCI-RTC-001-D004` — filter and multirate authority
   - Recommended: approve exact coefficients/normalization, IIR/notch state,
     edge rule, phase, representative time/support, and alias bound; persist full precision.

No owner choice was needed to complete the audit itself. These choices are
required only before repair scope can be authorized.

## Dependency and gate sequence

1. Coordinator/owner accepts or supersedes D001--D004 and preserves
   `existing_use_only`; no repair is authorized by this branch.
2. Complete and independently re-audit the approved ALIGN successor.
3. Complete and independently re-audit the approved CAL successor; AST remains
   behind ALIGN and must then close its coordinate/detector boundary.
4. Select a separate exact repair base and repair only the RTC interface and
   response contradictions. Do not combine mature numerical redesign.
5. At that exact successor SHA, run the focused deterministic protocol listed
   in local evidence: calibration, donor, impulse/constant/ramp/sinusoid/notch,
   non-finite/flag/synthesis, edge/short-scan, odd/even downsample, invalid
   coordinate, matched signal/kernel, product/provenance, simulation, and
   sequential/OpenMP cases. No required-data skip; zero unexpected errors.
6. Independently re-audit the successor. Only then may PTC, VAL, and BEAM
   disposition the three outgoing handoffs. Production status changes only by
   a later explicit owner decision.

No Unity, local Citlali reduction, external request, broad suite, or costly
study is requested. Any broad/costly evidence requires the separate
`FRAMEWORK-NUM-001` launch gate and owner authorization.

## Restrictions during review

- Continue only uses already authorized by existing package records; this audit
  adds no compatibility scope.
- Fail closed on new absolute calibration/photometry, transfer/covariance,
  kernel/beam response, PTC weight, VAL eligibility/exposure, or coordinate-aware
  mask/kernel claims.
- Do not equate finite/unflagged with original-valid, independent, or complete-support.
- Do not equate outer, inner, full, mini, diagnostic, simulated, or processed RTC products.
- Do not repair, re-audit, integrate, launch PTC/VAL/BEAM, contact Unity or
  external systems, change production, or treat a future test pass as authorization.

## Exact review artifacts

| Artifact | SHA-256 |
|---|---|
| `doc/audits/packages/SCI-RTC-001_INDEPENDENT_CORE.tex` | `d6cf49d1a5e17754c55cc4f2c8f4b4f5e276755f247496df888581d890be80b7` |
| `doc/audits/packages/SCI-RTC-001_SCIENTIFIC_CONTRACT_AUDIT.tex` | `169a0b6e013e727d5d04c25da5234e29def4ef79bc6f25a3f343bc675000301a` |
| `doc/audits/evidence/SCI-RTC-001_LOCAL_EVIDENCE_2026-08-08.yaml` | `1cb733544d6b9d6decb1794279d2e81249b00375071c01ee71f514135dcf6394` |
| `doc/audits/proposals/SCI-RTC-001_LEDGER_PROPOSAL_2026-08-08.yaml` | `3783fa2c61a9fd280ea3bc6ed2b4514c4a16418daebfab6474493b54f5616e88` |
| `doc/audits/handoffs/SCI-PTC-001/SCI-PTC-001-XAUD-005.yaml` | `89ce799a6526f87025bc1d57cc0bf8a0234b149374920027beaa4cb18b94e289` |
| `doc/audits/handoffs/SCI-VAL-001/SCI-VAL-001-XAUD-007.yaml` | `2c0a47c9f7f4eed4820db133304108f8f52d9702c22762b19e09cb851cce14b0` |
| `doc/audits/handoffs/SCI-BEAM-001/SCI-BEAM-001-XAUD-003.yaml` | `24cd271c5b27d9885c01002e063571a53aa7fd4517d34509ef8d2194e259e1a1` |

The final documentation commit identity is intentionally returned from Git
outside these bytes because a commit cannot embed its own hash. The frozen core
commit is `3620434eb988662210b2466ee357ffc8f891aa58`, parent the application SHA,
tree `3cafe771e15e0c6159e323f357ce3a5c76b01efa`, timestamp
`2026-08-08T17:50:49Z`.

After coordinator/owner review, the owner-controlled push command is:

```bash
git -C /Users/gwilson/.codex/worktrees/1697/citlali-refactor push --set-upstream origin codex/audit-sci-rtc-001
```

It was not executed by the auditor.
