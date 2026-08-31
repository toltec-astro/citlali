# WP-7 Network-Timing Authority Crosswalk

Date: 2026-08-29

Controlling decision:
[WP-7 Network-Timing Scientific-Owner Authority Correction](WP7_NETWORK_TIMING_OWNER_AUTHORITY_CORRECTION_2026-08-29.md)

Source authority: the WP-7 clean-room packet bound in
`validation/wp7_timestream_successor_authority.json`, including frozen
SCI-ALIGN v0.1/r0.3, SCI-RTC v0.1/r0.12, SCI-AST v0.1/r0.3, and SCI-PTC
v0.1/r0.5 content.

## Supersession Rules

| Package | Prior clause or symbol | Bounded successor meaning |
| --- | --- | --- |
| SCI-ALIGN | Purpose text: "one immutable, observation-scoped detector-reference sample identity" | One immutable network-scoped occurrence/time relation by default; an observation-wide common analysis relation exists only on explicit request. |
| SCI-ALIGN | `i_ref=D`, `phi_D`, `h_D`, `s`, `(o,s)`, `t_s`, `I_s` in notation and EQ-001--003 | For ordinary detector streams these quantities are network-scoped (`D_g`, `s_g`, `(o,g,s_g)`, or an equivalent exact occurrence identity). A separate analysis-grid identity is used for an explicitly requested cross-network relation. |
| SCI-ALIGN | REQ-002--004, REQ-012--015 | Preserve their clock, pairing, tolerance, failure, and distinct-time-fact semantics per network. Their singular-grid reading is superseded. Strict-half admission applies to a requested common analysis grid, not ordinary RTC ingress. |
| SCI-ALIGN | REQ-020--022 and the ALIGN-to-AST boundary profile | Observing state and pointing support bind to each exact network occurrence time. Cross-network analysis-grid time is used only by an explicitly requesting consumer. |
| SCI-ALIGN | REQ-047 and REQ-050 | AST and all consumers preserve network occurrence/time identity. Preservation of a requested common-analysis relation is additional and never replaces source identity. |
| SCI-RTC | Input index `j`, `x^A_dj`, `r^A_dj`, `G_pair`, `rho_dn`, and the statement that time is on the ALIGN-assigned grid | These are network-keyed for ordinary RTC. `G_pair` is the paired `x/r` occurrence/output axis within a network, not a cross-network common analysis grid. |
| SCI-RTC | DEF-001, DEF-046, DEF-048--052 | Preserve exact x/r pair identity, identical ordinary pair operator, coordinate-local availability, causes, support, and covariance distinctions on each network axis. "Common" in these definitions is pair-common, not cross-network. |
| SCI-RTC | REQ-001, REQ-006, REQ-008, REQ-028, REQ-041, REQ-050, REQ-085, REQ-116, REQ-133, REQ-135--136, REQ-139, REQ-141--143 | Preserve their non-timing semantics per network. `M=1` preserves exact network input occurrence and time. Sampling changes produce a new per-network output relation unless the named method is explicitly cross-network synchronous. |
| SCI-RTC | OWNER-096 and OWNER-099 | Pair-coherent action and distinct grid/support/availability/covariance axes remain authoritative per network; they do not authorize cross-network projection. |
| SCI-AST | `s`, `(o,s)`, `theta^A_ds`, current time `t_s`, and the exact ALIGN import text | Replace the singular slot with the exact network occurrence/time identity for ordinary AST. Preserve native/source timing and all existing geometry, frame, topology, support, and validity authority. |
| SCI-AST | REQ-016--021 and the ALIGN-to-AST boundary profile | Evaluate ordinary pointing association and field rotation at each network occurrence time. A common-analysis-grid AST relation is conditional on an explicit downstream request. |
| SCI-PTC | PTC-REQ-001--005 and the sample-by-detector matrix notation | A PTC matrix is group-local. Network-level groups use their network axis. Rectangular storage does not itself authorize cross-network timing projection. |
| SCI-PTC | PTC-REQ-008, 019, 023, 029, 031--032, 089--094 and EQ hierarchy | Preserve configured grouping, supports, masks, rank, and failure semantics. Network mode remains independent per network. Array mode must request an ALIGN-owned common analysis grid because its group-time estimates couple networks. |

## Preserved Requirements

All unlisted requirements, equations, predictions, owner-ledger states, source
bindings, and package limitations remain unchanged. In particular, this
correction does not authorize signal interpolation or synthesis, alter the
strict-half predicate for an explicitly requested common analysis grid, select
an RTC numerical method, choose PTC rank, create a CAL or AST numerical default,
or activate a successor route.

## Implementation Mapping

| Owner consequence | Implementation obligation |
| --- | --- |
| Network timing is primary | Ordinary RTC APIs key time and occurrence queries by network and native/network occurrence. |
| `M=1` is exact | Output identity, time, value, support, validity, and causes equal the admitted network input. |
| Per-network gaps are independent | A gap or missing occurrence in one network creates no ordinary-RTC slot or absence state in another. |
| Common analysis is explicit | Shared-slot carriers are isolated and named as common-analysis-grid relations; ordinary RTC source dependency guards exclude them. |
| Derived relation is non-destructive | A requested relation exposes both analysis-grid time and source-network occurrence/time plus validity, causes, and support. |
| Cross-network mathematics is the trigger | Array-wide PTC/PCA or an explicitly authorized cross-network RTC method requests the relation; network-level operations do not. |
