# WP-4 RTC Notation Semantic Mapping

Prepared: `2026-08-24`

Status: `WP4-OWNER-D001` approved; clean-room re-audit pending

Scope: notation-only resolution of `F-008` for frozen SCI-RTC v0.1/r0.12.
This mapping changes no frozen RTC byte, mathematical object, operator order,
plan lifecycle, response, replay rule, numerical behavior, or scientific
availability.

## Owner Decision

Question:

> Should frozen RTC r0.12 remain byte-unchanged while an authoritative
> semantic mapping assigns distinct symbols to the temporal-filter-stage count
> and final accepted-plan index?

Owner response:

> approved

Disposition: **approved**.

## Canonical Composed Notation

Use \(N_{\rm filt}\) for the number of ordered temporal-filter stages:

\[
F_1,\ldots,F_{N_{\rm filt}},
\qquad
L^x_\Omega
=D_{M,0}F_{N_{\rm filt}}\cdots F_1B^x_{\omega,\mathcal R}.
\]

Use \(k_{\rm fin}\) for the final accepted-plan index:

\[
k_{\rm fin}=k_{A+1}\le A,
\qquad
\mathbf u^{\rm final}
=\mathcal A_{\Pi_{k_{\rm fin}}}(\mathbf u^{(0)}).
\]

Reserve existing \(K^{a\leftarrow b}_\Omega\) and \(\mathcal K\) forms for
RTC response kernels or response-status objects. They are not counts.

## Exact Old-To-New Map

| Frozen r0.12 form | Canonical composed form | Meaning |
| --- | --- | --- |
| \(F_1,\ldots,F_K\) | \(F_1,\ldots,F_{N_{\rm filt}}\) | Ordered temporal-filter stages |
| \(F_K\cdots F_1\) | \(F_{N_{\rm filt}}\cdots F_1\) | Ordered temporal-filter composition |
| \(k\in\{0,\ldots,K\}\) | \(k\in\{0,\ldots,k_{\rm fin}\}\) | Accepted-plan index domain |
| \(\Pi_K\) | \(\Pi_{k_{\rm fin}}\) | Final accepted plan |
| \(K=k_{A+1}\) | \(k_{\rm fin}=k_{A+1}\) | Final accepted-plan index identity |
| \(K^{a\leftarrow b}_\Omega\), \(K_{dn,\cdot\cdot}\), or \(\mathcal K\) response forms | unchanged | Response kernel, response dependence, or typed response status |

The mapping is role-sensitive. It does not authorize substitution between
\(N_{\rm filt}\), \(k_{\rm fin}\), and any response-kernel/status object.

## Source Application

The mapping applies when composing or replaying frozen RTC clauses including
`SCI-RTC-EQ-004--006`, `SCI-RTC-EQ-020`, `SCI-RTC-EQ-022`, and
`SCI-RTC-EQ-029`, together with the accepted-plan notation row in
`src/common/notation.tex`. Citations to the frozen clauses remain citations to
SCI-RTC v0.1/r0.12; this artifact supplies only the disambiguating symbol map.

Future RTC successors may incorporate these names directly, but this mapping
does not require or authorize a successor.

## Claim Boundary

This artifact establishes owner-approved notation parity only. It does not
establish implementation conformity, numerical validation, response
availability, achieved performance, production readiness, MAP availability,
or clean-room closure of `F-008`.
