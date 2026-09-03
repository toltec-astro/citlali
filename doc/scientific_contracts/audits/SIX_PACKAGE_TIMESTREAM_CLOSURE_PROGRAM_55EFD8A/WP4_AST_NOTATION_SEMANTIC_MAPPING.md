# WP-4 AST Notation Semantic Mapping

Prepared: `2026-08-24`

Status: `WP4-OWNER-D002` approved with explicit compact-crosswalk constraint;
clean-room re-audit pending

Scope: notation-only resolution of `F-010` for frozen SCI-AST v0.1/r0.3.

This is a semantic crosswalk, not a runtime schema. It creates no object,
field requirement, serialization format, sidecar, payload, provenance record,
coordinate role, geometry quantity, or MAP authority. Frozen AST bytes and
scientific behavior remain unchanged.

## Owner Decision

Owner response:

> I approve D002. Keep it a compact semantic crosswalk for frozen AST
> notation; do not turn it into another runtime schema.

Disposition: **approved with explicit scope constraint**.

## Generic Coordinate Occurrence

The undeclared generic sample index \(i\) denotes one exact typed coordinate
occurrence \(\iota\):

\[
\iota=
\begin{cases}
(A,d,s,\mathrm{role}),&\text{ALIGN-grid coordinate},\\
(\mathrm{RTC},d,n,\mathrm{role}),&\text{RTC-grid coordinate}.
\end{cases}
\]

| Frozen form | Canonical role |
| --- | --- |
| \(\mathcal B_i^{\rm AST}\) | \(\mathcal B_\iota^{\rm AST}\), the base pre-MAP facts for the exact typed coordinate occurrence |
| \(\Pi_i^{\rm AST}\) | Exact AST direction, tangent, or pixel parent applicable to \(\iota\) |
| \(\Pi_i^{\rm RTC}\) | \(\Pi^{\rm RTC}_{dn}\), only for an RTC-grid-parented \(\iota\) |
| \(G_{pi}\) | \(G_{p\iota}\): MAP pixel \(p\) and exact AST occurrence \(\iota\); notation only, with no projection authorization |

## Realized Detector Displacement

Inside the already-admitted small-angle linear representation and its
preregistered adequacy domain,

\[
\boxed{
\mathsf B_{ds}\boldsymbol f_d
\equiv
\boldsymbol\xi_{ds}
=\mathcal G_\gamma
(\boldsymbol g_d;t_s,E_s,\mathrm{state})
}
\]

and the composed production expression is

\[
\boldsymbol\eta_{ds}^{\rm prod}
\simeq
\boldsymbol c_s+\boldsymbol\xi_{ds}.
\]

The representation-specific focal-plane vector \(\boldsymbol f_d\) is not
identified with the complete selected geometry datum \(\boldsymbol g_d\).
The two frozen descriptions meet only at the same realized detector
displacement \(\boldsymbol\xi_{ds}\); they are not independent physical
quantities.

## Atomic-Record Symbols

| Frozen symbol | Canonical role |
| --- | --- |
| \(\mathcal A_{\rm direction}\), \(\mathcal A_{\rm tangent}\), \(\mathcal A_{\rm pixel}\) | Role-factored atomic AST coordinate records |
| \(\nu_{\rm AST}^{\rm role}\) | Role-specific coordinate validity and cause |
| \(\mathcal U_{\rm role}\) | Role-specific uncertainty/Jacobian availability |
| \(\Pi_{\rm role}\) | Exact layered scientific parent for the coordinate role |
| \(\mathcal P\) | Role-local requested/effective/resolved/realized provenance reference |
| \(\mathcal M_{\rm AST}\) | Materialization service for an exact MAP-owned request, not a MAP policy owner |

## Preserved Boundaries

- No AST coordinate, geometry, field rotation, WCS, response, or availability
  changes.
- No new small-angle adequacy claim or threshold.
- No MAP projection, deposition, or \(G_{p\iota}\) authorization.
- No runtime representation or duplicated geometry state.
- No AST successor required.

Final closure of `F-010` belongs to WP-7 clean-room re-audit.
