# VAL evaluation and PTC named-use constraints

## Source cover

V02/V03 are byte-identical to the unchanged VAL Core at its owner-promoted
r0.3 candidate 3ad018e97e134a0b0324d3fa2674ef96d5a680d4 under V05.
V01 is the continuing Registry; its unsupported names and change rule control.
V04 is the independently owner-approved PTC common named-use fragment, exact
digest c1fc8370007b65307769fb966c8523251695924aaff84f3e5b4c89b6d3380b8c,
explicitly bound by V01. Only its scientific clauses below are admitted.
Its common fragment is not an independently evaluable profile.

A new complete coefficient-use proposition needs its own exact PTC policy
and immutable binding; V01 explicitly marks the generic coefficient-QC @1
identity unsupported. VAL supplies mechanics and never invents a restriction.
Its generic independent-exposure example is not imposed on other PTC uses.
The quoted structural-binding paragraphs, knowledge algebra, four axes and
role rules are reused without changing Core. Historical adjacent-source
snapshot tables and audit findings are excluded. All new predicates and their
relevance are for the PTC scientific author to propose and the owner to decide.

## Occurrence and knowledge

Source: V02; whole-source SHA-256 `b2101d551100fce6afb79153c3bd0e3419c945f874fa6663a31ee4d693516f44`.

<!-- EXCERPT VAL_RULES-Occurrence and knowledge -->
```tex
\subsection{The object of a decision}

An \emph{occurrence} is an exact sample--detector occurrence, not merely a
matrix cell. Its identity tuple is
\[
 \boldsymbol\iota=(o,s,t,d_{\rm occ},d_{\rm uid},a,n,q,k,\pi,\ell),
\]
where the components identify observation, scan or coherent segment,
sample/time occurrence, detector occurrence, stable detector UID, array,
network/group, stream or product role, stage, parent, and lifecycle. The tuple
may be represented in any lossless form. Dense row, column, detector, map, or
file order is not a substitute for stable external identity. Primary detector
timestreams have samples on rows and detectors on columns; cross-product
detector association uses the declared UID/occurrence relation. Actual
timestamps and gaps, rather than nominal cadence, define time support.

Let $\mathsf F_k$ denote one immutable, versioned producer-fact set about one
exact occurrence or detector at lifecycle generation $k$. Facts are typed and
retain producer, scope, stage, parent, asserted state, source binding, and
provenance. A producer-local support or Boolean composite is itself an opaque
producer fact with declared owner, inputs, truth-domain rule, missing-state
rule, scope, use, and version. VAL consumes it; VAL does not reconstruct it
from raw causes.

\subsection{Knowledge is not Boolean silence}

For a fact applicable to its declared domain, use
\[
 \mathbf K=\{\mathbf T,\mathbf F,\mathbf U,\mathbf C\}.
\]
$\mathbf T$ is an authoritative positive assertion; $\mathbf F$ is an
authoritative explicit negative assertion; $\mathbf U$ is missing or unknown;
and $\mathbf C$ is contradictory, ambiguous, or outside the asserted domain.
Fact applicability is recorded separately as applicable, inapplicable, or
unknown. For a cause family, $\mathbf F$ is admissible only when the owning
producer explicitly asserts absence under a declared complete family.
Silence is $\mathbf U$, never $\mathbf F$. These symbols state scientific
meaning, not an enum or bit layout.

A policy predicate uses the same four knowledge outcomes after authority and
domain checks. A predicate can be false because authoritative facts violate a
declared restriction; an unavailable fact cannot be coerced to false. A
structural contradiction prevents the question from being established. A
non-gating contradiction remains in the reasons and is unresolved by the
disposition algebra in Section~\ref{sec:equations}.

```
<!-- END EXCERPT VAL_RULES-Occurrence and knowledge -->

## Profile envelope and four axes

Source: V02; whole-source SHA-256 `b2101d551100fce6afb79153c3bd0e3419c945f874fa6663a31ee4d693516f44`.

<!-- EXCERPT VAL_RULES-Profile envelope and four axes -->
```tex
\subsection{Owner-bound profile registry}

Let $\mathfrak R$ be the immutable profile registry. A registry record is
\[
 \rho=(I_\rho,v_\rho,U_\rho,O_\rho,S_\rho,h_\rho,
        \mathcal D_\rho,R_\rho,X_\rho,Y_\rho,J_\rho,M_\rho),
\]
where $I_\rho,v_\rho$ are registry identity and version; $U_\rho$ is the
named use; $O_\rho$ is the actual scientific owner; $S_\rho$ and $h_\rho$
are authoritative source identity and exact version or digest;
$\mathcal D_\rho$ is the applicability domain and object type; $R_\rho$ is
the restriction set; $X_\rho$ states exception permissions and
non-exceptionable invariants; $Y_\rho$ assigns every response or uncertainty
availability fact its owner-supplied role in the closed set
\[
 \mathcal Y=\{\texttt{structural\_gate},
 \texttt{required\_permission},
 \texttt{decisive\_exclusion},
 \texttt{advisory}\};
\]
$J_\rho$ is the compatibility/supersession
rule; and $M_\rho$ is missing/conflict behavior. Every scientific element is
supplied or approved by $O_\rho$. The registry records and resolves the
binding. It does not make VAL the policy author.

The policy supplied for one evaluation is
\[
 \mathsf P_\rho=(\rho,A_\rho,G_\rho,R_\rho,X_\rho,Y_\rho,M_\rho,
                  \Gamma_\rho,L_\rho).
\]
$A_\rho$ is applicability; $G_\rho$ is the structural gate;
$\Gamma_\rho$ is any aggregation specification; and $L_\rho$ binds requested,
effective, observation-resolved, learned/resolved, and realized lineage. A
reserved name without this complete binding is not a policy and cannot be
evaluated as eligible.

When the proposition is an aggregate, its profile $\rho_\Gamma$ is a
separate registry record, not the atomic profile reused at a larger object
scope. It binds its own owner, version, domain, operator, threshold, and
propagation authority together with the exact compatible atomic source
profile identity and version.

\subsection{Four independent decision axes}

The domains are
\begin{align*}
 \mathcal R&=\{\mathrm{requested},\mathrm{not\_requested}\},\\
 \mathcal A&=\{\mathrm{applicable},\mathrm{inapplicable},
                    \mathrm{applicability\_unknown}\},\\
 \mathcal E&=\{\mathrm{eligible},\mathrm{ineligible},
                    \mathrm{decision\_unavailable}\},\\
 \mathcal Z&=\{\mathrm{realized},\mathrm{incomplete},\mathrm{failed},
                    \mathrm{not\_produced}\}.
\end{align*}
A record carries $(R,A,E,Z)$ without aliasing axes. Eligibility is a partial
proposition: if no eligibility question is evaluated because no request was
made or the profile is known inapplicable, the $E$ slot is uninstantiated,
written $\varnothing_E$. This is not a fourth disposition and not
\texttt{decision\_unavailable}. A failed or unproduced artifact likewise does
not manufacture a disposition.

```
<!-- END EXCERPT VAL_RULES-Profile envelope and four axes -->

## Registry structural binding

Source: V03; whole-source SHA-256 `fc2b07567bad39314776fda9453b010b482ba67525d5b14dc093df4cc459046e`.

<!-- EXCERPT VAL_RULES-Registry structural binding -->
```tex
Write $b(\rho,\mathfrak R)$ for registry binding integrity,
$g(\rho,\mathsf P_\rho,\mathsf F_k)$ for the structural interchange gate,
and $a(\rho,\mathsf P_\rho,\mathsf F_k)\in\mathcal A$ for applicability.
Binding integrity is true only when registry key/version, actual owner,
authoritative source/version or digest, use, domain/object type, restrictions,
exception permissions, compatibility, supersession, and missing behavior are
complete and mutually consistent. A reserved name is insufficient.

```
<!-- END EXCERPT VAL_RULES-Registry structural binding -->

## General structural gate

Source: V03; whole-source SHA-256 `fc2b07567bad39314776fda9453b010b482ba67525d5b14dc093df4cc459046e`.

<!-- EXCERPT VAL_RULES-General structural gate -->
```tex
The general gate succeeds only when $b=\mathbf T$ and identity, parent,
immutable fact set, producer authority, source bindings, policy lineage,
applicability authority, scopes, referenced supports, and influence records
are present and consistent. Missing or conflicting structural material makes
applicability unknown and the decision unavailable; it is not a use-specific
exclusion. Once this structural domain is established, a conflict in an
unrelated non-gating predicate is unresolved evidence rather than a structural
failure.

```
<!-- END EXCERPT VAL_RULES-General structural gate -->

## Restrictions and disposition

Source: V03; whole-source SHA-256 `fc2b07567bad39314776fda9453b010b482ba67525d5b14dc093df4cc459046e`.

<!-- EXCERPT VAL_RULES-Restrictions and disposition -->
```tex
\subsection{Restriction values and exceptions}

For each restriction $r_j\in R_\rho$, the owner-bound profile identifies an
applicability predicate, a base permission predicate, and its exception class.
After authority checking, contradiction $\mathbf C$ in a non-gating predicate
is retained and normalized to unresolved $\mathbf U$ for composition. An
inapplicable restriction contributes neutral permission $\mathbf T$. The base
permission of an applicable restriction is
\[
 p_j=\begin{cases}
 \mathbf T, & \text{its permission predicate is authoritatively true},\\
 \mathbf F, & \text{its permission predicate is authoritatively false},\\
 \mathbf U, & \text{restriction applicability or permission is unknown or
                         conflicting}.
 \end{cases}
\]
Only a resolved, permitted same-profile exception can transform this value:
\[
 q_j=\begin{cases}
 \mathbf T^{X}, & \text{that exception authoritatively applies to this exact
                         exceptionable restriction},\\
 p_j, & \text{otherwise, including unknown or conflicting exception
                         applicability}.
 \end{cases}
\]
$\mathbf T^X$ composes as $\mathbf T$ but requires the registry/profile
identity, exception, underlying restriction, and causes to remain in
$\mathsf V_k$. An unknown or conflicting exception cannot neutralize an
underlying $p_j=\mathbf F$; the disposition remains ineligible and the
exception conflict is preserved. No exception can be supplied by VAL,
another use, or a registry record that prohibits it. A changed exception
resolution changes immutable resolved-profile lineage and decision identity.

The normalized permission conjunction is
\begin{center}
\begin{tabular}{c|ccc}
$\wedge_K$ & $\mathbf T$ & $\mathbf U$ & $\mathbf F$ \\
\hline
$\mathbf T$ & $\mathbf T$ & $\mathbf U$ & $\mathbf F$ \\
$\mathbf U$ & $\mathbf U$ & $\mathbf U$ & $\mathbf F$ \\
$\mathbf F$ & $\mathbf F$ & $\mathbf F$ & $\mathbf F$ \\
\end{tabular}
\end{center}
Known exclusion dominates an unrelated unknown or conflicting non-gating
restriction; required unresolved information prevents eligibility when no
known exclusion exists. The table is commutative and associative, and every
conflict remains in the reason record.

\subsection{Disposition function}

For a requested record that can be realized, define
\[
 Q(\rho,\mathsf P_\rho,\mathsf F_k)
   =\bigwedge_{r_j\in R_\rho}q_j.
\]
The disposition is
\[
 D(\rho,\mathsf P_\rho,\mathsf F_k)=
 \begin{cases}
 \mathrm{decision\_unavailable},
   &b\ne\mathbf T\text{ or }g\ne\mathbf T
      \text{ or }a=\mathrm{applicability\_unknown},\\
 \varnothing_E,
   &b=\mathbf T,\ g=\mathbf T,\ a=\mathrm{inapplicable},\\
 \mathrm{ineligible},
   &b=\mathbf T,\ g=\mathbf T,\ a=\mathrm{applicable},\ Q=\mathbf F,\\
 \mathrm{decision\_unavailable},
   &b=\mathbf T,\ g=\mathbf T,\ a=\mathrm{applicable},\ Q=\mathbf U,\\
 \mathrm{eligible},
   &b=\mathbf T,\ g=\mathbf T,\ a=\mathrm{applicable},\ Q=\mathbf T.
 \end{cases}
\]
All restrictions are evaluated sufficiently to preserve every false,
unresolved, conflict, influence, exception, and policy-invalid reason; the
precedence does not authorize evidence-erasing short circuit.

\begin{center}
\begin{longtable}{@{}p{0.12\linewidth}p{0.17\linewidth}p{0.21\linewidth}p{0.18\linewidth}p{0.19\linewidth}@{}}
\toprule
Request & Applicability & Profile knowledge & Eligibility & Realization \\
\midrule
\endhead
not requested & not assessed or separately recorded & none evaluated &
$\varnothing_E$ & not produced \\
requested & inapplicable & domain known outside profile & $\varnothing_E$ &
realized \\
requested & applicability unknown & registry, gate, or applicability
unresolved & decision unavailable & realized \\
requested & applicable & at least one decisive false & ineligible & realized \\
requested & applicable & no false; at least one required unresolved &
decision unavailable & realized \\
requested & applicable & all required true & eligible & realized \\
requested & any & authoritative artifact not completed & no authoritative
eligibility assertion & incomplete, failed, or not produced \\
\bottomrule
\end{longtable}
\end{center}
A realized decision-unavailable record is a successful scientific account of
why no disposition can be made. It is not an execution failure.

```
<!-- END EXCERPT VAL_RULES-Restrictions and disposition -->

## Response uncertainty and payload ordering

Source: V03; whole-source SHA-256 `fc2b07567bad39314776fda9453b010b482ba67525d5b14dc093df4cc459046e`.

<!-- EXCERPT VAL_RULES-Response uncertainty and payload ordering -->
```tex
\subsection{Response, uncertainty, and payload ordering}

Response and uncertainty availability are typed facts, distinct from their
numerical values and from zero. The named-use owner assigns each fact exactly
one role from the closed set below; VAL does not choose or reinterpret it.

\begin{center}
\begin{tabular}{p{0.23\linewidth}p{0.64\linewidth}}
\toprule
Owner-supplied role & Deterministic evaluation consequence \\
\midrule
\texttt{structural\_gate} & Without authoritative satisfaction, the
proposition is not established: applicability is unknown and the decision is
unavailable. \\
\texttt{required\_permission} & Authoritative availability contributes
permission; authoritative unavailability excludes; unknown or conflict is
unresolved. \\
\texttt{decisive\_exclusion} & Authoritative failure or unavailability
excludes; authoritative nonfailure is neutral permission; unknown or conflict
is unresolved. \\
\texttt{advisory} & The exact state is preserved but contributes neutral
permission and cannot determine eligibility. \\
\bottomrule
\end{tabular}
\end{center}

The restriction conjunction then maps exclusion to \texttt{ineligible} and a
required unresolved state with no exclusion to
\texttt{decision\_unavailable}. VAL never substitutes identity response or
zero uncertainty.

Payload processing is ordered:
\begin{enumerate}
  \item establish identity, authority, parentage, applicability, and validity
        without numerically consuming an already invalid payload;
  \item exclude a producer- or policy-declared invalid occurrence before
        arithmetic; and
  \item only for an admitted required payload, evaluate its numeric domain.
        A non-finite value then causes failure or scientific unavailability at
        the owner-declared scope.
\end{enumerate}
Finiteness never supplies missing admission, response, uncertainty, parent
validity, or registry authority.
```
<!-- END EXCERPT VAL_RULES-Response uncertainty and payload ordering -->

## Common scientific semantics and limits

Source: V04; whole-source SHA-256 `c1fc8370007b65307769fb966c8523251695924aaff84f3e5b4c89b6d3380b8c`.

<!-- EXCERPT VAL_RULES-Common scientific semantics and limits -->
## Common semantics

For every registered PTC named use `U`:

1. `U` is a complete and distinct scientific proposition. Permission under
   another use `V` conveys no permission under `U`:

   \[
   E_V \not\Rightarrow E_U \qquad (U\ne V).
   \]

2. PTC preserves upstream facts, causes, and classifications and does not
   upgrade them. In particular, CAL `engineering-only` remains a producer
   fact; its admission consequence is declared by the exact named use rather
   than imposed globally by CAL, PTC, or VAL.
3. Only facts that `U` explicitly declares scientifically relevant to its
   decision may affect that decision. The mere existence, availability, or
   unknown state of other metadata has no admission consequence.
4. VAL evaluates the complete owner-supplied proposition under its frozen
   T/F/U/C and four-axis semantics. VAL supplies no missing predicate,
   threshold, exception, or permission.

## Explicit noncontents

This common fragment contains no direct-origin exclusion, fit population,
loading-estimator input, group or rank guard, output-retention rule, response
or uncertainty requirement, exception, or missing/conflict disposition. Those
remain in the complete named-use records where scientifically applicable.

It creates no runtime common-policy object, inheritance mechanism, sidecar,
payload, serialization requirement, duplicated provenance, or separate
engineering route.

## Version and compatibility

Every profile using this fragment binds this exact identity and byte digest.
Any content change creates a new fragment version and new dependent profile
identity or version. No change acts retroactively on an earlier evaluation.
<!-- END EXCERPT VAL_RULES-Common scientific semantics and limits -->

## Registry binding rule

Source: V01; whole-source SHA-256 `95b2ddeca2039aa2b3614e3dfede530cb7e024e6db33ca22baed867f19d2709d`.

<!-- EXCERPT VAL_RULES-Registry binding rule -->
## Registry Rule

The registry is a binding and replay mechanism. It does not write, approve,
or inherit scientific-use policy. A usable profile record must bind:

1. immutable registry key and profile version;
2. exact named scientific use;
3. actual scientific owner;
4. authoritative source and exact version or digest;
5. applicability domain and object type;
6. required restrictions and missing-fact behavior;
7. permitted exceptions, including any non-exceptionable invariants;
8. response and uncertainty roles, each selected from `structural_gate`,
   `required_permission`, `decisive_exclusion`, or `advisory`;
9. aggregation and propagation compatibility; and
10. supersession and incompatibility behavior.

Missing or conflicting binding information makes the profile unavailable for
evaluation. A reserved name is not a usable policy.

<!-- END EXCERPT VAL_RULES-Registry binding rule -->

## Unsupported names and immutable changes

Source: V01; whole-source SHA-256 `95b2ddeca2039aa2b3614e3dfede530cb7e024e6db33ca22baed867f19d2709d`.

<!-- EXCERPT VAL_RULES-Unsupported names and immutable changes -->
## Package-Qualified Names Unsupported, Deferred, Or Not Bound Here

The following names prevent broad labels from hiding scientifically different
questions. An `unsupported` disposition means the owner found no present
scientific proposition and VAL must not fabricate one. A `deferred` or
`unbound` name remains unavailable until its owner supplies a complete
immutable record satisfying the registry rule.

| Profile identity or template | Expected scientific owner | Status in this registry |
| --- | --- | --- |
| `SCI-PTC:coefficient_qc_population@1` | SCI-PTC | Explicitly unsupported under `WP5-OWNER-D008`. Informational diagnostics remain PTC-owned; a separately named complete profile is required only when a diagnostic gains decision authority |
| `SCI-PTC:empirical_or_simulation_population@1` | SCI-PTC | Explicitly unsupported under `WP5-OWNER-D010`. PTC v0.1 makes no scientific inference from an ensemble of alternative realizations |
| `<PACKAGE>:diagnostic_display` | Owning package or diagnostic consumer | Namespace template only; display permission conveys no stronger scientific use |

The earlier broad label `analysis_or_gridding_contribution` is not a v0.1
registry key. It is replaced by the narrower `map_upstream_admission`; no
automatic alias is permitted because the old wording could be mistaken for
actual numerical contribution.

## Registry Change Rule

A new source digest, profile version, domain, restriction, exception rule, or
compatibility declaration creates a new immutable registry record. It cannot
rewrite an earlier evaluated decision. A renamed profile is a new identity
unless its scientific owner publishes an explicit semantics-preserving alias;
VAL does not infer aliasing.
<!-- END EXCERPT VAL_RULES-Unsupported names and immutable changes -->
