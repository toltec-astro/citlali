# SCI-MAP v0.1 PTC-Handoff Stage B Manager Review r0.1

Date: `2026-08-26`

Reviewed author return:
`PTC_HANDOFF_STAGE_B_AUTHOR_RETURN_R0.1_2026-08-26.md`

Reviewed return SHA-256:
`3edaf5dbd9118bf1ae54066864c362f19ea005efbd361f1654127e5ca002bd76`

Status: return preserved as draft evidence; not accepted for integration;
bounded r0.2 author correction required

This is a manager review of scientific-contract content. It is not scientific
authority, implementation evidence, conformity assessment, validation,
performance evidence, freeze, or readiness approval.

## Overall Judgment

The implementation-blind author correctly landed the mandatory PTC-to-MAP
route, absence of a CAL fallback, MAP rather than VAL policy ownership,
ALIGN/AST versus MAP coordinate/projection ownership, honest response and
covariance states, immutable later derivatives, and the separation between
contribution membership and broader product/claim failure.

The return cannot be integrated as written. It introduces four substantive
rules that were not in the approved packet, and its resulting profile machinery
is substantially broader than the bounded horizontal repair. The good science
should be retained through a surgical r0.2 correction rather than discarded.

## Required Corrections

### MAP-B-MGR-F001 — Preserve x/r identity; do not rename r as response

Severity: **HIGH**

The approved owner decision concerns one physical paired x/r readout
occurrence. Independence is evaluated for the explicitly named x component, r
component, or downstream product aspect. The return repeatedly changes this
to a paired “signal/response” occurrence and treats r-origin preservation as
response-origin preservation.

That interpretation is not authorized. `r` is a readout coordinate; it is not
the MAP response/kernel product merely because both use the word “response” in
different contexts.

Required r0.2 action:

- replace every paired signal/response formulation in REQ-007, exposure
  definitions, narrative, and tests with exact paired x/r terminology;
- state that replacing x makes that x value non-independent, does not rewrite
  r origin, and does not universally invalidate the physical occurrence; and
- keep MAP response/kernel availability completely separate from x/r origin.

### MAP-B-MGR-F002 — Preserve producer-declared availability scope

Severity: **HIGH**

The return invents a universal split in which required-product PTC
unavailability aborts MAP while an unavailable occurrence is always local
noncontribution unless a new broader gate applies. The approved packet says
that PTC availability is necessary, MAP cannot rescue unavailable PTC output,
and causes do not determine downstream consequences. It does not authorize MAP
to narrow or widen the scope assigned by PTC.

Required r0.2 action:

- consume each PTC availability fact at its exact producer-declared scope;
- preserve the producer-declared cause and failure scope;
- prohibit MAP from promoting, narrowing, widening, or inferring another
  scope from the cause name;
- retain the frozen route rule that an unavailable required realized PTC
  product, disabled PTC, or invalid rank yields no ordinary MAP product; and
- define contribution nonmembership only when the source-bound availability
  fact is itself declared at contribution scope or the MAP-owned policy
  excludes an otherwise available contribution.

No new universal `A_P` versus `A_i` ontology is authorized by this revision.

### MAP-B-MGR-F003 — Keep policy identity semantic; do not legislate JSON/SHA serialization

Severity: **HIGH**

The return makes an exact JSON key order, whitespace rule, escaping rule, and
SHA-256 digest algorithm part of scientific authority. The science requires a
complete immutable profile identity, exact versioned predicate bindings,
declared scopes, and replayable provenance. It does not require this contract
to standardize a wire serialization or digest construction.

That representation choice belongs to the separately governed engineering
serialization lane. It would also overtake SCI-VAL's deliberately deferred
serialization questions.

Required r0.2 action:

- retain the semantic base profile identity
  `SCI-MAP:map_upstream_admission@v0.1` or an equivalently clear versioned
  identifier;
- require the realized identity to bind the effective plan, complete predicate
  set, source/version bindings, evaluation/failure scopes, and outcomes;
- require any semantic change to create a distinguishable versioned identity;
  and
- remove normative JSON layout, key order, escaping, byte encoding, digest
  algorithm, and `profile_instance_id` construction.

An implementation may use a digest if another engineering authority requires
one, but this scientific revision must not select it.

### MAP-B-MGR-F004 — Do not close OD-003 with a new closed-world claim registry

Severity: **HIGH**

The owner decided that response unavailability must be labeled honestly and
does not invalidate the map or prohibit later scientific analysis. The owner
also approved immutable versioned later response and corrected-map products.
The return adds a stronger rule: every response-bearing or
response-independent claim must enter an exhaustive owner-authorized registry,
and omission fails closed. That rule was neither requested nor approved and
could turn honest product labeling into a prohibition on legitimate later
analysis.

Required r0.2 action:

- keep OD-003 **OPEN**, narrowed to any minimum response obligations for
  specific claims made by the original MAP product;
- require every MAP-authored response-dependent claim to identify the response
  information and domain supporting it, or remain unsupported;
- require response-unavailable products to state that limitation clearly;
- do not require all later scientific analyses or external claims to register
  `required`/`not_required` records with MAP;
- preserve the exact versioned-derivative and immutable-parent rules; and
- preserve the distinction between unsupported MAP claims and scientific use
  that may add new evidence later.

The analogous covariance language should govern claims made by MAP products
without creating a universal registry for all downstream research. OD-004
remains open as already proposed.

### MAP-B-MGR-F005 — MAP composes facts for MAP use

Severity: **HIGH**

Several proposed profile keys say that producer eligibility or flag precedence
must “pass for the named MAP use.” That wording lets an upstream producer
author a MAP-use consequence. The approved boundary is different: producers
own their facts, causes, and any validity conclusion at their explicitly named
scope; MAP owns how those source-bound facts compose for map contribution.
VAL evaluates the MAP-authored composition.

Required r0.2 action:

- name the exact source-bound producer facts and their scopes;
- keep their meanings and causes immutable;
- make MAP's contribution rule the explicit composition of those facts plus
  MAP-owned coefficient, coordinate, projection, boundary, and companion
  predicates; and
- do not call a producer fact a MAP-use decision unless the producing authority
  explicitly owns that named use.

### MAP-B-MGR-F006 — Reduce the revision to the approved delta

Severity: **MEDIUM**

The return proposes replacements across a large fraction of the 52
requirements and 25 predictions. Some propagation is necessary for semantic
consistency, but much of the expansion is generated by the unapproved profile
serialization and claim-registry machinery.

Required r0.2 action:

- retain exact replacement wording only where a clause's premise or consequence
  changes;
- prefer concise amendments to repeated inventories;
- preserve all 52 requirement and 25 prediction identifiers;
- leave threshold, publication, mode, common-grid, and projection-class
  questions untouched; and
- return an explicit list of unchanged IDs after the correction.

## Science To Retain

The r0.2 author should preserve these successful parts of the return:

- realized PTC-transformed Stokes-I input in `mJy/beam`;
- no direct CAL-to-MAP fallback and neutral PTC retaining PTC identity;
- MAP-owned, versioned, use-specific admission with VAL as evaluator/registry;
- contribution admission separated from product- and claim-scoped failure;
- exact ALIGN/AST coordinate ownership and MAP grid/projection ownership,
  without selecting projection science;
- response and covariance states that report actual meaning, domain, and
  limitations without fabricating zeros;
- unavailable response or incomplete covariance not automatically invalidating
  the raw map;
- response/covariance-dependent MAP claims remaining unsupported when their
  required evidence is insufficient;
- later response, covariance, and corrected-map products as immutable-parent,
  versioned derivatives; and
- OD-004 and OD-008 narrowed but open, with all unrelated owner questions
  unchanged.

## Required r0.2 Return

The corrected author return must remain a revision specification rather than
editing canonical sources. It must provide:

1. corrected definitions and equations without a new availability ontology or
   serialization standard;
2. exact concise replacement wording for affected existing requirements and
   predictions;
3. OD-003, OD-004, and OD-008 as narrowed **OPEN** questions;
4. corrected x/r tests separate from response/kernel tests;
5. tests proving producer-scope preservation and MAP-owned use composition;
6. tests proving product/claim failures do not silently alter contribution
   membership;
7. a reduced crosswalk-impact list and explicit unchanged-ID list; and
8. an internal audit showing that no prohibited local MAP decision was made.

The r0.1 author return remains immutable evidence. A corrected r0.2 author may
receive this manager review only through a successor content-bound packet
approved by the scientific owner.
