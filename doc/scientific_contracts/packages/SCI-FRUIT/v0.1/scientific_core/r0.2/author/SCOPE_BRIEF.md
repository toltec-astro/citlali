# SCI-FRUIT — Conditional Feedback Core Scope Brief

Status: draft Stage A successor r0.2; exact packet not owner-approved.
Scientific owner: Grant Wilson. Contract version: v0.1. Date: 2026-09-06.
Approved direction: SCI-FRUIT-SCIENTIFIC-CORE-SCOPE-R0.1, 2026-09-06.

## Program Adherence And Prior-Work Recovery

This brief follows the [program charter, pilot workflow and downstream
roadmap extract](PROGRAM_REFERENCE.md). Completed manager recovery is
identified in the [input manifest](AUTHOR_INPUT_MANIFEST.md); its internal
dossier is excluded from the author channel. The scope investigator reviewed
the template and recovery on 2026-09-06. Owner review of this exact opening and
packet is **pending**; no author has been launched.

Adopt scientific identity conventions, adjacent ownership, the PTC fixed-state
application distinction and the approved NOI conditioning boundary from the
[permitted scientific extracts](BOUNDARIES_AND_CONVENTIONS.md). Abstract only
the approved scientific questions from earlier scope work; defer numerical
choices and exclude implementation, audit and experimental evidence. New work
is limited to the conditional science of model feedback, response, causal
continuation and termination. It does not rederive PTC or mapmaking.

The [manifest](AUTHOR_INPUT_MANIFEST.md) lists the five proposed author inputs
and their roles. None becomes an approved input merely by appearing here.

## 1. Package Name And Scientific Purpose

SCI-FRUIT asks how a model of astronomical signal can participate in iterative
residual processing while retaining a truthful account of what the resulting
signal measures. A feedback model may protect signal from a named operation;
that does not by itself establish that the model is correct, that lost sky
modes were measured, or that the result has unit response.

The approved deliverable is a conditional scientific core. It should explain
the obligations and mathematical consequences of a *declared* feedback method.
It selects no numerical method and certifies no existing reduction.

## 2. Scientific Boundary

The operation begins with identified upstream measured-signal products,
declared scientific operators and an initialization or continuation state.
It includes the scientific relation between a target sky quantity and a
selected model; projection and removal of that model; its bypass of named
residual-processing operations; rejoining; successor state; and termination
and product selection.

It ends at an identified iteration result or selected terminal result with
the state and claim disclosures needed to interpret that result. Downstream
source inference and fitness for Pointing or OOF are separate operations.

This boundary fixes the roles that the core must describe. It does not fix
their numerical operators, observation/coadd grouping, support or learning
schedule. An executable method must supply those choices. Conditional
derivations must identify their input and output spaces and their fixed and
varying state before making an equality or response claim.

## 3. Legitimate Inputs

The following are scientific input classes for the conditional core, not
admission of a numerical parent or a required file format.

| Input class | Scientific content and known constraints | Still required for a numerical method |
| --- | --- | --- |
| Measured signal | Exact producer, quantity, unit, calibration/beam convention, observation/group, sample or pixel domain, parent generation and validity. A detector-time object is indexed by exact detector/sample occurrences; a map is indexed by its declared grid and frame. Equal shapes do not establish an identity join. | Select an available upstream route and bind its actual domains, payloads and compatibility. |
| Target and proposed feedback model | Identify the astronomical quantity the model represents and its relation to a response-affected measurement. State model units, spatial domain, frame, support and response assumptions. The model and measured map are distinct objects. | Choose the target representation, construction/selection/sign policy and any physical prior. |
| Projector and processing operations | Exact domains/codomains, unit transformations, geometry, removal and restoration support, order and validity behavior; named operations retain their upstream scientific ownership. | Supply numerical projection, rejoin, processing and response operators. |
| Initialization or continuation state | A seed initializes a lineage. Exact continuation identifies a completed iteration, its parent/model state and all information that can change subsequent required results, including any consequential selection history. | Supply the chosen recurrence's complete state and a compatible continuation realization. |
| Learning and termination declarations | Distinguish requested policy, resolved plan, fitted state and realized application. Identify what can change, the information available when it changes, the convergence proposition and any resource limit or terminal-selection rule. | Choose learning populations/cadence, numerical criteria and terminal policy. |
| Response or uncertainty information, when used | Bind the exact quantity, axes/domains, units, support, conditioning state, approximation and omitted terms. An uncertainty input used by a later iteration is dependent input to a new generation. | Supply an available response/covariance or NOI method for the particular claim; missing information is not zero. |

For the inherited ordinary PTC-to-MAP handoff, the known signal convention is
calibrated-x-derived, nonpolarimetric total-intensity-equivalent signal in
top-of-atmosphere, point-source-equivalent mJy/beam, bound to its originating
fixed nominal beam/template and calibration lineage. This does not establish
Stokes I or select that numerical route. The target/model unit is still a
method declaration. The core does not assign sky, deconvolution or covariance
meaning from a unit label. Timing is an identified sample attribute, not a
replacement for exact occurrence identity. Missing, non-finite and unavailable states must retain
their scientific causes and use-specific consequences. No universal signal
veto is inferred merely from unavailable uncertainty.

## 4. Required Outputs

The core must define the meanings of these outputs of a declared method.
These are scientific objects and claims, not the documents an author writes.

| Output role | Required scientific meaning | What this core leaves unselected |
| --- | --- | --- |
| Iteration result | The identified signal after the declared residual/model composition, with its quantity, domain, unit, support, parent and validity. It is not automatically a true-sky estimate. | A numerical MAP/JINC/filter product or storage bundle. |
| Successor feedback state | The accepted model/state used by subsequent actions, distinguished from the measured iteration result and from a diagnostic difference. | A numerical state representation, additive recurrence or mandatory increment product. |
| Continuation information | The condition under which all future consequential state is available and compatible for exact continuation. Seed, restart and relearning have distinct meanings. | Checkpoint format, retention horizon or a claim that a particular checkpoint is sufficient. |
| Termination and terminal identity | Separate completed iteration, resource maximum, convergence status, selected terminal iteration and consumer fitness. A terminal claim identifies the product and selection rule to which it applies. | A stopping threshold, default iteration count or downstream acceptance rule. |
| Response, uncertainty and validity disclosures | Name the exact conditional or adaptive claim, its assumptions and domain; identify unavailable claims, lost information, omitted uncertainty and failure causes. | A numerical response/covariance product, complete uncertainty or any unavailable producer companion. |

A method need not persist every intermediate. It must make the information
required by a claimed output or continuation available in its declared form.
Selecting the actual numerical bundle and retention policy remains an owner
decision. This packet makes no such selection.

## 5. Upstream And Downstream Responsibilities

ALIGN/AST own coordinate and event identities; RTC/CAL retain conditioning and
calibration; PTC owns cleaning and coefficient families; MAP/JINC own their
estimands, normalization, grouping and admission; NOI owns its ensemble method.
FRUIT owns the feedback composition and the scientific identity of its state
and iteration/terminal claims. VAL may evaluate owner-defined criteria.

Source fitting, catalogue inference, Pointing and OOF retain their scientific
targets and fitness requirements. No normalized map becomes a feedback sky
model merely because it is available. No FRUIT rule supplies a missing
upstream coefficient, response, covariance, numerical route or filter authority.

## 6. Externally Imposed Conventions

- Iteration identifiers are absolute and zero-based; after completed N, exact
  continuation begins at N+1. This convention selects no stopping limit.
- Requested, effective, observation-resolved, fitted and applied facts have
  distinct identities. Unit, frame, grouping, support and generation travel
  with the quantity, not just its container.
- Fixed-subspace PTC application recomputes application coefficients; frozen
  numerical-component subtraction is a different affine family. Their exact
  inherited restrictions are in the PTC reference, not new FRUIT policy.
- MAP and JINC remain separate producer authorities. Base JINC has no
  inherited ordinary-MAP coadd or response/covariance publication role.
- The approved NOI boundary separates fixed-state uncertainty, dependent
  successor generations and per-realization relearning. It admits no numerical
  FRUIT uncertainty route by itself.

No checkpoint schema, FITS/HDF5 layout or implementation-specific convention
is imposed. Numerical geometry, units and sample timing must be inherited
from the exact method's actual producers before execution.

## 7. Questions The Contract Must Answer

1. What distinguishes the astronomical target, measured map, selected model,
   applied model and resulting signal, and what makes subtraction or rejoin
   scientifically meaningful between their domains?
2. How do removal, residual processing, bypass and restoration compose under
   explicitly fixed operators and support? Which assumptions are necessary
   for any additive or cancellation identity?
3. How does conditional response differ from the full procedure when the model,
   support, cleaner or selection is learned? Which derivatives exist, and
   what information remains lost or unidentifiable?
4. What scientific condition makes continuation equivalent to an uninterrupted
   procedure, and how do causal state, seed, restart and relearning differ?
5. Which propositions are asserted by completion, maximum, convergence and
   terminal selection, and which further premises belong to a named consumer?
6. What do zero feedback, identity residual processing, unequal removal and
   restoration support, and a perfect model with non-identity output response
   predict under explicit assumptions? What would falsify those claims?

The author may formalize these conditional questions and derive their
consequences. That delegation does not authorize choosing numerical policies.

## 8. Non-Goals

No new cleaning algorithm, numerical recurrence, parent admission, sky prior,
filter/inverse, stopping rule, uncertainty realization, source catalogue or
consumer acceptance policy is selected. Implementation, empirical studies,
performance tuning, qualification, production and Unity activity are outside
this author task. Later policy claims retain their separate evidence gates.

## 9. Allowed References

The proposed exact set is this brief, the scientific decision ledger, the
boundary/convention extract, the frozen PTC application excerpt and the
program reference, as bound by [AUTHOR_INPUT_MANIFEST.md](AUTHOR_INPUT_MANIFEST.md).
All remain pending exact packet approval. Only the PTC fragment is a verbatim
scientific-source excerpt; other extracts are identified abstractions with
restricted roles. Source hashes are provenance, not permission to retrieve
unlisted originals or follow their links.

## 10. Owner Decisions And Remaining Ambiguities

The owner approved the conditional-core scope and narrow sequence exception.
The [decision ledger](SCIENTIFIC_OWNER_DECISION_LEDGER.md) maps every existing
FRUIT question to approved direction, bounded author derivation, deferred
numerical choice or pending packet review. Deferral does not close a decision.

State definition, transition law, update-contribution meaning and persistence
remain separate. The core may formalize their conditional relation; it cannot
select a numerical answer for any of them. The ledger states exactly which
claims remain unavailable and who may resolve each question. If a core
derivation cannot proceed under these boundaries, return its precise missing
premise instead of treating the entire topic as a numerical detail.

## 11. Independence Statement

This brief identifies the problem, inputs, outputs, boundary and questions.
It does not prescribe current implementation as the scientific answer. A
fresh author receives only the approved version of this brief, its permitted
references and subsequent owner answers. Internal recovery, code, schemas,
tests, audits, reductions and observed outcomes remain excluded.
