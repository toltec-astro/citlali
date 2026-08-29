# SCI-NOI v0.1 — Scientific-Owner Decision Ledger

Status: Stage A owner-review candidate

Scientific owner: Grant Wilson

Updated: `2026-08-29`

Implementation defaults, historical products, successful tests, or accepted
reductions cannot resolve an open item.

## Decided Package And Boundary Questions

| ID | State | Approved substance | Exact consequence |
| --- | --- | --- | --- |
| `SCI-NOI-ODQ-001` | decided | One package contains two primary families: realization generation and empirical uncertainty inference | No separate NOI-001/002 package ownership split |
| `SCI-NOI-ODQ-002` | decided | Generation and inference have a hard typed boundary | A realization ensemble has no automatic variance/covariance/S/N/significance claim |
| `SCI-NOI-ODQ-003` | decided | Standardized signal is a third separate method/product role within SCI-NOI | It is not an uncertainty estimate and needs exact signal/scale parents |
| `SCI-NOI-ODQ-004` | decided | Without separate justification/validation, standardized signal means only “standardized by the stated empirical scale” | No automatic Gaussian significance or detection probability |
| `SCI-NOI-ODQ-005` | decided | Fixed-state and relearned realizations are different possible methods and shall not be mixed | Each has distinct conditioning, method, ensemble, and inference identity |
| `SCI-NOI-ODQ-006` | decided | Sample validity, MAP/PTC coefficients, assignments, realization QC, variance/covariance, empirical weight, and standardized signal remain separately typed | Numerical or unit similarity cannot create an identity join |
| `SCI-NOI-ODQ-007` | decided | Empirical NOI weight is not a MAP-facing PTC coefficient | Future cross-boundary use requires explicit successor/feedback authority |
| `SCI-NOI-ODQ-008` | decided | MAP/JINC parents remain immutable; NOI results attach as versioned companions | Later evidence does not rewrite original claims or validity |
| `SCI-NOI-ODQ-009` | decided | Dense full covariance is not universally required | Diagonal, stationary/kernel, structured, ensemble, projected, full, or unavailable may be honest states |
| `SCI-NOI-ODQ-010` | decided | Persistence is plan-controlled; exact regeneration or streaming may replace stored members | The plan must record reconstruction/audit limitations |

## Recovered Historical Constraints

These constrain the proposed packet but do not select the new v0.1 baseline.

| ID | State | Recovered substance | Limitation |
| --- | --- | --- | --- |
| `SCI-NOI-HIST-001` | recovered approved policy | Compact deterministic assignment identity may replace dense sign persistence when exactly reconstructible | Does not select an RNG or current implementation |
| `SCI-NOI-HIST-002` | recovered approved policy | Enabled generation requires positive effective/completed membership; disabled is explicit zero-ensemble/no-work state | Does not choose a default or adequate count |
| `SCI-NOI-HIST-003` | recovered approved policy | Historical `source_imprinted_current` products are conditional finite-stack diagnostics | Does not authorize physical-noise uncertainty or make this the ordinary future method |
| `SCI-NOI-HIST-004` | recovered approved policy | Mathematically distinct S/N-like values require distinct identities; invalid denominators are unavailable, not numeric zero | Does not authorize significance or a source threshold |
| `SCI-NOI-HIST-005` | recovered approved policy | Package-level semantic provenance with stable joins can avoid redundant metadata | Does not weaken exact parent/method/restriction identity |

## Open Scientific Questions

| ID | Owning authority | State | Decision or bounded analysis needed | Exact blocked claim or output |
| --- | --- | --- | --- | --- |
| `SCI-NOI-ODQ-101` | SCI-NOI scientific owner | **open — first walkthrough question** | Decide whether v0.1 ordinary authority is fixed-state only; fixed-state baseline plus explicitly optional relearned methods; fixed and relearned peers with no default; or another exact availability rule | Ordinary method family, Scope finalization, and Stage B task shape |
| `SCI-NOI-ODQ-102` | SCI-NOI generation | open | Select or bound permitted coherence/randomization families: detector/channel, scan, subscan/block, observation, observation-map/coadd, balance/pairing/replacement, and cross-observation coupling | Named Family G methods and assignment laws |
| `SCI-NOI-ODQ-103` | SCI-NOI generation and target authority | open | Define the source-suppression/null target and acceptable leakage statement for each initial method, including deterministic sky, scan-synchronous residual, filtering, and source-model error | Any “noise-like,” source-cancelled, or physical-noise adequacy claim |
| `SCI-NOI-ODQ-104` | RTC/PTC/AST/MAP/JINC/FLT/FRUIT boundaries | open | Enumerate exactly which state is fixed and which steps are rerun for each initial method; decide whether any partial-relearning method is in v0.1 | Relearned method identities and complete conditioning claims |
| `SCI-NOI-ODQ-105` | SCI-NOI inference | open | Select initial target(s), centering, second-moment/covariance distinction, finite normalization/design correction, missingness, and effective-information reporting | Empirical variance/covariance product semantics |
| `SCI-NOI-ODQ-106` | SCI-NOI inference | open | Select the initial covariance representation/product set and domains, including what is required, optional, or explicitly unavailable | Minimum Family U product inventory; no dense-matrix requirement implied |
| `SCI-NOI-ODQ-107` | SCI-NOI inference and future consumers | open | Decide which empirical weight roles, if any, v0.1 publishes: marginal inverse variance, regularized precision, consumer-effective weight, or descriptive scale only | Empirical-weight products and consumer permissions; MAP-facing use remains prohibited |
| `SCI-NOI-ODQ-108` | SCI-NOI standardization and consumer authorities | open | Select initial numerator/scale pairings, compatibility rules, canonical product names/aliases, and whether any stronger-than-empirical-scale claim is in scope | Family Z product inventory and claim ceiling |
| `SCI-NOI-ODQ-109` | SCI-NOI lifecycle/product owner | open | Select initial persistence/reconstruction plans, required audit facts, partial-completion behavior, and whether any members must be written for ordinary operation | Product bundle and audit/reconstruction capability |
| `SCI-NOI-ODQ-110` | SCI-NOI plus FLT/BEAM/MODE/FRUIT owners | open | Bound v0.1 support for filtered/Wiener, Beammap, Pointing, OOF, and FRUIT-attached methods versus explicit deferral | Mode/consumer method inventory and dependency interfaces |

## Stage A Approval Question

| ID | State | Decision required | Exact blocked action |
| --- | --- | --- | --- |
| `SCI-NOI-STAGE-A-Q001` | open | Approve or revise the exact Scope Brief, cover, conventions extract, taxonomy, owner-question routing, firewall, and author-packet hashes | Commissioning an implementation-blind Stage B author |

Implementation conformity, validation, achieved uncertainty/performance,
readiness, and production state are not questions in this scientific ledger.
They remain separately governed future assessments.
