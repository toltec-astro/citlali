# SCI-NOI v0.1 — Prior-Work Recovery

Status: reviewed Stage A recovery; awaiting scientific-owner review

Investigator/date: Codex manager, `2026-08-29`

This internal record follows the
[program charter](../../../README.md),
[pilot review](../../../PILOT_PROCESS_REVIEW_2026-08-16.md),
[prior-work registry](../../../PRIOR_WORK_REGISTRY.md), and
[downstream roadmap](../../../DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md).
It is not part of the proposed author packet.

## Starting Authority And Search Coverage

The package starts from
`codex/scientific-contract-library@5f206cf46bb2868aadb00f37dbbbc3944ac4ec8c`.
Recovery rechecked:

- the complete reachable branch and object inventory containing `SCI-NOI`,
  noise realization, jackknife, empirical weight, variance, covariance,
  `sig2noise`, filtering, and fruit-loop work;
- the earlier independent NOI-001 and NOI-002 mathematical cores and their
  predecessor/successor identities;
- the owner/coordinator decision records, repair/re-audit records, application
  integration record, and cross-package handoffs;
- the current internal noise-estimation plan and configuration authority;
- current implementation surfaces for configuration, RNG/assignment,
  ordinary and JINC map paths, observation/coadd products, deterministic and
  Wiener filtering, Beammap, Pointing/OOF, FRUIT, persistence, and product
  metadata;
- frozen SCI-MAP and SCI-JINC parent and downstream-attachment rules; and
- current validation/product registries only to inventory historical evidence
  and vocabulary.

No Unity system was contacted. No reduction, source audit, validation run,
scientific derivation, implementation modification, or conformity assessment
was performed.

## Exact Reference Identities

A digest records provenance, not scientific authority.

| Object | Lines | SHA-256 |
| --- | ---: | --- |
| `5a027c94ef9fc9c4a6e6cadc84af1c8a550d3508:doc/audits/packages/SCI-NOI-001_INDEPENDENT_CORE_R3.tex` | 920 | `27263ab3bf29ac8f098463455e540f13e783241a688ef2bc5cb15b1f2a4319da` |
| `4f1fec36f7802f3b5e8ac067377679946930983c:doc/audits/packages/SCI-NOI-002_INDEPENDENT_CORE.tex` | 1031 | `36781b766a2f57c9a3bd7e173ee8f1d85cba7f3d08afe2e67a403166f6b6d72d` |
| `d03ef80b31f704859ef836e368801dc17d92e76e:doc/audits/packages/SCI-NOI-001_COORDINATOR_DECISION_BRIEF_2026-08-06.md` | 121 | `bfbfcfa6673830d0f43f5ff2e8a8ed044edf98308441027b7e97c43b7d3c6a3f` |
| `64ba81795110d89d8baf0ad7d645d16472c254c5:doc/audits/packages/SCI-NOI-002_OWNER_DECISION_BRIEF_2026-08-06.md` | 244 | `3520172cfc11e8e34f280f9ebdf147ea414c7a3a4ca6109bad55354a5ff3cf71` |
| `4846fa4db39bd2f7d4ddc41f693836834cbc5ff4:handoff/SCI-NOI-002_APPLICATION_INTEGRATION_DECISION_2026-08-08.md` | 118 | `9901e1343633cd6b3a69d2b421730c4884e8757cb2fe01f823a58c8a6a79d6b0` |
| `38ef72860743636f59d226c9e1ff5ff776d0e9c0:include/citlali/core/pipeline/noise_realization_identity.h` | 449 | `83f6a107aabd35ffafd702367b9a572dc4b5193492c16373b98b903703505eea` |
| `5f206cf4...:doc/citlali_noise_estimation_plan.tex` | 1091 | `c0f6d8342993a5b1bd55dff77e9457607b2588f584fa92c7811dcceb737ad2bd` |
| `5f206cf4...:doc/NOISE_PRODUCTS_CONFIG_AUTHORITY.md` | 44 | `be78f56052b2043f2bbd0d65da75cfb43735037bd588142828bd259db492ef46` |
| `5f206cf4...:doc/SCIENTIFIC_CONVENTIONS.md` | 660 | `24c8397b130de0fb1c0dcfcd87c057c06e4f095ee6a54472759a6ef276bb5add` |
| `5f206cf4...:validation/product_contracts.json` | 2351 | `3ce4d6c40d5f2a14416f3acfe6cf1e3c26ad8d7114ba0076fb16be8b2c6eabcd` |

The R3 core identifies its immutable original and R2 predecessor. The R3
change is mechanical only. The NOI-002 core explicitly admits R3 as its
predecessor science.

## Recovered Materials And Dispositions

| Material | Classification | Reusable content | Conflict or limitation | Disposition |
| --- | --- | --- | --- | --- |
| Program charter, pilot review, roadmap, frozen MAP/JINC | Governing program and adjacent scientific authority | Recovery-first firewall; immutable parent attachment; same-fixed-operator distinction; no significance promotion | Does not choose NOI method | **Adopt** |
| NOI-001 R3 core | Reusable implementation-independent science | Coherence-unit sign law; conditional moments; randomization versus physical covariance; source imprint; fixed observation/coadd/filter propagation; finite assignment/dependence; compact regeneration; use-specific adequacy | Fixed-state sign ensemble only; old separate-package organization; no relearned method | **Cite** under binding cover; **supersede** scope limits |
| NOI-002 core | Reusable implementation-independent science | Finite-stack identities; centering/divisor/dependence; rank and inversion; covariance representations; projected uncertainty; standardization versus significance; adequacy hierarchy; provenance | Conditions specifically on old source-imprinted fixed-state ensemble and old NOI split; discusses broader consumer policies beyond initial scope | **Cite** under binding cover; **bound** to selected future methods |
| NOI-001 coordinator decision | Approved historical owner/coordinator policy mixed with audit status | Compact deterministic realization key; cross-observation separation; named pass/iteration identity; no dense sign stream; enabled-positive/disabled-zero; current source-imprinted mode is diagnostic | Repair, validation, count, production, and current-mode status are historical; does not select v0.1 ordinary method | **Abstract** policy only; **exclude** raw record from authorship |
| NOI-002 owner decision | Approved historical scientific policy mixed with implementation disposition | Current `1/R` centered scatter is descriptive conditional finite-stack scatter; MAP coefficient scaling is nonprecision diagnostic; S/N-like identities distinct; fixed estimator projection; package provenance; no universal count | Several decisions govern existing-use compatibility, not future scientific authority; FLT/FRUIT work remains open | **Abstract** stable meanings; **defer** implementation/status |
| NOI-002 integration decision | Historical implementation/evidence record | Confirms a bounded candidate existed and records exact evidence boundaries | Different application tree; no physical-noise, precision, significance, or production claim; validation cannot become science | **Defer** to later conformity; **exclude** from packet |
| Internal noise-estimation plan | Reusable scientific candidate mixed with implementation and recommendations | Estimator-dependent uncertainty; off-diagonal covariance; realization-level propagation; streaming sufficient statistics; optional persistence; product vocabulary | Treats formal MAP weight as inverse variance; promotes empirical scale into primary map weight; sometimes calls jackknife products noise/significance too strongly; recommends defaults | **Abstract** sound concepts; **supersede** conflicts; **exclude** raw plan from packet |
| Noise-products config authority | Current implementation/config authority | Six requested inputs; requested/effective distinction; disabled-zero behavior; fixed seed; realized cardinality | Explicitly not scientific estimator authority | **Record** in dossier; **exclude** from authorship |
| Current implementation at `5f206cf4...` | Implementation scope evidence | Actual placement, RNG, detector/scan grouping, map/coadd/filter routes, product names, persistence switches, mode dependencies | Current tree is not the accepted historical NOI repair tree and cannot select science | **Quarantine** in dossier; **exclude** |
| Historical deterministic assignment repair at `38ef7286...` | Implementation/repair evidence illustrating a possible representation | Versioned counter/key identity and compact regeneration | Not an ancestor of the current library tree and not scientific authority | **Defer**; retain as later implementation candidate evidence |
| Convolve/FLT material | Adjacent reusable mathematics mixed with audit | Fixed linear propagation and operator parity questions | Filtering is next tranche; current parity finding remains conditioned | **Defer** to SCI-FLT; include only interface question |
| Beammap, Pointing, OOF, FRUIT records | Adjacent mode/feedback scope and evidence | Shows distinct consumers, existing ratios, and adaptive/relearned paths | They own interpretation and adaptive procedure; cannot validate NOI | **Defer**; preserve boundaries |
| Accepted runs, tests, product registries, audit/re-audit reports | Validation and implementation evidence | Exact observed behavior and product vocabulary under named revisions | No target truth; no scientific authority; differing ancestry | **Exclude** from authorship and scientific decisions |

## Science Already Available Without Repetition

The recovered cores already establish:

1. changing detector/sample/scan/observation coherence units changes the
   ensemble law;
2. balance, complement pairs, no-replacement designs, duplicates, and shared
   streams alter marginal or cross-realization dependence;
3. a fixed sign-randomization covariance is not automatically repeated
   physical-noise covariance;
4. deterministic signal and scan-synchronous structure can imprint the
   realization ensemble even when the target sign mean is zero;
5. fixed observation, coadd, and linear filter operators propagate ensemble
   moments, while re-estimated operators require a different joint method;
6. finite-ensemble covariance normalization depends on centering and the joint
   design, not on `R` alone;
7. per-pixel variance, marginal inverse variance, full precision, and
   consumer-effective precision are different objects;
8. off-diagonal covariance can be handled by structured summaries, retained
   ensembles, or projection of each realization rather than a dense matrix;
9. standardized signal is not automatically a null-calibrated significance;
   and
10. exact regeneration can replace dense sign persistence when algorithm,
    key/seed, parents, partition, and membership are bound.

Stage B should reconcile and present this science, not derive it again.

## Recovered Method-Family Census

| Level or route | Recovered state | Classification |
| --- | --- | --- |
| Detector/channel signs | Current source draws a separate sign for each realized detector column when detector randomization is enabled | Existing implementation method; science not yet selected |
| Scan/chunk-coherent signs | Current source draws one sign shared across detector columns for each processed scan/chunk when detector randomization is disabled | Existing implementation method; exact scan/chunk identity must be bound |
| Subscan or residual-block resampling | No exact current requested subscan method was found; the earlier NOI-002 owner brief admitted residual block resampling only as a possible future physical-noise design preserving correlations | Conceptual future method; unavailable in current Stage A authority |
| Observation-level ensemble | Current source generates observation-map realization stacks; prior NOI-001 requires observation-specific assignment identity and cross-observation law | Existing method family; exact ordinary law unresolved |
| Coadd by fixed observation realization index | Recovered cores derive fixed observation-to-coadd propagation and the internal plan proposes common-index coaddition | Reusable mathematical method; not yet selected |
| Resampled observation-level coadd | The internal plan proposes reassigning/resampling observation realizations | Conceptual method; requires new assignment and parent identity |
| Direct combined-data coadd randomization | The internal plan and current coadd-related source surfaces expose a distinct direct route | Distinct method; reachability and science unresolved |
| Balanced interleaved scan-split null maps | Earlier owner reasoning prioritized this as a future physical-noise candidate | Approved research direction, not an implemented or validated ordinary method |
| Source-imprinted sign randomization | Earlier owner decision truthfully labels the current fixed-state ensemble `source_imprinted_current` | Existing-use diagnostic only; not physical-noise authority |
| Fixed source-residual randomization | NOI-001 R3 derives a conceptual fixed residual successor | Reusable definition; requires exact FRUIT/source-model boundary |
| Full or partial relearning | Owner launch requires separate method identity; no complete approved rerun plan was recovered | Genuine owner/Stage B question; not ordinary authority |

## Contradictions And Gaps Requiring Explicit Treatment

1. **Old split versus one package.** The two cores divided ownership between
   NOI-001 and NOI-002. The owner now requires one SCI-NOI package with hard
   internal method/product boundaries.
2. **Fixed-state-only versus fixed/relearned methods.** NOI-001 R3 conditions
   on realized RTC/PTC/MAP/filter state. The owner requires relearned
   generation to remain a distinct possible method and an unresolved choice.
3. **Generation versus inference.** Historical filenames and prose sometimes
   call sign-flipped maps “noise.” The new contract cannot infer an uncertainty
   target from the generation mechanism.
4. **Weight collision.** The internal plan and existing code can rescale and
   overwrite a MAP coefficient using realization scatter. Frozen MAP now
   declares that coefficient nonprecision, and the owner explicitly forbids
   automatic NOI-to-MAP promotion.
5. **`sig2noise` collision.** Current and historical products include pixel,
   point-source, source-finder, fitted-amplitude, and dynamic-range meanings.
   Standardized signal must have method identity and is not uncertainty or
   significance by name.
6. **Current versus historical implementation ancestry.** The accepted NOI-002
   candidate and NOI-001 repair are not ancestors of the current contract-
   library tree. Current code therefore cannot be described using their
   conformance status.
7. **Coadd route ambiguity.** Recovered science permits fixed coaddition of
   observation realizations, direct combined-data randomization, and resampled
   observation assignments as distinct methods. Current source exposes more
   than one coadd-related surface. Stage B must define methods, not infer one
   from a filename.
8. **Filter parity and inference.** A fixed deterministic filter can propagate
   an ensemble, while Wiener/noise-estimated filters may be data-dependent.
   Historical exact-application work left strict signal/realization edge-
   operator parity conditioned on future FLT authority.
9. **FRUIT residual versus relearned procedure.** Source-subtracted residual
   randomization, fixed-state post-subtraction inference, and replay of the
   complete adaptive FRUIT procedure answer different questions.
10. **Count and defaults.** Existing values such as 5, 10, 25, or optional
    capacity 64 are configuration or evidence facts, not general adequacy.

## Material Prohibited From Stage B Authorship

- all source code, implementation paths, schemas, product contracts, tests,
  accepted runs, audit findings, repair/re-audit reports, integration records,
  Unity or observational results;
- the raw historical owner/coordinator decision briefs, whose stable policy
  content is already sanitized;
- the internal noise-estimation plan, current config authority, Convolve
  audit, and full adjacent package documents;
- current implementation filenames, legacy aliases, defaults, counts, and
  product reachability; and
- any unlisted repository, web, paper, or model-memory source.

## Recovery Conclusion

Prior work is substantial and sufficient to prevent a new derivation. The
proposed author packet should reuse both mathematical cores under one cover,
then ask the author to reconcile them with the owner-approved three-operation
taxonomy and resolved owner decisions. The implementation inventory is useful
only for ensuring that the scope asks the right questions.

Stage B remains unlaunched. ODQ-101 subsequently approved fixed-state
conditional-sign as the ordinary conditioning family while keeping relearned
methods separate and never mixed. It selected no numerical route. The next
owner question is `SCI-NOI-ODQ-102A`, exact ordinary route or explicit route
unavailability.
