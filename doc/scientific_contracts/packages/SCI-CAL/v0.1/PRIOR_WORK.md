# SCI-CAL — Prior-Work Recovery

This recovery record follows the
[Citlali Scientific Contract Library Program](../../../README.md). It is an
internal Stage A artifact and is not part of the scientific-author packet.

Status: reviewed for Scope Brief drafting; owner disposition pending

Investigator/date: Codex manager, `2026-08-16`

Scope revision examined:

- integrated discovery line:
  `origin/codex/refactor-mainline@46ad23888a40f5102cdfd50c06e49a549bdf8a20`;
- coordination corpus:
  `codex/register-sci-map-003-audit-disposition@8c581bfb26f01b187f4f1e0565f4457bcc25f099`;
- frozen CAL independent-core/audit line:
  `codex/audit-sci-cal-001@27b0916e725696597c3ba84fb6a82bf6cf0ea356`;
- adjacent TolProj, TolAPT, and TolTECA ownership documentation available
  locally; and
- the active ALIGN B3c checkout was not inspected or modified.

## Search Coverage

The recovery began from [`PRIOR_WORK_REGISTRY.md`](../../../PRIOR_WORK_REGISTRY.md)
and searched:

- the living Citlali architecture, scientific-convention, product-contract,
  and configuration-contract records on the integrated discovery line;
- every recovered `SCI-CAL-001_*` decision, amendment, evidence disposition,
  repair/re-audit handoff, and successor owner record in the coordination
  corpus;
- the frozen implementation-independent CAL core and its associated audit;
- the accepted but unactivated canonical baseline and observation-specific APT
  ADRs, plus APT identity and end-to-end audit material as boundary evidence;
- TolProj's workflow description for cohort APT selection, calibrator
  interpretation, and pointing-derived flux correction;
- TolAPT's immutable input/output and design-to-measured matching contract;
- TolTECA's Citlali input-delivery boundary; and
- Citlali calibration, timestream, observation-resolution, Beammap,
  provenance, configuration, and product paths for scope ownership only.

Unity was unavailable and was neither contacted nor required. Historical
external executions were inventoried but not treated as scientific authority.

## Recovered Materials

| Material and exact reference | Classification | Scientific content already available | Limitations or conflicts | Disposition |
| --- | --- | --- | --- | --- |
| `MAIN:doc/SCIENTIFIC_CONVENTIONS.md` | Governing scientific authority | TolTEC array/network/detector identity, sample-by-detector shape, map unit and weight semantics, calibration ownership boundaries, requested/effective/realized state distinctions | Broad living document; some accepted unit tokens do not themselves authorize conversions | Adopt only the CAL-relevant conventions through a sanitized extract |
| `MAIN:validation/product_contracts.json` and `MAIN:tools/config/config_leaf_contract_resolved.json` | Implementation-informed scope evidence | Current declared product units and observation-resolved calibration/extinction controls | Executable contracts describe current interfaces, not necessarily the intended scientific answer | Abstract scope and consumer obligations; exclude raw files from authorship |
| `COORD:.../SCI-CAL-001_COORDINATOR_DECISION_2026-07-31.md`, decisions `CAL-D001`--`CAL-D004` | Approved scientific decision | Top-of-atmosphere extinction identity; factor semantics; initial `mJy/beam` point-source policy; uncertainty decomposition | `CAL-D002`'s original identity-only clauses are superseded; `CAL-D005` is production disposition, not contract science | Adopt D001--D004 subject to named amendments; exclude D005 from authorship |
| `COORD:.../SCI-CAL-001_OPACITY_DECISION_AMENDMENT_2026-07-31.md` | Approved scientific decision, later narrowed by model-scope decisions | Geometric transmission interpolation, equivalently linear line-of-sight optical depth, between zero and the first nonzero anchor | Existing q-models were later demoted to evidence rather than automatically approved physical anchors | Adopt the zero-limit algebra; do not promote the legacy anchor set to authority |
| `COORD:.../SCI-CAL-001_APT_IDENTITY_DECISION_AMENDMENT_2026-08-01.md` | Approved scientific decision | Layered acquisition, measured-APT, cross-observation association, and design identities; verified row-order or explicit-key admission | Cross-observation association remains imperfect and must not be relabeled as exact identity | Adopt; explicitly supersede the earlier universal-UID/row-is-never-identity wording |
| `COORD:.../SCI-CAL-001_ACCURACY_AND_MODEL_SCOPE_AMENDMENT_2026-08-01.md` | Approved scientific decision | Separates exact contract correctness, atmosphere representation fidelity, and observational calibration performance; sets provisional targets | Does not select a successor atmosphere operator or prove performance | Adopt the hierarchy and objectives; retain operator/domain as an explicit question |
| `COORD:.../SCI-CAL-001_ATMOSPHERE_CONFIRMATION_DECISION_2026-08-02.md` and `SCI-CAL-001_PASSBAND_AUTHORITY_001.json` | Approved scientific/instrument decision | Exact content-bound TolTECA v1 modeled array passband set and known limitations | Confirmation quadrature, source spectra, and domain were study-specific, not universal production choices | Cite the passband identity and limitations; exclude study execution rules |
| `COORD:.../SCI-CAL-001_TAU025_ENGINEERING_AVAILABILITY_DECISION_2026-08-03.md` | Approved scientific policy | Separates science-qualification target, engineering-availability target, and outside-supported calibration by opacity; one state per coherent observation/segment | Planning policy; no continuous operator to `tau225=0.25` was adopted | Adopt the quality-class distinction; defer exact operational realization |
| `COORD:.../SCI-CAL-001_SUCCESSOR_2_OWNER_DISPOSITIONS_2026-08-09.md`, especially F002, F005--F008 | Mixed approved decision and historical repair disposition | Retains a fixed continuous operator structurally; conditional variance/weight transfer; narrow `mJy/beam` boundary; package-level reconstructibility; once-only factor composition | Mostly implementation/repair authority; expressly does not establish atmosphere truth, model fidelity, observational calibration, response fidelity, total uncertainty, or production authority | Abstract only the owner-approved scientific statements; exclude repair findings, paths, gates, and status |
| `codex/audit-sci-cal-001@27b0916...:doc/audits/packages/SCI-CAL-001_INDEPENDENT_CORE.tex` | Reusable scientific reference | Independent factor algebra, extinction and unit-transfer equations, response/normalization, conditional and nuisance covariance, validity, limiting cases | Its original identity rule is superseded; its general target-unit treatment is broader than the approved initial `mJy/beam` domain | Cite through a supersession cover sheet after owner approval |
| `codex/audit-sci-cal-001@27b0916...:doc/audits/packages/SCI-CAL-001_SCIENTIFIC_CONTRACT_AUDIT.tex` and all CAL repair/re-audit/evidence packages | Historical audit, repair, or validation evidence | Version-specific findings, numerical evidence, repairs, and conformity states | Implementation-contaminated and not scientific authority | Defer in full to any later implementation-conformance program; exclude from authorship |
| `codex/repair-apt-prod-001-canonical-baseline-v1@d4a808c59f383a5f77059b83083af2a69802a12a:doc/adr/0010-canonical-baseline-apt-v1.md` | Approved architectural/scientific identity decision; unactivated | Artifact-local UID; complete raw `(network, local channel)` relation; semantic, occurrence, and byte identity are distinct | Does not activate a CAL input and leaves physical authority of some APT fields unresolved | Adopt the identity principles; defer schema and transport mechanics |
| `codex/repair-apt-prod-002-observation-contract@20feebc26f5ab36f3db04d05835de6ac907fd2e6:doc/adr/0011-canonical-observation-apt-contract.md` | Approved architectural/scientific identity decision; unactivated | Cross-artifact correspondence uses occurrence-scoped endpoint references; target/source sequences and local keys are not persistent identity | Does not activate the contract or choose matcher policy | Adopt the identity principles; defer schema, protocol, and publication mechanics |
| `MAIN:doc/astrometry_photometry_config_transition.md` and current Citlali calibration-related source paths | Implementation-informed scope evidence | Locates the current application boundary, configuration, factor application, Beammap production, and product/provenance consumers | Current behavior cannot prescribe the scientific contract | Use only in `INTERNAL_DOSSIER.md`; exclude from authorship |
| TolProj `docs/WORKFLOW_V0_2.md` and local ownership rules | Approved upstream boundary/reference | TolProj selects cohort APTs, interprets calibrators, and creates pointing-derived flux-corrected APT products without mutating inputs | Does not define Citlali's internal estimator or uncertainty model | Abstract the responsibility boundary; cite upstream only if Grant approves it |
| TolAPT `docs/output_contract.md` and local ownership rules | Approved upstream boundary/reference | Immutable design/measured inputs and provenance-bearing match outputs | Design match is neither required nor exact for ordinary measured Beammap CAL fields | Abstract the responsibility boundary; cite upstream only if needed |
| TolTECA operational-line Citlali delivery code and local ownership rules | Implementation-informed boundary evidence | Selected input/configuration delivery to Citlali | Contains implementation-specific conversions/defaults and is not scientific authority | Exclude from authorship; retain only the upstream-delivery responsibility in the brief |

`MAIN` and `COORD` are the exact references named in the scope-revision
header. Ellipses in the table abbreviate only the repeated directory
`doc/audits/packages/`; filenames and repository references are exact.

## Recovery Synthesis

### Questions already answered

1. **Reference plane and opacity identity.** Beammap-derived flux calibration
   is top-of-atmosphere; `tau225` is zenith optical depth; eligible samples
   require full sample airmass and no silent extrapolation (`CAL-D001`).
2. **Factor roles.** Relative detector responsivity, absolute `flxscale`,
   sensitivity, atmospheric extinction, and any target-unit transfer are
   distinct named factors; composition is once-only and reconstructible
   (`CAL-D002` plus the later F008 disposition).
3. **Initial scientific unit.** The first contract supports only
   top-of-atmosphere `mJy/beam` with point-source peak normalization; other
   unit families and extended/integrated photometry require later contracts
   (`CAL-D003` and F006).
4. **Conditional uncertainty.** A multiplier `a` transforms conditional
   variance and inverse-variance weight as `a^2 v` and `w/a^2`; calibration
   and response systematics remain named correlated nuisances and missing
   uncertainty is never zero (`CAL-D004` and F005).
5. **Identity.** Observation-local acquisition identity, measured-APT
   identity, cross-observation source association, and design identity are
   separate. Verified row order is admissible only when proven; a design
   match is not required for ordinary measured Beammap calibration fields
   (APT identity amendment).
6. **Accuracy claims.** Exact algebraic conformance, atmospheric
   representation fidelity, relative repeatability, and absolute
   photometric performance are different claims. The prior provisional goals
   are at most one-percent representation error, about five-percent relative
   repeatability, and about five-to-ten-percent absolute accuracy per band
   (accuracy amendment).
7. **Passband identity.** The prior work selected one exact v1 modeled-array
   passband set while recording that detector/network aggregation,
   telescope-measured uncertainty, normalization, and photon-versus-energy
   convention remain unknown (BAND-001).
8. **Quality classification.** Science qualification, engineering
   availability, and outside-supported calibration are distinct claims and
   cannot be switched sample by sample within an otherwise coherent product
   (CAL-ATM-D006).
9. **Artifact identity.** A UID is artifact-local, and occurrence, semantic
   content, and byte transport are separate identities. Cross-artifact
   correspondence needs explicit occurrence-scoped endpoint references; row
   positions, local keys, and equal integer spellings do not establish it
   (ADRs 0010 and 0011).

### Reusable definitions, equations, and reasoning

The frozen independent core already provides a coherent starting derivation:
detector coefficient normalization, an extinction operator referenced to an
explicit airmass pivot, factor composition, calibrated signal, conditional
covariance transfer, nuisance covariance, validity behavior, and analytic
limits. Its retrieved content SHA-256 is
`106755520b048f601bc60fd04e7b6020e6fa470480ac3105fa7ba269c730a4fe`.

It is suitable for the author packet only with a cover sheet stating that:

- the later layered identity decision supersedes its stronger row/UID rule;
- the v0.1 boundary is restricted to top-of-atmosphere `mJy/beam` point-source
  normalization; and
- its equations and reasoning are reusable science, while the associated
  audit and all later repairs are not author references.

### Conflicts and unresolved choices

1. The first low-opacity amendment approved a q25-anchored expression, but the
   later accuracy amendment said the legacy q models were evidence rather than
   automatically approved anchors. A later owner disposition retained
   `am12_fixed_djf25_piecewise_linear_los_tau_v1` as structurally closed while
   explicitly withholding atmosphere-truth and observational claims. The
   contract must preserve the reusable operator work without claiming more
   authority than the evidence provides.
2. The exact v1 passband set is selected, but the physical response convention
   and detector/network aggregation remain unknown. The contract must say
   which uses are binding and which passband-derived claims remain
   unavailable.
3. TolProj may deliver a pointing-derived correction through a new selected
   APT. CAL must define the scientific factor lineage and prevent double
   application without absorbing TolProj's estimator or APT-selection role.
4. Prior quality-class policy names opacity ranges but does not by itself
   establish the final elevation support, continuous engineering operator, or
   product behavior for an unsupported calibration request.
5. The recovered authority does not yet settle whether v0.1 calibrates only
   the ordinary `xs` detector stream or gives any other measured channel the
   same scientific meaning.

These are stated as owner/author questions in the Scope Brief rather than
being silently resolved from current code.

### Material excluded from authorship

- every Citlali source file, source trace, current config encoding, and
  implementation-specific product layout;
- the SCI-CAL audit, findings, repair candidates, repair handoffs, re-audits,
  tests, exact-SHA results, numerical execution packages, and Unity history;
- current implementation/validation/production status labels;
- active ALIGN B3c materials; and
- TolTECA compatibility conversions, defaults, or other current delivery
  mechanics.

### Genuinely new work

The new author should not rederive the entire calibration model. The smallest
remaining scientific work is to:

1. reconcile the retained structural atmosphere operator with the explicit
   limits on its physical and operational authority;
2. specify a single canonical once-only factor/lineage model at the selected
   APT and Citlali boundary, including a TolProj-derived correction without
   redefining it;
3. define the minimum scientifically meaningful validity, quality,
   uncertainty, response-basis, and provenance outputs without copying a
   current file layout; and
4. derive falsifiable predictions and limiting cases for the adopted domain
   and unavailable claims.

## Proposed Author Reference Packet

The following minimal packet is proposed for owner approval:

1. the owner-approved successor of [`SCOPE_BRIEF.md`](SCOPE_BRIEF.md), which
   will itself contain the binding sanitized decision digest;
2. the frozen `SCI-CAL-001_INDEPENDENT_CORE.tex` identified above, accompanied
   by a one-page supersession cover sheet for identity, initial unit scope,
   and authority limits;
3. `SCI-CAL-001_PASSBAND_AUTHORITY_001.json` at `COORD`, solely as the exact
   instrument-reference manifest; and
4. a short sanitized CAL convention extract prepared from
   `MAIN:doc/SCIENTIFIC_CONVENTIONS.md` and the TolProj/TolAPT ownership
   documents, containing identities, units, immutable-artifact principles,
   and inter-package responsibilities only, including owner-approved
   abstractions from APT ADRs 0010 and 0011.

The raw owner-decision files need not enter the packet because their binding
scientific content will be consolidated in the approved Scope Brief. The
audit, implementation, repair, validation, and internal dossier are expressly
not proposed references.

## Investigator Attestation

Prior work was recovered before commissioning new derivation. The proposed
scope reuses rather than rewrites the earlier independent mathematics and
owner decisions, explicitly records their supersessions and limitations, and
does not promote current implementation behavior or historical conformity
evidence to scientific authority.
