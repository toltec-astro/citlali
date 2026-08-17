# SCI-BEAM — Prior-Work Recovery

This recovery record follows the
[Citlali Scientific Contract Library Program](../../../README.md) and the
[pilot process review](../../../PILOT_PROCESS_REVIEW_2026-08-16.md). It is an
internal Stage A artifact and is not automatically part of an author packet.

Status: reviewed by the Codex manager; awaiting scientific-owner review

Investigator/date: Codex manager, `2026-08-16`

## Exact Recovery Snapshots

- library worktree:
  `codex/scientific-contract-library@82bde188647d65363b5f6bbe73e49addeb0cae52`;
- integrated Citlali discovery line:
  `origin/codex/refactor-mainline@46ad23888a40f5102cdfd50c06e49a549bdf8a20`;
- coordination corpus:
  `codex/register-sci-map-003-audit-disposition@8c581bfb26f01b187f4f1e0565f4457bcc25f099`;
- `toltec_beammap` current local status authority:
  `main@958a2a15f43189846a24556a63ef908da789c7b8` (local branch one commit ahead
  of its remote, with three untracked audit scripts left untouched); and
- TolAPT current development authority:
  `codex/rework-foundation@3a07cc551faf903da3e1d49d7d3a6b20381afc3d`.

These are recovery identities, not automatic scientific authorities.

## Search Coverage

Recovery began from the program registry and re-verified:

- the current library program, pilot lessons, CAL and MAP boundaries;
- the historical `SCI-BEAM-001` ledger entry and all three recovered incoming
  handoffs from CAL, AST, and RTC;
- living Citlali architecture, scientific conventions, Beammap configuration
  authority, design review, product contracts, analysis-flow records, and
  tracked prior artifacts;
- topic refs and audit-package paths containing `SCI-BEAM`, `beammap`, or
  Beammap handoffs;
- current `toltec_beammap` ownership, supported surface, and evidence limits;
- current TolAPT ownership, output contract, status, and soft-prior reliability
  contract; and
- primary literature candidates describing TolTEC optical expectations,
  commissioning use of beammaps, detector-level beam mapping, and calibration
  uncertainty.

No Unity system was contacted. No reduction, source audit, A/B test,
validation campaign, or numerical derivation was repeated.

## Reference Digests

| Object | SHA-256 |
| --- | --- |
| `doc/BEAMMAP_CONFIG_AUTHORITY.md` | `0bc660dc4a112811259ea3ddd21a7448587fe73c01bcc8beb5b31cc0589de61a` |
| `handoff/BEAMMAP_AUTHORITY_DESIGN_REVIEW_2026-07-14.md` | `6133397e712f2568750400134b203bb66d99c9da8fbaf46eeef7a0c272e97790` |
| `doc/ARCHITECTURE.md` | `d78571ab2da351fb2884b966b7dcbbd9e3d68d0c8c692a18f47159c2c7196f31` |
| `doc/SCIENTIFIC_CONVENTIONS.md` | `24c8397b130de0fb1c0dcfcd87c057c06e4f095ee6a54472759a6ef276bb5add` |
| `validation/product_contracts.json` | `3ce4d6c40d5f2a14416f3acfe6cf1e3c26ad8d7114ba0076fb16be8b2c6eabcd` |
| `data/beammap_priors/README.md` | `cf915bb299e4790b4b57b18c1eda253ff662a9e37b6516b359af51ef9b3a8615` |
| TolAPT soft-prior reliability contract | `59705b07832b1646806e2bc0ea5566f14cd8eeefae95705799d4e6455ab9df71` |
| `toltec_beammap/docs/STATUS.md` | `2885a3fe8cd5ccd674004f9ee81bc216dbb321ca484fa68279f26784c920d782` |
| `tolapt/docs/STATUS.md` | `30cf75439027ebadd7c1eea0ffbcb7ac626d716167d7481d73acd8e9e877d99a` |

## Recovered Materials

| Material | Classification | Reusable content | Limitation and disposition |
| --- | --- | --- | --- |
| Current library program and pilot review | Governing process authority | Recovery, firewall, package layout, shared-core, versioning, review, and stop rules | Adopt; process, not BEAM science |
| `COORD:doc/audits/audit-ledger.yaml`, `SCI-BEAM-001` | Historical scope inventory | Detector-map iteration, priors, per-detector fits, convergence, calibration candidates, QC, APT, TOD, split products; internal loop distinct from a fruit loop | The audit never launched; no independent derivation, audit, validation, or verdict. Use only to seed scope |
| `SCI-BEAM-001-XAUD-001` from CAL | Historical post-core evidence | Identifies a Beammap-to-CAL boundary involving source model, fitted/template amplitude, extinction, detector identity, and covariance | Does not establish correct calibrator flux, beam estimator, fitting, priors, flags, or iteration. Abstract questions only; exclude raw handoff from author packet |
| `SCI-BEAM-001-XAUD-002` from AST | Historical post-core evidence | Identifies detector-row identity, coordinate validity, frame, and unit dependencies | Does not establish Beammap fitting, photometry, priors, or selection. Abstract boundary only; exclude raw handoff |
| `SCI-BEAM-001-XAUD-003` from RTC | Historical post-core evidence | Identifies a conditional response/kernel and causal-validity dependency for conditioned detector maps | Does not audit BEAM or quantify bias. Abstract response questions only; exclude raw handoff |
| Living `SCIENTIFIC_CONVENTIONS.md` | Current shared convention authority | AltAz tangent-plane az/el offsets in arcsec; stable UID binding; explicit units and missing states; TolProj/TolTECA photometry boundary; one-way lifecycle | Broad, mixed-status document. Prepare a sanitized convention extract only after owner approval |
| `BEAMMAP_CONFIG_AUTHORITY.md` and design review | Approved architecture/config boundary mixed with implementation history | Requested/effective/realized separation, adjacent photometry exclusion, current policy families, lifecycle/product scope | Explicitly does not redesign Gaussian fitting, priors, flags, or numerical behavior. Internal dossier only; no author access |
| `ARCHITECTURE.md`, analysis flow, product/config contracts | Implementation-informed scope evidence | Apparent producers, transformations, products, identities, units, conditions, and consumers | Current interfaces cannot establish correct science. Dossier only; abstract questions, never source details |
| Tracked Citlali Beammap prior artifacts and builders | Historical implementation/data evidence | Existing network and soft-slot prior shapes and current consumer format | Derived from historical measured APTs; not independent truth or validated production authority. Dossier only |
| TolAPT reliability contract | Producer-side approved application contract | Soft initialization/gating prior; array/network/slot identity; broad weak-region priors; blind fallback; reliability sidecars | A producer contract, not validation inside live Citlali and not a BEAM scientific derivation. Candidate for a sanitized boundary extract, pending owner approval |
| TolAPT output contract and status | Current sibling authority | Immutable matched-APT inputs/outputs, matching and hero/reference ownership, application maturity | Engineering and production scope; exclude from scientific author packet |
| `toltec_beammap` guide/status/README | Current sibling authority | Downstream analysis, calibrator handling, planet calibration, APT diagnostics, and sensitivity ownership | Its current A/B evidence is missing and three local scripts are unreviewed. Boundary only; exclude raw repository material |
| [Bryan et al. 2018, *Optical Design of the TolTEC Millimeter-wave Camera*](https://arxiv.org/abs/1807.00097) | Primary instrument reference candidate | Instrument bands and optical/sensitivity design context | Pre-commissioning design expectations do not establish realized per-detector beams or current calibration; owner must decide author use |
| [Golec and the TolTEC Collaboration 2024, *Early high-resolution millimeter-wave maps from TolTEC*](https://doi.org/10.1051/epjconf/202429300022) | Primary commissioning reference candidate | Public statement that Citlali beammaps populate detector location, flux calibration, and FWHM products | Proceedings gives scope, not the required estimator, covariance, QC, or thresholds; owner must decide author use |
| [Wilson et al. 2008, *The AzTEC mm-wavelength camera*](https://doi.org/10.1111/j.1365-2966.2008.12980.x) | Primary methodological analogue | Per-detector beam maps, 2-D Gaussian fits, flux conversion, beam-shape use, and propagated calibration factors | Different instrument and operating context; analogy cannot set TolTEC policy. Optional reference only if owner approves |
| [Bendo et al. 2013, *Flux calibration of the Herschel-SPIRE photometer*](https://doi.org/10.1093/mnras/stt948) | Primary methodological analogue | Finite-source correction, detector-level calibration, and explicit uncertainty budget | Different instrument; relevant only to general calibration reasoning, not TolTEC facts. Optional reference only if owner approves |

`COORD` is the exact coordination snapshot named above.

## Recovery Synthesis

### Questions already answered

1. **Package coherence.** One package may coherently cover observation-local
   per-detector source/beam inference, fit state and uncertainty, iteration and
   convergence, QC state, and the resulting detector-product bundle.
2. **Frame boundary.** Current Beammap detector maps are declared in an AltAz
   tangent plane about the Beammap source, with azimuth/elevation offsets in
   arcseconds. Persisted WCS, not array order, defines sign and handedness.
3. **Stable identity.** Detector-resolved products require explicit stable
   identity binding; table row or `det_N` slot alone is not an external
   detector identity.
4. **Photometry ownership.** TolProj selects the calibrator and estimates its
   per-array flux; TolTECA supplies it; Citlali must not silently become the
   catalog/source-selection authority.
5. **Prior ownership.** TolAPT produces soft Beammap priors. They are
   initialization/gating information, not exact detector identity or measured
   position truth, and blind fallback remains required under the recovered
   producer contract.
6. **Repository boundary.** Citlali owns reduction behavior and product
   conventions; `toltec_beammap` owns downstream analysis/calibration use;
   TolAPT owns matching and prior/reference-APT production.
7. **Loop boundary.** BEAM's internal measurement/fit iteration is not a
   general science fruit loop and must not absorb FRUIT recurrence or restart
   semantics.
8. **Dependency status.** Strong coordinate, response, precision,
   calibration, and validity claims remain conditional on ALIGN/AST, RTC, PTC,
   CAL, VAL, and MAP authority.

### Reusable scientific derivation

No dedicated, approved, implementation-independent SCI-BEAM core was found.
The prior work establishes boundaries and scientific questions, but it does
not provide a complete derivation of the beam/source model, likelihood,
parameter identifiability, finite-source coupling, response, covariance,
convergence, QC, or promotion rules. Unlike SCI-CAL and SCI-MAP, this package
cannot truthfully claim that its central mathematics is already frozen.

The external papers may prevent generic calibration and Gaussian-fit reasoning
from being invented anew, but none is TolTEC's contract. Their admission is an
owner choice.

### Conflicts and unresolved choices

1. The historical ledger includes flux/sensitivity calibration in BEAM, while
   current ownership puts calibrator selection and per-array flux upstream and
   calibration meaning in SCI-CAL. The package must distinguish a
   Beammap-derived candidate factor from promoted calibration authority.
2. Current products contain fitted offsets and derotated coordinate variants,
   while active ALIGN/AST authority is intentionally separate. BEAM may infer
   relative centroids in an admitted frame but cannot claim absolute placement
   without that upstream authority.
3. Current Gaussian-fit and prior machinery exists, but neither software nor a
   producer prior contract establishes the correct scientific likelihood,
   model family, bounds, convergence, or quality thresholds.
4. The current APT product mixes detector identity, beam, pointing,
   calibration, sensitivity, flags, and KIDs quantities. Whether v0.1 owns one
   atomic product or several typed promotion states is an owner decision.
5. `toltec_beammap` performs downstream calibration and APT updates, while
   Citlali emits an observation-local APT candidate. The contract must prevent
   either artifact from silently superseding the other.

### Material excluded from authorship

- Citlali, TolAPT, and `toltec_beammap` source code and current interfaces;
- the Beammap configuration authority review, audit ledger, all three raw
  handoffs, implementation findings, repairs, tests, A/B claims, validation,
  Unity evidence, and production status;
- tracked prior catalogs and historical measured APTs;
- the internal dossier and this complete recovery record; and
- active ALIGN/AST material or any inferred physical timing solution.

### Genuinely new work

After owner approval, a fresh author must derive and explain:

1. the estimand and admissible source/beam model family, including finite
   calibrator size and background treatment;
2. the likelihood or objective, support, parameterization, identifiability,
   degeneracies, uncertainty/covariance, and failure states;
3. the distinction between source centroid, detector focal-plane coordinate,
   telescope pointing, beam shape, amplitude, response, and calibration
   candidate;
4. scientifically meaningful iteration, convergence, fallback, and QC logic;
5. prior causality and softness, including when a prior may initialize, gate,
   regularize, or invalidate a fit;
6. product identity, validity, provenance, and promotion boundaries to CAL,
   TolAPT, and `toltec_beammap`; and
7. falsifiable limiting cases and tests without inspecting implementation.

## Proposed Author Reference Packet

No packet is approved yet. The minimal proposal is:

1. the owner-approved successor of [`SCOPE_BRIEF.md`](SCOPE_BRIEF.md);
2. a content-bound sanitized convention/ownership extract to be prepared from
   only the stable frame, unit, identity, lifecycle, and repository boundaries;
3. Bryan et al. 2018 and Golec and the TolTEC Collaboration 2024 as TolTEC
   instrument/context
   references; and
4. at most one owner-selected detector-beam/calibration methodology reference
   (AzTEC or SPIRE), or a different primary reference supplied by the owner.

CAL and MAP may be passed only as explicit conditional interface extracts, not
as evidence that their still-open scientific authority has frozen. Raw
implementation and audit material does not enter the packet.

## Investigator Attestation

Prior work was recovered before new derivation was commissioned. No dedicated
BEAM independent core was found, no existing audit or validation was repeated,
and no scientific author has been commissioned. The package remains at the
owner Scope Brief gate.
