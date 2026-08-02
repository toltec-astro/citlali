# SCI-MAP-001 MAP-UNITY-ED1 successor-campaign handoff — 2026-08-02

Status: owner-authorized bounded implementation handoff; dedicated task may
amend the evidence protocol in a new sibling package; no Unity action or MAP
application edit is authorized

Package: `SCI-MAP-001`

Evidence request: `SCI-MAP-001-UNITY-001`

Owner decision: `MAP-UNITY-ED1`

## Frozen inputs

- Repair branch: `codex/repair-sci-map-001`.
- Repair candidate: `ed28dafb37f9113c0d3c95297148157129a90886`.
- Repair tree: `cf75c36557178f351fb62781108a6f4b41b19225`.
- Selected application base:
  `9aae0e669384c5c0c0dda93debc194d6b8dac787`.
- Campaign-preparation commit:
  `1b824f138754eeb1856ae5f102027db4b31598be`.
- Campaign-preparation tree:
  `e0f98af4ace754f90c682960a44af590cfa5013d`.
- Existing frozen package tree:
  `dbf486e30c9b78ca16e05bccafc2d027562d0746`.
- Existing package checksum-list digest:
  `ecf080cce98ad3aef6d6dbf52e72dd53be5d659a40285ec6c9bfbb0aee185a69`.
- Existing campaign JSON digest:
  `2cc7f31a5913af346e470c48f3f2e03863ef6072ced6bcbfa6175e26a508f1b3`.
- Coordinator review digest:
  `a1931af55f684c89cdaec9cd55c481c82287e0d90444f3e916489912eb83a484`.
- Coordinator identity digest:
  `97a2484d1599d01be27d11b0bb1c9b617b6e9cff8282f3cad60129f65858ec00`.
- Coordination parent before this decision:
  `291d8771458a02599a95ea7b81c005afc0232178`.

The existing package at
`validation/campaigns/SCI-MAP-001-UNITY-001/repair-ed28dafb` is immutable
historical preparation evidence. Do not edit, delete, regenerate, or relabel
it. Create the sibling successor package at
`validation/campaigns/SCI-MAP-001-UNITY-001/repair-ed28dafb-ed1` with request ID
`SCI-MAP-001-UNITY-001` and new revision
`repair-sha-ed28dafb-ed1-2026-08-02`.

The dispatch prompt will supply the exact coordination decision-content commit
after this handoff and the owner decision are committed. The successor must
bind that post-approval authority; the pre-approval coordination parent above
is provenance only.

## Objective

Implement and locally verify the smallest independent evidence representation
that makes the unchanged seven-case exact-candidate campaign operationally
preparable without storing the preliminary 1,717,082,860-term, 118.34-GiB
uncompressed ledger design.

The task owns the successor evidence package and validation-only producer. It
does not own the MAP estimator, repair acceptance, Unity execution, or re-audit.

## Invariant reduction matrix

Preserve exactly:

| Case | Observations | Mode | Coadd | Products | Threads |
| --- | --- | --- | --- | --- | --- |
| `P-SEQ` | 152389 | Point/pointing | no | enabled | 1 |
| `P-OMP` | 152389 | Point/pointing | no | enabled | 6 |
| `S-C-SEQ` | 152390, 152392 | Science | yes | disabled | 1 |
| `S-C-OMP` | 152390, 152392 | Science | yes | disabled | 16 |
| `S-E-SEQ` | 152390, 152392 | Science | no | enabled | 1 |
| `S-E-OMP` | 152390, 152392 | Science | no | enabled | 16 |
| `S-X-SEQ` | 152390, 152392 | Science | yes | enabled | 1 |

All cases cover a1100, a1400, and a2000. All must exit zero. Preserve the
coverage cuts, 64-realization seed and count, numbered-config order, resources,
product counts, exact inventories, product contracts, WCS checks, provenance,
support-floor checks, and seq/OpenMP bounds of the frozen campaign unless a
literal mechanical change is required by the successor path or revision. Any
such change must be named, justified, and shown not to weaken a gate.

## Required successor evidence design

### 1. Automatically generated raw authority

Generate one digest-bound Point manifest for observation 152389 and one
digest-bound Science manifest for ordered observations 152390 and 152392.
Their producer, invocation, raw/KIDs/APT/calibration/pointing/projection facts,
array/network and detector order, scan/sample cardinalities, target, map,
sample-rate, and response identities must be explicit and fail closed.

No owner hand-authoring is allowed. Unresolved Unity-only paths remain typed
owner inputs and must not be guessed locally.

### 2. Compact reconstruction authority

Replace persisted per-term ledgers with a streaming producer that binds term
order and population through digests while accumulating only the sufficient
statistics needed for the independent observation, realization, and coadd
claims retained by the campaign. Stored cardinality may scale with scans, map
pixels, arrays, cases, and a fixed trace budget; it must not scale with the full
detector-by-sample term population.

Provide separately identified compact streaming-digest/sufficient-statistic
sets for each of the nine observation/array combinations: Point 152389 and
Science 152390/152392, each at a1100/a1400/a2000. These are compact successor
evidence groups, not full-term NPZ ledgers.

Each compact group must provide the typed information needed to reconstruct
the retained claims without final FITS numerical payloads as inputs. At
minimum, freeze and test:

- per-scan/per-pixel binary64 signal numerator, weight, kernel numerator,
  upstream-eligible exposure, and retained exposure;
- aggregate `int64` geometric and contributing hits;
- the pinned `int8 [nscan, 64]` realization signs;
- exact scan/pixel/domain counts; and
- domain-separated canonical stream digests binding the complete primitive
  order and population.

The analyzer must use these compact groups to reconstruct signal, weight,
kernel, every retained F010 plane and alias, all 64 realizations, and centered
coadds. The implementation may refine the exact field factoring only when it
proves equivalent or stronger coverage and records the change explicitly.

Document every statistic's identity, dtype, units, shape, indexing, reduction
order, finite/missing policy, and exact claim it supports. Do not use a final
FITS plane as its own independent input authority.

### 3. Deterministic actual-data primitive trace

Define and preregister this minimal deterministic selection rule before reading
reduction outputs:

- select the first, middle, and last scan identity, deduplicated for short
  observations;
- for every active network, select the SHA-256-min valid detector and the
  SHA-256-min flagged detector where each class exists;
- define that detector-state class from the frozen detector/APT flag authority;
  retain sample flags as a separate primitive field and do not use them to
  silently reclassify the detector;
- record an explicit class-absence fact instead of fabricating a member; and
- retain all declared primitive fields for the selected detector-by-scan
  sample sequences.

The domain-separated selection hash must bind the candidate, campaign
revision, raw-manifest digest, observation, array/network, scan class,
detector-state class, and detector UID. The trace must also:

- span every active network in each selected observation;
- include the preregistered scan identities;
- include the hash-selected valid and flagged detector classes above;
- preserve the required sample/eligibility/coefficient/signal/kernel/duration/
  scan/realization-sign terms for the selected trace;
- bind the selection seed and inputs to the candidate, campaign revision, raw
  manifest, observation, array/network, scan, and detector identities; and
- have a fixed, measured trace budget with no routine per-sample identity
  product.

The task must justify the exact fixed selection policy as an engineering
falsification surface. It must not claim exhaustive external term coverage.
A schema gate must reject any stored full-term axis and bound ordinary arrays
by `nscan * npixel` plus this declared trace budget.

### 4. Escalation on discrepancy

Provide a fail-closed, separately invoked focused expansion that can emit
broader term traces only for a named observation/array/network/scan/detector or
pixel discrepancy, or when a named re-auditor requests it. It must never
silently expand to an unbounded full-population artifact.

## Required package work

Create the new sibling revision with all internally consistent replacements or
successors for the campaign manifest, source/result schemas, evidence contract,
analyzer, preparation driver, wrappers, verifier, owner template, runbook,
launch checklist, provenance, evidence-boundary note, and `SHA256SUMS`.

The package must:

- preserve the driver-only plan boundary: no SSH client and no job submission;
- reject unresolved, missing, stale, symlinked, overwritten, or identity-
  mismatched state;
- make the current large-ledger representation neither required nor silently
  accepted by the successor;
- make the compact coverage tradeoff visible in machine and human records;
- preserve every named product-level and sequential/OpenMP gate;
- retain the exact scan-farm gamma lane as external N/A and the local F011
  suite as its exact policy authority; and
- remain unable to close MAP findings or CAL/AST/PTC/VAL dependencies.

## Resource and proportionality gates

Use representative local metadata or safe local fixtures to report:

1. exact input term cardinality by observation and array/network;
2. projected and measured stored bytes for every successor evidence artifact;
3. peak memory and elapsed time for producer/analyzer self-check fixtures;
4. asymptotic stored cardinality; and
5. the reduction relative to the 118.34-GiB preliminary v1 ledger estimate.

Do not invent a numerical storage ceiling. The design passes this gate only if
stored evidence is structurally sublinear in the full term population and the
task demonstrates a clearly practical bounded package. If that cannot be
shown, stop with the smallest owner decision brief.

## Application-source and immutability gates

- No application file under `include/`, `src/`, or another application build
  surface may change.
- The exact Citlali executable source remains repair candidate `ed28dafb...`.
- The existing frozen campaign package must remain byte-for-byte identical to
  package tree `dbf486e3...` and pass its original 20-member verifier.
- The successor package must have a new revision, path, checksum list, package
  tree, and coordinator-reviewable identity.
- Do not merge, rebase, integrate, or push.

## Local validation gates

At minimum:

1. run the original package verifier and prove the old package is unchanged;
2. run the successor package verifier;
3. run the successor analyzer/driver self-checks, syntax and compilation
   checks, and focused tests for all fail-closed paths;
4. exercise deterministic selection stability, every-network coverage,
   valid/flagged selection, empty-class handling, digest tamper, reordered
   input, non-finite input, missing authority, chunk-size invariance, and
   discrepancy expansion;
5. prove the full seven-case/product contract remains unchanged;
6. prove the successor does not require or accept the nine full NPZ ledgers;
7. prove small-fixture numerical parity between the successor compact
   reconstruction and the original full-ledger reconstruction;
8. prove the driver cannot contact Unity or submit a job; and
9. obtain an independent read-only review of claims, resource bounds,
   identities, and stop conditions.

Use `$HOME/tolteca/bin/python`. Local synthetic or metadata results are not
Unity evidence.

## Stop conditions

Stop and return to the coordinator if the work requires:

- any MAP/application-source edit;
- a change to observations, arrays, cases, configs, products, or numerical
  acceptance gates;
- a new scientific or operational tradeoff beyond `MAP-UNITY-ED1`;
- stored evidence proportional to all primitive terms;
- final outputs as the sole authority for an independently reconstructed fact;
- owner hand-authoring of evidence;
- Unity access, transfer, build, reduction, or Slurm action;
- repair integration, finding/dependency closure, re-audit, or production
  expansion.

## Handback

Return one coherent task commit and, if needed, a final identity-binding commit
with:

- branch, exact commit(s), parent(s), and tree(s);
- successor path, revision, package tree, member count, and checksum-list
  digest;
- proof of unchanged application source and unchanged v1 package;
- exact changed-path inventory;
- verifier, self-check, focused-test, and independent-review results;
- measured cardinality/storage/runtime report;
- any unresolved evidence gap or owner-return item; and
- explicit confirmation that Unity was not contacted, external evidence was
  not supplied, no MAP finding/dependency was closed, and re-audit was not
  launched.

Stop after this handback. The coordinator reviews the exact successor before
asking the owner for operational values or preparing a human launch.
