# SCI-MAP-001 Unity campaign coordinator review — 2026-08-02

Status: verified preparation artifact; not dispatched; evidence-design and
producer gaps block launch; external evidence unsupplied

Review ID: `SCI-MAP-001-UNITY-001-COORD-REVIEW-001`

Package: `SCI-MAP-001`

Evidence request: `SCI-MAP-001-UNITY-001`

Machine-readable identity:
`SCI-MAP-001_UNITY_CAMPAIGN_IDENTITY_2026-08-02.json`

Identity-file SHA-256:
`97a2484d1599d01be27d11b0bb1c9b617b6e9cff8282f3cad60129f65858ec00`

## Coordinator disposition

Accept campaign commit
`1b824f138754eeb1856ae5f102027db4b31598be` as a locally verified
campaign-preparation artifact only. Do not classify it as external evidence or
as ready for human launch.

The campaign commit is the direct child of exact repair candidate
`ed28dafb37f9113c0d3c95297148157129a90886`, whose tree is
`cf75c36557178f351fb62781108a6f4b41b19225` and whose parent is exact selected
application base `9aae0e669384c5c0c0dda93debc194d6b8dac787`. The campaign
commit changes documentation and validation artifacts only; it does not change
the application candidate. The candidate remains unintegrated and
nonconformant pending external evidence and fresh re-audit.

The campaign package has Git tree
`dbf486e30c9b78ca16e05bccafc2d027562d0746`. Its 20-member `SHA256SUMS`
file has SHA-256
`ecf080cce98ad3aef6d6dbf52e72dd53be5d659a40285ec6c9bfbb0aee185a69`,
and `campaign.json` has SHA-256
`2cc7f31a5913af346e470c48f3f2e03863ef6072ced6bcbfa6175e26a508f1b3`.

## What passed

- Branch, commit, parent, tree, repair-base, and package identities match.
- The package verifier accepts all 20 named artifacts and leaves the worktree
  clean.
- The analysis program passes 506 synthetic self-checks covering fail-closed
  paths, product inventories, persisted F010 reconstruction, aliases, WCS,
  centered coaddition, 64 realizations, and registered seq/OpenMP comparisons.
- The seven fixed cases cover Point seq/OMP, Science coadd seq/OMP, Science
  empirical seq/OMP, and the repaired combined coadd-plus-empirical success
  case for all three arrays. `S-X-SEQ` retains a historical expected-failure
  jobkey only; success is now the required repaired outcome.
- Owner values fail closed while unresolved. The required remote alias is
  `unity_toltec`; no credential material was found.
- The driver contains no SSH, rsync, or other remote client and does not submit
  jobs. It emits plans whose execution is explicitly reserved for the owner.

These are local protocol and synthetic-program results, not Unity evidence.

## Blocking readiness gap

The campaign is not launch-ready because no approved frozen raw-manifest and
processed-term-ledger producer is included or otherwise present in the
reviewed repository. This is an engineering/evidence-preparation gap, not a
scientific-owner decision and not a request for the owner to hand-author data.

Before the campaign can emit its seven-case submission plan, one frozen
producer must generate and bind:

1. one Point raw-input manifest for observation 152389;
2. one Science raw-input manifest for ordered observations 152390 and 152392;
3. nine independent NPZ processed-term ledgers: one for each Point/Science
   observation and a1100/a1400/a2000 combination.

The ledgers must expose exact pre-output projection, detector/sample
eligibility, coefficient, signal, kernel, sample-duration, scan, and
realization-sign terms. They cannot be reconstructed from final FITS planes,
and the package correctly refuses to substitute templates or final products.
Until a producer and all eleven outputs exist and verify, there is nothing
responsible for the owner to deploy or submit.

There is a prior resource and engineering-scope gate before that producer
should be built. The pre-selection dimensions of the three selected local
validation observations imply 42,483,156, 836,181,713, and 838,417,991
network-sample-by-tone terms: 1,717,082,860 in total. The schema stores five
`int64`, two `uint8`, and four `float64` values per term, or 74 bytes. That is
127,064,131,640 uncompressed bytes (118.34 GiB), with the largest single
observation/array ledger approximately 33.2 GiB. Exact admitted cardinality may
be lower and redundant columns may compress, but the high-entropy physical
values will not make this a small metadata sidecar.

Do not implement this representation merely because the schema exists. First
perform a bounded evidence-design review that measures the exact cardinality
and determines whether redundant identities can be removed or the evidence can
be streamed, chunked, or represented more narrowly while preserving every
claim that truly requires independent per-term reconstruction. Any weaker
claim or reduced gate requires an explicit scope decision rather than a silent
optimization.

### Recommended bounded amendment — owner decision `MAP-UNITY-ED1`

The independent review finds the full per-term stored identity design to be
engineering creep beyond the original F012 request. The coordinator recommends
a bounded protocol amendment that:

- retains all seven exact-SHA cases and the complete output inventory, F010
  product/alias, WCS, centered-coadd, 64-realization, provenance, and
  seq/OpenMP checks;
- retains the exhaustive local F011 truth suite as the primitive-semantics
  authority;
- generates compact raw-input manifests automatically;
- uses streaming hashes and compact per-scan/per-pixel sufficient statistics
  for independent full-output reconstruction instead of storing every term;
- adds a deterministic actual-data primitive trace using preselected scans and
  hash-selected valid and flagged detectors from every active network; and
- escalates to broader per-term traces only when a discrepancy appears or a
  named re-auditor requests them.

This is an intentional evidence-coverage tradeoff, not a purely additive
change. It replaces exhaustive retention of every actual-data primitive term
with deterministic actual-data trace coverage plus streaming hashes and
sufficient statistics, while retaining exhaustive primitive-semantics coverage
in the local F011 truth suite. Owner approval would accept reduced external
term-level coverage and escalation on discrepancy in exchange for bounded
storage and processing; it would not reduce the seven-case execution,
full-product, F010, WCS, coadd, realization, provenance, or seq/OpenMP gates.

This recommendation is not yet authority. Do not amend the frozen campaign or
implement its successor until the owner approves, rejects, or modifies
`MAP-UNITY-ED1`.

The ordinary 22-field owner-values record is a later operational input. It
contains Unity checkout, project, path, executable, site, evidence-operator,
Slurm, dependency, and local-retrieval facts. The owner should not spend time
filling it until the producer gap is closed.

## Coordination snapshots and dependencies

The frozen package correctly retains preparation-time coordination head
`846128c8ee6dc27851bd6c71aeecbe4739e1d24a`, when ALIGN phase zero was active.
That is immutable historical provenance, not the live coordination state.
This review was performed against clean coordination head
`8fc9263a2f502656b51d32cb60655481f83509f1`, with ALIGN phase one and the CAL
EL25 confirmation active. Do not rewrite the package merely to refresh status;
bind a new execution-time coordination snapshot immediately before any future
human launch.

The conservative dependency boundary remains correct. This campaign cannot
close ALIGN, CAL, AST, PTC, VAL, or `SCI-MAP-001-F013`. The late CAL and AST
handoffs remain held for the fresh MAP re-audit: CAL handoff
`SCI-MAP-001-XAUD-002` has SHA-256
`2a70dae061827d18126d1bd6776f5a709d3315b5c1ae4f61d063116aeac76ac6`, and AST
handoff `SCI-MAP-001-XAUD-003` has SHA-256
`829ca1e5d9122461d011cae85ce0024d46b3bd67cb494779ea8d003a90c29aee`.
A later state change does not
invalidate the campaign mechanics by itself, but any change to the MAP
candidate, requested cases, product contract, TolProj/TolTECA setup semantics,
or acceptance program requires a new campaign revision and review.

## Prioritized next sequence

1. Obtain owner disposition `MAP-UNITY-ED1` on the recommended bounded
   evidence-protocol amendment.
2. After that decision, amend and freeze the exact protocol, then identify or
   implement the independent evidence producer required by the owner-approved
   protocol without changing the MAP estimator.
3. Freeze and locally validate that producer, its source contract, and its
   exact output schema; generate and independently verify every evidence output
   required by the owner-approved protocol.
4. Only then ask the owner to confirm the Unity operational values, inspect the
   emitted seven-case plan, execute the two human-only plans, and return the
   immutable evidence bundle.
5. Coordinator-review the returned bundle. Do not launch the fresh MAP
   re-audit until its exact evidence and dependency conditions are satisfied.

## Stop boundary

This review does not contact Unity, deploy or submit anything, change the MAP
candidate, integrate it into the application line, supply external evidence,
close F012/F013 or any upstream dependency, launch the re-audit, or expand
production use.
