# SCI-MAP-001 existing-corpus evidence closeout

## Disposition

The human owner already executed the seven `SCI-MAP-001-UNITY-001` cases and
returned them under
`/Users/gwilson/work_toltec/local_data/2026-ENG-citlali-MAP`. **Do not launch,
request, or prepare another reduction campaign.** This note reconciles that
read-only corpus with the exact repair candidate; it is not an independent
re-audit, finding closure, or production admission.

The returned reduction indexes identify
`v4.0.0-3628-ged28dafb`, binding the products to repair candidate
`ed28dafb37f9113c0d3c95297148157129a90886`. Every one of the seven captured
executables independently hashes to
`693c14898faa1d41a854030b86cdde2729bf121442eb8427feffb4d4e57686c5`.
The executed minimal transfer package identifies candidate tree
`cf75c36557178f351fb62781108a6f4b41b19225`; its 13-entry `SHA256SUMS`
verifies, and each installed `99_zzz_sci_map_case.yaml` is byte-identical to
the corresponding packaged overlay.

The cases completed on 2026-08-03. They were analyzed locally on 2026-08-05.
No explicit retrieval manifest exists, so the exact transfer time is
unavailable. Local inode timestamps only show that the case trees were present
locally between approximately 11:17 and 11:23 EDT on 2026-08-03; those
timestamps are observations, not durable transfer authority.

## Seven-case evidence matrix

Product counts are `observation map/noise; coadd map/noise` FITS files.

| Case | Accepted job and completed interval | Realized configuration | Products | Evidence use | Explicit gap or condition |
| --- | --- | --- | ---: | --- | --- |
| `P-SEQ` | `62569979`; 13:29:10–13:31:51 UTC | ordinary naive; seq; 1 thread; cut 0.1; coadd off; products on | 3/3; 0/0 | Direct observation bundle, F010 planes/aliases, provenance, WCS, and 64-realization files; sequential point reference | No independent raw manifest or per-sample ledger |
| `P-OMP` | `62569983`; 13:29:20–13:30:59 UTC | ordinary naive; OMP; 6 threads; cut 0.1; coadd off; products on | 3/3; 0/0 | Same persisted evidence; paired point SEQ/OMP lane | No independent raw manifest or per-sample ledger |
| `S-C-SEQ` | `62570206`; 13:50:12–15:01:03 UTC | ordinary naive; seq; 1 thread; cut 0.5; coadd on; empirical products off | 6/0; 3/3 | Observation bundle plus atomic coadd-admission, centered-embedding, coadd-map, and coadd-realization evidence | Observation realizations are not serialized in this products-off lane; sibling products are supporting evidence only |
| `S-C-OMP` | `62570232`; 13:50:43–14:00:54 UTC | ordinary naive; OMP; 16 threads; cut 0.5; coadd on; empirical products off | 6/0; 3/3 | Same persisted evidence; paired observation/coadd SEQ/OMP lane | Same independent-reconstruction limitation |
| `S-E-SEQ` | `62570227`; 13:50:23–14:59:44 UTC | ordinary naive; seq; 1 thread; cut 0.5; coadd off; products on | 6/6; 0/0 | Direct empirical F010/statistical products and observation realizations; sequential science reference | A later canceled job `62600841` overwrote root-level transient files, but did not modify the accepted `redu00`, accepted stdout, or its candidate-bound provenance |
| `S-E-OMP` | `62570235`; 13:50:56–13:59:32 UTC | ordinary naive; OMP; 16 threads; cut 0.5; coadd off; products on | 6/6; 0/0 | Same persisted evidence; paired empirical SEQ/OMP lane | No independent raw manifest or per-sample ledger |
| `S-X-SEQ` | `62570230`; 13:50:31–15:02:40 UTC | ordinary naive; seq; 1 thread; cut 0.5; coadd on; products on | **6/0; 3/3** | Repaired all-enabled-plus-coadd execution completed; direct observation/coadd maps, F009/F010 provenance, atomic admissions, and coadd realizations are present | Six same-case observation-noise FITS expected by the later frozen package are absent; see below |

All accepted scheduler logs record the executable digest above, `citlali is
done`, `Citlali Process finished`, and `PipelineRuntime: work's done!`, with no
error or critical record. Requested, effective, and realized policy/thread
facts agree with the case matrix. The noise provenance records
`boost::random::mt19937`, fixed seed 5489, and 64 realizations per scientific
map. Mapmaking, runtime, noise-product, coadd, config-source, product-index,
and diagnostic provenance are present.

## Requirement classification

### Satisfied by returned evidence

- All seven required case and observation identities, all three arrays, exact
  jobkeys, successful completion records, executable digest, and candidate
  version are present.
- Installed case overlays, merged configurations, and config-source manifests
  preserve the realized later-overlay semantics. The executed minimal package
  used the ordinary nine TolProj/TolTECA numbered sources followed by the
  deliberate `99_zzz_sci_map_case.yaml`; it must not be rewritten after the
  fact to resemble the later idealized package.
- Persisted ordinary-naive maps expose the typed F010 plane bundle,
  compatibility aliases, shape and WCS relationships, admitted bundle and
  raw-parent digests, required-companion inventories, thresholds, response
  identity, and realized product inventories. Coadd provenance records ordered
  atomic admissions, centered integer embeddings, coefficient stages, and
  exposure totals.
- The directly serialized noise files contain one metadata-only primary HDU
  plus realization HDUs 0 through 63.

### Derivable and routed to fresh re-audit

- Exhaustive persisted-plane, alias, binary-mask, within-file WCS, coadd
  recombination, and registered SEQ/OMP numerical comparisons can be derived
  from the returned products without duplicating the multi-gigabyte corpus.
- `S-E-SEQ` and `S-E-OMP` observation realizations and the sibling non-noise
  identities can support interpretation of the products-off and `S-X-SEQ`
  lanes. They are supporting comparisons, not substitutes for missing
  same-case bytes or independent sample authority.
- Successful output and atomic-admission provenance exercise the F009 success
  path. Local mismatch, tamper, and failure-injection tests remain the
  rejection/rollback evidence.

### Genuinely unavailable

- Independent raw-input byte manifests, exact sample-membership authority,
  per-sample ledgers, and full processed-time-chunk primitive captures.
- The ideal frozen wrapper's owner-values, request-root preflight, case-log
  JSON, submission/exit record, Slurm accounting/MaxRSS, environment and
  pre/post integrity manifests, frozen result collection, and explicit
  retrieval record.
- Submission-time executable selection: accepted logs report `job-start
  fallback`. The exact executed digest is nevertheless recorded in each log
  and matches every downloaded snapshot binary.

These unavailable lanes are evidence limitations, not instructions to repeat
the reductions.

### Unnecessary duplication

- Another seven-case campaign or another copy of the returned products.
- Reconstruction of wrapper-only Slurm metadata after the fact.
- Retrofitting the executed ten-source later-overlay layout to the later
  nine-source idealized campaign layout.

## Open discrepancies preserved for re-audit

### `S-X-SEQ` observation realization serialization

The later frozen campaign expected six observation-noise FITS plus three
coadd-noise FITS. The returned `S-X-SEQ` tree contains zero observation-noise
FITS and three coadd-noise FITS. Its effective config has empirical products
enabled. `noise_products_provenance.yaml` records 384 observation and 192
coadd realizations generated, but `realization_image_write_count` is 192,
exactly the coadd set.

The missing same-case observation realization bytes are unavailable. Sibling
`S-E` products may be cited as supporting evidence only; they are not promoted
to same-case evidence or closure. Fresh re-audit must decide whether the
remaining exact-candidate evidence is sufficient.

### Typed WCS serialization and Stokes identity

A frozen-analyzer spot check of `S-X-SEQ`, observation 152390, `a1100`, found
that cardinality, string WCS cards, shape, epoch, and orientation checks pass,
but the typed-to-FITS numeric adapter does not meet its exact expectation:

| Card | Persisted FITS | Typed adapter expectation |
| --- | ---: | ---: |
| `CRVAL1` | `187.0463` | `187.04632568359375` |
| `CRVAL2` | `44.09356` | `44.09355926513672` |
| `CDELT1` | `-0.0005555556` | `-0.0005555555690079927` |
| `CDELT2` | `0.0005555556` | `0.0005555555690079927` |
| `CRVAL3` | `272538600000.0` | `272538599424.0` |

The sky-coordinate differences exceed the registered `1e-12` degree adapter
bound. The typed slot records `stokes_identity=0`, while a separate frozen
analyzer assertion expects `1`; that analyzer derives `CRVAL4` from the
recorded zero. This closeout does not loosen the tolerance, change a
scientific convention, or patch either side. Fresh re-audit must adjudicate
whether this is FITS serialization loss, an application-contract defect, or a
verifier-contract inconsistency.

## Read-only reconciliation methods

The closeout used only the downloaded corpus and local candidate authorities:

```sh
find <case-root> -type f
du -sh <case-root>
shasum -a 256 <each .tolproj snapshot executable>
cd <corpus>/repair-ed28dafb-minimal && shasum -a 256 -c SHA256SUMS
rg 'runtime selection|citlali is done|Process finished|work.s done' <accepted-log>
$HOME/tolteca/bin/python  # read-only YAML/FITS inventory and frozen-adapter checks
```

The checks covered exact case/product cardinality, overlay and merged-config
identity, executable/version/log identity, provenance inventories, and
representative FITS contract inspection. No external product was copied into
the repository. The ideal frozen analyzer remains unchanged because its
independent manifests, ledgers, and wrapper records were not captured by the
executed minimal package.

## Workflow effect and next gate

The human execution requested by F012 has occurred, and the returned corpus is
now identified and reconciled as the candidate F012 evidence corpus. This
advances the workflow from **repair awaiting human execution** to **repair and
external evidence awaiting fresh independent re-audit**. It does not itself
satisfy F012, close F009/F010, establish conformance, or alter production
status.

Create a fresh `codex/reaudit-sci-map-001` worktree from the exact committed
repair/evidence-closeout SHA. The independent re-auditor must examine:

1. repair implementation F001-F011 and the local truth/gate evidence;
2. this closeout and the read-only seven-case corpus at the supplied local
   path;
3. persisted F009/F010 plane, alias, WCS, coadd, realization, and SEQ/OMP
   relationships;
4. the unavailable independent-ledger and wrapper lanes;
5. the `S-X-SEQ` observation-realization serialization gap; and
6. the typed-WCS/Stokes discrepancy above.

Only that re-audit may decide F012 sufficiency, finding status, conformance,
or production disposition. F009 and F010 remain
`addressed_pending_reaudit`. F013 remains conditioned on the named ALIGN, CAL,
AST, PTC, and VAL work; this MAP evidence closes none of those dependencies.
