# Human launch checklist — MAP-UNITY-ED2

This checklist is not launch authorization. The coordinator must first accept
the returned package and provide the exact implementation commit.

## Identity gate

- [ ] The local task is `codex/map-unity-ed1` at the exact coordinator-approved
  commit; its parent chain includes stop commit
  `3e014f11decbcf17ad372391e5e960e6c0c54461`.
- [ ] `scripts/verify-package.sh` passes and `SHA256SUMS.ed2` verifies every
  package member.
- [ ] The historic `SHA256SUMS` digest and the other three ED1 stop artifacts
  match `PROVENANCE.md`.
- [ ] The predecessor package tree is
  `dbf486e30c9b78ca16e05bccafc2d027562d0746` and its checksum-list digest is
  `ecf080cce98ad3aef6d6dbf52e72dd53be5d659a40285ec6c9bfbb0aee185a69`.
- [ ] The candidate is clean at
  `ed28dafb37f9113c0d3c95297148157129a90886`, tree
  `cf75c36557178f351fb62781108a6f4b41b19225`; no application/build file
  differs.
- [ ] The frozen owner-values file contains no placeholder, uses
  `unity_toltec`, and every path/identity check passes.

## One-build gate

- [ ] The ordinary `unity_release` candidate was compiled exactly once with
  disconnected FetchContent.
- [ ] Candidate SHA/tree, build inputs, compiler/dependencies, version output,
  and executable SHA-256 are recorded.
- [ ] CAP-POINT, CAP-SCIENCE, and all seven cases bind the same executable
  digest. No second or instrumented executable exists.

## Fresh project and staging gate

- [ ] Both project destinations are absent before `tolproj init-test`.
- [ ] The Point and Science JSON specifications match the package byte-for-byte
  and the realized `project.yaml` inventories are checked.
- [ ] The explicit canonical raw root has been owner-verified as `/work/toltec`.
- [ ] Each required raw basename resolves to one canonical regular file with
  matching observation identity; there are no missing, ambiguous, unresolved,
  duplicate-basename, or non-regular sources.
- [ ] Only individual raw-file symlinks were installed at fresh destinations;
  no directory link or raw copy occurred.
- [ ] Point has only `apt_152389_matched.ecsv`; Science has only
  `apt_152390_matched.ecsv` and `apt_152392_matched.ecsv`; exactly one
  `ppt_*.ecsv` is bound for each of 152389/152391/152393.
- [ ] Every copied ECSV and link target has a recorded size, identity, resolved
  path, and SHA-256. `tolproj copy-raw` is prohibited from this point onward.

## Capture gate

- [ ] CAP-POINT targets only 152389. CAP-SCIENCE targets ordered 152390/152392;
  152389/152391/152393 are only pointing support.
- [ ] Each capture's fully merged config, included fragments, and numbered-file
  order were compared with P-SEQ or S-E-SEQ and differ only at enabled/full/all.
- [ ] Before CAP-POINT, CAP-SCIENCE, and every later output stage, all generated
  roots, directories, regular files, and symlinks under all governed roots are
  inventoried without following symlink payloads.
- [ ] The before-stage record is under
  `compact-groups/_campaign/resource-records` and binds current logical and
  allocated usage, selected-filesystem availability, and the frozen local
  planning estimate. That estimate is not a full/all-PTC serialization bound
  or a guarantee of realized use.
- [ ] The owner reviewed each before-stage record and separately invoked only
  the next approved stage. The owner stopped if observed use approached or
  exceeded 214,748,364,800 bytes, or capacity was inadequate. No evidence was
  deleted, cleaned up, or reused from cache to make room.
- [ ] The matching after-stage record is under the same governed compact root.
- [ ] Each full PTC is binary64 full/all with signal, flags, kernel, detector
  pointing, APT columns, per-scan weights, and complete scan indices.
- [ ] PTC `SAMPRATE` equals native `telescope.fsmp`. The digested raw provenance
  separately supplies finite-positive `telescope.d_fsmp`, decimal/hex binding,
  bit-equal `1/d_fsmp`, a positive integral factor with bit-exact
  `d_fsmp == fsmp / factor`, and realized scan/cardinality cross-check.

## Compact evidence gate

- [ ] Exactly nine observation/array groups exist in campaign order.
- [ ] Each group binds its raw manifest, capture digest, mapmaking identity,
  native/effective rates, scan/detector order, population, and all complete
  streaming digests.
- [ ] Sufficient-statistic arrays are scan-first and have exact binary64/int64
  dtypes and shape; 64 pinned realization signs are present.
- [ ] First/lower-middle/last scans and valid/flagged SHA-min representatives
  cover every active network, with typed absence when a class does not exist.
- [ ] Fixed trace budgets hold and no artifact contains an accidental complete
  primitive-term axis.
- [ ] Any focused expansion has a named discrepancy and re-auditor, stays under
  the fixed term bound, passes the two digest-identical passes, and reads the
  same retained capture. Otherwise no expansion was run.

## Seven-case and analysis gate

- [ ] The case order is exactly P-SEQ, P-OMP, S-C-SEQ, S-C-OMP, S-E-SEQ,
  S-E-OMP, S-X-SEQ; observations, arrays, coadd/products policy, CPU counts,
  tolerances, 64 signs, inventories, WCS, provenance, and scientific gates are
  unchanged from `repair-ed28dafb`.
- [ ] All seven reductions exit zero with zero unexpected error-level messages.
- [ ] Sequential/OpenMP comparisons, independent signal/weight/kernel/noise,
  F010 products and aliases, centered coadds, support floor, WCS, provenance,
  and exhaustive local F011 analysis complete.
- [ ] Final FITS payloads were comparison subjects, never reconstruction
  authority.
- [ ] ANALYSIS and FINAL-BUNDLE each have before/after Unity-root records; the
  resource-completion verifier confirms preparation, the two captures, nine
  groups, any requested expansion, analysis, and bundle.
- [ ] The bounded returned bundle excludes full PTC payloads unless the
  coordinator specifically requested a focused expansion.

## Retention and handback gate

- [ ] CAP-POINT and CAP-SCIENCE remain present through fresh MAP re-audit and
  requested focused expansion.
- [ ] No cleanup command ran and cleanup eligibility remains false.
- [ ] The final-plan temporary TAR remains inside the governed compact return
  directory and is accounted for by the final post-stage record; this package
  does not remove, reuse, or replace it.
- [ ] Result/resource records contain both logical and allocated usage and an
  exhaustive generated-file inventory.
- [ ] The returned bundle and digest are retrieved with `unity_toltec`; the
  coordinator receives exact identities and all negative confirmations.
- [ ] No finding/dependency is closed, repair integrated, re-audit launched, or
  production expanded by the collection workflow.
