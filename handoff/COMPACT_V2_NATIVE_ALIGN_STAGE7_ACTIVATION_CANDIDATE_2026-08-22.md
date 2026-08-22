# Compact-v2 Native ALIGN Stage 7 Activation Candidate — 2026-08-22

## Disposition

Stage 7 is implemented and locally validated as an **activation candidate** at
exact commit `36f6ada25d06f2236dfcd279d53c6afc40298cb1`, tree
`40099545347326aed03df7be22bcc7cfe74e0e7d`.

This is not an accepted Stage 7 result and is not a production-readiness
claim. Acceptance remains blocked on the five owner-run Unity campaigns in
the accepted reconstruction plan. No Unity evidence has been inferred or
prefilled by this record.

## Activated boundary

An ordinary Science observation now takes the native-required route only when
one complete verified compact-v2 matched detector relation and its exact
native alignment/pointing carriers are present. Partial authority fails
closed. A native-required Science scan:

1. gathers the exact measured detector scan from the retained raw sources;
2. derives detector map identities from the typed resolved map grouping;
3. runs the established RTC body on packet-contiguous network-local runs;
4. runs the established PTC body on admitted typed cohorts;
5. projects measured cells only through the admitted native pointing plan;
6. populates the established naive or JINC map products; and
7. commits the complete scan lineage only with the map occurrence.

This is an exclusive route. The legacy common-grid RTC/PTC pass does not run
before or after the native RTC/PTC pass. Legacy RTC/PTC TOD and diagnostic
files are not advertised or created for a native-required observation.

The production path uses the existing RTC, PTC, naive, and JINC numerical
bodies through the Stage 4–6 adapters. It does not change their numerical
kernels, defaults, or established valid-path accumulation arithmetic.

## Deliberately asymmetric mode policy

| Mode | Candidate behavior |
| --- | --- |
| Science | Native-required only with the exact matched-v2 relation/carrier pair; otherwise legacy-inactive |
| Pointing | Fails closed if native authority is present because the current low-level reduction identity also represents OOF |
| OOF | Native activation unavailable pending a distinct mode contract |
| Beammap detector/automatic | Remains the raw/APT producer and cannot request matched-consumer lineage |
| Beammap non-detector | Retains the existing calibration-table lane and cannot request matched-consumer lineage |

The accepted plan allowed reviewed Science and Pointing activation. This
candidate activates Science only. Promoting Pointing while its low-level mode
identity is collapsed with OOF would be an inference across an unresolved
mode boundary, so it remains fail closed.

## Product and provenance contract

The raw-timestream provenance sidecar embeds
`citlali-native-cohort-product-provenance-v2`. The observation binding covers
the matched bundle/relation identity, raw manifest, alignment plan, and native
pointing plan. Each committed scan covers:

- observation binding and exact scan/chunk operation identity;
- ordered RTC run support, selected anchors, detector partitions, native
  sample identities, and ORed input flags;
- admitted PTC groups and detector memberships;
- transactional input/output revision transitions;
- ordered detector map identities;
- the exact eligible detector-weight digest;
- the eligible native input digest;
- the map product occurrence and product identity; and
- for JINC, both the admitted processing-configuration digest and the actual
  native scan processing-trace digest.

Incomplete, missing, stale, foreign, duplicate, or partially committed
lineage rejects publication. Final product-index replacement is deterministic
and atomic per index file and is attempted only after required products exist.

## Candidate execution domain

The first Unity campaigns must use a deliberately bounded Science
configuration. Native-required preflight rejects any operation whose support,
selection, iteration, or product lineage has not yet been reconstructed. In
particular, the candidate requires:

- mapmaking disabled or method `naive`/`jinc` only;
- polarimetry disabled;
- extinction correction and RTC kernel products disabled;
- impulsive-coincidence and coherent-IQ cross-network observers disabled;
- raw line audit and AltAz destriping disabled;
- raw and processed lower/upper TOD inverse-variance detector cuts all equal
  to zero;
- learning, noise maps, TOD output, fruit loops, PTC second pass, and weight
  validation disabled;
- one established PTC cleaning grouping at most;
- PTC and variance source-mask radii equal to zero;
- an exact `duplicate_tone` APT column with every value zero; and
- scan science bounds identical to the loaded outer-context bounds.

These are candidate-domain restrictions, not silent behavior changes. An
ordinary Science configuration with nonzero default detector outlier cuts,
duplicate-tone exclusions, noise maps, or outer filter context will fail
before scan mutation. The Unity candidate config must therefore record each
intentional override explicitly.

## Local validation

The local build used AppleClang 21.0.0, Release mode, OpenMP 5.1, and the
accepted disconnected dependency-source set.

| Gate | Result |
| --- | --- |
| Stage 7 focused routing, lineage, publication, and execution cases | passed |
| Complete SCI-ALIGN executable | 66/66 passed at OpenMP thread counts 1, 2, 4, and 8 |
| Public-header isolation | passed |
| Complete CTest | 788/788 runnable passed; one established disabled test not run |
| Existing current-JINC suite | 22/22 passed unchanged |
| Existing production FITS suite | 35/35 passed unchanged |
| Baseline-tool unit suite | 203/203 passed |
| Required config preflight | 127/127 unit tests; four mode kits; 8/8 compact cases; zero skips/gaps; all audits passed |
| Frozen raw-execution census | 48 records; digest `efd347b41857542b770de90c9c383a254fbb5a4890988f3b1da43f27de4bcf9f`; zero review-required; no drift |
| Validation ledger | valid, 60 records |
| Intended-science-change ledger | valid, 3 changes and 5 integration commits |
| Phase 5 readiness | valid and still `preparing`; not promotion-ready |
| Session-exit audit | 733 dependencies; zero library/CLI exits; zero growth |
| Frozen gap fixture | SHA-256 `a4dfdfe4b45638952f57f5f258badfab84f5d6ce1d022abfefc47a9e84091701` |
| Diff and log hygiene | `git diff --check` passed; zero unexpected error-level messages |

The exact candidate CLI is
`v4.0.0-3678-g36f6ada25` with binary SHA-256
`95c53af60db30a353b6bfd8e2badcbb368a16d112fb04f876218183cdab84a7a`.
The local dependency identity is reported as `kids 04088da-dirty`; the
Citlali source worktree itself was clean at the implementation boundary.

CTest discovered 789 tests. The established disabled
`MapFitterLifecycle.ExactProductSequence` test did not run; all 788 runnable
tests passed.

## Required owner-run Unity campaigns

All five campaigns must use the pushed exact candidate, a freshly configured
and built CLI, retained merged configuration, and retained source/binary/input
identity. Because Pointing remains deliberately inactive, the first two
consumer campaigns must be Science.

| Campaign | Required comparison and pass condition |
| --- | --- |
| 1. Native-gap Science | Run a small Science observation with a verified matched-v2 bundle and retained gap diagnostics. The native sidecar must cover every scan and prove no sample synthesis or gap bridging. |
| 2. Identical-time/no-gap | Compare candidate native-required Science against the legacy-inactive Stage 6 parent using the same raw inputs, verified bundle, merged config, and products. Any numerical delta requires explanation and review. |
| 3. Same-scan naive/JINC | Run the same admitted native Science scan once with naive and once with JINC, changing only the mapmaking method and its method-specific settings. Both must retain complete lineage and expected products. |
| 4. Detector/automatic Beammap | Re-run the accepted relevant Beammap regression and prove the raw/APT producer lane needs no matched-v2 consumer relation and retains the accepted numerical behavior. |
| 5. Non-detector Beammap | Re-run an existing array/network/frequency-group calibration-table Beammap and prove that the lane is unchanged and acquires no matched-consumer lineage. |

For every run retain:

- full Citlali commit and tree, CLI version, binary SHA-256, and KIDs identity;
- requested and merged YAML plus SHA-256;
- compact-v2 bundle/relation occurrence and component identities where used;
- raw source manifest, paths, byte counts, and SHA-256 values;
- complete stdout/stderr log and scheduler command/resource identity;
- raw-timestream/native-cohort provenance sidecar;
- deterministic product index and complete retained-product inventory; and
- reduction audit plus the appropriate numerical/product comparison report.

Acceptance requires all expected scans and products, zero unexpected
error-level messages, no unexplained detector/flag changes, and no unreviewed
numerical delta. The large FITS products need not be transported to the local
workstation when their Unity-side path, byte count, checksum, headers, audit,
and comparison evidence are retained.

## Unity preparation after push

The owner controls Unity. Use the required SSH alias and do not reuse the
pre-amend `8918189e4` candidate:

```bash
ssh unity_toltec
cd ~/work_toltec/citlali_dev/citlali_refactor
git fetch origin
git switch codex/converge-apt-align-jinc
git pull --ff-only
git rev-parse HEAD
git rev-parse 'HEAD^{tree}'
cmake -S . -B build
cmake --build build --target citlali_cli -j 8
build/bin/citlali --version
sha256sum build/bin/citlali
```

Before launching a reduction, confirm the source identity is exactly
`36f6ada25d06f2236dfcd279d53c6afc40298cb1` and the tree is exactly
`40099545347326aed03df7be22bcc7cfe74e0e7d`. The Unity binary SHA-256 is
expected to differ from the local macOS binary and must be recorded as its own
identity.

## Stop boundary and next action

The candidate commit is complete. The next action is to push the candidate
and this documentation record, then prepare—not yet launch—the bounded
native-gap Science campaign with an explicit merged-config review. Stage 7
must not be marked accepted, integrated, or production-ready until all five
Unity campaigns are complete and independently dispositioned.
