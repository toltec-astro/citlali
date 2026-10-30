# Isolated Mac Spack qualification — 2026-10-30

Grant authorized installation of the 44 dependencies and completion of the
isolated candidate's Spack qualification. The [work order](../validation/spack_qualification_2026-10-30/WORK_ORDER.json)
records the directive, Tier 2 preflight, scope and bounded reassessments.
The supported local **Mac developer native gate passes**. This closes the
44-dependency/native-Mac prerequisite recorded by the earlier combined candidate.

## Identity and environment

The starting candidate is `b090da0458271c618568802fde1cf5c39f5dd5d2` on
`codex/pointing-engineering-candidate-20261030`. Its approved pointing and
engineering ancestries remain intact. Canonical stays at
`e32c4584dec07617d6caa857d39bf13047b6c4b0`. The only source change is explicit
`run_polarization = false` in two unpolarized JINC test fixtures; production
sources, numerical policies, configuration defaults and scientific contracts
are unchanged.

The [qualification receipt](../validation/spack_qualification_2026-10-30/QUALIFICATION.json)
binds the tested base, exact test patch, gate results and evidence digests.
The external packet's `COMPLETION.json` binds the final documentation/test commit,
refreshed executable, final checks and independent exact-SHA review. This avoids
embedding a self-referential commit identity in the committed receipt.

| Item | Actual realization |
| --- | --- |
| Environment | Candidate-owned `build/spack-candidate/environment`; Spack 1.2.2 |
| Dependency store | Candidate-owned `build/spack-candidate/store`; existing upstream store read only |
| Installed graph | All 44 nonexternal dependencies installed; four external providers; Citlali root remains a developer build |
| Profile | `macos-llvm20`, LLVM Clang 20.1.8 through concrete Spack wrappers |
| Native build | Fresh `build/spack-native`, strict C++23, Release, OpenMP and Wiener OpenMP enabled |
| Platform | Actual macOS 27 / `goldengate`, SDK 26.5, deployment target 27.0 |
| Architecture | Host/spec `m4`; observed wrapper argument `-mcpu=apple-m1` |
| Root graph identity | `kqk3pl4f6jgsb5yksobnynlx2gmnqks6` |
| Lock SHA-256 | `00964e0c57c1df96b0ddc3f6d214bef1c1ab8f59cc4e5c4da653cefbcb915238` |

Pinned clean first-party sources remain Tula CMake `6433cdabe7010d0af2d0ba69da7af27510391b80`,
Tula `aa16c853c6b596c04ccdc90dc3acc4ce2006d947` and Kidscpp
`498ece1113001ae2d42d96d9fc29152aea3eaaef`. The package recipe's Citlali version
label does not identify or modify a protected legacy branch.

## Gates and bounded repairs

| Gate | Result |
| --- | --- |
| Fresh native CLI and complete ordinary test aggregate | Pass |
| Ordinary native CTest | 1,388 enabled pass; one inherited disabled case remains |
| Repaired population fixtures, both registrations | 4 pass |
| Native-capable baseline tests | 207 pass |
| Complete configuration preflight | `--require-all` passes, no missing required baseline |
| Installed Tula and Kidscpp synthetic consumers | Each compiles, links and passes its CTest |
| Four public standalone headers | Strict C++23 compilation passes without PCH/forced include |
| Runtime identity and linkage | Standalone help/version work; explicit managed environment reports `binding=dag-match` |

The disabled case is the inherited `MapFitterLifecycle.ExactProductSequence`.
No new skip or relaxed assertion was introduced. Installed-consumer templates
retain their upstream GNU C++23 setting; the main build and four standalone
header probes use strict C++23. Duplicate-rpath linker warnings are retained in
the logs. Successful validation has no unexpected error-level output.

Three failures were resolved within this build/test work order:

1. LLVM 20's strict C++23 `<complex>` failed against SDK 27's `NAN`/`INFINITY`
   header protocol. A minimal probe reproduced it independently of the project.
   Selecting the already installed SDK 26.5 in the isolated process fixed it,
   keeping strict C++23, compiler, real OS, deployment target, graph and source
   pins unchanged. The failed Kidscpp stage and logs were preserved. Retained
   dependencies have mixed SDK evidence, itemized per package; cache records for
   header/interface packages do not prove native compilation. This is an exact
   local realization, not a blanket claim that LLVM 20 supports SDK 27 headers.
2. Both registrations of one JINC population fixture segfaulted. OS crash
   addresses and independent disassembly located the uninitialized polarization
   flag entering an absent `fg` vector. Production already initializes the flag.
   Explicit `false` in the two unpolarized population fixtures fixes the test
   setup; the full native rerun passes with application sources unchanged.
3. The isolated Clingo bootstrap `PYTHONPATH` omitted the repository for direct
   Python-script imports. Prepending the candidate root, while retaining the
   bootstrap path, makes the full configuration preflight pass. No script or
   dependency change was necessary.

Reproduction uses the retained `build/spack-candidate/process-environment.json`
with `SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk`, the
TolTECA venv first on `PATH`, and the candidate root prepended to the retained
`PYTHONPATH`. The packet runners record complete argument lists and task-owned
cache paths. The supported runner is `tools/build/run_spack_citlali_dev.py` with
`--profile macos-llvm20`, this exact environment and `build/spack-native`.
The install command used `--only-concrete --only dependencies --fail-fast
--keep-stage --show-log-on-error -j 8` and the exact root hash above.

## Review, preservation and remaining acceptance

Independent review covers scientific/behavioral conformance, architecture and
ownership, and repository/evidence hygiene separately. SDK and fixture repairs
have independent reassessments; the packet contains the final exact-SHA verdict
and final preservation comparison. Existing backups and the earlier combined
candidate evidence packet are retained, including initial failures. Preservation
verification states its coverage explicitly: protected refs, original workspace
status lists, upstream Spack database, original dependency file inventory,
756 backup metadata records and 46 protected artifact hashes. It is not a claim
to have rehashed every historical product or dirty file.

The native gate qualifies this Mac developer candidate. It does not install or
qualify an immutable Citlali root/export/release package, reproduce the accepted
Unity-backed V2 campaign, exercise a full real-data reader, or establish new
observational, calibration, SCIENCE/coadd or production acceptance. Hosted CI
and owner-run Unity qualification remain separate. No mainline move, push,
cleanup, pointing-runtime change or legacy 4.x/5.x modification follows from
these passing checks. Exact candidate integration still requires Grant's decision.
