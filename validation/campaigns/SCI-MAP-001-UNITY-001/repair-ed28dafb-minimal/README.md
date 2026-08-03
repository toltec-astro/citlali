# SCI-MAP-001 minimal Unity transfer package

This small, human-run package prepares the MAP evidence reductions for the
unchanged Citlali candidate
`ed28dafb37f9113c0d3c95297148157129a90886`, tree
`cf75c36557178f351fb62781108a6f4b41b19225`.

The fixed scientific cases are:

| Case | Observations | Arrays | Configuration |
| --- | --- | --- | --- |
| P-SEQ | 152389 | a1100, a1400, a2000 | pointing; no coadd; `seq`, 1 thread/CPU |
| P-OMP | 152389 | a1100, a1400, a2000 | pointing; no coadd; `omp`, 6 threads/CPUs |
| S-C-SEQ, S-C-OMP | 152390, 152392 | a1100, a1400, a2000 | science coadd; `seq` 1 or `omp` 16 threads/CPUs |
| S-E-SEQ, S-E-OMP | 152390, 152392 | a1100, a1400, a2000 | science empirical/no coadd; `seq` 1 or `omp` 16 threads/CPUs |
| S-X-SEQ | 152390, 152392 | a1100, a1400, a2000 | science coadd plus empirical products; `seq`, 1 thread/CPU |

The Point TolProj workspace contains only 152389 under `1146+399`. The Science
workspace contains science observations 152390/152392 under `NGC4449`, with
152389/152391/152393 as `1146+399` pointing support. The included full/all
processed-time-chunk overlay is for the two auxiliary primitive captures only;
it does not replace any of the seven cases.

`case-overlays/` contains the seven exact case configurations. Install one as
`99_zzz_sci_map_case.yaml` in a fresh case directory after TolProj setup. Its
name deliberately sorts after TolProj's `99_zz_tolproj_submission_runtime.yaml`,
so the case binds its runtime policy and thread count after the scheduler
overlay. Never use `parallel_policy: omp` with `n_threads: 1`: the candidate's
GRPPI pipeline waits indefinitely in that unsupported combination.

Expected return material is the two realized `project.yaml` files, the seven
completed reduction directories (including their generated numbered YAML,
`02_redu.sh`, scheduler stdout/stderr, and ordinary reduction products), and
the one ordinary candidate executable snapshot selected by TolProj. The
machine-readable list is in `campaign.json`.

## Boundaries

This package does not authorize Codex to contact Unity, transfer data, build,
submit a reduction, change the candidate, alter a case or acceptance criterion,
close a finding or dependency, launch a re-audit, or admit production. Grant
runs every Unity command in `OWNER_RUNBOOK.md` after confirming the local
candidate and the exact raw/APT/pointing inputs. Stop rather than substituting
an observation, array, configuration, executable, or input source.
