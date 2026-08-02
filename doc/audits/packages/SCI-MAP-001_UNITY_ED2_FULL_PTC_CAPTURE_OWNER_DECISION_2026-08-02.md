# SCI-MAP-001 MAP-UNITY-ED2 full/all PTC capture owner decision — 2026-08-02

Status: owner approved; bounded successor implementation may resume; push and
Unity execution remain blocked

Package: `SCI-MAP-001`

Evidence request: `SCI-MAP-001-UNITY-001`

Decision ID: `MAP-UNITY-ED2`

Authority: project owner

Amends: `MAP-UNITY-ED1` evidence protocol only

## Decision

After reviewing the verified `MAP-UNITY-ED1` stop return, the project owner
selected Route 1: use two segregated full/all processed-TOD captures to supply
primitive authority to the compact successor producer.

This decision authorizes only the output configuration, temporary-resource,
input-staging, and retention terms below. It does not change the MAP estimator,
the exact repaired application candidate, the fixed seven acceptance cases, or
the scientific acceptance contract.

## Frozen identities and unchanged acceptance lane

- Exact application candidate:
  `ed28dafb37f9113c0d3c95297148157129a90886`.
- Candidate tree: `cf75c36557178f351fb62781108a6f4b41b19225`.
- Successor task resumes from the clean stop commit
  `3e014f11decbcf17ad372391e5e960e6c0c54461` on
  `codex/map-unity-ed1`.
- The frozen predecessor package at
  `validation/campaigns/SCI-MAP-001-UNITY-001/repair-ed28dafb` remains
  byte-for-byte immutable.
- The original seven cases remain exactly `P-SEQ`, `P-OMP`, `S-C-SEQ`,
  `S-C-OMP`, `S-E-SEQ`, `S-E-OMP`, and repaired-success `S-X-SEQ`, with the
  same observations, arrays, configurations, products, tolerances,
  realizations, provenance, resources, and sequential/OpenMP gates.
- One ordinary binary compiled from the exact candidate must be reused for
  both captures and all seven acceptance reductions. No observer, derived
  executable, or second instrumented binary is authorized.

## Authorized auxiliary captures

Two capture reductions may be prepared in addition to, and never in place of,
the fixed seven cases:

1. one Point capture for observation 152389; and
2. one ordered Science capture for observations 152390 and 152392, with
   observations 152389, 152391, and 152393 admitted only as the existing
   pointing support required by the science setup.

Relative to each named fixed source configuration, the only permitted
effective-configuration differences are these processed-time-chunk output
leaves:

```yaml
reduce:
  steps:
    0:
      config:
        low_level:
          timestream:
            processed_time_chunk:
              output:
                enabled: true
                mode: full
                indices: all
```

The successor package must automatically compare each fully merged capture
configuration with its fixed reference and fail unless the complete difference
is this allowlist. The captures are evidence inputs, not new production
profiles and not additional acceptance cases.

## Sample-rate authority

The existing full processed-TOD `SAMPRATE` field must remain truthfully labeled
as native `telescope.fsmp`. It must not be reinterpreted as the mapmaking rate.
Sample intervals, exposure identities, stream grouping, and independent map
reconstruction must bind separately to finite, positive effective
`telescope.d_fsmp`, including full-precision realized provenance and explicit
cross-checks. If the effective-rate authority cannot be recovered and bound
without changing the candidate, the task must stop.

## Temporary-resource and retention contract

The combined retained full-capture lane has a hard 200-GiB
(`214748364800`-byte) temporary ceiling.
This ceiling covers generated capture products, intermediate capture outputs,
logs, and compact successor evidence below the dedicated capture/output roots;
it excludes the pre-existing canonical raw source files that are only reached
through individual symlinks.

Before each future human-run stage, the package must measure current governed
usage and fail closed unless current usage plus the projected incremental
stage output is at or below `214748364800` bytes and the filesystem has enough
free space for that incremental output. It must record actual staged usage
after each capture and before subsequent work. A projected or measured
footprint whose logical apparent size or allocated size exceeds 200 GiB, or
inadequate available capacity, blocks further execution and returns to the
coordinator; it does not authorize deletion or a larger ceiling.

The full captures must remain on Unity through the fresh MAP re-audit and any
focused discrepancy expansion requested by that re-audit. Nothing may delete
them automatically. The package may prepare a precisely targeted, guarded,
human-run cleanup procedure, but cleanup becomes eligible only after the
coordinator records explicit acceptance of the fresh MAP re-audit. This
decision does not execute or pre-authorize that future cleanup.

## Lightweight TolProj source projects

Prepare fresh project-setup JSON for exactly these projects:

```json
{
  "description": "SCI-MAP-001 Point source project",
  "project_id": "SCI-MAP-001-POINT-SOURCE",
  "obsnums": [152389],
  "1146+399": {"obsnums": [152389]}
}
```

```json
{
  "description": "SCI-MAP-001 Science source project",
  "project_id": "SCI-MAP-001-SCIENCE-SOURCE",
  "obsnums": [152389, 152390, 152391, 152392, 152393],
  "NGC4449": {"obsnums": [152390, 152392]},
  "1146+399": {"obsnums": [152389, 152391, 152393]}
}
```

The grouped target entries are authoritative and prevent target-name
inference. Project initialization must fail if either destination already
exists.

Stage only individual symlinks to verified canonical regular raw files already
present below the owner-verified `/work/toltec` source root on Unity. The
stager must reject missing or non-regular canonical targets, unresolved links,
duplicate basenames, and pre-existing destinations, and must record the
resolved target identity and digest. Do not symlink an entire data directory.
Do not run `tolproj copy-raw` after link staging because it is a copying path
and may follow or alter staged symlink targets.

Copy, digest, and provenance-bind only the small required ECSV authorities:

- Point: `apt_152389_matched.ecsv`;
- Science: `apt_152390_matched.ecsv` and `apt_152392_matched.ecsv`; and
- pointing support: exactly one matching `ppt_*.ecsv` for each of 152389,
  152391, and 152393.

Missing or ambiguous raw, APT, or pointing authority is a stop. Existing local
or Unity reductions may be consulted as references only; no wholesale old
reduction is an evidence input or approved rsync payload.

## Authorized continuation work

The existing MAP task may now implement and locally validate the successor
package, capture overlays, lightweight project specifications, bounded link
stager, compact streaming producer/analyzer, deterministic traces, focused
expansion, verifier, manifests, resource checks, human runbook, and explicit
`unity_toltec` commands.

The original nine observation/array compact groups, every-active-network trace
coverage, discrepancy expansion, complete product/F010/WCS/coadd/realization/
provenance/seq-OpenMP gates, exhaustive local F011 authority, and independent
read-only review remain required. Final FITS products may not become their own
independent authority.

## Stop and non-authorization boundary

This decision does not authorize:

- editing application source, build configuration, or numerical behavior;
- changing candidate identity or using a second executable;
- changing a fixed case, observation, array, product gate, tolerance, or
  scientific claim;
- broadening the output-only configuration allowlist;
- exceeding the 200-GiB ceiling;
- guessing or weakening raw/APT/pointing/effective-rate identity;
- pushing, contacting Unity, transferring files, building, reducing, or
  submitting Slurm work;
- filling owner operational values before coordinator review of the returned
  package;
- integrating the repair, supplying external evidence, closing a finding or
  dependency, launching the re-audit, expanding production, or deleting data.

If another scientific or operational choice is required, the task must return
to the coordinator with a bounded decision brief.

## State effect and next gate

`MAP-UNITY-ED2` resolves only the route choice raised by the verified stop. It
authorizes local successor-package implementation in the existing task. It
does not itself supply external evidence or change MAP conformance, finding,
dependency, validation, production, repair, or re-audit state.

The next gate is a new exact task handback with package/tree/digest identities,
proof of unchanged application and frozen predecessor, local verifier and
self-check results, a measured resource envelope, human-only operational
instructions, and independent review. The coordinator must review that
handback before asking the owner to push or perform any Unity action.
