# MAP-UNITY-ED1 bounded decision brief — 2026-08-02

Status: **stopped at a governing handoff condition; owner/coordinator decision
required**

Request: `SCI-MAP-001-UNITY-001`

Reserved successor revision: `repair-sha-ed28dafb-ed1-2026-08-02`

Branch: `codex/map-unity-ed1`

## Bound identities

- The app-supplied worktree began clean and detached at the exact
  campaign-preparation commit
  `1b824f138754eeb1856ae5f102027db4b31598be`.
- That commit has the sole parent
  `ed28dafb37f9113c0d3c95297148157129a90886`.
- The unchanged Citlali application candidate is
  `ed28dafb37f9113c0d3c95297148157129a90886`, tree
  `cf75c36557178f351fb62781108a6f4b41b19225`.
- Before branch creation, `codex/map-unity-ed1` did not exist and no worktree
  used it. The branch was created only in this worktree.
- Governing handoff SHA-256:
  `8ce9d12f93b2cf60e6fb281b67afcfdcaeb9f2ffde755b9c0f640a38c98c0c5b`.
- Owner-decision SHA-256:
  `4cec0fbbd172a32f51cfe95d5ef1712f091297fb43bd2efee5a1c4eecf99e5fa`.
- Decision-content commit:
  `db74fe293436b59eecfa5c36b2a2ea186b05e9b6`.
- Coordination identity-binding head:
  `4257265dba44aac3b29d985e1f7bc01b2a50368c`.
- Frozen predecessor package tree:
  `dbf486e30c9b78ca16e05bccafc2d027562d0746`.
- Frozen predecessor `SHA256SUMS` digest:
  `ecf080cce98ad3aef6d6dbf52e72dd53be5d659a40285ec6c9bfbb0aee185a69`.

## Stop decision

The successor cannot populate its nine compact reconstruction groups and
fixed actual-data traces from the unchanged seven-case products. Continuing
would require at least one of the following actions, each outside the current
authorization:

1. change the frozen Point/Science timestream-output configuration and add a
   segregated full/all processed-TOD product-producing reduction;
2. build and execute a separately instrumented Citlali binary whose observer
   duplicates candidate projection and primitive-admission logic, together
   with a new equivalence and execution contract; or
3. add an application-owned pre-map streaming hook, which edits application
   source and changes the exact candidate.

These are respectively a config/product and storage tradeoff, a new
scientific/operational equivalence tradeoff, and an application-source change.
The handoff says to stop rather than choose any of them implicitly.

## Evidence for the stop

### Existing products are incomplete authority

- Point processed TOD is enabled for all scans but the fixed policy is
  `mini`. Mini output stores signal as `float32` and omits kernel and detector
  projection geometry.
- Science processed TOD is disabled. Its frozen selection is also mini and
  selected-scan rather than full/all if merely enabled.
- Existing PTC diagnostics provide detector summaries and identities, not the
  complete per-sample signal, kernel, sample flag, geometry, duration, and
  coefficient population.
- Existing contribution diagnostics retain only admitted signal/coefficient
  records for bounded diagnostic targets. They omit the full geometric and
  upstream-eligibility population, kernel, duration, sample flags, realization
  signs, and every-network fixed traces.
- Provenance binds configuration and realized identities but does not contain
  the required per-scan/per-pixel pre-normalization sufficient statistics.
- Final FITS products contain derived aggregate planes only. The owner decision
  forbids using them as the sole authority for the same independently
  reconstructed facts.

Consequently, no package-only analyzer can truthfully reconstruct signal,
weight, kernel, all retained F010 planes and aliases, all 64 realizations, and
centered coadds while also proving primitive order/population and the required
actual-data trace coverage.

### Technically sufficient full PTC output is not currently authorized

The existing full PTC writer would provide binary64 signal, flags, kernel,
detector geometry, APT columns, per-scan weights, and scan layout immediately
before map population. Using it for this campaign would nevertheless require:

- changing Point from mini to full;
- enabling full/all Science PTC output;
- adding a product-producing capture execution outside or in place of the
  fixed cases; and
- retaining or consuming a transient artifact proportional to the primitive
  population.

Those are exactly the config, product, and operational choices this task may
not make on the owner's behalf. The output cadence also needs an explicit
binding to the effective mapmaking sample rate `telescope.d_fsmp`; the TOD
`SAMPRATE` field is populated from the native `telescope.fsmp`.

### Package-local binary instrumentation is not a neutral workaround

An include-path observer around `NaiveMapmaker` is technically conceivable,
but it would produce a binary that is not the ordinary exact-candidate binary,
repeat the candidate's projection and admission calculations in a second
implementation, and require either an additional reduction or substitution in
the fixed cases. Proving that observer to be non-perturbing and numerically
equivalent is a new acceptance contract absent from `MAP-UNITY-ED1`. Treating
that as already approved would broaden scope.

## Local metadata and proportionality measurement

Safe local metadata, not Unity evidence, independently reproduced the
preliminary source-term total of `1,717,082,860`:

| Observation | a1100 | a1400 | a2000 | Total |
| --- | ---: | ---: | ---: | ---: |
| 152389 | 24,429,421 | 10,531,812 | 7,521,923 | 42,483,156 |
| 152390 | 480,826,960 | 207,303,104 | 148,051,649 | 836,181,713 |
| 152392 | 482,113,090 | 207,857,076 | 148,447,825 | 838,417,991 |

Active networks are exactly `0`–`5`, `7`–`9`, and `11`–`12`. Local candidate
timing logic gives 12 Point scans with post-RTC counts
`[289, 305 x 10, 289]` and 124 Science scans per observation with counts
`[1188, 1220 x 122, 1188]`. With 5,518 detectors this is
`1,688,839,080` processed detector-sample terms.

A full PTC capture has a core-array lower bound of
`94,254,679,616` bytes (`87.7815 GiB`) for one Point and the two Science
observations: 40 bytes per Point term for signal/flags/kernel/lat/lon and 56
bytes per Science term after adding RA/Dec. This excludes weights, telescope
streams, APT metadata, NetCDF/HDF5 overhead, manifests, and the compact
successor output. It is therefore a lower bound, not a storage ceiling. The
comparison v1 estimate remains `127,064,131,640` bytes (`118.34 GiB`).

No successor artifact/runtime report is claimed because the stop occurred
before an operational producer could be authorized and implemented.

## Required owner/coordinator choice

Choose and explicitly bound one route before resuming:

1. **Full/all PTC capture authorization.** Permit one segregated Point capture
   and one ordered Science capture, the required output-only config changes,
   proportional transient storage, effective-sample-rate binding, and exact
   cleanup/retention policy. The unchanged seven acceptance cases would remain
   separate.
2. **Application-owned compact hook authorization.** Permit a narrowly scoped
   pre-map read-only stream hook and a new candidate identity so compact
   statistics/digests/traces are produced without a full PTC intermediate.
3. **Instrumented validation-binary authorization.** Permit a package-local
   derived executable, define whether it is an auxiliary run or replaces named
   case binaries, and approve explicit non-perturbation/equivalence gates.
4. **No successor producer.** Keep the evidence request blocked and retain the
   existing findings/dependencies without relabeling or closure.

The coordinator should not request operational Unity values or authorize a
human launch until one route is selected and its new tradeoffs are recorded.

## Preserved boundaries

- No Citlali/MAP application source, build configuration, numbered config,
  case, product gate, observation, array, tolerance, or scientific gate was
  changed.
- The frozen `repair-ed28dafb` package remains byte-for-byte unchanged.
- Unity was not contacted or queried; no transfer, build, reduction, or Slurm
  action was attempted.
- No owner operational value was filled.
- No external evidence was supplied.
- The repair was not integrated.
- No MAP finding or CAL/AST/PTC/VAL dependency was closed.
- Re-audit was not launched and production was not expanded.

The exact task commit/tree and package checksum identity are intentionally
reported in the external handback, avoiding a self-referential commit identity
inside this brief.
