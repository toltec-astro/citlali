# SCI-CAL-001 tau225 engineering-extension exact execution request

Request ID: `SCI-CAL-001-TAU025-ENGINEERING-EXTENSION-001-EXECUTION-REQUEST-001`
Status: documentation only; **owner approval is required before any cache is
created or AM is invoked**
Decision authority: `CAL-ATM-D007`, coordination commit
`b227043d2a1ac9caad0d2b18d357fe732fcc9a6d`
Decision-record SHA-256:
`67690533211ec47b9f01d269a85e0dcc2296009b50b11d1ee701d1957dfe76b4`
Achieved-coordinate gate amendment:
`CAL-ATM-D007-ACHIEVED-COORDINATE-001`, coordination commit
`1bffc48b6e72191ed2c9125ac405eabf4b2eae3c`, SHA-256
`1aa5e20c521b204f5f6c130fb0bf3ebf4ef80850899c8a9e706577acfd336894`

## Scope and explicit non-authorizations

This is the exact execution request required by D007. It requests a future,
fresh-cache AM 12.2 direct-truth study only. It does not create the cache,
execute AM, fit or define a candidate/operator, evaluate a candidate, or
interpret a numerical result. It does not authorize a Citlali or TolTECA
source change, Unity action, repair, re-audit, operator or operational-domain
adoption, production change, or a new application output format.

The preceding protocol remains binding, including its no-AM
`nextafter(.15)` evaluator diagnostic and its quality/segment policy:
`SCI_CAL_001_TAU025_ENGINEERING_EXTENSION_PROTOCOL.md`, SHA-256
`2552a8fee94bc64719504528d3a763c402bfedd9c7ec1380c2d4f5d1775b6967`.

## Immutable input binding

The future runner shall verify these inputs before it creates its cache lock or
any cache directory.

| Input | Required identity |
| --- | --- |
| AM model | AM 12.2; native executable SHA-256 `78e721d45b08990069a2d67a5fb337446bcbfb728046940c0d473bea340205fb` |
| AM source payload | aggregate SHA-256 `0cd4ea9d48c3c6da2100a692af1dc24dce5b3c903ced2b07b7372e8e85182fe8` |
| AMC inventory | `copied_am_manifest.json`, SHA-256 `714ad24329e625625da281d3b31ac2d28d04ab3c516980d516cff5ddadb027a9`; 25 files, 121,065 bytes; canonical NUL aggregate `b7dd766852b4f422bdc861337e04d8f0184732045ea1a06a962560e86d2ce87c` |
| Frequency grid | 0--500 GHz inclusive, 10 MHz (`0.01` GHz), 50,001 rows |
| Passbands | `toltec-passband-set-v1:sha256:5e6f38f14bcae93a29ffe8362c52b15209f51aee4e48373b23aaa5ec2f8a6433`; index `74465637294e536c44818099e4858a916fc6b9acbb1ea21b40427d15fb6532d5`; a1100 `13b8fd009bb8d7c375d3c46d21e26d0a779f7f00a949a2a5ccd619d1fe56fd72`, a1400 `a7b671d9f659cbc98dad99d3015ce81a3d7a3486c702819d9b3305703e7c682e`, a2000 `77e4b33c7bbc2c345ef94d41480d5fee5cb096d789f4fe78e1b4f80a37e0d6ff` |
| Integration convention | exact ECSV `f` (GHz) and `throughput`, alpha `{-1,0,2,4}`, composite trapezoid, denominator normalization, no spectral extrapolation; LOS optical depth `-log(T)`; full modified-secant airmass and `X_ref=0` |

The 25 approved immutable AMC inputs are the following. Their listed paths
are relative to the supplied AM root; no generic q95 product is an input,
target, or substitute.

| AMC file | Bytes | SHA-256 |
| --- | ---: | --- |
| `LMT_DJF_5.amc` | 4837 | `fcb3b70f44cad98cf0586fede9dcd3b2e35f3cb45023d0485c782c108b25b474` |
| `LMT_DJF_25.amc` | 4841 | `aeeeeb48bef422f2d9392b5d7a3d62ab1887fd9e7c10322d5246d914841ba866` |
| `LMT_DJF_50.amc` | 4841 | `d7c256d04d922beb51c9f8ab715e5be1a962252580eff2d08ba1be4d206eb5b0` |
| `LMT_DJF_75.amc` | 4841 | `b63503c7f4170404d18f3797735b64fb947ce73eed35f0315155d0a29d499721` |
| `LMT_DJF_95.amc` | 4841 | `b87b918b302425ef3d85aeedc285863a987579923289a37b97c6de5c935175e6` |
| `LMT_JJA_5.amc` | 4837 | `f5a3a92f41803247da504271eafe8a62af6df51e7a3ec8740c5e89c2b97409fc` |
| `LMT_JJA_25.amc` | 4841 | `13ea1837e3f2afeb605d8f8e8329472032f27c7a9d526d1b381bd7e75830e9b6` |
| `LMT_JJA_50.amc` | 4841 | `00b86aa0f2331f6a138c8efcc89ed5a4d918baef948f0da55feb114b0df2eb76` |
| `LMT_JJA_75.amc` | 4841 | `1b54ac2d0d5c7cd8f0805d1117d44ad1ff938ccb74433a93e2a20cbc77b3fc95` |
| `LMT_JJA_95.amc` | 4841 | `54a4345d487babbffcc9b36b9ccbaec2904b58d35458f6476a382da8d70cf437` |
| `LMT_MAM_5.amc` | 4837 | `ecdf228e34ca8f4b0f5930865179fd8afe7c8e602b1863d7ee8ac4352c65351f` |
| `LMT_MAM_25.amc` | 4841 | `82ac1e2a49a528244c1571daadcc8d42bd6d13c0ba8a7b5d2f81d10ebc13caee` |
| `LMT_MAM_50.amc` | 4841 | `2f282452f3932024be26b4579ed765996c08ff1e74cafb7d2750396d234fa6ac` |
| `LMT_MAM_75.amc` | 4841 | `937ecf9b3725b03a2745a61546f3659a9506bbaead72afbb41141b51e88a630e` |
| `LMT_MAM_95.amc` | 4841 | `dc7a3acc1fbc5ce92ef98dd5d00f45db997c6f2680c3c88e485f94bdbac398b2` |
| `LMT_SON_5.amc` | 4837 | `79a50d0275d026886feefb83e68e88b84801c5e3efa0066edbfc925e2b134926` |
| `LMT_SON_25.amc` | 4841 | `e9b06bae87e742801a751270aadd7939f94b39bfb04c7196ecf12c7586cce627` |
| `LMT_SON_50.amc` | 4841 | `b47140d5680449c83327ec8ffaa3a36d2472f7d7d042119d95b84689f06b42b2` |
| `LMT_SON_75.amc` | 4841 | `c2d6a7b6aee60639168dfcb03d85dc07b85ed23ea7ea2b8e202033d16c14a770` |
| `LMT_SON_95.amc` | 4841 | `a4348f003b44205c9c4f367da42ea9a5962689cdfd6f1c12580c28c853526984` |
| `LMT_annual_5.amc` | 4849 | `f58921d3cb222965df86b05f89cbf716f92f8193465d18f0106bf09b52fd718d` |
| `LMT_annual_25.amc` | 4853 | `a9524553a5808a549eb18046a9ed6f8bd67ca1e29ccd1c91e05b351b64ea23e6` |
| `LMT_annual_50.amc` | 4853 | `ee3946b48db6049b26231ff22d456c8fc2f2dc96ecabd1a861c4d8002c81c3c3` |
| `LMT_annual_75.amc` | 4853 | `8e7a250764c8583ef23f9ca140248e62670e3e3d9b709baf005b31c24dc52387` |
| `LMT_annual_95.amc` | 4853 | `687218c4633e03f61e179cd41314ca720572eabd6015404fd7a8149e2280b1e5` |

## Exact target and tuple inventory

The AM coordinate is the full modified-secant zenith `tau225` construction at
EL80:

```text
X80 = 1.01538872688246729
T225,target = exp(-tau225,requested * X80)
tau225,achieved = -log(T225,parsed_literal) / X80
```

The target literal is the existing AM parser's seven-significant-digit
scientific decimal. A scale search succeeds only if its parsed 225-GHz EL80
transmission equals that literal exactly. The achieved coordinate is reported
from that literal, not substituted for the requested decimal or silently
rounded back to it.

| Node ID | Role | Requested `tau225` | Exact target literal | `tau225,achieved` from literal |
| --- | --- | ---: | --- | ---: |
| `tau015` | construction | `.15` | `8.587235e-01` | `0.1499999859125433062628881602402745` |
| `tau01625` | held-out | `.1625` | `8.478931e-01` | `0.1625000436670042842458733986011408` |
| `tau0175` | held-out | `.175` | `8.371994e-01` | `0.1749999782159755418032064132046966` |
| `tau01875` | held-out | `.1875` | `8.266405e-01` | `0.1874999959892568741794020809655989` |
| `tau020` | construction | `.20` | `8.162148e-01` | `0.1999999783213567867059666712638576` |
| `tau02125` | held-out | `.2125` | `8.059206e-01` | `0.2124999488193856859985890648134455` |
| `tau0225` | held-out | `.225` | `7.957562e-01` | `0.2249999585478593390938136819948858` |
| `tau02375` | held-out | `.2375` | `7.857200e-01` | `0.2374999620652431274454965339427345` |
| `tau025` | construction | `.25` | `7.758104e-01` | `0.2499999377860148032413478624431719` |

The achieved coordinate is derived provenance, not a second AM target, a
scale-selection criterion, or a substitute for the exact parsed target
literal. For each node, recompute it with decimal precision of at least 100
significant digits from the displayed target literal and `X80` above; record
the serialized recomputation and the absolute difference from the printed
reference. The provenance gate passes only when every difference is at most
`1e-12`; changing a requested tau or target literal, or failing exact
parsed-literal equality, remains fail-closed.

| Node ID | Recomputed achieved `tau225` (70 fractional digits) | Absolute difference from printed reference |
| --- | --- | --- |
| `tau015` | `0.1499999859125433062628881602402745171307972530967402260405210510624037` | `1.7130797253096740226040521051062403692417361011312637574839939399400000e-35` |
| `tau01625` | `0.1625000436670042842458733986011407957621884281834432958559048571995763` | `4.2378115718165567041440951428004237069387897970290426112394215591000000e-36` |
| `tau0175` | `0.1749999782159755418032064132046966137541183699181855319794028821003869` | `1.3754118369918185531979402882100386882805634625455719561607182201000000e-35` |
| `tau01875` | `0.1874999959892568741794020809655989327280331348631778269363229169776814` | `3.2728033134863177826936322916977681416022175522525825924502205232300000e-35` |
| `tau020` | `0.1999999783213567867059666712638576458165224486770253649622899349654778` | `4.5816522448677025364962289934965477815605672981946970080440532402300000e-35` |
| `tau02125` | `0.2124999488193856859985890648134455471372731474165985305638558361114224` | `4.7137273147416598530563855836111422444518198162564286785174055222200000e-35` |
| `tau0225` | `0.2249999585478593390938136819948857561002096611296179412824915384024972` | `4.3899790338870382058717508461597502818566419949401467431757753970600000e-35` |
| `tau02375` | `0.2374999620652431274454965339427344793402346942304927484185036454082416` | `2.0659765305769507251581496354591758379998155087822689984947818935400000e-35` |
| `tau025` | `0.2499999377860148032413478624431718813116558655350862065273554110308605` | `1.8688344134464913793472644588969139511323682289333463073794501897400000e-35` |

Let `P` be the 25 AMC filenames in the preceding table. The exact full-grid
run inventory is the following lexical Cartesian expansion; the run identifier
is part of every request, raw-output filename, sidecar, and digest manifest.

| Set | Tuple expansion | Full grids/profile | Full grids |
| --- | --- | ---: | ---: |
| Construction | `P × {tau015,tau020,tau025} × {25,35,45,55,65,75,80}` | 21 | 525 |
| Held-out | `P × {tau01625,tau0175,tau01875,tau02125,tau0225,tau02375} × {29,41,53,67,79}` | 30 | 750 |
| Total | construction union held-out | 51 | **1275** |

The canonical identifier is
`tau025e001/<role>/<AMC-stem>/<node-id>/el<two-digit-elevation>`; for
example, `tau025e001/heldout/LMT_SON_95/tau0225/el67`. The corresponding
integer AM zenith angle is exactly `90 - elevation_deg`. No other tuple is
permitted. The union is disjoint by both requested opacity/elevation role and
identifier, and held-out tuples must be anti-joined against every construction
tuple before AM execution.

The `nextafter(.15,-inf)`, `.15`, and `nextafter(.15,+inf)` triplet is absent
from this inventory. It is a no-AM evaluator diagnostic, performed only after
a future candidate is defined, with the same candidate identity at all three
values; it has no AM target, scale trace, or cache record.

## AM invocation and scale-search plan

Only the documented final AMC argument `Nscale troposphere h2o` through argv
`%9` may vary. The profile file bytes themselves must never be edited. The
runner uses `LC_ALL=C`, `LANG=C`, `OMP_NUM_THREADS=1`, the AM working directory
`Big_Atmosphere`, and these argv forms:

```text
# Scale-search anchor at EL80 (zenith angle 10 deg)
<am> LMT_am_inputs/<profile>.amc 224.99 GHz 225.01 GHz 10 MHz 10 deg <scale>

# Direct truth at each registered elevation E
<am> LMT_am_inputs/<profile>.amc 0 GHz 500 GHz 10 MHz <90-E> deg <scale>
```

For each of the 25 profiles and nine target literals, use the existing
nonnegative parsed-transmission plateau search, without a bandpass residual or
any additional atmospheric degree of freedom:

1. Run and retain scale `0` and scale `1` at EL80 once per profile.
2. Seed the target from the scale-0 parsed optical depth and the copied
   scale-1 225-GHz optical-depth increment; evaluate the seven-digit decimal
   round-trip binary64 scale.
3. If necessary, expand an upper nonnegative bracket at most 64 times, then
   bisect at most 48 iterations to find the parsed-transmission target.
4. If a target plateau is exact, bound its lower and upper edges using at most
   64 expansions and 48 bisections per edge, and choose the plateau midpoint
   (falling back to the lower inside value only if that midpoint no longer
   parses to the required literal).
5. Reject a non-finite or negative scale, an unbracketed target, nonmonotone
   parsed optical depth/transmission sequence, or a result whose parsed 225-GHz
   EL80 transmission is not exactly the table literal. Do not select a
   nearest residual or change a tuple.

Each `profile × node-id` writes one scale trace containing every trial's
round-trip decimal and binary64 hexadecimal scale, argv, parsed 225-GHz row,
bracket/plateau role, raw-output digest, warning diagnostics, and final scale.
Thus the plan contains 225 scale traces. Once a scale passes, each associated
full-grid tuple uses only that trace's final scale.

## Immutable fresh-cache and provenance layout

The owner-approved execution must supply one previously absent external cache
path whose basename is exactly
`sci_cal_001_tau025_engineering_extension_001_root`. This request deliberately
does not name a host-specific parent path or create that path. Before creation,
the runner records the absolute proposed path and verifies it is absent; an
existing, partial, writer-locked, or reused path fails closed.

After approval, the runner may create only this layout below that fresh root:

```text
<fresh-cache>/
  .tau025-engineering.lock                 # POSIX whole-cache lock
  execution_context.json                   # canonical, SHA-256 bound
  inputs_manifest.json                      # copied input identities
  scale_traces/<AMC-stem>__<node-id>.json
  raw_outputs/<run-id>.txt                 # unchanged combined AM stdout/stderr
  sidecars/<run-id>.json                   # existing raw-grid sidecar schema
  am_spectral_cache/shard_00/ ... shard_07/
  failed_attempts/<run-id>__<digest>.txt
  failed_attempts/<run-id>__<digest>.failure.json
  manifests/raw_and_sidecar_sha256.json
  manifests/cache_inventory_sha256.json
```

`execution_context.json` cryptographically binds this request ID; D007 and
package input digests; runner source and runner-file digest; AM executable and
source-payload digests; profile table; target table; argv templates; locale;
8 jobs/8 shards/one OMP thread; cache-ID algorithm; the complete 1,275-run
inventory; and the exact inherited warning policy. Cache IDs are derived as
the first 24 hexadecimal characters of SHA-256 over canonical UTF-8 JSON
(`sort_keys=true`, separators `,` and `:`) of the RunSpec request, executable
SHA-256, profile SHA-256, OMP thread count, shard count, and execution-context
SHA-256. Shard assignment is the big-endian first 64 bits of
`SHA256(cache_id) mod 8`.

Raw output is never altered. The existing sidecar schema records raw and
numeric-text SHA-256 values plus a normalized warning-bearing-output digest
that replaces only documented volatile runtime and dcache-counter header
values. It preserves AM identity, warnings, numeric grid, configuration, and
all nonvolatile output. Failed attempts are retained under `failed_attempts`
but are never evidence and stop the study at the first failure.

## Inherited warning/cache stop register

This request does not decide a new warning policy. It binds and enforces the
approved text verbatim:

> WARN-001 admits AM status 1 only as explicitly warning-bearing numerical
> evidence with all 50,001 rows, solely the preregistered unresolved-line
> warnings and canonical summary count 86, 87, or 88, and zero unknown
> warnings, cache mutation, or errors. Every other nonzero status fails closed.

The following conditions stop the execution, preserve the failed attempt and
its sidecar, and prohibit its use as evidence: any status other than 0 or a
WARN-001-admitted 1; a non-50,001-row full grid; a new/unknown warning;
summary count outside 86/87/88; an error header; an AM cache-mutation warning;
missing or mismatched raw/sidecar pair; a changed input digest; a duplicate or
unexpected run ID; a cache lock failure; any cache preexistence; or any source
or runner identity different from the preflight record. No retry may reuse or
overwrite an affected cache; a later owner-approved request would be required.

## Resource estimate and readiness gates

This is an arithmetic storage/invocation estimate, not an unrun performance
claim. The direct-grid count is 1,275. At the largest preserved 50,001-row raw
output size (3,314,862 bytes), raw direct grids require at most 4,226,449,050
bytes (3.936 GiB) before sidecars, traces, AM's internal shard cache, failed
attempt preservation, or filesystem overhead. The scale plan has 225 target
searches, 50 shared scale-0/scale-1 anchors, and a worst-case ceiling of
75,875 narrow-band anchor invocations under the registered expansion/bisection
limits. Reserve at least 12 GiB free storage and 8 concurrent processes; the
runner must record observed peak storage and wall time rather than infer them
from this estimate.

The owner may authorize execution only when all of these readiness gates are
recorded as passing before any cache creation:

1. D007 decision commit and decision-file digest match the values above, and
   the package is clean at a recorded commit with this request's SHA-256.
2. Every executable, source payload, AMC input, passband index/member, and
   protocol digest matches this request; all 25 profiles and all 1,275 unique
   tuple IDs expand exactly once.
3. The requested tau and seven-significant-digit target-literal columns of the
   scale-target table byte-compare with this table. Every achieved coordinate
   is recomputed at high precision from its unchanged target literal and
   `X80`, serialized with its absolute difference from the printed reference,
   and must differ by at most `1e-12`. Achieved tau is derived provenance only,
   never a second AM target or scale-selection coordinate.
4. The proposed external cache parent has at least 12 GiB free, the exact
   basename target is absent, and no existing cache is inspected, copied,
   reused, or mutated.
5. The runner review confirms only argv `%9` varies, binds the cache layout
   and cache-ID identity above, keeps raw output/sidecars atomically paired,
   and enforces WARN-001 verbatim. No new warning class or deviation is
   admitted.
6. Static dry-run expansion only (no AM process and no directory creation)
   emits the 1,275 expected run IDs, 225 scale-trace IDs, profile/target
   anti-join, elevations, zenith angles, and expected output paths with no
   duplicates or omissions.
7. The owner explicitly approves this exact request after reviewing these
   gates. Until then, stop.

## Requested owner action

Approve or reject this execution request. Approval would permit only the
subsequent preflight and fresh-cache direct-AM evidence run described here; it
would not permit candidate fitting, numerical interpretation, operator
selection, adoption, implementation, repair, re-audit, or production use.
