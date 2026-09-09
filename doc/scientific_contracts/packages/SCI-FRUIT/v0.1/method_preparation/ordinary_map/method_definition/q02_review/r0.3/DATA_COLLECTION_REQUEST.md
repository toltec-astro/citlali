# Short pointing data request for the weighting screen

2026-09-09. Owner-run ordinary PTC collection; no FRUIT intervention or replay
is requested. This supplies missing input candidates for the approved test
design. It does not itself admit the later experiment under frozen contracts.

## Program adherence and prior-work recovery

Use the [current disposition](README.md), its charter/recovery routing and the
[approved design](../r0.2/WEIGHTING_TEST_PLAN.md). The metadata sources and exact
local candidate identities are in [INPUT_METADATA.json](INPUT_METADATA.json).
The existing [population-access record](../../../../../empirical_lane/EL_G0_POPULATION_ACCESS_AND_LINEAGE_PLAN_R0.1.md)
keeps all previously exposed fruit-loop lineage outside untouched qualification.

## Requested observations

| Observation | Source / date | Local header duration | Role |
| --- | --- | --- | --- |
| **123424**, subobs 0, scan 2 | Neptune, 2024-11-27 | Requested integration 60 s; telescope time span 62.730 s | New full PTC bundle for discovery alongside 152389. Prior UID 4460 investigations remain closed; this is a whole-population mapping comparison. |
| **129081**, subobs 0, scan 2 | 3c273, 2025-03-04 | Requested integration 60 s; telescope time span 63.151 s | Reserve for later replication; keep its new detector-noise and weighting outcomes out of discovery tuning. |
| **152389**, subobs 0, scan 2 | 1146+399, 2026-02-19 | Existing PTC has 3,628 output samples, 5,518 detector slots and all 12 chunks | Reuse the existing uninjected, FRUIT-disabled candidate if its bundle passes input review. No new export requested initially. |

The two requested targets are short pointings. Different dates and sources
provide context diversity; they do not prove detector-noise contrast or full
statistical independence. No beammap or extended science scan is requested.
153481's header was also checked as a possible short candidate; it is not in
the requested batch. Selection does not use the old outcome-based quality rank.

## What to create and retain

For each requested observation, use one ordinary calibrated PTC pass from its
original inputs, with FRUIT, injected-source tests and restart/model input
disabled. Preserve the owner's exact ordinary cleaning recipe, calibration,
flags and support. Record its actual grouping/rank; do not silently convert it
to the proposed future FRUIT array recipe. Record all ordered configurations
and the effective/realized settings. The old population ten-iteration configs
must not be launched unchanged for this request.

Export every PTC chunk and retained detector/sample, without diagnostic chunk
subsampling. The necessary contents are:

- processed signal and exact units/reference; per-sample flags and their
  meanings; detector IDs, array/network IDs, chunk bounds and sample/time IDs;
- detector/sample coordinates at the mapping boundary if that output is
  available; otherwise all telescope/pointing-offset/APT inputs needed to
  reconstruct and verify the exact same-occurrence AST association;
- the exact APT/calibration and telescope files, nominal beam/quantity
  metadata, source position and any fixed source-protection region;
- existing per-detector/per-chunk weights and noise/quality facts with their
  names, units, support and lifecycle, without labeling them inverse variance;
- executable/version/build identity, ordered input configs, merged config,
  runtime, PTC/timestream, astrometry, mapmaking and input-source provenance,
  chunk-completeness record and log. Include hashes and sizes of all files.

PTC output is sufficient for this mapping study. A second RTC timestream dump
is not requested. Preserve the ordinary reduction products in their own new
output directory; never overwrite prior reductions. Keep the 129081 bundle
separate from discovery inputs and mark it `reserved-weighting-replication`.
Receiving its manifest/identity is allowed now; opening its signal, noise
statistics or weighted-map outcomes waits for the replication access decision.

The existing 152389 PTC file is 138,244,964 bytes (about 132 MiB) in `mini`
mode with all chunks. Full mode can add substantial coordinate/kernel arrays;
this is a reference size, not an estimate for the requested exports. Check the
actual size before downloading, and retain lossless signal/flag precision.
Do not truncate the detector or sample population to hit a size target.

Send the small manifests/provenance first if the full bundles are inconvenient.
For new working files on Unity, use the named test area beneath
`~/work_toltec/wilson/citlali_testing`; no new files belong directly in `~/`.
No shell, Slurm, download or login command is issued in this request, and Codex
must not connect to Unity. The owner chooses the available ordinary executable
and supplies its identity; no unverified path is invented here.

## Checks before T2

The manager will verify exact bytes, complete chunks, current quantity/flag
meanings, and the occurrence-coordinate join. Header labels alone are not
conformity evidence. The 152389 file's `GROUPING=array` describes map grouping;
its effective PTC cleaning grouping is `nw`, which must remain explicit.
The current `mini` file lacks per-sample/per-detector coordinate arrays, so the
join must be supplied or established from the retained inputs before gridding.

Only after the applicable input/preflight authorization, estimate the declared
training scatter and detector-noise dispersion for discovery data. Compare the
two fixed discovery observations before opening U/N evaluation maps. Neither
APT `sens`, an old map-quality label, nor the word `validated` establishes the
required current residual-noise distribution. If the intended contrast or
required input meanings are unavailable, report that fact and keep T2 gated.
