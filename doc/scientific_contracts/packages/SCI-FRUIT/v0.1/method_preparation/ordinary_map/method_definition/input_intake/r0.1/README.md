# Weighting-screen input intake: full files, wrong iteration state

2026-09-09. Status: received candidates are **not admitted as single-pass
weighting inputs**. They remain preserved historical products. No weighting
comparison or signal/noise analysis was performed.

## Program adherence and prior-work recovery

Follow the [charter](../../r0.4/inputs/program/README.md),
[pilot workflow](../../r0.4/inputs/program/PILOT_PROCESS_REVIEW_2026-08-16.md),
[roadmap](../../r0.4/inputs/program/DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md),
[frozen recovery](../../r0.4/PRIOR_WORK.md) and the
[current Q02 preparation record](../../q02_review/r0.3/README.md).
The [collection request](../../q02_review/r0.3/DATA_COLLECTION_REQUEST.md)
requires one ordinary uninjected pass with FRUIT disabled. This intake uses
the owner's supplied directory as read-only evidence of delivered inputs.
It does not change A/B approval, C's open decision, any frozen source or the
independent-author reference inventory.

The owner supplied:

```text
/Users/gwilson/work_toltec/local_data/beammaps/pointings/reduced/redu01
```

## What arrived

| Observation | PTC bytes | Signal/flag shape | Coordinates / chunks |
| --- | ---: | --- | --- |
| 123424 / Neptune | 760,674,549 | 3,628 samples by 5,905 detectors, float64 | det_lat and det_lon have the same shape, in radians; 12 declared chunks |
| 129081 / 3c273 | 710,131,605 | 3,628 samples by 5,512 detectors, float64 | det_lat and det_lon have the same shape, in radians; 12 declared chunks |

Both files carry signal units `mJy/beam`, flags, detector/array/network IDs,
chunk bounds, per-chunk weights and detector-by-sample coordinates. Thus the
requested full output surface is present. Exact coordinate frame/association,
flag meanings and current named-use permissions have not been established by
this schema check. The 12 chunk-bound rows have gaps; no padding/overlap samples
were silently promoted to science support. Those rows are recorded as metadata.

The directory contains 22 files, including one saved configuration for both
observations, index records and ordinary map/diagnostic files. The two PTC files
total 1,470,806,154 bytes. Embedded Citlali version is `v4.0.0-62-ge0090e2d`,
with KIDs `4a3428e` and Tula `28875ba`. These are reported identities, not a
verified complete executable/build identity. No run log or modern provenance
sidecars are present in this supplied directory.

## Blocking mismatch

The saved `citlali_o123424_0_2_c2.yaml` explicitly has:

```yaml
timestream:
  fruit_loops:
    enabled: true
    max_iters: 10
    save_all_iters: false
```

Both PTC files report `FRUITLOOPS_ITER=9` and
`CONFIG.FRUITLOOPS.MAXITER=10`. Taken together, these identify a multi-iteration
FRUIT run's final saved state, not the requested ordinary initial PTC pass.
The legacy `CONFIG.FRUITLOOPS` scalar contains 22529, which is not interpreted
as a reliable Boolean; the disposition rests on the explicit saved YAML and
iteration metadata. No implementation bug is diagnosed or repaired here.

The model/residual processing may already have changed the signal, learned
state, weights and support. Gridding these arrays with U/N would therefore
answer a different question. Do not relabel them bootstrap inputs, try to undo
feedback from a final file, or substitute this population/state into T2.
The original data and all products remain untouched.

## Required correction to collection

For the same two observations, retain full PTC output and the ordinary recipe,
but set these two existing configuration leaves before an ordinary run:

```yaml
timestream:
  fruit_loops:
    enabled: false
    max_iters: 1
```

The [settings snippet](SINGLE_PASS_SETTINGS.yaml) is for these leaves in the
complete configuration; it is not a standalone runnable config. Keep original
inputs, no injected source or restart/model parent, and all requested chunks.
Keep RTC output disabled, as it already is in the delivered configuration.
Use a fresh output directory and preserve `redu01`. Save the actual configuration
and run log with the replacement export; include executable/input identities
where the older executable does not emit modern provenance sidecars.

This is correction of the existing collection request, not a new FRUIT
experiment or permission for Codex to run on Unity. The owner performs any
new ordinary reduction/export. On receipt, check the actual single-pass state
against both configuration and log before scientific use; do not rely on the
legacy Boolean header alone. Then complete the existing T2 input/evidence and
execution bindings. The proposed numerical tolerances are still unapproved.

## Access and verification

129081 remains reserved for new weighting outcomes. Inspection was limited to
file identities, schemas, scalar input/configuration headers and small chunk
index arrays; no signal, flag, coordinate, weight, PSD, fitted-source or map
values were examined. SHA-256 hashing reads payload bytes only for identity.
Neither discovery noise statistics nor reserved replication outcomes were
computed. No files in the supplied directory were altered.

[INPUT_MANIFEST.json](INPUT_MANIFEST.json) binds the two PTC files, saved config
and index records, lists the supplied filenames/sizes, and records the exact
observations above. Its sidecar hashes the manifest. The manifest also binds
this disposition and settings snippet. Earlier Q02 reviews and all frozen
packets remain unchanged; no implementation, numerical test, replay, transfer,
Unity access or push occurred.
