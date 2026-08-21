# JINC Working-Support Unity Validation — 2026-08-21

## Verdict

The owner-run `redu04` Beammap of observation 148670 passes the targeted Unity
gate for the JINC working-support repair. With the failed `redu03` requested
and merged configurations held byte-identical, the repaired executable removes
the pathological near-cancellation tail and restores healthy detector yield,
map RMS, empirical-template construction, reference selection, and kernel
amplitude.

This accepts the bounded defect repair for the tested observation. It does not
create a full accepted-run ledger record or establish general JINC production
readiness because four indexed large FITS products were intentionally omitted
from the local retrieval.

## Evidence Scope and Identity

The evidence is the owner-retrieved local snapshot at:

```text
/Users/gwilson/work_toltec/local_data/citlali-validation/v1/beammaps/3c273/reduced/redu04
```

The Unity live system was not accessed by Codex. The snapshot records:

- observation: 148670;
- reduction: Beammap, JINC detector-grouped mapmaking;
- Citlali: `v4.0.0-3657-ge77460cf`;
- KIDs: `04088da`;
- Tula: `f30f81d`;
- requested config SHA-256:
  `d81ac8b1aa52c06c0ef7d69158c802850499695aa9d614ebaf996147ba736788`;
- merged config SHA-256:
  `5035d010d4cf0d2ce1a6d9b58626b0beefcc09c8266342756e7fbf8fbe04e1eb`.

Those two config hashes exactly match failed JINC run `redu03`. The repaired
Unity commit `e77460cf` is the Unity cherry-pick of local repair commit
`59a142b334a4d7882f85f031ba090cdd74171839` on top of the retained
inactive-notch hotfix.

## Controlled Result

| Metric | Healthy naive `redu02` | Failed JINC `redu03` | Repaired JINC `redu04` |
| --- | ---: | ---: | ---: |
| Final good detectors | 4,829 | 462 | 4,973 |
| Final good fits | 5,135 | 5,135 | 5,135 |
| Sig2Noise rejections | 244 | 4,754 | 181 |
| Median map RMS, all detectors | `1.79956e-7` | `4.12372e-6` | `1.03564e-7` |
| Empirical templates | 500 per array | none | 500 per array |
| Template/fallback calibration | 5,120 / 114 | 0 / 5,234 | 5,133 / 101 |
| Reference candidates | 490 | 51 | 494 |
| Final kernel peak maximum | `1.085` | `1.403e4` | `7.798` |
| Error or critical log records | 0 | 0 | 0 |

The repaired 4,973-detector yield exactly matches the older controlled JINC
result cited in the incident record. The repaired run's 295 warning records
are comparable to the healthy naive run's 296; they are dominated by the
existing per-map insufficient-weighted-pixel fit warnings.

The repair was active rather than merely benign. The three normalization
summaries report working-support threshold downgrades for 7,810,611,
9,901,143, and 9,859,003 pixels. In the final iteration the failed run's kernel
peak tail (`p90=3.982`, maximum `1.403e4`, mean `8.599`) becomes bounded
(`p90=1.149`, maximum `7.798`, mean `0.9135`).

The reduction completed normally at 2026-08-21 19:25:38 UTC-equivalent log
time after 7,386.889 seconds of reduction iterations.

## Retained Artifact Digests

| Artifact | SHA-256 |
| --- | --- |
| `citlali.log.gz` | `43c6c1432f9ceafff0a9800d6d781dcc83208c84f5bbb30a86a5d9503b731c53` |
| `apt_commissioning_beammap_148670_citlali_fit_qc.ecsv` | `c59c30b469d6ba8251394e5a2e98c4a9ba07b151655e42224802d76580e26d03` |
| `mapmaking_provenance.yaml` | `73ccbe6b890ef1c828fc9c146da734094aa65d385c14c1b14223bef41ce8e0b7` |
| `config_source_manifest.yaml` | `a2716582f2b267e7d40de939b8c9cb344b23789b87b20da6904bc416a755a5e5` |
| compact APT v2 `manifest.ecsv` | `b6294bf93cba7e5ebbd0b16c4061d2735eaf9299c72b8e786f13b974ba50b7e9` |
| compact APT v2 `manifest.ecsv.sha256` | `1176dead63b4cc2c43c7063536d75f7daf9302e8e65abb3558a3ddf940d3b1fb` |

## Deliberately Unavailable Local Products

The downloaded `148670/raw/index.yaml` names six split map cubes. The following
four were not downloaded:

```text
toltec_commissioning_a1100_beammap_148670_citlali_flag0_good.fits
toltec_commissioning_a1100_beammap_148670_citlali_flag1_bad.fits
toltec_commissioning_a1400_beammap_148670_citlali_flag0_good.fits
toltec_commissioning_a2000_beammap_148670_citlali_flag0_good.fits
```

The a1400 and a2000 bad-detector cubes are present locally. Unavailable FITS
content must remain recorded as unavailable; no digest or product comparison
is inferred for it.

## Promotion Boundary

The evidence closes the observed JINC numerical incident and supports carrying
the bounded repair forward. Before creating a complete accepted-run ledger
record or claiming full product-level validation, retrieve or otherwise make
available the four omitted FITS products and run the governed strict Beammap
product audit against the selected predecessor/successor profile. Preserve
`redu03` as rejected diagnostic evidence and `redu04` as the repair snapshot.
