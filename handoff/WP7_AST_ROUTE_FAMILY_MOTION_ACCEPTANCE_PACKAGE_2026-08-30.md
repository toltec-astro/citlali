# WP-7 AST Route-Family Motion Acceptance Package

Date: 2026-08-30

Authority: `wp7-ast-scan-motion-v2`

Implementation revision:
`672f907355a3f15f3ee987d92a5f7e95bbdc38b5`

Status: **clean exact-SHA implementation and seven-case representative
execution pass; fresh independent exact-SHA conformance review pending**

## Bounded claim

The implementation preserves the complete accepted v1 numerical operator and
adds only the approved v2 route-family and physical-membership authority:

- exact Science, OOF, and Pointing Lissajous profiles;
- the exact zero-offset, no-hold-during-turns, azimuth-coordinate,
  continuous rectilinear Beammap profile;
- the non-hold inclusive realized-footprint membership predicate;
- compact physical-segment identity and segment-bounded estimator support;
- typed unsupported-profile, non-finite membership, hold, and outside-footprint
  causes; and
- unchanged network-specific ALIGN mapped views with no common analysis grid.

The immutable source owns each producer plane once. The derived product owns
one compact record plane and one compact first-record entry per physical
segment; it does not duplicate the source time, direction, or seven Beammap
membership planes.

## Exact implementation and acceptance identities

The exact observation-152390 acceptance record is stored at:

```text
/Users/gwilson/work_toltec/local_data/citlali-validation/wp7/ast-route-family-motion/672f907355a3f15f3ee987d92a5f7e95bbdc38b5/ast-scan-motion-152390-v2.json
```

Its SHA-256 is:

```text
67e31b5f862609527f0dedeb3464513676aba5305b95bd972f3287aa0aaa8a5a
```

The exact acceptance executable SHA-256 is:

```text
bdbe4037b998fb4e7935b69f7a98356a26da52c82dd7b28930ddef86decf2042
```

The v2 exact-package validator authenticates both byte streams and passes. The
62,109-record Science witness retains 62,067 valid derivatives, the two
accepted telemetry defects at records 2504 and 12971, and the exact maximum
`221.40490828695155 arcsec/s` at telescope record 16973. The sole physical
segment begins at record zero. All record, support, summary, network identity,
mapped value, and chunk-partition mismatch counts are zero.

## Clean seven-case corpus

The complete compact corpus is stored at:

```text
/Users/gwilson/work_toltec/local_data/citlali-validation/wp7/rtc-filter-census/672f907355a3f15f3ee987d92a5f7e95bbdc38b5
```

`corpus.json` has SHA-256:

```text
4fbb9e62c318d05a839205e3d529cf3fc1f7e6cb31955ce2fa277b6562c62e33
```

The exact census executable SHA-256 is:

```text
95ce5e852534a476682078339ee730b91ac641a1ffd79f2e5374214349d398ce
```

The corpus declares `source_clean: true`, binds all input bytes and canonical
APT bundles, and contains these post-approval results:

| Case | Profile | Members | Segments | Valid derivatives | Maximum (`arcsec/s`) |
| --- | --- | ---: | ---: | ---: | ---: |
| Beammap 148670 | rectilinear continuous Beammap | 61,816 | 252 | 57,563 | 75.29115202170063 |
| OOF 152385 | OOF Lissajous | 3,271 | 1 | 3,251 | 135.38178144221268 |
| OOF 152386 | OOF Lissajous | 3,353 | 1 | 3,333 | 158.40684078739127 |
| OOF 152387 | OOF Lissajous | 3,266 | 1 | 3,246 | 140.54776730507191 |
| Science 152390 | Science Lissajous | 62,109 | 1 | 62,067 | 221.40490828695155 |
| Pointing 152391 | Pointing Lissajous | 3,309 | 1 | 3,289 | 171.69626792459061 |
| Science 152392 | Science Lissajous | 62,339 | 1 | 62,319 | 281.4108785752698 |

Every maximum is available with zero maximum causes. Whole-product and a
shuffled three-part engineering schedule agree exactly for every record and
summary. Across 77 independent network inputs and 7,904,409 native
occurrences, identity mismatch, missing mapped support, and unexpected-error
counts are zero. Beammap has 1,540,101 available mapped occurrences and
2,680,604 explicit unavailable occurrences produced by physical membership
and estimator support; it does not manufacture a cross-network slot.

Per-case result SHA-256 values are:

| Case | Result SHA-256 |
| --- | --- |
| Beammap 148670 | `b835c571a17b031f1dcb87f8560243665355af2e4bd9a931fbd8d9bd47de43b4` |
| OOF 152385 | `53051404bbddb6e67a8d5464829a5b34906bee1bfe31f160ad327f36c1dc632c` |
| OOF 152386 | `d77425ec11af231fd0a3c8125ba15696952d015bb3074bf7aca430e196e24cd9` |
| OOF 152387 | `402edbafa3ff44506ae23c1d9cd77f990da9dfcff202a99caf84cd6a6877936b` |
| Science 152390 | `83a12f012f5d54b67700e4eb60100fbd8eef992f5f80da4c97e3ae96466134d0` |
| Pointing 152391 | `8ae67be37c2c5a439a275851e496a64212c48f1e14a670a46b4e6b7245bff309` |
| Science 152392 | `4387efb58c0456afa358cfe2d745cbb94b3911e7f0bf5d06bcfb49578c8969b3` |

## Verification

The exact implementation passes:

- all 16 focused AST/ALIGN CTests, including below/exact/above footprint,
  nonzero hold, fail-closed profiles, short segments, support isolation,
  network timing, and chunk invariance;
- all 893 enabled repository CTests, with the one established disabled test
  unchanged;
- the clean ignored-source and exact dependency-state gate;
- all 207 baseline-tool tests;
- all 14 census/acceptance verifier mutation tests;
- all 129 config-preflight unit tests and every downstream audit;
- the local `citlali_cli`, acceptance-runner, and census-runner builds; and
- the clean exact-SHA observation-152390 and seven-case representative runs.

## Disposition

The post-approval execution closes local implementation and representative
evidence construction. It makes the Beammap, OOF, Pointing, and Science motion
census eligible for the F0 conformance decision, but it does not claim final
acceptance before the required fresh independent exact-SHA review.

This package does not select a decimation factor, certify a filter, estimate
`M>1` support erosion, activate a nonidentity RTC route, publish a persistent
AST/RTC product, or alter CAL, PTC, MAP/JINC, or legacy Beammap behavior.
