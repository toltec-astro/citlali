# WP-7 AST Route-Family Motion Review Repair

Date: 2026-08-31

Authority: `wp7-ast-scan-motion-v2`

Repair implementation revision:
`adbc013e2d4287fb5a32db8bc7f2b0112c1c88d7`

Status: **bounded repair and clean exact-SHA representative replay pass;
independent exact-SHA re-review pending**

## Reviewed package and finding

The first independent read-only conformance review examined exact package
`93de2cd9ca37f3740ceab98bf994ed684e9281ee`, whose direct implementation
revision was `672f907355a3f15f3ee987d92a5f7e95bbdc38b5`. It returned `HOLD` with
zero `BLOCKER`, one `MAJOR`, and zero `MINOR` findings.

The finding was confined to non-finite Beammap profile metadata. Source-level
profile admission classified non-finite `XLength`, `YLength`, or `ScanAngle`
as `unsupported_beammap_profile`, so occurrence-level membership was never
evaluated. This bypassed the approved ordering in which producer `Hold` is
tested first and a non-held occurrence with any non-finite required footprint,
angle, trajectory, correction, or source value receives
`nonfinite_membership_field`.

The review passed the other eight bounded conformance areas: exact custody,
route registry, source ownership, physical segments, preserved v1 numerical
operator, network-specific ALIGN semantics, exercised representative corpus,
and activation/status boundaries.

## Bounded repair

Revision `adbc013e2d4287fb5a32db8bc7f2b0112c1c88d7` separates finite unsupported
Beammap-profile values from non-finite values needed by occurrence-level
membership classification:

- categorical and finite unsupported profiles still fail closed with
  `unsupported_beammap_profile`;
- a held occurrence retains `producer_hold_active` even when a numeric
  footprint, offset, or angle field is non-finite;
- a non-held occurrence with such a non-finite field receives
  `nonfinite_membership_field` and makes the scan maximum incomplete; and
- no numerical estimator, source ownership, physical-segment, ALIGN mapping,
  RTC, filter, or publication behavior changed.

Focused coverage exercises both NaN and infinity for `XOffset`, `YOffset`,
`XLength`, `YLength`, and `ScanAngle`, plus finite invalid-footprint
fail-closed behavior.

## Exact repaired evidence identities

The observation-152390 acceptance record is:

```text
/Users/gwilson/work_toltec/local_data/citlali-validation/wp7/ast-route-family-motion/adbc013e2d4287fb5a32db8bc7f2b0112c1c88d7/ast-scan-motion-152390-v2.json
```

Its SHA-256 is:

```text
366adc630a03ff3d5fc2bb641ee45c8dc7a76e747bc1d79e93deb5d8d07d5c2c
```

The exact AST acceptance executable SHA-256 is:

```text
9659d13df686be5370430c069bd02b9df82e625a33df61210ee4d2c33e725b5e
```

The exact-package validator authenticates both byte streams and passes. The
Science witness remains exactly 62,109 physical members, 62,067 valid
derivatives, two accepted telemetry defects, and maximum
`221.40490828695155 arcsec/s` at telescope record 16973. All product,
support, mapping, and chunk-partition mismatch counts remain zero.

The repaired seven-case corpus is:

```text
/Users/gwilson/work_toltec/local_data/citlali-validation/wp7/rtc-filter-census/adbc013e2d4287fb5a32db8bc7f2b0112c1c88d7/corpus.json
```

Its SHA-256 is:

```text
c8dcc459e9df3c84e5f7c7bdaf73bed0672500d381af3acf3ad5b63bc62b1bee
```

The exact census executable SHA-256 is:

```text
ebf9b32622e97e32fcc5914a8b32bcfe654d45e47656218b6da8ed94924eac35
```

The corpus declares a clean source, contains all seven required cases, and
binds these exact result identities:

| Case | Members | Segments | Valid derivatives | Maximum (`arcsec/s`) | Result SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| Beammap 148670 | 61,816 | 252 | 57,563 | 75.29115202170063 | `ad148dbd5b5d12f88cf9bf0bf89a81a603587ad9d350428aab25e260f2d46c7f` |
| OOF 152385 | 3,271 | 1 | 3,251 | 135.38178144221268 | `d7e4c3098d5bf308a6e6b4074cfb49c04c80d172a1242bd8302fa1ca2b8c3033` |
| OOF 152386 | 3,353 | 1 | 3,333 | 158.40684078739127 | `d6835913d92cfe172c00b85633a55d23d5c4cf3acb97ed3f5cbe027e1bbf3644` |
| OOF 152387 | 3,266 | 1 | 3,246 | 140.54776730507191 | `f5599fe31691cb6fca79eb415602de7514a30d383256e12784ebb78c535f75a8` |
| Science 152390 | 62,109 | 1 | 62,067 | 221.40490828695155 | `4e15a805f4dcfdbad5ec9e9e553af9880283aadf4bef28e73ad144abf3ea4f2c` |
| Pointing 152391 | 3,309 | 1 | 3,289 | 171.69626792459061 | `6f47da649b21216191d16523a552cc0b104fa8612462c55ddc5a900f9f79b081` |
| Science 152392 | 62,339 | 1 | 62,319 | 281.4108785752698 | `30b7b120769e0a8c4d9086bf96404e5725e66be16c6b832a55edfbeaf5e7f844` |

After removing only the new source revision, Citlali version, and executable
SHA identities, every case's complete scientific payload is byte-for-byte
equivalent as canonical JSON to the first package. No result, support, cause,
mapping, participant, cadence, factor-census, or activation field changed.

## Verification

The repair passes:

- all 24 focused AST/ALIGN/census CTests;
- all 894 enabled repository CTests, with the one established disabled test
  unchanged;
- all 207 baseline-tool tests;
- all 14 census/acceptance verifier mutation tests;
- all 129 config-preflight unit tests and every downstream audit;
- the clean ignored-source and exact dependency-state build gate;
- the local CLI, acceptance-runner, and census-runner builds; and
- the authenticated observation-152390 and seven-case exact-SHA replays.

## Disposition

This repair addresses only the independent review finding. It does not revise
the owner authority, select a decimation factor, certify a filter, activate an
RTC route, publish a persistent product, or alter CAL, PTC, MAP/JINC, legacy
Beammap, or nonidentity numerical behavior.

The AST v2 conformance gate remains open until a fresh independent exact-SHA
re-review passes the repaired package. The first reviewed package remains
preserved as historical evidence; this document is its bounded successor
handoff.
