# WP-7 RTC Occurrence-Speed D0/F0 Evidence

Date: 2026-08-30

Authority: `wp7-rtc-occurrence-speed-admission-v1`

Harness source revision:
`b9589d28a85c1f348f02416d230cb4db1555dd98`

Status: **exact-SHA custody/cadence census complete; F0 motion census complete
for the authorized Science/Lissajous cases; Beammap and OOF motion evidence
pending; no factor selection, filter certification, RTC route activation, or
production readiness claim**

## Scope

This record covers only the first measurement increment of the accepted
[filter/downsampling certification plan](../doc/WP7_RTC_FILTER_DOWNSAMPLING_CERTIFICATION_TEST_PLAN_2026-08-30.md):

- exact fixture/build/input custody;
- independent network-native timing, cadence, gap, and mapped-AST inventory;
- analytic structural upper-speed ceilings for each array, observed cadence
  family, and integer factor `M=1..256`;
- raw occurrence admission and retained-run accounting where the accepted AST
  product supplies valid mapped motion; and
- exact `M=1` occurrence-local support accounting.

It does not design a filter, estimate `M>1` support erosion, select a factor,
run a nonidentity RTC operator, publish a timestream product, or close any
Beammap, MAP/JINC, OOF/fruitloops, response, noise, alias, or performance gate.

## Immutable evidence identity

The complete local corpus is at:

```text
/Users/gwilson/work_toltec/local_data/citlali-validation/wp7/rtc-filter-census/b9589d28a85c1f348f02416d230cb4db1555dd98
```

Its compact root record is `corpus.json` with SHA-256:

```text
90443f6490764b938f7c9c0e3c4e3635eaa0eecc777431b4974d106c5fa1cc33
```

The exact census executable SHA-256 is:

```text
da93313d673516cdcb67c54093aecce6ce5449215c153adf1b47bf7245b46cf0
```

The corpus declares `source_clean: true`, contains seven cases, and binds each
raw detector, telescope, configuration, canonical APT bundle, and declared
auxiliary input by path, byte count, and SHA-256. The approximately 16 MB
evidence package contains compact counts and summaries, not copied timestream
planes or per-sample provenance.

## Harness correction proved

The v2 harness no longer accepts a scan maximum as an eligibility input. It:

1. derives fixed structural ceilings from array identity, exact input cadence,
   factor, the accepted optical-band and four-samples-per-Airy-FWHM rules, and
   the `1.05` velocity and `0.9999` cadence margins;
2. treats the ceiling as inclusive and assigns
   `scan_speed_above_mode_support` only for a mapped valid speed strictly above
   it;
3. independently maps and classifies every network occurrence without a
   common analysis grid;
4. reports raw counts and durations using the primitive acquisition duration
   `AccumLen/FpgaFreq`;
5. breaks retained runs at native packet gaps, unavailable AST, the lower-speed
   boundary, and the candidate upper-speed boundary;
6. records zero extra support erosion for unfiltered `M=1`; and
7. records `M>1` support erosion as pending exact coefficients and half-support.

Every domain explicitly records that automatic factor selection is not
authorized. The raw telescope maximum remains present only as an AST
diagnostic.

## Fixture and mapping census

| Case | Networks | Native occurrences | Detectors | Matched APT relation | Accepted mapped motion | D0 identity ready |
| --- | ---: | ---: | ---: | --- | --- | --- |
| Beammap 148670 | 11 | 4,220,705 | 5,234 | no | unavailable | no |
| OOF 152385 | 11 | 86,468 | 5,518 | yes | unavailable | yes |
| OOF 152386 | 11 | 86,742 | 5,518 | yes | unavailable | yes |
| OOF 152387 | 11 | 86,559 | 5,518 | yes | unavailable | yes |
| Science 152390 | 11 | 1,666,908 | 5,518 | yes | available | yes |
| Pointing 152391 | 11 | 85,661 | 5,518 | yes | unavailable | yes |
| Science 152392 | 11 | 1,671,366 | 5,518 | yes | available | yes |

Across all 77 network inputs:

- native timing identity mismatch count is zero;
- available mapped-AST support loss count is zero;
- every measured cadence family is `122.0703125 Hz`;
- maximum observed contiguous-interval fractional deviation is
  `2.1489337086677551e-5`, within the accepted 100 ppm margin;
- no common analysis grid was requested; and
- no RTC route or persistent scientific product was activated.

`wp7-ast-scan-motion-v1` intentionally admits physical motion only for the
approved Science/Lissajous family. Therefore the zero upper-speed exclusion
counts in Beammap, OOF, and Pointing are not evidence of slow motion: every
mapped occurrence is AST-unavailable, and those route families remain
indeterminate for F0 motion admission.

Beammap's verified baseline compact-v2 APT bundle supplies the required source,
network, array, and detector inventory, but it has no matched detector relation.
The harness therefore truthfully records `d0_fixture_identity_ready: false` for
that case.

## Structural domains

At `122.0703125 Hz`, the exact `M=1` structural upper bounds are:

| Array | Upper speed (arcsec/s) | Governing constraint | Factors with a nonempty `v >= 1 arcsec/s` structural domain |
| --- | ---: | --- | ---: |
| a1100 | 135.9681197283374 | four samples per Airy FWHM | 135 |
| a1400 | 172.81929236498959 | four samples per Airy FWHM | 172 |
| a2000 | 246.55552377405178 | four samples per Airy FWHM | 246 |

The ceiling scales inversely with `M`. Candidates above the listed factor
counts have no overlap between the accepted inclusive lower boundary
`v >= 1 arcsec/s` and their structural upper bound. This is an analytic domain
fact, not a filter rejection or a production factor choice. Every candidate
ceiling remains an upper bound pending passband, mapped-response, phase, alias,
support, and edge certification.

## Exact M=1 Science/Lissajous admission

Counts below are network-native occurrences aggregated only across the
networks belonging to the named array. They are neither telescope-record
counts nor detector-occurrence-cell counts.

| Observation | Array | Lower-speed/AST-valid base | Above-ceiling occurrences | Base fraction | Primitive duration (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| 152390 | a1100 | 904,402 | 17,313 | 1.914304% | 141.828096 |
| 152390 | a1400 | 452,208 | 1,514 | 0.334802% | 12.402688 |
| 152390 | a2000 | 301,465 | 0 | 0% | 0 |
| 152392 | a1100 | 907,368 | 12,960 | 1.428307% | 106.168320 |
| 152392 | a1400 | 453,683 | 1,688 | 0.372066% | 13.828096 |
| 152392 | a2000 | 302,454 | 90 | 0.029757% | 0.737280 |

Independent network times produce slightly different exact counts within an
array. For example, the six a1100 networks in observation 152390 exclude
between 2,883 and 2,890 occurrences each at `M=1`; no shared slot or copied
network time axis is involved.

The complete artifacts retain per-network counts, durations, retained-run
counts, and longest runs for all 256 factors. They deliberately contain no
weighted-exposure, spatial-coverage, or `M>1` support-erosion claim because the
required weights, pointing/product comparison, and filter footprints do not
yet exist.

## Verification

Exact harness revision `b9589d28a85c1f348f02416d230cb4db1555dd98`
passes:

- all 63 focused WP-7 timestream/AST/census tests;
- all four fail-closed Python orchestrator tests;
- all 889 runnable repository CTests, with the one established disabled test
  unchanged;
- the required config preflight, including all 129 unit tests and downstream
  audits;
- JSON parsing for every case and the root corpus; and
- the seven-case exact clean-SHA corpus replay.

## Disposition and next prerequisites

This execution closes the harness correction and the available Science/
Lissajous portion of F0. It does not close the complete D0/F0 gate for the
declared certification corpus.

Before filter-family research can govern the complete required route matrix:

1. Beammap 148670 needs a matched detector relation or an explicitly approved
   equivalent D0 identity closure.
2. The owner must resolve how the already-required Beammap and OOF route tests
   obtain authorized occurrence-level motion/admission evidence without
   contradicting the accepted physical-scan membership of
   `wp7-ast-scan-motion-v1`. No local velocity estimator or silent family
   broadening is authorized.
3. `M>1` support erosion remains pending F1 coefficients and exact
   half-support.
4. Weighted exposure and spatial coverage remain pending their existing
   scientific products and definitions.

Science-case PSD and line discovery can be measured separately, but no bank
entry, automatic factor-selection rule, or production nonidentity RTC route may
be frozen from this evidence.
