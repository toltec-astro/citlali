# Discovery input preflight

2026-09-09. Manager evidence for [Q02 r0.4](README.md); scientific input
admission remains pending. The earlier single-pass intake passes are retained.

## Access and identity

Read only the fixed 152389 and 123424 discovery files' schemas, scalar
metadata, detector identities, flags, telescope/pointing coordinates and chunk
records. No signal matrix, stored weight array, PSD, new noise statistic or
weighted-map outcome was examined. The 129081 file was not opened in this
preflight; its identity/access record remains the replacement intake.

Exact external paths, bytes and hashes are in [the counts](DISCOVERY_PREFLIGHT.json)
and [review manifest](REVIEW_MANIFEST.json). The received reductions were not
edited, copied, replaced or repaired. The reproducible
[preflight checker](check_discovery_inputs.py) writes only to its named scratch
directory and has no weighting or mapping implementation.

## Chunk labels do not describe stored support correctly

Both discovery files have 3,628 stored rows. Their twelve reported inclusive
intervals each have length 289 and cover only 3,468 rows. These 160 omitted
rows must not be called padding merely because the labels omit them.

At both reported source revisions, the writer appends each chunk at the
current file length. It then adds the current chunk length to *both* endpoints
of the preceding reported interval. That preserves the first interval's length
even when later chunks have different lengths. For example, the second stored
chunk is rows 289–593, but its reported interval is 305–593.

Let (a_j,b_j) be the reported inclusive interval. The exact writer recurrence
recovers l_0=b_0+1 and l_j=b_j-b_(j-1) for j>0. Both endpoints' increments
agree. The recovered lengths are:

```text
289, 305, 305, 305, 305, 305, 305, 305, 305, 305, 305, 289
```

Their cumulative half-open boundaries are:

```text
0, 289, 594, 899, 1204, 1509, 1814, 2119, 2424, 2729, 3034, 3339, 3628
```

152389's twelve original chunk-summary logs independently give these lengths;
its output_scan_index is 1..12. Its raw_scan_indices are rewritten output
indices and repeat the defective labels; they are not independent evidence.
123424's raw_scan_indices instead retain original raw inner bounds, whose
lengths divided by the configured factor two give the same output lengths.
Both stored telescope time vectors are finite and strictly increasing.

The future adapter should bind these recovered intervals to the exact file
hashes and preserve every original index field. It must not generalize this
recovery to arbitrary exports. Adoption belongs in the eventual T2 input
decision. No Citlali writer patch or reduction rerun is commissioned here.

## Coordinates, quantities and detector identity

Both exports declare altaz tangent coordinates in radians. In their reported
source, with elevation E, detector APT offsets x_t,y_t in arcseconds and stored
pointing offsets p_az,p_alt in arcseconds, the exported relation is:

```text
x = az_phys  + (cos(E) x_t - sin(E) y_t + p_az) pi/648000
y = alt_phys + (cos(E) y_t + sin(E) x_t + p_alt) pi/648000
```

152389 contains all of these fields with the same sample/detector layout as
its signal and flags. The writer binds them in the same append operation;
this is not a proposed cross-file join by matching array length or time.
Its lat_phys/alt_phys and lon_phys/az_phys aliases agree exactly. For 123424,
reconstruction over all twelve stored chunks agrees with its saved det_lon and
det_lat to at most 4.336808689942018e-19 rad in either axis, with identical
finite/nonfinite patterns. Neither discovery file has nonfinite reconstructed
coordinates where its saved sample flag is zero. This supports recovery of
the *legacy export relation*, not full contract conformity or equivalence to
every possible native science projection in the producer.

Flags take only 0/1 in both files. The 152389 schema explicitly says 0=good,
1=flagged; the exact historical writers/consumers corroborate that meaning.
There are 4,443 and 4,480 detector slots respectively with any flag-zero
occurrence. Their UIDs are finite and unique within each observation, and so
are their (network,UID) pairs. 123424's full rectangular inventory has duplicate
UIDs among inactive slots: never merge or re-key all columns by UID alone.
152389 has nonfinite APT flags among inactive slots. No flag-zero occurrence
in either file has a nonzero/nonfinite final embedded APT flag. Preserve the
per-occurrence flags; an APT array overwritten during output is not a substitute
for each chunk's retention state. Signal finiteness has not yet been checked.

152389's mini signal is float32, while 123424 is float64. A future paired test
must retain those exact input precisions and cannot describe 152389 as the
unrounded internal double-precision state. Both declare mJy/beam. Beam headers
are producer APT-derived array summaries in arcseconds, not newly fitted
pointing results or proof of an admitted nominal-beam contract.

The proposed source guard instantiated from the largest BMAJ/BMIN header is
27.38359774550894 arcsec for 152389 and 28.59742792952955 arcsec for 123424.
These exact values were used only to check the existing proposal's feasibility.
They are not adopted scientific beam/guard identities. No measured bright pixel
or fitted centroid defines either guard.

TelTime is incorrectly labeled rad; its values and source context describe
time. SAMPRATE is 122.0703125, while the median stored time step is approximately
0.016384 s after downsampling. Do not infer output cadence from SAMPRATE or
manufacture exposure seconds from output row count. The planned split uses
integer stored indices, with all time/frame/reference limitations explicit.

## Training feasibility under the unchanged proposal

For each recovered chunk of length l, take its first floor(l/2) rows as the
training candidate and the rest as evaluation. Count only flag-zero, finite
coordinates outside the fixed guard for training. A group is counted below
when it has at least one flag-zero, finite-coordinate evaluation occurrence.
No signal values, noise estimates, weights, pixel grid or support scores enter.

| Observation | Array | Groups with evaluation | Training count below 64 | Training count zero |
| --- | --- | ---: | ---: | ---: |
| 152389 | a1100 | 30,378 | 1,745 | 305 |
| 152389 | a1400 | 11,223 | 622 | 111 |
| 152389 | a2000 | 10,155 | 508 | 91 |
| 123424 | a1100 | 32,819 | 1,293 | 166 |
| 123424 | a1400 | 9,498 | 361 | 15 |
| 123424 | a2000 | 10,363 | 347 | 46 |

Of the groups below 64, 2,028 in 152389 and 645 in 123424 have evaluation
occurrences inside the central guard disk as well. This is not merely an
outer-field issue. The per-chunk/array counts are in DISCOVERY_PREFLIGHT.json.
These counts precede signal-finiteness and final MAP population admission;
they are not a finalized science denominator. They show that the present
whole-output proposal cannot simply proceed. Future input admission must not
hide this issue with exclusions selected to make N available.

The approved design requires N to be unavailable when a required training
group fails. It forbids silently clipping, substituting U or dropping detectors.
Zero-training groups also show why merely lowering 64 is not an adequate
repair. No alternative training law was tested here.

## Provenance recovered and still missing

152389's retained log identifies c31a60a0b74a7149d03d542966d6e35b77b8091c,
reports all twelve chunks completed and a completed Citlali process. Its
embedded Spack DAG identity is nuefnc6fkaul2m5x7l2x6qbblvsxezoe; deploy profile
is unmanaged and lock hash unavailable. A version string/DAG identity does not
verify executable bytes. Its configuration, telescope, APT-v2 inputs and
runtime/PTC/astrometry sidecars are locally available.

For 123424, the local
`2026-ENG-hero-multiyear-pointings-v1/apts/apt_123424_matched.ecsv`
matches all 5,905 embedded rows exactly in uid, nw, array, x_t, y_t, flag,
flxscale, responsivity, a_fwhm and b_fwhm. Four other same-basename candidates
do not. This strongly identifies a matching calibration candidate, but does
not establish the original executed path's file hash. The replacement's
reported source resolves to e0090e2da68ec118876aa7d3506b4dfb10ae696d.

The small remaining owner-supplied provenance, if retained on Unity, is the
replacement run log, exact executable/build identity, and hashes/identities of
its actual APT, telescope and ordered raw inputs. No new reduction, RTC dump or
large timestream export is requested. Local candidates are not silently
substituted for the executed inputs. No Unity query or transfer was attempted.

Both legacy bundles still lack an admitted complete frozen PTC/AST/MAP parent
chain and original-occurrence exposure authority. A future bounded empirical
permission must name its exact adapter and limited claims; reading these files
does not register new upstream profiles. Real-data noise uncertainty, source
response, grid/support and numerical execution remain unresolved decisions,
not implementation tasks that can be filled by inference.
