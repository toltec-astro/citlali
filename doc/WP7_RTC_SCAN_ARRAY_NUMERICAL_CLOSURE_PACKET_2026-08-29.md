# WP-7 RTC Scan/Array Numerical-Closure Packet

Prepared: 2026-08-29

Scientific-owner disposition: approved 2026-08-30

Authority identity: `wp7-rtc-scan-array-numerical-policy-v1`

Status: **approved numerical authority; nonidentity RTC implementation remains
blocked on conforming AST velocity/validity authority and implementation
gates**

Controlling structural authority:
[WP-7 RTC Scan/Array Planning Scientific-Owner Authority](WP7_RTC_SCAN_ARRAY_PLANNING_OWNER_AUTHORITY_2026-08-29.md)
at `38644343a4f8cfa213c8cab87c06753377704e12`

This packet fixes one explicit owner-approved value or rule for every numerical
field left open by the controlling decision. It does not alter the accepted
network-timed `M=1` route, promote the feasibility filter estimates to
certified coefficients, or construct the nonidentity RTC method.

## Approved owner disposition

The scientific owner approved the following as version 1 of the bounded TolTEC
RTC scan/array numerical policy. A change to any value below requires a new
policy identity; it must not appear as a software default or observation-local
override.

1. Use exact nominal center frequencies `272 GHz`, `214 GHz`, and `150 GHz`
   for `a1100`, `a1400`, and `a2000`, respectively.
2. Use an exact `50.0 m` clear circular aperture convention and the normalized
   uniformly illuminated, unobscured circular-aperture Airy intensity profile
   as the planning beam.
3. Use Airy intensity FWHM coefficient
   `C_beam = 1.028993969962188`, with width `C_beam lambda/D`.
4. Preserve the full nonzero one-dimensional temporal support of that scanned
   beam: `f_sci = v_plan D/lambda`, rather than selecting a conventional
   crossing or attenuation point.
5. Require each point-source product metric in the table below to pass
   independently at `1e-3`; do not trade one metric against another.
6. Require filter passband amplitude ripple at most `1e-4`, exact centered
   zero phase after occurrence-time compensation, and DC-gain error at most
   `1e-12`.
7. Bound the total folded alias power gain by `1e-6` at every retained-band
   frequency.
8. Require at least `4` output samples per Airy intensity FWHM.
9. Permit every integer factor `M` in the finite set `[1, 256]`.
10. Permit one initial realization family: an odd-length, real, symmetric
    Type-I Kaiser-windowed-sinc FIR with the exact construction and tie rule
    below. Direct convolution and an exactly equivalent polyphase execution
    are implementation strategies, not different scientific operators.
11. Limit filter half-support to `5.0 s`; do not pad or extrapolate across an
    admitted-run boundary.
12. Use binary64 policy calculations, coefficients, inputs, outputs, and an
    increasing-source-occurrence binary64 fused-multiply-add accumulation.
    Disable contraction changes and reassociation outside that declared
    operation. Compare admission and plan boundaries directly in binary64
    after applying the declared safety margins.
13. Plan with `v_plan = 1.05 v_max` and
    `f_sample,safe = 0.9999 f_sample,in`. A source whose accepted uncertainty
    exceeds either allowance fails planning instead of silently receiving a
    wider margin.
14. Select `M=1` only when the input cadence itself still meets the science
    passband and four-samples-per-FWHM requirements. Otherwise publish no
    admitted ordinary astronomical product and use typed cause
    `input_cadence_inadequate_for_science_band`.

Item 14 closes a necessary edge case in the earlier fallback language. An
identity operator cannot recover beam bandwidth or sampling absent from its
input. Treating every `M=1` fallback as conforming without this check would
make the preservation claim false at sufficiently high scan speed.

## Array-model consequence

Use the exact SI speed of light `c = 299792458 m/s`. The approved array artifact
therefore contains:

| Array | `nu_0` (GHz) | `lambda` (mm) | `lambda/D` (arcsec) | Airy intensity FWHM (arcsec) | temporal cutoff per `1 arcsec/s` (Hz) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `a1100` | 272 | 1.1021781544 | 4.5468112694 | 4.6786413788 | 0.2199343542 |
| `a1400` | 214 | 1.4008993364 | 5.7791246041 | 5.9466843693 | 0.1730365875 |
| `a2000` | 150 | 1.9986163867 | 8.2448844352 | 8.4839363668 | 0.1212873277 |

The 50 m telescope and the 1.1/1.4/2.0 mm bands are established instrument
facts in the [NIST TolTEC instrument publication](https://www.nist.gov/publications/millimeter-wave-polarimeters-using-kinetic-inductance-detectors-toltec-and-beyond).
The original optical design describes 150/220/280 GHz focal planes in
[Bryan et al. 2018](https://arxiv.org/abs/1807.00097). Current TolTEC beammap
calibration code consistently uses 272/214/150 GHz. This policy selects that
current calibration-software convention exactly; the historical optical-design
values remain evidence and are not silently mixed into the artifact.

The full ideal-aperture optical cutoff is the conservative bandwidth boundary:
real-beam broadening is not credited to reduce it. The Airy FWHM is used only
for the independent output-sampling guard and the declared product metrics.

## Science-band and product rule

For array `a` and scan `s`, define

```text
v_plan,s = 1.05 * v_max,s
lambda_a = c / nu_0,a
f_sci,a,s = v_plan,s * D / lambda_a
```

where angular velocity is converted to radians per second. The ideal circular
aperture optical transfer function has no support beyond `D/lambda` cycles per
radian. Preserving the complete temporal image of that support avoids an
arbitrary `3 dB`, FWHM-crossing, or historical-frequency decision.

Compare the realized, sampled and filtered temporal point-source response with
the unfiltered authoritative Airy response at the same output epochs. Every
metric is a separate maximum bound:

| Metric | Approved bound |
| --- | ---: |
| relative point-source peak error | `1e-3` |
| relative integrated-flux error | `1e-3` |
| maximum absolute normalized-profile residual | `1e-3` of the reference peak |
| relative Airy-FWHM error | `1e-3` |
| absolute centroid displacement | `1e-3` Airy FWHM |
| relative calibration-transfer magnitude error | `1e-3` |

Test the full sub-sample phase domain, including the extremizing phase, not
only a source centered on an output sample. A plan passes only if every metric
passes. The `1e-4` passband-ripple allocation reserves most of the product
budget for sampling, coefficient rounding, and independent validation rather
than consuming the complete `1e-3` allowance in the filter response.

## Alias norm

Let `f_out = f_sample,safe/M`. At every `|f| <= f_sci`, fold every nonzero
downsampling image that exists below input Nyquist and require

```text
sum over nonzero images k of |H(f + k f_out)|^2 <= 1e-6
```

with the exact signed/mirrored frequency mapping used by real sampled data.
This is a worst-frequency operator power-gain bound. It does not assume a
particular detector noise spectrum and does not average a failure over
frequency. A sufficient per-image amplitude allocation for factor `M > 1` is

```text
delta_s,M = sqrt(1e-6 / (M - 1)).
```

The final candidate must be checked against the summed norm; the per-image
bound is a design allocation, not a replacement for that check.

## Exact approved FIR rule

For every candidate `M > 1`:

```text
f_p  = f_sci
f_sb = f_out - f_sci
f_c  = (f_p + f_sb) / 2
```

Reject the factor if `f_sb <= f_p`, if output beam sampling is below four, or
if no qualifying filter fits the support bound. Allocate

```text
delta = min(1e-4, sqrt(1e-6/(M-1)))
A = -20 log10(delta) dB
```

and derive Kaiser `beta` exactly as

```text
beta = 0                                      when A < 21
beta = 0.5842*(A - 21)^0.4 + 0.07886*(A - 21) when 21 <= A <= 50
beta = 0.1102*(A - 8.7)                       when A > 50.
```

For odd tap counts `N = 3, 5, 7, ...`, let `L = (N-1)/2` and construct

```text
h_ideal[n] = 2*f_c/f_sample,in
             * sinc(2*f_c/f_sample,in * (n-L))
w[n] = I0(beta * sqrt(1 - ((n-L)/L)^2)) / I0(beta)
h[n] = h_ideal[n] * w[n]
sinc(x) = sin(pi*x)/(pi*x), with sinc(0) = 1.
```

Normalize `h` once to unit DC gain in binary64. Select the smallest `N` whose
independently certified response satisfies all passband, phase, DC,
folded-alias, direct product, and support bounds. With one permitted family and
fixed `f_c` and `beta`, this is the deterministic simplest-realization tie
rule.

The coefficient artifact records the complete binary64 coefficient bit
patterns, policy identity, array/scan/cadence inputs, construction version, and
certification result. A dense FFT grid alone is not a proof at the band
boundaries; implementation must add a conservative between-grid response
bound or an equivalent certified evaluator.

Apply the centered filter to paired `x/r` in increasing source-occurrence
order. Phase zero is bound independently to each immutable network-native axis:
a source center is selected exactly when its stable native-axis ordinal modulo
`M` is zero. A run boundary does not reset that phase. The output event time is
the selected central source occurrence time; the complete filter footprint is
its support. Any selected center without complete support inside one admitted
run is unavailable. There is no reflection, replication, zero padding, or
filter-state carry across a gap, slow interval, invalid occurrence, scan
boundary, or network boundary. Equal ordinals on different networks do not
construct or imply a common analysis grid.

## Reproducible feasibility evidence

The evidence-only calculator
[`tools/wp7/analyze_rtc_scan_array_policy_candidate.py`](../tools/wp7/analyze_rtc_scan_array_policy_candidate.py)
and its checked result
[`validation/wp7_rtc_scan_array_policy_candidate_2026-08-29.json`](../validation/wp7_rtc_scan_array_policy_candidate_2026-08-29.json)
apply the approved policy values to the exact representative detector cadence
`122.0703125 Hz`. They use the conventional Kaiser order estimate only; tap
counts are not certified artifacts and can change when exact response and
point-source tests are performed.

| admitted `v_max` (arcsec/s) | `a1100` prototype | `a1400` prototype | `a2000` prototype |
| ---: | ---: | ---: | ---: |
| 1 | `M=125`, 1207 taps | `M=139`, 1215 taps | `M=158`, 1217 taps |
| 10 | `M=13`, 131 taps | `M=17`, 175 taps | `M=24`, 243 taps |
| 25 | `M=5`, 49 taps | `M=6`, 57 taps | `M=9`, 87 taps |
| 50 | `M=2`, 19 taps | `M=3`, 29 taps | `M=4`, 37 taps |
| 100 | `M=1` | `M=1` | `M=2`, 19 taps |

This sweep demonstrates the intended scan/array dependence, the usefulness of
arbitrary integer factors, and the bounded-support consequence near the exact
`1 arcsec/s` admission boundary. It does not prove that any row is the final
realization.

## Observation 152390 evidence boundary

The locally retained telescope file for observation 152390 is a 62,109-row
Lissajous record. `Header.Lissajous.ScanRate` stores
`0.00024240684055476798`; interpreting that stored scale as radians per second
gives exactly `50 arcsec/s`, but the variable attribute says `arcsec/sec`.
Direct finite differences of the telescope coordinate planes contain large
spikes and depend materially on unapproved smoothing choices.

Therefore:

- the nominal 50 arcsec/s value is representative workload evidence only;
- neither the mislabeled header nor a local derivative is `v_max,s` authority;
- no smoothing window, percentile, clipping threshold, or spike repair is
  introduced by this packet; and
- the first real scan/array RTC plan remains blocked until AST provides the
  accepted science-scan membership, scalar velocity, validity, cause, and
  actual-maximum product required by the controlling authority.

The synthetic 50 arcsec/s row shows a plausible expected plan but is not an
accepted observation-152390 result.

## Scope of owner approval

The approval closes the twelve numerical fields listed by the controlling
authority and the inadequate-input `M=1` edge case. It authorizes engineering
to prepare exact array/policy artifacts and the bounded nonidentity
learn-consider-apply increment only after the AST prerequisite is conforming.

It does not approve the prototype tap counts, waive coefficient certification,
claim 152390 `v_max`, introduce a common analysis grid, add a persistent TOD
schema, activate production, retire the identity route, or begin despiking,
level shifts, CAL, VAL, PTC, MAP/JINC, or another numerical RTC method.

The exact owner disposition received on 2026-08-30 was:

```text
I Approve the WP-7 RTC scan/array numerical policy proposed in
WP7_RTC_SCAN_ARRAY_NUMERICAL_CLOSURE_PACKET_2026-08-29.md, including the
input-cadence-inadequate M=1 disposition.
```
