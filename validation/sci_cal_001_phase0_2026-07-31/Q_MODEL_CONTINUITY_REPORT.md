# SCI-CAL-001 q-model continuity preflight

## Identity and authority

- Governing application source: `9aae0e669384c5c0c0dda93debc194d6b8dac787`.
- Audit-framework dispatch: `e6174dd9f49afe9ec57c150a7a97db3f0f4910e0` (read-only and clean at dispatch verification).
- Bounded repair handoff SHA-256: `9d2c0ae89244d355070d6b300f431ac1799787b835c7e4cb76c8d7fc51cde106`.
- Opacity amendment SHA-256: `64fd3ae9788c6a8e3db18ac5ea4f04799586b548f9e7ec12cc8c18f9cbf96e09`.
- `include/citlali/core/timestream/rtc/calibrate.h` SHA-256: `d70a55278227b43cdd7de19bc67e4ddb332524d40e1455c5fa7a80ae5e2d11ee`.
- `include/citlali/core/timestream/extinction_model_selection.h` SHA-256: `45cf86bbb2318c22514411f6d2a0e0371e22e9e355e61b293d93c628d9f3469d`.

The generator rejects source bytes that do not match those frozen digests.

## Domain statement

Neither the exact-base application source nor the supplied owner decision, opacity amendment, or bounded handoff gives a numeric approved elevation/airmass interval. This preflight therefore does not invent one. It evaluates the audit's representative elevations 30, 50, and 70 degrees and the selector's exact 80-degree reference. The 30--80 degree grid is diagnostic only, not a production-validity declaration. The stop result is already present at 80 degrees, so it does not depend on choosing a wider eventual validity domain.

## Method

The selector threshold for model `m` is derived from the exact source literal as `-log(T225_m) / A(80 deg)`, using the source pi and modified-secant coefficient. The left limit uses the preceding model and the exact-boundary/right limit uses model `m`, matching the source `<=` selection. Each band transmission is the source-order degree-six elevation polynomial multiplied by `exp(-A(e) * tau225_boundary)`. Line-of-sight optical depth is `-log(T)`.

Above q25, analytic equality is tested at decimal precision 80 using the exact coefficient literals: the common 225-GHz attenuation cancels, so continuity requires identical adjacent polynomial values. Runtime values use IEEE-754 binary64. For each row, `kappa = sum(|c_i e^(6-i)|) / |P(e)|` and the conservative line-of-sight optical-depth comparison bound is `4096 * u * ((1 + kappa_left + kappa_right) * max(1, |A|, 1/|A|) + 1 + |tau_left| + |tau_right|)`, where `u = 2^-53`. The 4096-u envelope covers coefficient conversion, powers/products/sums, modified-secant arithmetic, exp/log, division, and independent left/right rounding. All reported above-q25 jumps exceed this conditioned bound by many orders of magnitude.

## Exact source-derived thresholds

| Boundary | Left model | Right model | tau225 binary64 | hex |
| --- | --- | --- | ---: | --- |
| `am_q25` | `am_q0` | `am_q25` | `5.04874104674104401e-02` | `0x1.9d97c61a26f5ep-5` |
| `am_q50` | `am_q25` | `am_q50` | `8.83393725904400573e-02` | `0x1.69d68bc39a014p-4` |
| `am_q75` | `am_q50` | `am_q75` | `1.58313198574890929e-01` | `0x1.4439b5d33c071p-3` |
| `am_q95` | `am_q75` | `am_q95` | `3.04868387190534607e-01` | `0x1.382f6b22453bfp-2` |

## Above-q25 result at the selector reference

| Boundary | Band | T left | T right | LOS tau left | LOS tau right | signed LOS tau jump | abs T jump | relative T jump | LOS tau roundoff bound | Analytically equal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `am_q50` | `a1100` | `8.93479033466319295e-01` | `8.79237520498917124e-01` | `1.12632410280450554e-01` | `1.28700201076605542e-01` | `1.60677907961549876e-02` | `1.42415129674021701e-02` | `1.59393924579865585e-02` | `3.02910733845122051e-11` | `false` |
| `am_q50` | `a1400` | `9.18190299803817944e-01` | `9.20655472619352322e-01` | `8.53506115837226326e-02` | `8.26693923870560760e-02` | `-2.68121919666655661e-03` | `2.46517281553437773e-03` | `2.68481687953040947e-03` | `8.04814787793336519e-12` | `false` |
| `am_q50` | `a2000` | `9.37385171660801486e-01` | `9.58432622329524531e-01` | `6.46610122011069405e-02` | `4.24560138764483730e-02` | `-2.22049983246585675e-02` | `2.10474506687230445e-02` | `2.24533642146615836e-02` | `4.58781394377469715e-11` | `false` |
| `am_q75` | `a1100` | `8.18934777413460324e-01` | `7.94896607981050995e-01` | `1.99750835158052364e-01` | `2.29543225639434773e-01` | `2.97923904813824081e-02` | `2.40381694324093287e-02` | `2.93529718060477965e-02` | `4.67836795834754680e-11` | `false` |
| `am_q75` | `a1400` | `8.57512068088479573e-01` | `8.61877701825846998e-01` | `1.53720026468502940e-01` | `1.48641895616166586e-01` | `-5.07813085233635442e-03` | `4.36563373736742477e-03` | `5.09104641185874427e-03` | `1.24505585945245690e-11` | `false` |
| `am_q75` | `a2000` | `8.92698261771000978e-01` | `9.30061266615362925e-01` | `1.13506647957895307e-01` | `7.25048169236655049e-02` | `-4.10018310342298020e-02` | `3.73630048443619467e-02` | `4.18540132140936966e-02` | `9.50098267902649439e-11` | `false` |
| `am_q95` | `a1100` | `6.84988169257569224e-01` | `6.42917456809809607e-01` | `3.78353712025893185e-01` | `4.41738934975871744e-01` | `6.33852229499785591e-02` | `4.20707124477596173e-02` | `6.14181592265430620e-02` | `1.23934562044593865e-10` | `false` |
| `am_q95` | `a1400` | `7.42707948643909321e-01` | `7.51235322315289666e-01` | `2.97452382002624804e-01` | `2.86036331007057387e-01` | `-1.14160509955674172e-02` | `8.52737367138034497e-03` | `1.14814627835211000e-02` | `5.52263078835588394e-11` | `false` |
| `am_q95` | `a2000` | `8.01463936214734107e-01` | `8.81858483264816195e-01` | `2.21315303310123696e-01` | `1.25723685661927626e-01` | `-9.55916176481960700e-02` | `8.03945470500820880e-02` | `1.00309625196338459e-01` | `8.62273336636451658e-10` | `false` |

## Disposition

**Phase 0 fails.** All 36 of 36 above-q25 band/elevation rows are analytically unequal and exceed the documented binary64 roundoff bound. The largest row bound is `8.62273336636451658e-10` and the smallest observed absolute-jump-to-bound ratio is `1.10859994837691516e+08`. The q25 mismatch in the assessed source is recorded separately in the table but is the already authorized low-opacity repair and is not itself used as the stop condition.

Per the bounded handoff, application-code work must stop. No q25/q50/q75/q95 model is modified. Only these phase-0 evidence artifacts are committed for the project owner's successor scope decision.

## Artifact digests

- `generate_q_model_continuity.py`: `a46211c007bdc1fa11d1408c6db4c4a68264ca00cd383806fd421ba978fffe78`.
- `q_model_continuity_table.csv`: `6de859fbc3e3f91376e2a6ad841f6f4f5d1eac0b773c25b521b8d5fffa5ec50f`.
- `SHA256SUMS` additionally records the report digest without creating a self-referential report.
