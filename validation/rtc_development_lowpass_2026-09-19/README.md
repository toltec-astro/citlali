# Explicit development low-pass plans

The owner selected development of the a1100/a1400 low-pass plans before the
all-network replay on 2026-09-19. These are fixed application inputs for the
existing RTC Learn → Consider → frozen-original Apply path. They are not a
runtime design service or a certified production filter bank.

Authority: ADRs 0019–0022, their linked owner authorities, the fixed-filter
speed-support decision of 2026-09-16, and the owner-directed successor
development default. The beam remains the 50-m circular Airy model at
272/214/150 GHz; nominal native interval is 0.008192 s. The retained design
ceiling is 235 arcsec/s, with the existing cadence and motion margins.
Occurrence-center speed admission remains a separate decision. Selecting a
235 arcsec/s filter does not admit every center at that speed.

| Array | Optical support at design ceiling | Output factor | Taps | Support on each side |
| --- | ---: | ---: | ---: | ---: |
| a1100 | 51.68457 Hz | 1 | 67 | 0.270336 s |
| a1400 | 40.66360 Hz | 1 | 33 | 0.131072 s |
| a2000 | 28.50252 Hz | 2 | 307 | 1.253376 s |

At this ceiling, factor two would put part of the shorter arrays' optical
bands beyond output Nyquist (30.51758 Hz). Their explicit selection is real
low-pass filtering **without decimation**, at 122.0703125 Hz. No largest-factor
search or new speed threshold is introduced. Factor one introduces no new
decimation aliases; it cannot remove contamination already aliased at readout.
It also does not remove lines inside the astronomical passband. No notch is
selected for these arrays.

`prepare_rtc_array_lowpass.py` constructs the two new entries offline. Its
80-dB Kaiser order estimate is an engineering design input, not a measured
attenuation claim. Dense response measurements give maximum passband magnitude
errors of 0.00851% and 0.01155%, and near-Nyquist amplitude bounds of
7.060e-5 and 2.268e-4, respectively. The checked coefficients, identities,
array/factor and cadence domain are bound by the application adapter; altered
or foreign profiles fail explicitly.

The a2000 coefficient bytes and identity are copied unchanged from the accepted
network12 input, SHA256
`bbb85e2b943c0f6d21ebebf990aef1286906fab8f6971c3026020603e9dcd8e3`.
Its coefficient SHA256 is
`e25377075b9b20147bfc11b9c4b9ab792dd60576166f6c25dbc8b9aaf75e8967`.
The previous inline request remains supported and unchanged.

Focused tests exercise direct convolution through actual RTC Consider/Apply,
paired invalid support, physical gaps, native output phase and immutable input.
Numerical tests cover both cadence endpoints and matched-support paired Airy
injections at four speeds and four sub-sample phases. They retain the existing
provisional uniform readout average. These are engineering transfer checks,
not naive/JINC/OOF/FRUIT or cleaned-PSD science qualification. Those broader
ADR0020 qualifications remain outstanding; MAP and FRUIT remain out of scope.

Use a `filter_plan: {path: <absolute artifact>, sha256: <file digest>}` in the
existing exact network input, replacing its inline `fir`/`lowpass_identity`.
The ordinary development request still consumes that input. An exact APT
array match, original paired ingress, initial VAL, AST motion, physical support
and the existing complete RTC plan remain required. The per-detector operation
must be `lowpass` on the two new arrays. CAL consumes the resulting actual RTC
grid at its realized cadence. Learning, consideration and application keep
their existing owners; offline coefficient preparation does not replace any
runtime phase.
