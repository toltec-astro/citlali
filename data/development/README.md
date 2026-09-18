# Default development invocation

Build `citlali_cli` and run the ordinary `citlali development.yaml` command.
`citlali --dump_config` emits the current request format; no successor option
or alternate executable is needed.

```yaml
schema: citlali-development-v1
input:
  path: /absolute/path/to/rtc-input.json
  sha256: EXACT_SHA256_OF_THAT_FILE
output: /absolute/path/to/new-output-directory
terminal: cal
```

The currently connected adapter consumes the existing bound native x/r exports,
raw/Tune files, verified canonical APT, telescope/motion/timing/processing-scan
bindings and explicit treatment plan for 152390/0/2 network12. The accepted
reference input is `/private/tmp/citlali-rtc-output-handoff-ast-2026-09-17/input.json`;
the verified ordinary invocation is recorded in
`/private/tmp/citlali-successor-development-default-20260917`. File digests are
checked before use. No experimental injection or diagnostic rejection is added.

This runs the existing RTC Learn/Consider/Apply, its output/VAL terminal, AST
coordinates on that exact output grid, and CAL Learn/Consider/Apply. An omitted
`terminal` also selects CAL; `rtc-only` is an explicit early stop. CAL consumes
conditioned x, applies the selected child APT's `flxscale` and the frozen
atmosphere correction once, and preserves r and all inherited support through
the exact RTC parent. PTC/MAP, generic legacy `data_items` conversion and other
unimplemented requests report their specific limitation; they never invoke the
old pipeline. Existing
filter coefficients, scientific thresholds and original-input execution policies
are unchanged. Each run requires a fresh output directory.

The current 152390 telescope file has only a scalar radiometer header. The
approved WVR contract forbids using that as an observation-wide correction.
The default run therefore executes CAL, publishes `cal/receipt.yaml` and
complete support causes with `no-calibrated-output`, preserves RTC values,
and exits 2. It does not publish zero or unity as a substitute calibrated
signal. This is a missing producer-input binding, not an open calibration
policy question. A newly supplied WVR series needs its actual source-time and
validity layout bound at ingress; unknown layouts fail explicitly.

Conditional covariance, nuisance uncertainty and the complete conditioned
response remain unavailable. Otherwise supported CAL signal is a development
capability; it is not a literal point-source-peak or science-qualified claim.

For old-route comparisons use the frozen baseline recorded in
`validation/timestream_successor_development_baseline.json`, with its original
configuration. The baseline is not a selectable fallback in this executable.
