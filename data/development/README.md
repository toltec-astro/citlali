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
terminal: rtc-only
```

The currently connected adapter consumes the existing bound native x/r exports,
raw/Tune files, verified canonical APT, telescope/motion/timing/processing-scan
bindings and explicit treatment plan for 152390/0/2 network12. The accepted
reference input is `/private/tmp/citlali-rtc-output-handoff-ast-2026-09-17/input.json`;
the verified ordinary invocation is recorded in
`/private/tmp/citlali-successor-development-default-20260917`. File digests are
checked before use. No experimental injection or diagnostic rejection is added.

This runs the existing RTC Learn/Consider/Apply and its output/VAL terminal. It
produces an immutable CAL source binding but stops before calibration. CAL,
PTC/MAP, generic legacy `data_items` conversion and other unimplemented requests
report their specific limitation; they never invoke the old pipeline. Existing
filter coefficients, scientific thresholds and original-input execution policies
are unchanged. Each run requires a fresh output directory.

For old-route comparisons use the frozen baseline recorded in
`validation/timestream_successor_development_baseline.json`, with its original
configuration. The baseline is not a selectable fallback in this executable.
