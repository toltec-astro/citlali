# EL-F8 analysis attempt R0.2

## Status

Analysis r0.2 stopped before writing any result artifact.  It passed the
scientific map calculations and corrected UID 4460 application-accounting
check, then rejected the first otherwise successful execution log because its
resource parser assumed each timing field occupied a separate line.

On macOS, `/usr/bin/time -l` records all three values on one line, for example:

```text
31.25 real        29.69 user         0.70 sys
```

All four R0.4 logs contain `citlali is done!`, complete resource records, and
zero error/critical messages.  The bounded parser repair accepts either
value-before-label or label-before-value tokens separated by whitespace,
regardless of whether the three fields share a line.  A focused test freezes
the observed macOS form.

No replay, checkpoint, map, scientific calculation, metric definition, or
claim limit changes.  A new analysis revision must identify the repaired
tool and use a fresh output directory.
