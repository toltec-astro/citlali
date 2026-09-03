# EL-F8 analysis attempt R0.1

## Status

The first full analysis attempt against the complete, successful R0.4 replay
set stopped before writing any result artifact.  The four replay products and
logs remain valid and unchanged.

The analyzer required all 305 proposed post-downsampling samples in the UID
4460 mapmaking application to be *newly* flagged.  The realized evidence was:

- proposed samples: 305;
- already flagged for other reasons: 34;
- newly flagged by the carried detector penalty: 271;
- excluded after application: 305;
- source-protected samples: 0;
- application accepted: true.

The registered scientific contract requires that UID 4460 not be hard
excluded before RTC/PTC and that it be fully excluded before final map
accumulation.  It does not require the samples to be otherwise pristine on
arrival at mapmaking.  The realized `271 + 34 = 305` evidence satisfies that
contract.

The bounded analyzer repair therefore replaces the false equality
`newly_flagged == proposed` with the complete-state requirement
`newly_flagged + already_flagged == proposed`, while retaining the exact
proposed count, zero source protection, one valid matching record, accepted
application, and absence of pre-RTC/pre-PTC detector-exclusion records.  The
three counts are retained separately in the result.

This repair changes no replay, checkpoint, scientific map, component
definition, region, metric, or interpretation rule.  A new analysis revision
must identify the repaired tool and use a fresh output directory.
