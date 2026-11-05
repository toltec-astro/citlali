# Configured weighted POINT CAL/APT handoff

Grant authorized this Tier 2 increment from local mainline
`cbe2ef64372e1353bf86901ff0e1f1a3699e5adf`. The comparison application is
`5b0d63bd670a2db94eddba4f6c0c7fa9f18eeb62`. In the fresh six-worker
152389 run, six map passes spent 66.29 of 78.93 seconds in saved-input loading
and final verification. Grant executes all builds, tests and reductions.

## Product ownership

The configured coordinator launches one observation child for ordinary
non-retained weighted POINT. It authenticates the upstream request, runs the
existing native RTC/CAL owners, and acknowledges completion to Python. CAL
emits the same per-detector payload declarations from its produced bytes.
There are no named channel files on this route. One worker owns one network
slot. All network workers join before health. The ordered network, input
generation, emitted CAL receipt, detector identity, lengths, slot join, cause
bits and declared digests must match before a channel enters prepared CAL.
Source slots and cause semantics stay with CAL/VAL. The handoff buffer is
released one network at a time. Observation-local values, masks, coordinates,
health and feedback keep their existing numerical owners.

The APT owner keeps the same admitted table and relation from upstream setup
through health. Bundle checks still cover the writable external paths at each
map entry and publication. The old duplicate manifest hash in the original
CAL ledger is covered by that complete APT check. Detector factor and peer
population labels use the admitted APT digest. The worker consumes the
authenticated setup network request and the handoff also binds its locator.
Network receipts, native times, config and telescope data remain file-backed.
There is no assumption that a fresh APT receipt, a filename or a timestamp
makes an external source immutable. Endpoint checks reject observed changes;
they do not detect every possible change-and-restoration between checkpoints.
Stronger guarantees for writable external files require a producer-backed
immutable generation or a reliable write-exclusion contract.

After final fits and before transport release/publication, the child verifies
the remaining original ledger and full APT, writes a receipt and sends its
producer digest over the private pipe. Python authenticates the receipt and
joins the child before publication.
A failure leaves the public observation directory unpublished. Final verification
is outside the teardown timeout. Standalone MAP, comparison, retained
intermediate output and recovery retain their file-backed paths.

## Gates

1. Exact-source independent static review and a source/tree/binary-bound Mac
   Spack build run by Grant. Focused native handoff and byte-count regressions,
   CLI rejection, Python lifecycle, retain/non-retain dispatch, APT mutation
   and final publication failure. Enabled CTest, baseline and required config
   preflight.
2. Fresh matched six-worker 152389 through M5 against `5b0d63bd6`. Require
   matching CAL payload declarations, health seed, segment decisions, every
   scientific map and feedback plane, final fits and pointing table. Check
   the new memory transport designation separately. Expect zero channel file
   hashes and bytes in health and all maps, one APT admission shared by RTC
   and health, eleven health network selections reusing that generation, and
   two APT generation checks per map for the one-locator fixture. Require
   a successful final-originals receipt. Record remaining original, pass and
   retained-APT hash I/O, full and per-stage time and process-tree peak RSS.
   The counters describe their named readers; they are not syscall tracing.

No speed target, scientific tolerance or change in worker defaults is part of
the gate. Keep every slow pass. Standalone replay uses retained files; the
ordinary memory-route CAL receipt identifies its required transport explicitly.
Acceptance, integration and production decisions stay with Grant.
