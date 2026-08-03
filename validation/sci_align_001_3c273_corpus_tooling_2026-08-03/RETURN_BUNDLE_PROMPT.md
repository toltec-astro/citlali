# Recommended prompt after the owner returns Unity evidence

Copy the block below into a new Codex task and attach either the inventory
directory/archive or the completed compact corpus archive.

```text
Act as the bounded SCI-ALIGN-001 3C273 corpus evidence reviewer. Use the
attached owner-returned compact bundle only. Do not connect to Unity, launch a
Citlali reduction, edit application or production configuration, infer a raw
row reassociation, design a production correction, merge, rebase, or push.

First verify the outer archive SHA-256, every nested SHA256SUMS, the selected
manifest digest, frozen protocol digest, tool/commit identity, and the claim
that no raw timestream or reduction product is present. Report any mismatch
and stop without weakening the protocol.

Retain this producer authority: NTP supplies integer-second T0 at ROACH
initialization; all internal clocks share the Octo-distributed 10 MHz
reference; PPS shares that system but does not restart detector integration
cadence; PPS is observed at detector-frame cadence through an incremental
ISR-updated counter; UDP contains T0, PPS counter, and internal-clock counter;
FPGA source is unavailable. Therefore arbitrary millisecond NTP error and
differential oscillator drift are strongly disfavored, but distinct stable
per-network integration phase, detector-frame quantization of PPS observation,
adjacent/non-atomic metadata association, and start/end/centroid timestamp
semantics remain distinct possibilities. The prior Stage-A proof starts at
the delivered D[n]/Ts[n] pair and cannot exclude upstream FPGA association
error. Do not call a first/second-half difference clock drift unless counters
contradict the shared-clock account.

If the attachment is inventory-only, review every source identity, exclusion,
duplicate group, canonical proposal, exact network T0 vector, counter-field
availability, and core/enhanced eligibility without inspecting timing
outcomes. Identify the exact owner choices needed to freeze one primary
reduction per observation; do not choose an ambiguous duplicate for the owner.
Return a proposed owner_selection.csv diff and the exact freeze, sentinel,
batch, and resume commands, then stop for owner approval.

If the attachment is a completed corpus bundle, verify that grouping was
frozen before aggregate timing inspection and that duplicate reductions do
not count as independent observations. Recompute or independently check the
compact decision statistics and synthetic/control fingerprints. Keep native
detector-frame phase and native-to-assigned-slot residual as separate
predictors; test the slope near -1; check within-T0-session phase stability,
across-initialization changes, 122/123 PPS spacing, exact 128-second/15,625-row
repeat, modulo-2^32 counter increments, and same/adjacent/variable metadata
transition association. Evaluate only genuinely held-out predictions; with
exactly three independent groups report models but do not select one.

Return one frozen category (GLOBAL-STABLE, NETWORK-STABLE, SESSION-STABLE,
SLOT-PREDICTABLE, TIME-VARIABLE, UNPREDICTABLE, or INSUFFICIENT), its exact
evidence and limitations, timing error translated by measured scan speed into
arcseconds and beam-FWHM fraction, every exclusion/duplicate sensitivity, and
the minimum preregistered non-3C273 confirmation needed later. A stable or
T0-session-predictable native phase may support a later bounded structural
native-time/fractional-slot investigation, not fixed physical clock
corrections. State explicitly that 3C273 alone authorizes no production
correction and stop before any application change.
```
