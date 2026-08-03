# SCI-MAP-002 JINC kernel identity owner decision — 2026-08-03

Status: owner approved contract clarification; no implementation work
authorized

Package: `SCI-MAP-002`

Decision ID: `SCI-MAP-002-D003-KERNEL-001`

Authority: project owner

## Decision

The JINC kernel plane is the **realized, processing-filtered source-template
response projected through JINC**. It is accumulated and normalized as

\[
K/C = \frac{\sum_i q_i c_i k_{i,\mathrm{processed}}}{\sum_i q_i c_i}.
\]

The kernel timestream is created upstream and, when enabled, undergoes the
same relevant temporal filtering, notch filtering, mean/mask handling, and
PTC common-mode/PCA cleaning operations as the signal before JINC deposition.
It is therefore not an unfiltered analytic JINC response, a measured PSF, or
a generic beam product. Disabled processing stages have no corresponding
effect.

The plane requires the same formal-support validity mask as the signal map.
Its provenance must bind the kernel-template identity and the realized enabled
processing/operator state: temporal filters, notches, cleaning/PCA realization,
masks, flags, JINC support/phase/array parameters, conditioning, and product
identity. This is a compact configuration and realization record, not a
per-sample or per-pixel diagnostic payload. Downstream users may not
renormalize or relabel it without a separately approved response contract.

Future validation may compare it to a checksum-pinned independently declared
source-template response only if such an input already exists. No new
simulation, code change, Unity evidence, repair, re-audit, or production-status
change is authorized.
