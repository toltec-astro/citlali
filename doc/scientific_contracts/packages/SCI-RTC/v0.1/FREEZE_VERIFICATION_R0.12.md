# SCI-RTC v0.1/r0.12 freeze verification

Date: 2026-08-21

Status: Complete implementation-blind verification of the status-only owner
freeze. This record is not implementation conformity, representation fidelity,
validation, performance evidence, science qualification, or production
promotion.

## Promotion identity

- Verified candidate commit:
  `ffce339abbb3c89ae1bf622c5395e28a5e727ea4`.
- Owner action: freeze SCI-RTC v0.1/r0.12 without scientific change.
- Normative inventory remains 52 definitions, 44 equation tags, 12
  assumptions, 143 requirements, and 108 predictions.
- Owner ledger remains 103 entries: 63 open, one conditional, 34 resolved,
  and five deferred.

## Canonical PDF artifacts

| Artifact | Pages | SHA-256 | Metadata |
| --- | ---: | --- | --- |
| `pdf/SCI-RTC-SCIENTIFIC-RATIONALE-v0.1.pdf` | 12 | `b0060b28253906f83f2f106d9df761864d8277317ebd5e3742ff963e11e30b3d` | US Letter; unencrypted; no form; no JavaScript; title ends `v0.1/r0.12` |
| `pdf/SCI-RTC-ENGINEERING-CONFORMANCE-v0.1.pdf` | 62 | `9211091e71830295a8fe5febb102704c95f8397b017584cbeb4575728081da42` | US Letter; unencrypted; no form; no JavaScript; title ends `v0.1/r0.12` |

## Verification disposition

- Tectonic compiled both views without warnings.
- The engineering view imports the six-file shared normative core in order and
  exactly once; the rationale imports none of it and contains no independent
  displayed normative equation.
- Poppler extraction finds the frozen status, EQ-042, REQ-143, PRED-108, and
  the exact r0.12 end-of-core marker.
- All 74 pages were rasterized and inspected through complete contact sheets;
  both title pages and representative formal pages were also inspected at full
  size.
- No clipping, overlap, blank or sparse spill page, footer collision, table
  truncation, malformed glyph, equation collision, unexpected rotation, or
  inconsistent page geometry is present.
- The mechanical contract verifier passes the exact inventories, crosswalk,
  ledger states, source bindings, frozen status, and canonical PDF hashes.

Disposition: pass. The promotion is status-only and changes no scientific
meaning or unavailable-state consequence.
