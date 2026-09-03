# SCI-CAL v0.1 Rationale r0.5 / Engineering r0.4 Build Review

Date: `2026-08-20`

Scope: artifact, mechanical, and visual review of the Q01--Q09 owner-decision
revision. This review does not establish implementation conformity or execute
scientific validation.

## Result

- Canonical science PDF: 14 nonempty letter-size pages; SHA-256
  `d4024db374f361854060ef4939796ae8c2fec910a33935852f832384f7d692a3`.
- Canonical engineering PDF: 26 nonempty letter-size pages; SHA-256
  `994a641b21c0f4af0701c3eb5c09d86669bb7943b0be02d2080024d00331ac0d`.
- Tectonic completed all reruns without LaTeX warnings, overfull/underfull
  boxes, undefined references, or multiply defined references.
- Poppler rendered all 40 pages at 120 dpi. Every page was inspected through
  whole-document contact sheets; dense decision, notation, atmosphere,
  requirement, Q09, evidence, and disposition pages were also inspected at
  full rendered size. No clipping, overlap, missing glyph, unreadable table,
  or bad pagination was found.
- The package verifier passes exact wrapper, shared-authority, and PDF hashes;
  11 sequential assumptions; 50 sequential requirements; 30 sequential edge
  predictions; 50 crosswalk rows; all Q01--Q09 identifiers; the r0.5/r0.4
  revision relationship; and closure/transfer terminology.

## Remaining gate

The documents specify but do not execute the Beammap closure and
associated-pointing transfer workflow. Systematic uncertainty products remain
unavailable where recorded. Achieved-performance acceptance is an owner decision
based on the evidence actually achieved; the numerical benchmark values are
not automatic pass/fail ceilings.
