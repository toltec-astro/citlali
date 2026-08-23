# SCI-PTC v0.1 PDF Outputs

Status: Scientific authority frozen; implementation conformity not yet assessed
under this contract.

The canonical v0.1/r0.4 frozen PDFs are:

- `SCI-PTC-SCIENTIFIC-RATIONALE-v0.1.pdf`: 11 pages; SHA-256
  `7cb358eec6633e06ca2559741d4f32ca2cf62607fac2fe6efb73365863832fd0`; and
- `SCI-PTC-ENGINEERING-CONFORMANCE-v0.1.pdf`: 22 pages; SHA-256
  `1e73d3e001dafce4dd6a9025553af95da58075fb49ea2b4eb41222431d658b85`.

Both PDFs were generated from the audience views under `../src/` at the r0.4
freeze commit. The 11-page
scientific rationale is a standalone science-team document with compact
traceability and no duplicated full register. The 22-page engineering view
imports the six shared normative modules exactly once and is the complete
formal contract view. These files are the frozen scientific authority. Their
presence does not claim implementation conformity, representation/response
fidelity, validation, achieved performance, science qualification, or
production readiness.

Repeatable packet-hash, source-identifier, crosswalk, audience-separation, and
PDF coverage checks are provided by `../src/verify_contract.py`. Both PDFs
were rendered with Poppler and inspected page by page at revision `r0.4`; the
support-composition rule, nonrestoring-centering statements, repaired
traceability, response chain, and formal registers are clean and readable.
Future substantive edits require explicit owner authority and a versioned
successor or formally reopened revision.

A separately marked, non-authoritative r0.5 review candidate is held under
`r0.5-candidate/`. Its presence does not amend these canonical r0.4 PDFs.
