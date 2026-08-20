# SCI-PTC v0.1 PDF Outputs

The bounded Stage B freeze-candidate revision is complete at document revision
`r0.3`. The canonical owner-review PDFs are:

- `SCI-PTC-SCIENTIFIC-RATIONALE-v0.1.pdf`; and
- `SCI-PTC-ENGINEERING-CONFORMANCE-v0.1.pdf`.

Both PDFs are generated from the audience views under `../src/`. The 11-page
scientific rationale is a standalone science-team document with compact
traceability and no duplicated full register. The 20-page engineering view
imports the six shared normative modules exactly once and is the complete
formal contract view. These files are contract drafts for scientific review;
their presence does not claim implementation conformity, validation, achieved
performance, freeze, or production readiness.

Repeatable packet-hash, source-identifier, crosswalk, audience-separation, and
PDF coverage checks are provided by `../src/verify_contract.py`. Both PDFs
were rendered with Poppler for targeted visual sanity at revision `r0.3`; the
new science table, response chain, appended formal entries, and named orphan
heading are clean and readable. Final
pagination and cosmetic layout polish are deferred to the final editorial
revision so this pass remains focused on scientific content.
