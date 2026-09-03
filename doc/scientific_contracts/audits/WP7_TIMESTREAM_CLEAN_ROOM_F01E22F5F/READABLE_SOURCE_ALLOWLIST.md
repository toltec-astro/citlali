# WP-7 Readable Source Allowlist

Source commit: `f01e22f5f8d8d92e49ae70312bdc59a81c1540ec`

The fresh auditor may read only the source-commit paths below plus the
packet-local inputs named in `SOURCE_MANIFEST.md`. Git history,
sibling files, history directories, dossiers, manager reviews, change logs,
implementation material, and unlisted audit records are not admitted during
independent extraction.

For each package, `src/common/*.tex` means the six canonical modules named
`notation.tex`, `definitions.tex`, `equations.tex`, `assumptions.tex`,
`requirements.tex`, and `edge_cases.tex`. SCI-CAL additionally admits
`src/common/preamble.tex`.

## Primary Package Authorities

### SCI-ALIGN

- `doc/scientific_contracts/packages/SCI-ALIGN/v0.1/README.md`
- `doc/scientific_contracts/packages/SCI-ALIGN/v0.1/SOURCE_MANIFEST.md`
- `doc/scientific_contracts/packages/SCI-ALIGN/v0.1/CROSSWALK.md`
- `doc/scientific_contracts/packages/SCI-ALIGN/v0.1/SCIENTIFIC_OWNER_DECISION_LEDGER.md`
- `doc/scientific_contracts/packages/SCI-ALIGN/v0.1/src/scientific-rationale.tex`
- `doc/scientific_contracts/packages/SCI-ALIGN/v0.1/src/engineering-conformance.tex`
- `doc/scientific_contracts/packages/SCI-ALIGN/v0.1/src/common/*.tex`
- `doc/scientific_contracts/packages/SCI-ALIGN/v0.1/pdf/README.md`
- `doc/scientific_contracts/packages/SCI-ALIGN/v0.1/pdf/SCI-ALIGN-SCIENTIFIC-RATIONALE-v0.1.pdf`
- `doc/scientific_contracts/packages/SCI-ALIGN/v0.1/pdf/SCI-ALIGN-ENGINEERING-CONFORMANCE-v0.1.pdf`
- `doc/scientific_contracts/packages/SCI-ALIGN/v0.1/SCI-ALIGN_TO_SCI-AST_BOUNDARY.md`

### SCI-AST

- `doc/scientific_contracts/packages/SCI-AST/v0.1/README.md`
- `doc/scientific_contracts/packages/SCI-AST/v0.1/SOURCE_MANIFEST.md`
- `doc/scientific_contracts/packages/SCI-AST/v0.1/CROSSWALK.md`
- `doc/scientific_contracts/packages/SCI-AST/v0.1/SCIENTIFIC_OWNER_DECISION_LEDGER.md`
- `doc/scientific_contracts/packages/SCI-AST/v0.1/src/scientific-rationale.tex`
- `doc/scientific_contracts/packages/SCI-AST/v0.1/src/engineering-conformance.tex`
- `doc/scientific_contracts/packages/SCI-AST/v0.1/src/common/*.tex`
- `doc/scientific_contracts/packages/SCI-AST/v0.1/pdf/README.md`
- `doc/scientific_contracts/packages/SCI-AST/v0.1/pdf/SCI-AST-SCIENTIFIC-RATIONALE-v0.1.pdf`
- `doc/scientific_contracts/packages/SCI-AST/v0.1/pdf/SCI-AST-ENGINEERING-CONFORMANCE-v0.1.pdf`

### SCI-RTC

- `doc/scientific_contracts/packages/SCI-RTC/v0.1/README.md`
- `doc/scientific_contracts/packages/SCI-RTC/v0.1/CROSSWALK.md`
- `doc/scientific_contracts/packages/SCI-RTC/v0.1/SCIENTIFIC_OWNER_DECISION_LEDGER.md`
- `doc/scientific_contracts/packages/SCI-RTC/v0.1/src/scientific-rationale.tex`
- `doc/scientific_contracts/packages/SCI-RTC/v0.1/src/engineering-conformance.tex`
- `doc/scientific_contracts/packages/SCI-RTC/v0.1/src/common/*.tex`
- `doc/scientific_contracts/packages/SCI-RTC/v0.1/pdf/README.md`
- `doc/scientific_contracts/packages/SCI-RTC/v0.1/pdf/SCI-RTC-SCIENTIFIC-RATIONALE-v0.1.pdf`
- `doc/scientific_contracts/packages/SCI-RTC/v0.1/pdf/SCI-RTC-ENGINEERING-CONFORMANCE-v0.1.pdf`

### SCI-CAL

- `doc/scientific_contracts/packages/SCI-CAL/v0.1/README.md`
- `doc/scientific_contracts/packages/SCI-CAL/v0.1/CROSSWALK.md`
- `doc/scientific_contracts/packages/SCI-CAL/v0.1/DECISION_LOG.md`
- `doc/scientific_contracts/packages/SCI-CAL/v0.1/SCIENTIFIC_OWNER_DECISIONS_R0.5.md`
- `doc/scientific_contracts/packages/SCI-CAL/v0.1/SCIENTIFIC_OWNER_DECISION_LEDGER.md`
- `doc/scientific_contracts/packages/SCI-CAL/v0.1/src/scientific-rationale.tex`
- `doc/scientific_contracts/packages/SCI-CAL/v0.1/src/engineering-conformance.tex`
- `doc/scientific_contracts/packages/SCI-CAL/v0.1/src/common/*.tex`
- `doc/scientific_contracts/packages/SCI-CAL/v0.1/pdf/README.md`
- `doc/scientific_contracts/packages/SCI-CAL/v0.1/pdf/SCI-CAL-SCIENTIFIC-RATIONALE-v0.1.pdf`
- `doc/scientific_contracts/packages/SCI-CAL/v0.1/pdf/SCI-CAL-ENGINEERING-CONFORMANCE-v0.1.pdf`

### SCI-PTC

- `doc/scientific_contracts/packages/SCI-PTC/v0.1/README.md`
- `doc/scientific_contracts/packages/SCI-PTC/v0.1/CROSSWALK.md`
- `doc/scientific_contracts/packages/SCI-PTC/v0.1/AUTHOR_DRAFT_DECISIONS.md`
- `doc/scientific_contracts/packages/SCI-PTC/v0.1/SCIENTIFIC_OWNER_DECISION_LEDGER.md`
- `doc/scientific_contracts/packages/SCI-PTC/v0.1/src/scientific-rationale.tex`
- `doc/scientific_contracts/packages/SCI-PTC/v0.1/src/engineering-conformance.tex`
- `doc/scientific_contracts/packages/SCI-PTC/v0.1/src/common/*.tex`
- `doc/scientific_contracts/packages/SCI-PTC/v0.1/pdf/README.md`
- `doc/scientific_contracts/packages/SCI-PTC/v0.1/pdf/SCI-PTC-SCIENTIFIC-RATIONALE-v0.1.pdf`
- `doc/scientific_contracts/packages/SCI-PTC/v0.1/pdf/SCI-PTC-ENGINEERING-CONFORMANCE-v0.1.pdf`

### SCI-VAL

- `doc/scientific_contracts/packages/SCI-VAL/v0.1/README.md`
- `doc/scientific_contracts/packages/SCI-VAL/v0.1/CROSSWALK.md`
- `doc/scientific_contracts/packages/SCI-VAL/v0.1/DECISION_LOG.md`
- `doc/scientific_contracts/packages/SCI-VAL/v0.1/SCIENTIFIC_OWNER_DECISION_LEDGER.md`
- `doc/scientific_contracts/packages/SCI-VAL/v0.1/PROFILE_REGISTRY.md`
- `doc/scientific_contracts/packages/SCI-VAL/v0.1/SOURCE_BINDING_REGISTER.md`
- `doc/scientific_contracts/packages/SCI-VAL/v0.1/src/scientific-rationale.tex`
- `doc/scientific_contracts/packages/SCI-VAL/v0.1/src/engineering-conformance.tex`
- `doc/scientific_contracts/packages/SCI-VAL/v0.1/src/common/*.tex`
- `doc/scientific_contracts/packages/SCI-VAL/v0.1/pdf/README.md`
- `doc/scientific_contracts/packages/SCI-VAL/v0.1/pdf/SCI-VAL-SCIENTIFIC-RATIONALE-v0.1.pdf`
- `doc/scientific_contracts/packages/SCI-VAL/v0.1/pdf/SCI-VAL-ENGINEERING-CONFORMANCE-v0.1.pdf`

## Composition Authorities

- `doc/scientific_contracts/boundaries/v0.1/SCI-RTC_TO_SCI-AST_SAMPLE_GRID_BOUNDARY.md`
- `doc/scientific_contracts/boundaries/v0.1/DETECTOR_GEOMETRY_FIELD_ROTATION_BOUNDARY.md`
- `doc/scientific_contracts/boundaries/v0.1/TIMESTREAM_EXPOSURE_LINEAGE_BOUNDARY.md`
- `doc/scientific_contracts/producer_interfaces/v0.1/TUNE_READOUT_NATIVE_XR_PRODUCER_INTERFACE.md`
- `doc/scientific_contracts/audits/SIX_PACKAGE_TIMESTREAM_CLOSURE_PROGRAM_55EFD8A/PTC_NAMED_USE_COMMON_SEMANTICS_R0.1.md`

## Packet-Local Readability View

- `SANITIZED_COMPOSITION_NOTES.md`

The exact source/freeze manifests are bound by `SOURCE_MANIFEST.md`. Their
hashes establish authority identity, but the fresh auditor must not open
unlisted review, approval, repair, or closure records during independent
extraction.
