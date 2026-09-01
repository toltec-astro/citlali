# SCI-FLT-MATCHED v0.1 r0.6 standalone output bundle

Status: conditional-freeze preflight only; owner title and AO dispositions are
pending, and scientific authority is not frozen.

This archive contains exactly the current authority/freeze-candidate object
rows in `STAGE_B_DRAFT_MANIFEST.md`, that manifest and its SHA-256 sidecar, and
this note. Historical repository context is represented only by immutable
digest anchors in the manifest. It is intentionally not copied into the
standalone archive.

`verify_stage_a.py` is intentionally excluded because its complete approved
packet and repository firewall context are not bundle-local. Its successful
repository-context execution is recorded in `build/BUILD_VERIFICATION.md`.

Run `verify_stage_b_draft.py` from the extracted archive root to verify every
manifest-bound byte and the complete standalone Markdown-link closure. The
included `build/audit_bundle_links.py` may also be run directly against the
extracted root.

The archive makes no title or AO selection and no implementation, conformity,
response/covariance fidelity, observational validation, performance,
readiness, production, freeze, or Unity claim.
