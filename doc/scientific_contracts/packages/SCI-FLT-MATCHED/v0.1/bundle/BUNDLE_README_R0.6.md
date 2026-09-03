# SCI-FLT-MATCHED v0.1 r0.6 standalone output bundle

Status: standalone bundle of the frozen v0.1/r0.6 scientific authority.

This archive contains exactly the current frozen-authority object
rows in `STAGE_B_DRAFT_MANIFEST.md`, that manifest and its SHA-256 sidecar, and
this note, and the external exact-manifest freeze record. Historical repository context is represented only by immutable
digest anchors in the manifest. It is intentionally not copied into the
standalone archive.

`verify_stage_a.py` is intentionally excluded because its complete approved
packet and repository firewall context are not bundle-local. Its successful
repository-context execution is recorded in `build/BUILD_VERIFICATION.md`.

Run `verify_stage_b_draft.py` from the extracted archive root to verify every
manifest-bound byte and the complete standalone Markdown-link closure. The
included `build/audit_bundle_links.py` may also be run directly against the
extracted root.

The archive records the selected title and final AO/SODL dispositions. It makes
no concrete numerical-route, implementation-conformity, response/covariance-
fidelity, observational-validation, performance, readiness, production, or
Unity claim.
