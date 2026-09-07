# SCI-FRUIT v0.1 — Stage B r0.1 owner-review draft

This generic conditional scientific core was authored from the exact approved Stage A r0.4 amended packet. **It is not frozen. Every numerical method and route remains `unavailable_pending_separate_owner_approval`.** The next decision is Grant Wilson's separate review of the Stage B substance and possible conditional freeze.

- [Scientific rationale](pdf/scientist.pdf): ten-page main narrative, followed by eighteen pages of predictions and record appendices.
- [Engineering conformance view](pdf/engineering.pdf): the same complete canonical content, with the requirement crosswalk and method obligations first.
- [Source/PDF review bundle](SCI-FRUIT-v0.1-stage-b-r0.1-owner-review.tar.gz) and [archive digest](SCI-FRUIT-v0.1-stage-b-r0.1-owner-review.tar.gz.sha256).
- [Artifact manifest](ARTIFACT_MANIFEST.md) and [manifest digest](ARTIFACT_MANIFEST.sha256).
- [Independent review](review/REVIEW_REPORT.md), [document verification](DOCUMENT_VERIFICATION.md), [requirement crosswalk](REQUIREMENT_CROSSWALK.md), [owner ledger](OWNER_LEDGER.md), [author decision log](DECISION_LOG.md), and [source identities/read boundary](SOURCE_IDENTITIES.md).

The sole authored scientific authority is the six LaTeX modules [notation](src/common/notation.tex), [definitions](src/common/definitions.tex), [equations](src/common/equations.tex), [assumptions](src/common/assumptions.tex), [requirements](src/common/requirements.tex), and [edge cases/predictions](src/common/edge_cases.tex). Both [scientist](src/scientist.tex) and [engineering](src/engineering.tex) wrappers include them via the common preamble and render every content unit once. Markdown registers are derived navigation/decision records, not independent science. The [permitted PTC fragment](src/PTC_APPLICATION_REFERENCE.tex) is byte-exact under its source-local cover.

Two substantive independent-review rounds found and closed one low-severity omission: the explicit inherited independent-pointing evidence obligation for later numerical development, qualification and policy recommendations. No new scientific owner premise was identified. No evidence plan, method, recurrence, numerical default, support/learning/stopping/selection rule, covariance estimator or downstream policy was supplied.

All twenty stable conditional predictions, three A/B/C authorities, three model spaces, full additive-reference/gauge/null-space premises, exact seven response scopes, eight support roles, nine terminal fields and universal completion dependencies are retained. Compilation, rendering, consistency review and byte checks are **document verification only**. No numerical experiment, replay, qualification, validation, implementation assessment, production or Unity activity occurred.

Rebuild these documents locally, using an existing cached TeX bundle only:

```sh
/Users/gwilson/tolteca/bin/python qa/build_documents.py --render
/Users/gwilson/tolteca/bin/python qa/verify_documents.py
```

Run from this package directory, or use each script's absolute path. The build script also accepts `--bundle /absolute/path/to/local/TeX/resource/directory`. It writes only within this output directory, sets headless/task-specific cache variables, and performs no network source retrieval. Tectonic and Poppler are required. Generated caches, compilation intermediates and rendered PNGs are ignored and excluded from the review archive. The saved page-inspection record binds the delivered PDF hashes; changed PDFs require renewed visual inspection.

No approved-packet file, approval control, source archive, unlisted repository/status/navigation file or reduction product was changed. No push is authorized or performed.
