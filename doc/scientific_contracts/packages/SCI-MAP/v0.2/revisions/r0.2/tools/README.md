# Document packet tools

These three local tools support only the SCI-MAP r0.2 source/PDF delivery requested by the owner. They consume this package, isolated document-build directories, and the document tools explicitly recorded in the build recipe. They do not inspect or modify Citlali application sources, configuration, tests, or data.

- `build_packet.py` verifies the exact input packet, prepares acyclic cover identities in a new build directory, and records independent PDF opening/rendering results after an external Tectonic compilation.
- `verify_packet.py` checks bound package hashes, input preservation, 52/25 identifier continuity, TeX inputs, authored file links, PDF metadata, cover/source/build identity agreement, crosswalk continuity, and the syntax/initial values of new prospective ECS record templates. The template check uses local `jsonschema` and records its version and module hash; it grants no semantic conformance. Its result describes document consistency only.
- `seal_packet.py` writes the complete source manifest and sidecar, then optionally creates and verifies a new external archive and its sidecar.

Run these scripts with `/Users/gwilson/tolteca/bin/python -B`; this is the repository's required Python environment. The exact observed compiler, renderer, reader versions and executable hashes live in `../build/BUILD_ATTEMPT.json` and `../build/PDF_BUILD_INSPECTION.json`. The build sequence is in [the build recipe](../build/BUILD_RECIPE.md).

The immutable attempt directory and external archive are never overwritten. Source manifests are generated only in the uncommitted candidate assembly; after sealing and review, any later revision requires new exact bytes and review. No script changes an ECS requirement-result placeholder from `not_assessed` to `pass`.
