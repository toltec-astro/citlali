# SCI-ALIGN To SCI-AST Boundary Identity Proof

Profile: `SCI-ALIGN_TO_SCI-AST v0.1/r0.1`

Status: Stage B r0.2 packet-control proof; not scientific approval,
implementation conformity, validation, freeze, or readiness

AST installed path: `SCI-ALIGN_TO_SCI-AST_BOUNDARY.md`

SHA-256:
`359444fec10f35a3c7ab6d59c5d8d127d24f07dfce3f33590eac6268d07489cf`

The ALIGN-led coordinator installed the final shared boundary copy into this
package without AST-side editing and reported a byte-for-byte `cmp` match
against the ALIGN authority copy. The AST package independently recomputed the
installed-file digest above. The document verifier requires that exact digest
before either PDF is built.

Compatibility requires this exact profile identity plus a declared compatible
version/revision and preservation of every required identity, semantic state,
ownership boundary, and typed-availability rule. Similar shape or field names
do not establish compatibility. A successor must name the revision it
supersedes and provide an explicit semantic mapping for every changed, removed,
or newly required item.
