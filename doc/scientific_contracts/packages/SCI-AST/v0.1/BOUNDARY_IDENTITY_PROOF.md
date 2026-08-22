# SCI-ALIGN To SCI-AST Boundary Identity Proof

Profile: `SCI-ALIGN_TO_SCI-AST v0.1/r0.1`

Status: Stage B r0.3 packet-control proof; not scientific approval,
implementation conformity, validation, freeze, or readiness

AST installed path: `SCI-ALIGN_TO_SCI-AST_BOUNDARY.md`

SHA-256:
`04357d36b302d607b95950f529044e178deb2528d0c6f656d90da93067a5da36`

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
