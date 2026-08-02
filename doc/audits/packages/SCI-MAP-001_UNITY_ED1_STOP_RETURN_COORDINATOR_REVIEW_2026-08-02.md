# SCI-MAP-001 MAP-UNITY-ED1 stop-return coordinator review — 2026-08-02

Status: verified bounded stop; no external evidence supplied; owner route
decision recorded separately as `MAP-UNITY-ED2`

Package: `SCI-MAP-001`

Evidence request: `SCI-MAP-001-UNITY-001`

Returned task: `MAP-UNITY-ED1` successor protocol/producer

## Verified return identity

- Branch: `codex/map-unity-ed1`.
- Return commit:
  `3e014f11decbcf17ad372391e5e960e6c0c54461`.
- Sole parent:
  `1b824f138754eeb1856ae5f102027db4b31598be`.
- Return commit tree:
  `6db187a3f0f976cbd16dafbe17078438c0af1733`.
- Returned successor-package tree:
  `910f712c8153e997a908db3590f0c63c8dee312b`.
- Returned `SHA256SUMS` digest:
  `293b21ec162d407496c22db0b022cc512e8e4ebc8ac0c6d15765e8bbd844cc60`.
- Human decision-brief digest:
  `a1de515081bee6169811ac9a9f7ec14ab4e07135b6a30858384c7325e676d2bb`.
- Machine decision-brief digest:
  `6b2d061332f62fd6316c37c3efade3196a181377e219730923e05ae0b1062b92`.
- Independent read-only review digest:
  `ab1949e738c22544ede6ae9af449bfbe219f5e33794346426d8a81eb76bdca6d`.

The return changed only four files below the reserved sibling package
`validation/campaigns/SCI-MAP-001-UNITY-001/repair-ed28dafb-ed1`. The worktree
was clean at review. Application and build surfaces were unchanged. The exact
candidate remains `ed28dafb37f9113c0d3c95297148157129a90886`, tree
`cf75c36557178f351fb62781108a6f4b41b19225`.

The frozen predecessor package at
`validation/campaigns/SCI-MAP-001-UNITY-001/repair-ed28dafb` remains exactly
tree `dbf486e30c9b78ca16e05bccafc2d027562d0746`, with checksum-list digest
`ecf080cce98ad3aef6d6dbf52e72dd53be5d659a40285ec6c9bfbb0aee185a69`.

## Coordinator finding

The task stopped at the correct governing boundary. The unchanged fixed
seven-case products do not expose enough primitive authority to populate the
nine independent compact observation/array groups and deterministic
every-active-network traces required by `MAP-UNITY-ED1`:

- Point processed TOD is enabled only in `mini` mode, which stores signal at
  reduced precision and omits required kernel and projection authority.
- Science processed TOD is disabled; its dormant selection is not full/all.
- Existing diagnostics omit part of the complete primitive population and
  required fields.
- Final FITS products are derived aggregates and cannot be used as their own
  independent reconstruction authority.

The existing full processed-TOD writer is technically positioned to supply
the missing binary64 signal, flags, kernel, detector geometry, APT columns,
weights, and scan layout. Using it requires an explicit output-only
configuration/product authorization, proportional temporary storage, a
retention policy, and an effective-rate binding because its `SAMPRATE`
metadata records native `telescope.fsmp` while mapmaking uses
`telescope.d_fsmp`.

The task correctly did not choose among a full/all PTC capture, an application
hook, an instrumented executable, or no successor. Its 87.7815-GiB
core-array lower bound and 118.34-GiB preliminary comparison are planning
measurements, not Unity evidence and not ceilings.

## State effect

This return is accepted only as a verified engineering stop and owner-choice
brief. It supplies no external numerical evidence and makes no change to:

- the approved MAP scientific contract;
- the nonconformant implementation assessment;
- F009/F010 `addressed_pending_reaudit` or any other finding state;
- the open CAL/AST/PTC/VAL dependencies;
- validation `in_progress` or production `existing_use_only`;
- repair integration, re-audit, or production scope.

Unity was not contacted, no owner operational values were filled, no transfer,
build, reduction, or Slurm action occurred, and no cleanup was performed.

## Next authority

The project owner's subsequent selection of a bounded full/all PTC capture is
recorded as the separate additive decision `MAP-UNITY-ED2`. The original
`MAP-UNITY-ED1` decision, handoff, and this returned stop remain immutable
evidence.
