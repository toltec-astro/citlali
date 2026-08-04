# Provenance and immutable identities

## Application and task

- Request: `SCI-MAP-001-UNITY-001`
- Revision: `repair-sha-ed28dafb-ed1-2026-08-02`
- Branch: `codex/map-unity-ed1`
- Resume commit: `3e014f11decbcf17ad372391e5e960e6c0c54461`
- Resume tree: `6db187a3f0f976cbd16dafbe17078438c0af1733`
- Resume parent: `1b824f138754eeb1856ae5f102027db4b31598be`
- Application candidate: `ed28dafb37f9113c0d3c95297148157129a90886`
- Candidate tree: `cf75c36557178f351fb62781108a6f4b41b19225`

No application or build-configuration file is changed by the successor.

## Coordination authority

- ED1 campaign handoff SHA-256:
  `8ce9d12f93b2cf60e6fb281b67afcfdcaeb9f2ffde755b9c0f640a38c98c0c5b`
- ED1 owner decision SHA-256:
  `4cec0fbbd172a32f51cfe95d5ef1712f091297fb43bd2efee5a1c4eecf99e5fa`
- ED1 decision-content commit:
  `db74fe293436b59eecfa5c36b2a2ea186b05e9b6`
- ED1 identity-binding head:
  `4257265dba44aac3b29d985e1f7bc01b2a50368c`
- ED2 content commit:
  `ae2188dd4761afa85a772a1edd6b9d9571fa9d4b`
- ED2 identity-binding head:
  `c35333d4090e2bebae422538cb40fc063f7cb71a`
- ED2 stop-return review SHA-256:
  `ec98c2f3b8475e7aa4842363780cc247143ecca05053ea61e22e0a9d8e22f83d`
- ED2 owner decision SHA-256:
  `b03e410bf246fd4e3218d1114b59cf96f6019a901112fadea6074af0003a026a`
- ED2 continuation handoff SHA-256:
  `709873e1c3e325d9e1a0a2a85d6acd647b9a31b44f4074014abe674878ffa058`
- ED2 resource/operations amendment, coordination commit
  `a38ec92f28d63d543ad80d463bc99b5ec4606e52`, SHA-256:
  `85998ea7c078208ba6bcae939dd97919f5189cf776f727bd00651cf6ef07d8c4`
- ED2 Unity-cap operationalization, coordination commit
  `23ddec55b6bede06cb27342d00fd96bb9a919019`, SHA-256:
  `bb9fba34f6122a24268fd9fba3e92d8775b1c678fb908a4cd019e491b3a3b73b`

Coordination head `e7b09e0aff2b47f83da760c13ca4edd9f8e013ea` only recorded task resumption and
did not amend the governing ED2 content or identity binding. The two dated
resource decisions above amend only ED2 operational resource handling; they do
not alter the candidate, cases, scientific gates, retention, or repair status.

## Preserved package history

The predecessor `repair-ed28dafb` tree is
`dbf486e30c9b78ca16e05bccafc2d027562d0746`; its `SHA256SUMS` digest is
`ecf080cce98ad3aef6d6dbf52e72dd53be5d659a40285ec6c9bfbb0aee185a69`.

The four immutable stop-return artifacts in this successor are:

| Artifact | SHA-256 |
| --- | --- |
| `decision-brief.json` | `6b2d061332f62fd6316c37c3efade3196a181377e219730923e05ae0b1062b92` |
| `INDEPENDENT_READ_ONLY_REVIEW_2026-08-02.md` | `ab1949e738c22544ede6ae9af449bfbe219f5e33794346426d8a81eb76bdca6d` |
| `MAP-UNITY-ED1_BOUNDED_DECISION_BRIEF_2026-08-02.md` | `a1de515081bee6169811ac9a9f7ec14ab4e07135b6a30858384c7325e676d2bb` |
| `SHA256SUMS` | `293b21ec162d407496c22db0b022cc512e8e4ebc8ac0c6d15765e8bbd844cc60` |

`SHA256SUMS.ed2` is a separate exhaustive inventory so the historic checksum
file remains byte-for-byte immutable.

## Evidence state

All shipped results are local preparation evidence. Future capture, case,
compact, typed resource-projection/resource-record, analysis, and Slurm
records are absent until the owner and coordinator authorize human execution
after reviewing this package. Owner values are intentionally unresolved. The
future governed analysis, manifest, evidence, and return surfaces are all
rooted at `compact-groups/_campaign`; exact full PTC payloads remain retained
in their capture roots and are excluded only from the bounded return bundle.
