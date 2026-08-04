# SCI-MAP-002 provenance-correction acknowledgment — 2026-08-03

Status: verified documentation-only correction. It changes no audit conclusion,
owner decision, repair boundary, or evidence authorization.

## Verified correction

The completed scientific-contract audit remains the source package at commit
`abb8d8896d6a1cbaa912b9ac181bd649588acc62`. At that commit, its report's
actual SHA-256 was

```text
d77bb5c8e555b43d5303ad2ce0a81e5baef42df2beb85347f9f98cab759d5239
```

The initial closeout message transcribed a different digest. The original
coordinator review and repair handoff already bind the actual completed-audit
digest above; their historical identities therefore remain valid.

Audit commit `fe201b69be2764dc47dc0a1957bfc8e493f2905a` adds a correction note
to the audit report and updates only its companion provenance documents. The
current corrected artifact identities are:

| Artifact | SHA-256 |
| --- | --- |
| corrected report | `f2049d2e3cd677405559bd256c488199a22d0411d8f278b276183b94ee93f531` |
| corrected local evidence | `d599a058a0c378c49043342fc50fac231cb6f2c9b7a772bb3c374d58dd77f67e` |
| corrected ledger proposal | `2be6ff0f032532cce79004c78a5f8584f5364a98c7f38ef7fc5cfa7bdd311b76` |

The task verified YAML parsing, hash closure, LaTeX structure, whitespace, and
changed-file scope. The correction states, and coordinator review confirms,
that audited source claims, scope, evidence basis, findings, owner decisions,
and the proposed Unity protocol are unchanged.

## Effect

`SCI-MAP-002` remains integrated at the original audit-content identity with
its existing approved owner decisions and bounded repair/re-audit handoff.
This acknowledgment records the later provenance correction only; it neither
reopens the decisions nor authorizes repair, Unity evidence, or production use.
