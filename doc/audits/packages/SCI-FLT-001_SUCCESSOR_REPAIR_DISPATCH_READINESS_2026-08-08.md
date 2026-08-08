# SCI-FLT-001 successor-repair dispatch readiness — 2026-08-08

Record ID: `SCI-FLT-001-SUCCESSOR-REPAIR-READINESS-2026-08-08`

Status: readiness only; repair, evidence, Unity, re-audit, and application
integration are not authorized

## Exact admitted authority

- Canonical application mainline:
  `46ad23888a40f5102cdfd50c06e49a549bdf8a20`.
- Integrated MAP authority: exact accepted application source
  `af0c849ce59a5f80e5efc8db435bb6662863052f`, bounded application
  integration `d5015fe716971bf8ea617e8a187311bf5af05185`, production
  `existing_use_only`, and all recorded limitations. MAP supplies no precision
  claim and closes no PTC/VAL/CAL dependency.
- Integrated NOI authority: exact accepted application source
  `5b29e13548a6fec884c67b192dec20c92f0bbb62`, bounded application
  integration `4846fa4db39bd2f7d4ddc41f693836834cbc5ff4`, production
  `existing_use_only`, and all recorded limitations. F005/RA-B004 remain
  conditioned under FLT; F006 remains held under FRUIT.
- Immutable CAL-to-FLT handoff:
  `doc/audits/handoffs/SCI-FLT-001/SCI-FLT-001-XAUD-001.yaml`, SHA-256
  `8b39df22e82687a42281357e97f226a0e91e76130d2cf0d694b7d571e56d694f`,
  status `submitted`, late-arrival route `held_for_reaudit`.
- Approved D001--D003 amendment authority: Git object
  `192e0d9b5e3be4eb20522d3319cae346168c4bce:doc/audits/packages/SCI-FLT-001_COORDINATOR_AMENDMENT_2026-08-05.md`,
  SHA-256
  `965e9dd545c9b5bd7da4f15e08ab6ec3d96ac59b53c78873485eaead19ace661`.
  Its concise decision brief at the same commit has SHA-256
  `b381e0ed64b55a891e1251e6ff4e8d72484a961acce6b437d479c8ab30e887cd`.

The approved amendment resolves FLT's scientific-policy choices only:

- D001 retains same-map median fill solely as a numerical boundary device and
  requires scientific admission to be eroded so no admitted stencil reaches
  fill; fill-influenced pixels have no scientific use.
- D002 retains fixed convolved `signal_I`, requires the corresponding
  identically convolved `kernel_I` and truthful response/normalization
  metadata, and permits user-applied peak or aperture response correction
  without creating automatic photometry.
- D003 retains one robust global empirical calibration of the formal spatial
  pattern; direct per-pixel jackknife variance/S/N remain diagnostic, and
  aperture uncertainty comes from blank apertures or a future compact NOI
  product rather than independent-pixel summation.

These decisions close no implementation, evidence, dependency, or re-audit
gate. Existing FLT package axes remain `proposed`,
`conditionally_conformant`, `in_progress`, `fail_closed`, verdict `amend`, and
re-audit `required`.

## Unresolved exact repair base

CAL and MAP-002 bounded repair lanes are active from exact application base
`46ad2388...`. Their active-task results are not available here. The FLT repair
base therefore remains deliberately unresolved. It may be selected only after
SCI-CAL-001 has an accepted and independently re-audited successor integrated
into canonical application mainline, and only if that base also preserves the
bounded integrated MAP-001 and NOI-002 authorities above.

No existing FLT candidate, `46ad2388...` itself, a CAL task branch, or an
unintegrated CAL successor is selected by this readiness record.

## Future dispatch gate

After CAL integration, the coordinator may prepare a fresh scope checkpoint
that freezes the exact repair SHA, D001--D003 amendment object, MAP/NOI/CAL
dependency facts, included findings, exclusions, tests, cost classification,
and repair/re-audit separation. Until then:

- no FLT branch, source edit, test campaign, evidence request, Unity action,
  re-audit, or production change is authorized;
- filtered scientific weighting, significance, photometry, confidence,
  morphology, feedback, and covariance-dependent use remain fail closed; and
- `FRAMEWORK-NUM-001` applies before any future costly FLT execution.
