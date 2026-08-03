# SCI-CAL-001 tau225 engineering-extension execution approval — 2026-08-03

Status: owner approved; preflight may begin. Direct AM execution remains
fail-closed on all registered readiness gates and an owner-supplied fresh cache
root.

Decision ID: `CAL-ATM-D007-EXECUTION-001`

Authority: project owner

## Verified request

The coordinator verified source commit
`31c67119be604ac21905d39b4a91699bd62fffcf` from
`codex/sci-cal-001-atmosphere-operator`:

- `SCI_CAL_001_TAU025_ENGINEERING_EXTENSION_EXECUTION_REQUEST.md`
  SHA-256 `54a694d783e9f80c078707dcc1cac4a07da4e4ae37fd6e8e554bb620ba71c04f`;
- the bound TAU025 engineering-extension protocol,
  SHA-256 `2552a8fee94bc64719504528d3a763c402bfedd9c7ec1380c2d4f5d1775b6967`; and
- the request's checksum manifest binding.

The source commit is documentation-only, passes `git show --check`, and
contains no AM cache, AM execution, candidate/operator fit, Citlali/TolTECA
source change, Unity action, repair, re-audit, adoption, or production change.

## Approval

The owner approves the exact request above. This authorizes the CAL task to
perform the request's no-AM readiness preflight and, only after every one of
its seven stated gates passes, the registered 1,275-grid fresh-cache AM 12.2
direct-truth study:

- all 25 copied, digest-bound AM 12.2 profiles;
- construction nodes `tau225 = .15, .20, .25` at elevations
  `25,35,45,55,65,75,80`;
- held-out nodes `.1625,.175,.1875,.2125,.225,.2375` at elevations
  `29,41,53,67,79`;
- the exact passband/integration/airmass/scale-search and WARN-001 policies;
  and
- the proposed 5% maximum held-out fractional correction-error engineering
  screen.

The `nextafter(.15)` triplet remains a later no-AM candidate-evaluator
diagnostic only. It has no direct-AM target, scale search, or cache entry.

## Cache-root boundary

The approved request deliberately does not name a host-specific external cache
parent. Before creating a cache or invoking AM, the owner must supply the
absolute fresh cache-root path with basename
`sci_cal_001_tau025_engineering_extension_001_root`. The CAL task may
perform all no-AM checks now, but must stop if that path is absent, existing,
not writable, has insufficient free space, or fails any registered gate. It
may not choose, reuse, delete, or mutate a cache root by inference.

## Continuing prohibitions

This approval does not authorize candidate/operator selection or fitting,
numerical-result interpretation beyond the preregistered checks, application
repair, Unity activity, re-audit, operational-domain or production adoption,
a new output format, or any tolerance/warning-policy expansion. A completed
study returns to the coordinator for independent review and a separate owner
decision.

