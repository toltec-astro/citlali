# SCI-FRUIT EL-F11-R2 analysis registration manifest r0.4

Test ID:
`SCI-FRUIT-EL-F11-PROSPECTIVE-INFLUENCE-PERSISTENCE-R0.1`

Repair identity:
`SCI-FRUIT-EL-F11-R2-LEARNING-LEDGER-SCOPE-NORMALIZATION-R0.1`

Status: **frozen after owner approval and before JINC accounting values were opened**

`REGISTRATION_R0.4.yaml` is 17,433 bytes with SHA-256
`7e0ded9b04a6d1026b3abd3ad0bc7ceb10f6ea8683942cadc829ebe5ac52a707`.
All 39 registered source, method, compatibility, retained EL-F10, replay,
diagnostic, prior-registration, and R2-authorization files pass exact size and
SHA-256 checks.

The successor registration preserves all retained input and replay identities
from r0.3 and authorizes no additional Citlali replay. It changes only the
learning-ledger compatibility rule approved in EL-F11-R2:

- CSV headers must match exactly;
- every replay row must be from absolute iteration 4;
- the only reference-only iterations must be 0, 1, 2, and 3;
- the reference iteration-4 rows and all replay rows must match exactly in
  count, order, and every raw string field; and
- complete iteration counts and a canonical ordered-row hash must be reported.

The focused suite passes 10 tests, all 115 fruit-loop tests pass, Ruff lint and
format checks pass, and the complete required configuration preflight passes
127 unit tests, eight compatibility fixtures, and every configured audit.

The registered analysis command is:

```text
MPLBACKEND=Agg
MPLCONFIGDIR=/private/tmp/el_f11_r2_mpl
XDG_CACHE_HOME=/private/tmp/el_f11_r2_cache
/Users/gwilson/tolteca/bin/python
tools/fruit_loops/analyze_prospective_influence_persistence.py
--registration validation/fruit_loop_point_123424_el_f11_prospective_influence_2026-09-04/REGISTRATION_R0.4.yaml
--output-dir validation/fruit_loop_point_123424_el_f11_prospective_influence_2026-09-04
```

The analysis must still stop on the first failed compatibility, receipt,
ledger, closure, support, or identity gate. It may report the frozen
descriptive persistence metrics only after every preceding gate passes. No
predictive cutoff, intervention, safeguard, detector judgment, production
change, qualification, Gate D, Stage B, or Unity action is authorized.
