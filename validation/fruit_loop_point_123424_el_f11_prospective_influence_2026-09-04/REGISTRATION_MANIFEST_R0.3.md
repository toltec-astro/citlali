# SCI-FRUIT EL-F11 output registration manifest r0.3

Test ID:
`SCI-FRUIT-EL-F11-PROSPECTIVE-INFLUENCE-PERSISTENCE-R0.1`

Status: **frozen after replay and before scientific values were opened**

`REGISTRATION_R0.3.yaml` is 15,715 bytes with SHA-256
`f9f773232cd70b76ef70d984b7e4b2f6bcac6d4343e7b4edf31fb5333e6cfa55`.
All 35 registered source, method, compatibility, retained EL-F10, replay, and
diagnostic files pass their exact size and SHA-256 checks.

The registration incorporates the method-preserving restart binding recorded
in r0.2 and binds the single successful local replay. It does not alter the
test question, target, algorithms, source, metrics, support, numerical bounds,
regions, resource limits, or claim limits frozen before execution.

The replay completed in 31.59 seconds at 871,579,648 bytes maximum resident
set size with no error- or critical-level log records. The complete retained
root is 122,300 KiB. The earlier pre-iteration failure remains preserved and
did not consume the single scientific replay.

The registered analysis command is:

```text
/Users/gwilson/tolteca/bin/python
tools/fruit_loops/analyze_prospective_influence_persistence.py
--registration validation/fruit_loop_point_123424_el_f11_prospective_influence_2026-09-04/REGISTRATION_R0.3.yaml
--output-dir validation/fruit_loop_point_123424_el_f11_prospective_influence_2026-09-04
```

The analysis must stop on the first compatibility, receipt, ledger, closure,
support, or identity failure. It may report the frozen descriptive persistence
metrics only after every preceding gate passes. No cutoff, intervention,
safeguard, detector judgment, production change, Gate D, Stage B, or Unity
action is authorized.
