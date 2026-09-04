# SCI-FRUIT EL-F11 repaired replay registration manifest r0.2

Test ID:
`SCI-FRUIT-EL-F11-PROSPECTIVE-INFLUENCE-PERSISTENCE-R0.1`

Status: **method-preserving repair frozen before a scientific replay**

`REGISTRATION_R0.2.yaml` is 3,717 bytes with SHA-256
`55351780a97e684037e2150610721625f5f35ad15490faf3a7be5ab84219d08a`.
All seven repair-registration files pass their exact size and SHA-256 checks.
The original r0.1 registration remains immutable and all 23 of its inputs
remain valid except that its now-superseded override-config identity records
the rejected `start_iteration: 1` binding.

The first launch stopped before entering a FRUIT iteration and produced no
scientific product. Its complete log and empty reduction lock are retained.
The single authorized scientific replay therefore remains available.

The only corrected field is
`timestream.fruit_loops.injected_source_test.start_iteration`, changed from
`1` to the copied checkpoint's required `next_iteration` value `4`. This is
the restart binding that keeps the already-active, unchanged 100 mJy/beam
off-source injection enabled during absolute iteration 4. The correction and
its authority are documented in
`EL_F11_R1_ROUTINE_RESTART_INJECTION_START_REPAIR_2026-09-04.md`.

The frozen launch command and argument order remain exactly those in
`REGISTRATION_MANIFEST_R0.1.md`; only the registered contents of the final
EL-F11 override now carry the corrected value. The output `reduced` and `logs`
paths are absent before launch. All method, input, target, compatibility,
closure, metric, resource, and claim boundaries remain unchanged.

No accounting value has been opened. After the one local replay, its exact
outputs must be bound in a successor registration before scientific analysis.
No Unity activity is authorized.
