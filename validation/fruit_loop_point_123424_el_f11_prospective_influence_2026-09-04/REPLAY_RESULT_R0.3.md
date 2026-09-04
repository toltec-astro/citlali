# SCI-FRUIT EL-F11 replay result r0.3

Test ID:
`SCI-FRUIT-EL-F11-PROSPECTIVE-INFLUENCE-PERSISTENCE-R0.1`

Status: **single scientific replay complete; output-bound analysis pending**

The corrected local replay completed absolute FRUIT iteration 4 from the
registered copied iteration-3 checkpoint:

- wall time: 31.59 seconds;
- maximum resident set size: 871,579,648 bytes;
- complete retained EL-F11 root: 122,300 KiB, including the copied restart
  source and preserved failed pre-iteration attempt;
- one configured thread and `--grppiex seq`;
- zero error- or critical-level log records; and
- all expected maps, checkpoint, learning output, map diagnostic, JINC
  accounting receipt, and target ledger present.

The replay consumed the one authorized scientific run. No replacement replay
is authorized by this result.

The output hashes and sizes are bound in `REGISTRATION_R0.3.yaml`. Only log
completion, resource values, product presence, file sizes, and hashes were
examined before that registration was frozen. No JINC accumulator, target
ledger, map-plane, checkpoint value, learning row, or persistence result was
opened for scientific analysis.

The frozen analysis must now apply compatibility gates before reading the new
accounting values. A failed gate stops interpretation rather than authorizing
another run or a relaxed comparison.
