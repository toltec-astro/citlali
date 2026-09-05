# Input integrity and read record

Authoring run: `citlali-ptc-uniform-author-run-2026-09-05`  
Draft: `SCI-PTC-COEFFICIENT-UNIFORM v0.1-draft.1 / r0.1`  
Recorded: 2026-09-05

## Pre-read integrity result

The manifest was hashed before its contents were opened. Its SHA-256 was:

`3140f1769ce1d4ed970c21f8cb56ff50e697880d5676983314d7b0584afd8f00`

This exactly matched the owner-supplied release digest. Recursive inventory
then found exactly eleven regular files: the manifest plus the ten declared
content files. After reading only the manifest, all ten content byte counts
and SHA-256 values were checked before any content file was opened. Every
value matched.

| Content file | Bytes | Verified SHA-256 |
| --- | ---: | --- |
| `PRIOR_WORK.md` | 3437 | `2fc533d236ab04b98783c6acd932a23f1e6caa2de0548d273436a2e44e6f9d8d` |
| `README.md` | 2895 | `a9402af0d78f93be1bd1ecf693eac582e10b16ca78b9ba5baca21c0601245303` |
| `SCOPE_BRIEF.md` | 6943 | `d42df7ae3e6b2272eb42e420fd2b4c364db7df8cd1db96cea5eee88fd47a7481` |
| `references/JINC_BOUNDARY.md` | 8747 | `246f9c87d04bdbf09c878155810d089da9f7509f744847a61a83846824c9728c` |
| `references/MAP_BOUNDARY.md` | 8917 | `a02a98af1d381336b47bba10d908bd84999613fe5c60a1cc92fc6286b40606a3` |
| `references/NOI_BINDING.md` | 8903 | `b8bf96cf6e32e82b244d1f2f0eb45683435e3725f51ea4135a14703b10c9dde4` |
| `references/PROCESS_AND_SCOPE.md` | 9256 | `3665e6008399a7af49b7050709fa00c66e2f3a92070da1623c9bd0dfbc41501c` |
| `references/PTC_COEFFICIENTS.md` | 21015 | `ffe62f6a9110ac7d442944afaedde90d69e0f28cabc2dc2d2159a53bb011e014` |
| `references/REGISTRY_FRAMEWORK.md` | 6348 | `9b9a8426b26afdce584bb30189d6dc25dc03f090b6bbebfbf07f15c0e56fef46` |
| `references/VAL_RULES.md` | 20276 | `579d966ad64a378bfab05b5c8f4ffa3a1400dfb319e3bd8f43a9bd29a0b03d75` |

## Read order and firewall record

After the integrity preflight, content was read in the released order:

1. `README.md`
2. `SCOPE_BRIEF.md`
3. `PRIOR_WORK.md`
4. `references/PROCESS_AND_SCOPE.md`
5. `references/PTC_COEFFICIENTS.md`
6. `references/REGISTRY_FRAMEWORK.md`
7. `references/MAP_BOUNDARY.md`
8. `references/JINC_BOUNDARY.md`
9. `references/NOI_BINDING.md`
10. `references/VAL_RULES.md`

No Git checkout, repository instruction, implementation, test, audit, repair,
review receipt, validation result, other task, external scientific source, or
surrounding provenance source was inspected for scientific content. No input
was modified. All authored artifacts were confined to the designated output
directory. The only non-packet executables queried or invoked were local
artifact/rendering utilities supplied by the manager; they added no
scientific content.

The manifest's historical `release approval pending` labels were treated as
superseded only by the explicit owner release instruction carried in the
author task. They were not treated as scientific adoption.

