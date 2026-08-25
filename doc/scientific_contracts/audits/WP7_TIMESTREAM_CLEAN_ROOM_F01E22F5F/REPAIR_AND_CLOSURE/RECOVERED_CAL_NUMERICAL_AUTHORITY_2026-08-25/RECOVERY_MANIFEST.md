# WP-7 Recovered CAL Numerical Authority Manifest

Status: **exact digest recovery complete; successor admission staging only**

Recovered: `2026-08-25`

This staging bundle recovers the exact numerical objects named by the frozen
SCI-CAL authority and missing from the WP-7 clean-room packet. It does not
create a new atmosphere model, passband set, interpolation method, scientific
claim, or implementation claim.

## Source provenance

### Citlali atmosphere authority

- Repository: Citlali refactor Git object database
- Branch: `codex/sci-cal-001-atmosphere-operator`
- Commit: `7156881bd1a47e8cece97b8c541a013c93ac03e1`
- Repository license notice: `sources/citlali/licenses/LICENSE`

| Staged source object | Role | SHA-256 |
| --- | --- | --- |
| `sources/citlali/validation/sci_cal_001_atmosphere_operator_2026-08-01/SCI_CAL_001_FIXED_DJF25_FULL_DOMAIN_OWNER_DECISION.md` | Owner-directed scientific contract record | `c43aa932c633e336497547730f73278d3a5cf70d2a5fcfb19049d967c79dd469` |
| `sources/citlali/validation/sci_cal_001_atmosphere_operator_2026-08-01/sci_cal_001_fixed_djf25_full_domain_operator_contract.json` | Exact machine contract | `7a064ff768a3de4f427f1338d94ef6cb9026d248f3c3c816fc3dfc96d156e36a` |
| `sources/citlali/validation/sci_cal_001_atmosphere_operator_2026-08-01/sci_cal_001_fixed_djf25_full_domain_operator_nodes.csv` | Exact numerical node table | `fd688a4cd3f46585b08631bc63a562aed482feb9b24ec9ee0071b70db7eb8a5f` |

The contract identifies operator
`am12_fixed_djf25_piecewise_linear_los_tau_v1`, domain
`0 <= tau225 <= 0.25` and `25 <= elevation_deg <= 80`, fail-closed behavior
outside support, and the reference-spectrum/passband conventions. The node
table contains `1,368` data rows. These are recovered contract facts, not new
claims made by this manifest.

### TolTECA-v1 passband authority

- Repository: TolTECA Git object database
- Commit: `2791e6a1e6349ad1d3ac549a648f41cbc51b98c7`
- Repository license notice: `sources/tolteca/licenses/LICENSE.rst`
- Source root:
  `sources/tolteca/tolteca/data/cal/toltec_passband/`

| Relative member | Bytes | SHA-256 |
| --- | ---: | --- |
| `data/a1100_passband.ecsv` | `496,586` | `13b8fd009bb8d7c375d3c46d21e26d0a779f7f00a949a2a5ccd619d1fe56fd72` |
| `data/a1400_passband.ecsv` | `430,487` | `a7b671d9f659cbc98dad99d3015ce81a3d7a3486c702819d9b3305703e7c682e` |
| `data/a2000_passband.ecsv` | `370,199` | `77e4b33c7bbc2c345ef94d41480d5fee5cb096d789f4fe78e1b4f80a37e0d6ff` |
| `index.yaml` | `531` | `74465637294e536c44818099e4858a916fc6b9acbb1ea21b40427d15fb6532d5` |

Passband-set identity is computed in lexical order of the four relative member
names. For each member, append its UTF-8 relative name, a NUL byte, the raw
32-byte SHA-256 digest of its exact bytes, and a final NUL byte. The resulting
aggregate is:

`toltec-passband-set-v1:sha256:5e6f38f14bcae93a29ffe8362c52b15209f51aee4e48373b23aaa5ec2f8a6433`

The four members total `1,297,803` bytes.

## Evidence classification and boundary

The owner decision and machine contract are recovered approved scientific
authority. The node table and passband members are their exact numerical
objects. The archived task history was used only to locate the commits and is
historical recovery evidence; it is not admitted here as scientific
authority.

The included BSD three-clause notices preserve the source repositories'
redistribution terms. This manifest asserts no additional license grant.

This bundle completes `WP7-OWNER-D002` Gate A only. The exact WVR source/time
interpolation rule remains pending under Gate B. Until that rule is approved,
the bundle does not make the full ordinary CAL route source-closed.

## Verification

From this directory, run:

```sh
/Users/gwilson/tolteca/bin/python verify_recovery.py
```

The verifier is standalone: it checks the exact staged source inventory,
individual SHA-256 values, machine-contract/node linkage, node-table row count,
passband member bytes, and canonical passband-set aggregate without consulting
a repository checkout, task history, network source, or Unity.
