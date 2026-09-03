# FRUIT EL-F7 frozen executable and inputs

Frozen after registration verification and before either replay on 2026-09-03.

Test ID: `SCI-FRUIT-EL-F7-SHARED-START-RESPONSE-DECOMPOSITION-R0.1`

Repository registration commit: `7830713c2`

Branch: `codex/sci-fruit-v0.1-empirical-lane`

## Execution software

| Object | SHA-256 |
|---|---|
| external `setup/citlali-el-f5` | `6431c6653ed46ff6e1dfa5512cd27e8169525f7a110207b0b24505786f39dbbe` |

This is an exact copy of the frozen EL-F5 executable. No rebuild or production
code change is permitted within EL-F7.

## Fixed configuration stack

Both replays apply the first six files in order, followed by their named final
overlay and `--grppiex seq`.

| Order | File | SHA-256 |
|---:|---|---|
| 1 | `POINT_123424_BASE.yaml` | `f710d1c172b5655b136ef4d8ebbff918083d37cb4974036c7af8725978d51491` |
| 2 | `POINT_123424_INPUTS_LOCAL.yaml` | `d79e22e1fdfdc67e9235829f1fc2b904a82894c4fbfc10fbfa7b713409b9bc02` |
| 3 | `POINT_123424_COMMON_LOCAL.yaml` | `716c952133ee37b51d8ca4edb783741279aa8f9117149810ab94f987c5d4e269` |
| 4 | `POINT_123424_FITREPORTS_LOCAL.yaml` | `df63463a1a3e83ed5dc2969525b9600c60d88b3226f79af82fda6cba4750d629` |
| 5 | `POINT_123424_ALPHA_1P25_CONTROL.yaml` | `e0e84e693d75033e02e2af0b097f171bf84e1751d7aacff4a40a51bb82eafd56` |
| 6 | `POINT_123424_OFF_SOURCE_CONTROL.yaml` | `da215704b9becf1b941bc7ccdfede6aed924967e4d8f8f7d59693a2e7a6ea3ca` |
| 7a | `EL_F7_CONTROL_SHAM.yaml` | `681b615757c4aa06ce929ae953a6f74c176f8bccba55438381559e1c2e5a7138` |
| 7b | `EL_F7_SHARED_START_PROBE.yaml` | `4caa01dd1ec14da8c8a1c84dd97500f1400578b4f500b236bd8bf9ae44d5942f` |

The copied registration and analysis manifests retain SHA-256 values
`57a63f41e45766851b3f5fe1f9a261ee836486b0662833167612c8e369c558ff`
and `60f8e6dd57b87636a106b170a150fcdad4b1c255b586f0c75e1b6ab29ac38f4d`,
respectively.

## Common restart source

The source is the complete EL-F5 no-injection iteration-4 directory:

`/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f5-off-source-injection-r0.1/point-123424/control/reduced/redu04`

Its checkpoint SHA-256 is
`a77505ab0637c1f257016ee0d9e801b3bba17ed52ab88d52f417a5c1513b451f`.
The source and both isolated copies contain 39 files and 59,346,277 bytes.
A deterministic tree digest over relative-path lengths/bytes and file
lengths/bytes is
`45b81565b7cb6458482a6e619c025e6d7678a594317d250c98f595eee4e87ab0`
for all three directories. `diff -qr` also reports no difference.

The copied iteration-4 FITS products retain these identities:

| Array | SHA-256 |
|---|---|
| a1100 | `9ff2563d57d5dc3563f71977a3f3f6b6d0be56f2815a0c5cf139dbb29cf47b40` |
| a1400 | `e757196735f2d590964c21096f1579e37d3cb4a1be30795c4535f7cd9442f11b` |
| a2000 | `d795506c834e0b0a72182b8c432a2ca49779ea41e8b32ff3c6b04c96769be387` |

## Existing iteration-5 analysis inputs

| State | Checkpoint | a1100 FITS | a1400 FITS | a2000 FITS |
|---|---|---|---|---|
| `C5` control | `d10bb3a90920e8b55174ea1eb34c8126a0a2b6d265636034794967ac6ce17c9e` | `76172b85fa7577412f8ef56499528e693a7fe2489e59f2e37ceeae1053e6a44f` | `0473c0073944bc3ccb3cdb5486f0a637c91ce0f57b31ea0c2ca336f27d179478` | `0154c56b8af226237c27edd8c88b56ef15421c23d57598e31a33f248fa329b01` |
| `A5` adaptive injected | `2256bb5888543b032e073bf59a81f743336c81df865961e8a8638369ce0deaa9` | `fe6297ce17b811871706fcb3c69a54faa7eb327b17456eb2f1255eac94bb759f` | `8dca60369b279d4a54160420544c7d7f016b3b879cdd3e91045c7032cb3c2401` | `8bf185d640d09235363f83a9d49662754725c8bbf3d507c37d3fe0c7d92849e8` |
| `N5` without carried UID 4460 | `d7df0ee480ad99ab3e1b51bb9f311c69e5ae9ab7104a525490dc2fe32ff37faa` | `c60da5025d7412820c01aa59b069d5aa324a2b09b8f2fbbfd6ac47e3897c3ee7` | `5d8594aa566d3bd30f00e4ca3beecef69e3c69f26503f57ce4f0c7834670b0cd` | `ff1cfea4d8964b5e157811f62ac1c2ba260bf16c88488a882e9f9d81009add14` |

## Pre-run verification

- all 228 baseline and FRUIT-loop Python tests passed;
- the EL-F7 analyzer and tests passed Ruff and byte compilation;
- every registered YAML file parsed;
- the repository whitespace check passed;
- the external root retained 130,604 KiB before either replay; and
- 316,875,412 KiB was available on the containing filesystem.

No output directory, log, response map, metric, or scientific result existed
at freeze. The fixed order is sham, exact sham gate, shared-start probe,
registered analysis, then stop.
