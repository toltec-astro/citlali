# FRUIT EL-F6 result manifest r0.1

Generated after both registered one-iteration replays and the prospectively
fixed causal classification were evaluated on 2026-09-03.

The registration commit is
`fb7769b77`; preparation commit `e68a4ae75` adds the tested analyzer and fixed
overlays; freeze commit `19e592386` records the executable, copied states, and
intervention; and commit `5b9da679b` records the timing-only pre-result parser
repair. The frozen executable SHA-256 is
`6431c6653ed46ff6e1dfa5512cd27e8169525f7a110207b0b24505786f39dbbe`.

## Repository result artifacts

| Artifact | SHA-256 |
|---|---|
| `tools/fruit_loops/analyze_off_source_penalty_counterfactual.py` | `1b3dce81f1a3f3ae729f6eb90a8ef1bdf569dfed94490b955ad5fdb38c60fa70` |
| `tools/fruit_loops/test_analyze_off_source_penalty_counterfactual.py` | `96cc2babb004f360f36cfc73024dfda992dc60f884cb2cc2a87059e563ce7bb6` |
| `ANALYSIS_MANIFEST_R0.1.yaml` | `cfaf035ee9fa3ef9266dc0a423063ab3d5ec124bb8ab8749627382173b7ae183` |
| `EXECUTION_RESULT_R0.1.md` | `adfbb5ee0136f1ad92e3fcc780873930213c9f0c32e0150d337855a0d05b27c4` |
| `COUNTERFACTUAL_METRICS_R0.1.csv` | `0b1bdd1157387a0b4d37c4d58070d041ba61e792922af109f1309f21c53cdc45` |
| `PRIMARY_EXECUTION_R0.1.csv` | `71784a60e0ed7b47a1b7560888a4bf4b1252929b5809ad50887668828dc5f24e` |
| `COUNTERFACTUAL_RESULT_R0.1.json` | `64d585ab1d0249ddc76839c650dbb95625593a8e2c8369973055168214895b74` |
| `COUNTERFACTUAL_EFFECT_R0.1.png` | `fb86b17d52918a1ecc98096074c0a6ff7cffbbb36e519609e9f652e04609dab2` |

## External evidence

All external evidence is under
`/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f6-off-source-penalty-counterfactual-r0.1/`.

| Artifact | SHA-256 |
|---|---|
| `logs/untouched-injected-sham.log` | `adfda5f05c9966fce127aa2ac326824d10afcec3b7d84d372296c6dce30a1765` |
| `logs/injected-without-uid4460.log` | `a7fd210ff8b4e64431fb08c190eb00d5a7aa4fb62da7bbac4f541cd25201410c` |
| intervention audit | `6894a356b889d15bc0641ba0d66c220ed3de7a888c87b94c6f7128eca790892f` |
| sham output checkpoint | `2256bb5888543b032e073bf59a81f743336c81df865961e8a8638369ce0deaa9` |
| counterfactual output checkpoint | `d7df0ee480ad99ab3e1b51bb9f311c69e5ae9ab7104a525490dc2fe32ff37faa` |
| counterfactual a1100 FITS | `c60da5025d7412820c01aa59b069d5aa324a2b09b8f2fbbfd6ac47e3897c3ee7` |
| counterfactual a1400 FITS | `5d8594aa566d3bd30f00e4ca3beecef69e3c69f26503f57ce4f0c7834670b0cd` |
| counterfactual a2000 FITS | `ff1cfea4d8964b5e157811f62ac1c2ba260bf16c88488a882e9f9d81009add14` |
| three-file response-map name-and-content aggregate | `866ec414196f59c62fe81046f3c027cbb8b41790f748cd624b7ee6305c4a19cd` |

Each response-map path, SHA-256, and byte count is also recorded in
`COUNTERFACTUAL_RESULT_R0.1.json`. The full isolated restart inputs and output
products remain in the registered external development root. The non-copying
`redu05` aliases do not duplicate or modify product content.
