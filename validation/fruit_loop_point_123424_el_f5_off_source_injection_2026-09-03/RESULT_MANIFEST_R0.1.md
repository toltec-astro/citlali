# FRUIT EL-F5 result manifest r0.1

Generated after both registered trajectories completed and the prospectively
fixed location-control classification was evaluated on 2026-09-03.

The registered packet is commit
`0a7f24a4f46fe2e795593837b9abac8707c2371f`. The tested implementation and
fixed analysis classification are commit
`fd760cdbf59940f803ab38323088b35682f342cd`; the executable/input freeze is
commit `11e8be017dc0ab906d37f78a024a33d817d79464`. The frozen executable SHA-256
is `6431c6653ed46ff6e1dfa5512cd27e8169525f7a110207b0b24505786f39dbbe`.

After execution, the analyzer was extended only to persist the already
required complete response maps, expose the fitted source-centroid check,
count the target detector's existing contributor records, and state that an
above-unity amplitude moving toward unity is not a flux degradation. The
classification function, target event, sign test, fit window, annulus,
comparator, and scientific claim limits were not changed.

## Repository result artifacts

| Artifact | SHA-256 |
|---|---|
| `tools/fruit_loops/analyze_off_source_injection.py` | `0917fe4ff90836983385dd5cce796e8508368346cb1062dec127fa100b19b30e` |
| `tools/fruit_loops/test_analyze_off_source_injection.py` | `0aeed1ee705ad92b619bcf8d076aab22f64a0e2842ccd8f3be46e09a481fb5d8` |
| `tools/fruit_loops/analyze_compact_relaxation_screen.py` | `47ae85d85a8e01fe996f2fe3ec54756b24323e17efdacd4571ea2e8bbca6a5b2` |
| `ANALYSIS_MANIFEST_R0.1.yaml` | `a9e24373d5ce3223bdf34835a3369d2c69ca5fd09e1b9d212d487d6cb2513227` |
| `EXECUTION_RESULT_R0.1.md` | `c43f68bce04e530ee6186a597152409e32c0040a89f97e12c1adae254e663769` |
| `ITERATION_METRICS_R0.1.csv` | `d7d76c1cabb9ab754ef888a52e2df2b177ef64d5ee8b99ea6892d148b8040a53` |
| `PENALTY_INVENTORY_R0.1.csv` | `f7a5cc047aa7f83897e39bae6c28040623747c5367bf77a677198e385444d9af` |
| `PENALTY_COMPARISON_R0.1.csv` | `28ab2e265b4fae3b65bf3cc2712c06f30b10d3fbe4263ddfb9714ebd6cdfa587` |
| `PRIMARY_EXECUTION_R0.1.csv` | `f99dc350d8062a5d13a4bd26e525bef665d9757b93a9496f8b0012f349c7c07e` |
| `SCREEN_RESULT_R0.1.json` | `cd9a6eaca038be0907f00032524f06bd60c6195096f1522f5c35bc93b1f8e15d` |

## External evidence

All external files are under
`/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f5-off-source-injection-r0.1/`.

| Artifact | SHA-256 |
|---|---|
| `logs/point-123424-control.log` | `4c4383974be0b6c3b0ca40e7d57638db447af6821d9ed041d4ef8e409be56faf` |
| `logs/point-123424-off-source-injected.log` | `f2452be6338b814c2e386503898ce7b2163b877502980946745277de425a2f67` |
| complete 18-file `analysis/response-maps` name-and-content aggregate | `44d6c8cf650ed721c48a500c45de986f9c8d95a95b5fd3994e665fe1b8946476` |

Each individual response-map path, SHA-256, and byte count is recorded in
`SCREEN_RESULT_R0.1.json`. Repeating the response-map generation produced the
same aggregate digest. The complete trajectory products remain in the
registered external output root and are development evidence only.

