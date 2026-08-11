# Reproduce the bounded result

All commands are local. They require the repository at the committed branch,
the checksum-valid frozen map result root, and the already retained PTC/PPT
files named by the selection. They do not access Unity.

```bash
export SCI_REPO=/Users/gwilson/GitHub/citlali-refactor
export SCI_PYTHON="$HOME/tolteca/bin/python"
export SCI_PACKAGE="$SCI_REPO/validation/sci_align_001_lissajous_timestream_2026-08-10"
export SCI_SELECTION="$SCI_REPO/validation/sci_align_001_lissajous_pointing_2026-08-10/selected_pointings.json"
export SCI_MAP_ROOT=/Users/gwilson/.codex/visualizations/2026/08/03/019fc822-9f45-72b1-91e9-775e80768d2a/sci_align_001_lissajous_pointing_2026-08-10
export SCI_RESULT_ROOT=/Users/gwilson/.codex/visualizations/2026/08/03/019fc822-9f45-72b1-91e9-775e80768d2a/sci_align_001_lissajous_timestream_2026-08-10

cd "$SCI_REPO"
MPLBACKEND=Agg MPLCONFIGDIR="$(mktemp -d)" XDG_CACHE_HOME="$(mktemp -d)" \
  "$SCI_PYTHON" tools/diagnostics/test_analyze_sci_align_001_lissajous_timestream.py

for obs in 131920 131926 133542 133544 135396 135398 136278 136280 150818; do
  (cd "$SCI_RESULT_ROOT/o$obs" && shasum -a 256 -c SHA256SUMS)
done

(cd "$SCI_PACKAGE" && shasum -a 256 -c SHA256SUMS)
```

The original run used `fit-anchor` only for ObsNum 150818 and
`analyze-observation` for each subsequently opened observation. After the
documented 500-realization multimodality gate repair and complete synthetic
rerun, the exact frozen command for the stop case was:

```bash
MPLBACKEND=Agg "$SCI_PYTHON" \
  tools/diagnostics/analyze_sci_align_001_lissajous_timestream.py \
  extend-bootstrap \
  --protocol "$SCI_PACKAGE/frozen_protocol.json" \
  --selection "$SCI_SELECTION" \
  --map-root "$SCI_MAP_ROOT" \
  --obsnum 136280 \
  --output "$SCI_RESULT_ROOT/o136280"
```

The compact partial tables are a deterministic field projection of the nine
checksum-valid `result.json` documents. Recreate them in a temporary directory
and compare bytes with:

```bash
SCI_COMPACT_CHECK=$(mktemp -d)
"$SCI_PYTHON" "$SCI_PACKAGE/summarize_partial_results.py" \
  "$SCI_RESULT_ROOT" "$SCI_COMPACT_CHECK"

for name in \
  partial_observation_results.ecsv \
  partial_observation_results.json \
  partial_input_identities.json \
  partial_stop_summary.json; do
  cmp "$SCI_COMPACT_CHECK/$name" "$SCI_PACKAGE/$name"
done
```

No bootstrap arrays are committed.
