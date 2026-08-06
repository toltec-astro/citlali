# Owner return bundle

Return the complete diagnostic root after both observations finish, or after
the first hard failure. Preserve failed output; do not edit generated YAML,
retry in place, or substitute another observation.

The return must contain:

- `preparation/`, including source-config hashes and both rendered `all`
  configs;
- `jobs/`, including job IDs and Slurm stdout/stderr;
- `o150819/` and `o148670/`, each containing its one ordinary Citlali
  reduction tree;
- standard, `_left`, and `_right` map FITS, Beammap APT, and fit-QC ECSV files
  wherever the corresponding ordinary product is enabled;
- each observation's `beammap_direction_scan_registry_all.csv`;
- executable hash, repository commit, realized config/provenance products,
  final recursive checksum manifest, and Slurm `MaxRSS` evidence;
- `OWNER_NOTES.txt` identifying scheduler interruption, retry, or unexpected
  warning/error.

After all files are closed on Unity:

```bash
cd "$SCI_SPLIT_ROOT"
find . -type f ! -name RETURN_SHA256SUMS -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 shasum -a 256 > RETURN_SHA256SUMS
shasum -a 256 -c RETURN_SHA256SUMS

cd "$(dirname "$SCI_SPLIT_ROOT")"
tar -czf "$(basename "$SCI_SPLIT_ROOT").tar.gz" \
  "$(basename "$SCI_SPLIT_ROOT")"
shasum -a 256 "$(basename "$SCI_SPLIT_ROOT").tar.gz"
```

Download the tarball from the local machine with the `unity_toltec` SSH alias.
Do not omit failure logs merely because a later retry succeeds.
