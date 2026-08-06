# Owner return bundle

Return the complete diagnostic root after both observations finish, or after
the first hard failure. Preserve the failed root; do not edit generated YAML,
retry in place, or substitute another observation.

The return must contain:

- `preparation/`, including the two source-config hashes and all eight rendered
  mode configs;
- `jobs/`, including submitted job IDs and Slurm stdout/stderr;
- `o150819/{standard,left,right,all}/` and
  `o148670/{standard,left,right,all}/` for every attempted run;
- ordinary map FITS files, Beammap APT and fit-QC ECSV files, and every emitted
  nonstandard scan registry;
- the executable hash, repository commit, realized config/provenance products,
  and final recursive checksum manifest;
- a short `OWNER_NOTES.txt` identifying any scheduler interruption, retry, or
  unexpected warning/error.

On Unity, after all retained files are closed:

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

The owner can then download the tarball from the local source machine with the
`unity_toltec` SSH alias. Do not omit failure logs merely because a later retry
succeeds.
