# External Validation Fixtures

Large validation payloads remain outside Git. Each JSON manifest in this
directory gives a path-independent content identity so a fixture copied from
Unity, a validation-suite mirror, or another controlled store is accepted only
when its basename, byte size, and SHA-256 match.

## Kidscpp Real Reader

`kidscpp_real_reader_pointing_152389_v1.json` identifies the network-0,
scan-2 raw timestream from pointing observation 152389. Within the self-
contained Citlali validation suite, the payload is located at:

```text
point/data/toltec0_152389_000_0002_2026_02_19_06_34_38.nc
```

Run the installed-reader gate with the payload at any local path:

```bash
$HOME/tolteca/bin/python tools/build/test_spack_kidscpp.py \
  --kidscpp-source build/spack-sources/kidscpp \
  --require-real-data \
  --fixture /path/to/toltec0_152389_000_0002_2026_02_19_06_34_38.nc \
  --fixture-manifest \
    validation/fixtures/kidscpp_real_reader_pointing_152389_v1.json
```

The test rejects missing, renamed, truncated, or modified payloads before
opening the file through Kidscpp.
