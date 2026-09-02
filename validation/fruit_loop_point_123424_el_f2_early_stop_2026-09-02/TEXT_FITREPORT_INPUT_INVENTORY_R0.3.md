# FRUIT EL-F2 text fit-report input inventory r0.3

Status: **locally verified candidate development input; not yet authorized**

After the r0.2 file-type failure, a read-only local search found the exact text
form required by Citlali under:

`/Users/gwilson/work_toltec/local_data/tone-match-lab/c2025t_tune_text/beammaps/data`

Their acquisition provenance beyond that existing local path is unavailable.
They are therefore proposed only for the already development-only paired
screen and cannot support APT, recurrence, or production qualification.

For each expected network, the executable's observed regular expression
selects exactly one file. Every file parses as ECSV with the same 14 model
columns, and its metadata reports observation 123424, sub-observation 0, tune
scan 1, and the network encoded in the filename. Its row count equals the
`ntones` dimension of the corresponding r0.2 processed tune NetCDF.

| Network | Rows | Bytes | SHA-256 | File |
| ---: | ---: | ---: | --- | --- |
| 0 | 630 | 130398 | `b7e901b45849ab40de208e44e4758d7a166d32cc6529afd99bcec87b95dd4550` | `toltec0_123424_000_0001_2024_11_27_04_08_30_tune.txt` |
| 1 | 487 | 101199 | `4351a10f53b140a3fbbc44b87da351a86c4176f5a69e2a7bae7a2192af4771c3` | `toltec1_123424_000_0001_2024_11_27_04_08_30_tune.txt` |
| 2 | 522 | 108204 | `61f56bf0985a3542bc5e09e314e12018f4547dd0c1b4edbae7de41e88a619004` | `toltec2_123424_000_0001_2024_11_27_04_08_30_tune.txt` |
| 3 | 531 | 109708 | `4a4aea2d1420f285e6c439b53ccc06f5d4c000d928511057e5131d0be16e897a` | `toltec3_123424_000_0001_2024_11_27_04_08_30_tune.txt` |
| 4 | 421 | 87220 | `790fb017e3ab62638ad30acd1752ccc5db884448c7f8fe3440fcfb9265b5c3ca` | `toltec4_123424_000_0001_2024_11_27_04_08_30_tune.txt` |
| 5 | 484 | 100145 | `240023e8cf5214529fd173a9b56f93f4a804eb78114f33d50475c5bd4c880261` | `toltec5_123424_000_0001_2024_11_27_04_08_30_tune.txt` |
| 6 | 564 | 115504 | `b0f41d5e6e0010983ea5107d0f284847e9373166c671e0e110e500b0d89c3368` | `toltec6_123424_000_0001_2024_11_27_04_08_30_tune.txt` |
| 7 | 411 | 84975 | `023f7bc923dada4cae32ee62b640433eb10968602da97b1c4304d13b11952984` | `toltec7_123424_000_0001_2024_11_27_04_08_30_tune.txt` |
| 8 | 449 | 93155 | `d0c01cbbf4daec7d6cf5ccb1ca89d11135040e1bf6239b2cb95506f3f8fdb615` | `toltec8_123424_000_0001_2024_11_27_04_08_30_tune.txt` |
| 9 | 421 | 85673 | `2924ea377f92d91bb4b49644bfeb3dafb5ebfc7156cd6c5d13dfad0efb5c508f` | `toltec9_123424_000_0001_2024_11_27_04_08_30_tune.txt` |
| 11 | 479 | 99347 | `869dcbee034587706eca8795276e0f9c981fa4b0bd2d5a9113cf3017e2314833` | `toltec11_123424_000_0001_2024_11_27_04_08_30_tune.txt` |
| 12 | 506 | 104644 | `3268a6da23370136bb26b25f9820fd09573a9adb1bdde82e402c89bd76ee51c0` | `toltec12_123424_000_0001_2024_11_27_04_08_30_tune.txt` |

The 12 files contain 5,905 data rows and 1,220,172 bytes in total. They remain
external, immutable inputs; neither the r0.2 NetCDF files nor these text tables
are independent sky products.
