# FRUIT EL-F2 observation-123424 input inventory

Status: **locally verified; development input only**

The user retrieved the raw detector files from the Unity directory
`/work/toltec/commissioning2025-test/beammaps/data/` on 2026-09-02 after a
file-specific request. An exact retrieval time was not supplied. Codex did not
connect to Unity.

## Observation identity checks

- raw detector headers: observation 123424, sub-observation 0, scan 2;
- raw networks present: 0--9, 11, and 12; network 10 is not expected by the
  source configuration;
- raw time-axis lengths: 7493--7495 samples;
- telescope DCS and TelescopeBackend identity: observation 123424,
  sub-observation 0, scan 2;
- telescope source and pattern: Neptune, Lissajous;
- APT: 5905 rows, 89 columns, and exactly networks 0--9, 11, and 12; and
- required APT fields `uid`, `array`, `nw`, `responsivity`, `flxscale`,
  `sens`, and `flag` are present.

## Exact files

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `input/raw/toltec0_123424_000_0002_2024_11_27_04_09_50.nc` | 38202904 | `c555414fc281cc39b33bd46783e15bce2e8ac22c20775135308a419a11ddf6e6` |
| `input/raw/toltec1_123424_000_0002_2024_11_27_04_09_50.nc` | 29596424 | `89689337f32981488306801317401e201c54552bc37d1c4a57d4e0d77e89738b` |
| `input/raw/toltec2_123424_000_0002_2024_11_27_04_09_50.nc` | 31704576 | `36e6a1e91cbb5258b1a958b6ee1d4446dc6f6b08fdc1432d021c78f44f6f914a` |
| `input/raw/toltec3_123424_000_0002_2024_11_27_04_09_50.nc` | 32245608 | `7494f28d5b5e1ae90aacc264c84ddfff4f20182edec9f76b3a7facec0dfbf2bc` |
| `input/raw/toltec4_123424_000_0002_2024_11_27_04_09_50.nc` | 25634040 | `4bb26e04f6eea19116594bb5bfb108696e65d67e686e84c226dab5e214d4b7b4` |
| `input/raw/toltec5_123424_000_0002_2024_11_27_04_09_50.nc` | 29415944 | `35909a256b46658e86a64954e0fc5f324e907aaaef0157da76f5974fd37573e4` |
| `input/raw/toltec6_123424_000_0002_2024_11_27_04_09_50.nc` | 34229904 | `f625e0606b9359913d73537f64b537ae51fce5e696dee049137f9c2428cbc843` |
| `input/raw/toltec7_123424_000_0002_2024_11_27_04_09_50.nc` | 25034920 | `96b900a25d94528014326c50ca021264240e7115314ac02865a1bebc3dae7a1b` |
| `input/raw/toltec8_123424_000_0002_2024_11_27_04_09_50.nc` | 27313832 | `d054d4cf8391ef2c44a83bbf5be0cb529d61de09ca06e4ab3d5998310b31c7a8` |
| `input/raw/toltec9_123424_000_0002_2024_11_27_04_09_50.nc` | 25634584 | `a841df1dbd7d9ffa7f8de1c4ae4a9d2c4de81286e7bcb2b76bd094fbbd63cf0b` |
| `input/raw/toltec11_123424_000_0002_2024_11_27_04_09_50.nc` | 29116008 | `82f5c626d272daacd4116c1a5b0712108bd950ab5fd7e9c6a900f5da5d898b89` |
| `input/raw/toltec12_123424_000_0002_2024_11_27_04_09_50.nc` | 30744032 | `1ad4b9a875e011a5fdd8522c9549c70f6e5424986d9e72cb3fbbdaf53777c3cd` |
| `apt_123424_matched.ecsv` | 6207815 | `16389e5e58b76d39ef7fcedd3888c662e96db592bc3a8561530379e51c435626` |
| `tel_toltec_2024-11-27_123424_00_0002_recomputed.nc` | 1226332 | `bfb95deda33e8f5a3e86a0990db2698bddb9ed7c39c6284733900d91fd99defe` |
| `citlali_rc1_fruitloops10_o123424.yaml` | 16258 | `f710d1c172b5655b136ef4d8ebbff918083d37cb4974036c7af8725978d51491` |

The detector files live under
`/Users/gwilson/work_toltec/local_data/fruit-development/point-123424`.
The APT and telescope files remain at their existing paths under
`/Users/gwilson/work_toltec/local_data/2026-ENG-hero-multiyear-pointings-v1`.
All are read-only inputs to this experiment.

## APT limitation

`apt_123424_matched.ecsv` is a legacy matched table, not an APT v2 run
directory with a manifest and resolved configuration. Its exact bytes and
content identity are frozen here, and the same file must be used in all four
trajectories. This permits a paired development comparison but not an APT
qualification, a production-readiness claim, or a claim that the calibration
is independent of legacy processing.
