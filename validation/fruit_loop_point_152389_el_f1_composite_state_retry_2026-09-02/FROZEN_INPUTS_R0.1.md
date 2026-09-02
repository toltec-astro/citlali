# SCI-FRUIT EL-F1-R1 frozen executable and inputs

Frozen before any primary comparison run on 2026-09-02.

## Executable

- Path: `/Users/gwilson/work_toltec/local_data/fruit-development/point-152389/fruit-injection-development/el-f1-composite-state-retry-r0.1/setup/citlali-el-f1-r1`
- SHA-256: `a49082dde8f71d6f50edd8c378ad94195496b5eb0e0855b746e189f3442acbcc`
- Embedded Citlali version: `sci-noi-v0.1-stage-a-27-g2b59ad642 (2026-09-02T10:36:22)`
- Embedded kids version: `04088da-dirty (2026-09-02T10:36:22)`
- Embedded repaired checkpoint schema confirmed: `citlali-reduction-restart-checkpoint-v5-el-f1-r1`

The executable was built from the approved repair before that repair was
committed.  Its embedded version string is therefore not a complete source
identity.  The local Citlali proposal base was
`30a3deed965ba5fe156d0eb577a2765592d0bf8d`; the SHA-256 of the uncommitted
code-and-test diff over the five repaired implementation/test files was
`fc656038ee49f244ab098f5127b1b1b93a3c76549bfeaa075976db7e3277b31b`.
The executable SHA-256 above is the exact execution identity.  No rebuild is
permitted between this freeze and completion of the primary trajectories.

## Inputs

All files below are frozen copies in the same `setup` directory.

| File | SHA-256 |
|---|---|
| `BASE_POINT_152389.yaml` | `dc0df89b706f1af9f32d747861f8c23975ded7cb0cf5c706110e7a96126d5909` |
| `COMMON_LOCAL.yaml` | `3a3dce72481a27352ff1d6764cfc7d9071360a211f533720a8be73698f811ae3` |
| `ALPHA_1P00_CONTROL.yaml` | `9968a9490ef0d8bbd1849cb85538f03af276972f7eb93cfe06536c279704c2ff` |
| `ALPHA_1P00_INJECTED.yaml` | `0c2ffae2793ffe80a3e6166a55369bca5dfd7adf9942b59006f3d79348e2dae7` |
| `ALPHA_1P25_CONTROL.yaml` | `75edc6a168381b032067b44068ed36c35eb9f71abe3df8d882ffb810ab494b64` |
| `ALPHA_1P25_INJECTED.yaml` | `73b96dc8c9ca50d3fc31cb87771c00f2322552d79485c3b76f861908fa1bc288` |
| `ALPHA_1P50_CONTROL.yaml` | `3aced17dd3bb7097afe8ed6ed3b1784d28a1c8a8146d42c0886cb9c1f90de1d8` |
| `ALPHA_1P50_INJECTED.yaml` | `110a24a4eb22af0855c1b934746fe0486eccdc1af6032c3e2acb0cd080e8fef0` |

The copied overlays were byte-compared with the repository-authored overlays
at freeze time.  The base configuration hash matches the authorized source
configuration identity.
