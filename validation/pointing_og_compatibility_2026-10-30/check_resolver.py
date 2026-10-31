"""Run only the pinned TolTECA offset resolver on fixtures/saved headers.
No reduction, source fit, calibration issuance or numerical pipeline execution.
"""

import ast
import importlib
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import astropy.units as u
from astropy.table import Table
import netCDF4
import numpy as np
import yaml

OUT = Path(__file__).resolve().parent
PROJECT = Path(
    "/Users/gwilson/work_toltec/local_data/2026-refactor/projects/COM01-local-trial-20261027"
)
REV = "2791e6a1e6349ad1d3ac549a648f41cbc51b98c7"
SOURCE = "tolteca/reduce/engines/citlali.py"
source = subprocess.check_output(
    ["git", "-C", "/Users/gwilson/GitHub/tolteca", "show", f"{REV}:{SOURCE}"]
).decode()
node = next(
    n
    for n in ast.walk(ast.parse(source))
    if isinstance(n, ast.FunctionDef) and n.name == "_resolve_pointing_offsets"
)
node.decorator_list = []
namespace = dict(
    np=np, u=u, get_logger=lambda: logging.getLogger("offset-fixture"), pformat_yaml=str
)
exec(
    compile(
        ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
        f"{REV}:{SOURCE}",
        "exec",
    ),
    namespace,
)
resolve = namespace["_resolve_pointing_offsets"]
checked = {}


def bind(path):
    path = Path(path)
    checked[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def telescope(obs, az, alt):
    values = {"Header.Dcs.ObsNum": obs}
    for axis, pair in [("Az", az), ("El", alt)]:
        for name, value in zip(("UserOff", "PaddleOff"), pair):
            values[f"Header.PointModel.{axis}{name}"] = (value * u.arcsec).to_value(
                u.rad
            )
    return SimpleNamespace(
        nc_node=SimpleNamespace(variables={k: np.array([v]) for k, v in values.items()})
    )


def ppt(obs, mjd, x, y, az, alt):
    t = Table(dict(x_t=x, y_t=y))
    t.meta = dict(obsnum=obs, mjd=mjd)
    for axis, pair in [("Az", az), ("El", alt)]:
        for name, value in zip(("UserOff", "PaddleOff"), pair):
            t.meta[f"Header.PointModel.{axis}{name}"] = (value * u.arcsec).to_value(
                u.rad
            )
    return t


def vals(result):
    p = result[0]["pointing_offsets"]
    return np.array(
        [p[0]["value_arcsec"], p[1]["value_arcsec"], p[2]["modified_julian_date"]]
    )


a = ppt(100, 60000, [3, 6, 9], [-6, -3, 0], (1.2, 0.3), (-1, -0.5))
b = ppt(102, 60002, [-9, -6, -3], [6, 3, 0], (2, 0.1), (2, 0.5))
tel = telescope(101, (0.2, 0.1), (-0.2, -0.3))
cases = {
    "bracket_sorted": ([a, b], tel, [[-7.2, 4.2], [4, -6], [60000, 60002]]),
    "bracket_input_reversed": ([b, a], tel, [[-7.2, 4.2], [4, -6], [60000, 60002]]),
    "single_before": ([a], tel, [[-7.2], [4], [60000]]),
    "single_after": ([b], tel, [[4.2], [-6], [60002]]),
    "same_observation_once": (
        [a, b],
        telescope(100, (0.2, 0.1), (-0.2, -0.3)),
        [[-7.2], [4], [60000]],
    ),
}
results = {}
for name, (tables, headers, expected) in cases.items():
    actual = vals(resolve(None, tables, headers))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-14)
    results[name] = actual.tolist()

# Reproduce both archived science corrections using actual reference PPTs and TEL headers.
ref = PROJECT / "reference/v2-science"
root = ref / "refactor/NGC4449/reduced"
prov = yaml.safe_load(bind(root / "redu03/astrometry_provenance.yaml").read_text())
tables = []
for obs in [152389, 152391, 152393]:
    path = next((ref / f"pointings/reduced/redu00/{obs}/raw").glob("ppt_*.ecsv"))
    tables.append(Table.read(bind(path), format="ascii.ecsv"))
real = {}
for obs in [152390, 152392]:
    telpath = next(root.glob(f"tel*_{obs}_*_recomputed.nc"))
    # Hash only this scoped telescope file, not any detector raw data.
    bind(telpath)
    with netCDF4.Dataset(telpath, "r") as nc:
        actual = vals(resolve(None, tables, SimpleNamespace(nc_node=nc)))
        fields = {
            k: float(nc.variables[k][:].item())
            for k in [
                "Header.PointModel.AzUserOff",
                "Header.PointModel.AzPaddleOff",
                "Header.PointModel.ElUserOff",
                "Header.PointModel.ElPaddleOff",
            ]
        }
    saved = next(q for q in prov["observations"] if q["obsnum"] == obs)["requested"][
        "pointing_offsets"
    ]
    expected = np.array(
        [saved["az_arcsec"], saved["alt_arcsec"], saved["modified_julian_date"]]
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    real[str(obs)] = dict(
        resolved=actual.tolist(),
        saved=expected.tolist(),
        max_absolute_difference=float(np.max(abs(actual - expected))),
        telescope_offsets_rad=fields,
    )
# Test the existing successor configuration helper, without opening APT/raw inputs.
candidate = Path(
    "/Users/gwilson/.codex/worktrees/pointing-engineering-candidate-20261030/citlali-refactor"
)
sys.path.insert(0, str(candidate / "tools/timestream_successor"))
_calibration = importlib.import_module("configured_point_config")._calibration

config_cases = {}
for name in ["bracket_sorted", "single_before", "single_after"]:
    tables, headers, _ = cases[name]
    item = json.loads(json.dumps(resolve(None, tables, headers)[0]))
    config = {"inputs": [{"cal_items": [{"type": "array_prop_table"}, item]}]}
    try:
        _calibration(config)
        config_cases[name] = {"accepted": True}
    except ValueError as exc:
        config_cases[name] = {"accepted": False, "reason": str(exc)}
assert config_cases["bracket_sorted"]["accepted"]
assert not config_cases["single_before"]["accepted"]
assert not config_cases["single_after"]["accepted"]
item = json.loads(json.dumps(resolve(None, [a], tel)[0]))
item["pointing_offsets"] = item["pointing_offsets"][:2]
_calibration({"inputs": [{"cal_items": [{"type": "array_prop_table"}, item]}]})
config_cases["constant_without_epoch"] = {"accepted": True}
item = json.loads(json.dumps(resolve(None, [a, b], tel)[0]))
item["pointing_offsets"] = item["pointing_offsets"][:2]
try:
    _calibration({"inputs": [{"cal_items": [{"type": "array_prop_table"}, item]}]})
except ValueError as exc:
    config_cases["two_values_without_epochs"] = {"accepted": False, "reason": str(exc)}
else:
    raise AssertionError("Expected demonstrated epoch-support restriction")
bind(candidate / "tools/timestream_successor/configured_point_config.py")

# Same-time arithmetic comparison only; not a native RTC/t_common schedule replay.
for obs, result in real.items():
    v = np.asarray(result["resolved"])
    full = (v[2] - 40587.0) * 86400.0
    old = np.trunc((v[2] - 2440587.5 + 2400000.5) * 86400.0)
    domain = np.array([max(full[0], old[0]), min(full[1], old[1])])
    before = v[:2, 0, None] + (v[:2, 1] - v[:2, 0])[:, None] * (domain - old[0]) / (
        old[1] - old[0]
    )
    after = v[:2, 0, None] + (v[:2, 1] - v[:2, 0])[:, None] * (domain - full[0]) / (
        full[1] - full[0]
    )
    result["fractional_epoch_comparison"] = {
        "new_minus_legacy_epoch_seconds": (full - old).tolist(),
        "common_time_domain_unix": domain.tolist(),
        "max_same_time_abs_offset_difference_arcsec": np.max(
            abs(after - before), axis=1
        ).tolist(),
        "scope": "Maximum over overlapping bracket support from endpoint evaluation of linear difference; not sample-schedule or map-result parity",
    }

record = dict(
    configuration_cases=config_cases,
    scope="Pinned resolver arithmetic on fixtures and saved reference metadata; not successor end-to-end correction qualification",
    source=dict(
        repository="/Users/gwilson/GitHub/tolteca",
        revision=REV,
        path=SOURCE,
        sha256=hashlib.sha256(source.encode()).hexdigest(),
    ),
    synthetic=results,
    saved_reference=real,
    inputs=checked,
)
(OUT / "resolver_checks.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(dict(fixtures=len(results), saved_reference=real), indent=2))
