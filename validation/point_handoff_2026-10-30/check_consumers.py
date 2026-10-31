"""Exercise immutable operational consumers on successor synthetic POINT tables.

No reduction, saved-map fit, flux service, APT production or sibling import/edit.
The extracted functions are the actual pinned sources; their narrow dependencies
are supplied explicitly. Files are created only inside a temporary synthetic
project. Invoke with the local Python environment and --output /path/result.json.
"""

import argparse
import ast
import hashlib
import importlib
import json
import logging
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import astropy.units as u
from astropy.table import Table
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOLTECA_REV = "2791e6a1e6349ad1d3ac549a648f41cbc51b98c7"
TOLPROJ_REV = "6309a5a852a0412cee867992aff01341a56b0ebf"


def source(repo, revision, path, bindings):
    content = subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{revision}:{path}"]
    )
    bindings.append(dict(repository=str(repo), revision=revision, path=path,
                         sha256=hashlib.sha256(content).hexdigest()))
    return content.decode()


def extract(content, names, namespace, filename):
    """Execute only named definitions/assignments, with unchanged function bodies."""
    nodes = []
    for node in ast.walk(ast.parse(content)):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            node.decorator_list = []
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names for target in node.targets
        ):
            nodes.append(node)
    found = {node.name if hasattr(node, "name") else node.targets[0].id for node in nodes}
    if found != set(names):
        raise ValueError(f"pinned source definitions differ: {found} != {set(names)}")
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, filename, "exec"), namespace)


def telescope(observation, az, el):
    values = {"Header.Dcs.ObsNum": observation}
    for axis, pair in (("Az", az), ("El", el)):
        for kind, value in zip(("UserOff", "PaddleOff"), pair):
            values[f"Header.PointModel.{axis}{kind}"] = (value * u.arcsec).to_value(u.rad)
    return SimpleNamespace(nc_node=SimpleNamespace(
        variables={key: np.asarray([value]) for key, value in values.items()}))


def published_fixture(project, observation, mjd, x, y, az, el, publisher):
    dest = project / f"pointings/reduced/redu00/{observation}/raw"
    dest.mkdir(parents=True)
    meta = dict(obsnum=observation, source="synthetic-point-consumer-fixture",
                date="2023-02-25T00:00:00", mjd=mjd)
    for axis, pair in (("Az", az), ("El", el)):
        for kind, value in zip(("UserOff", "PaddleOff"), pair):
            meta[f"Header.PointModel.{axis}{kind}"] = (value * u.arcsec).to_value(u.rad)
    metadata = dict(status="available", cause=None, meta=meta,
                    applied_pointing_offsets=dict(az=[0.], alt=[0.]),
                    telescope=dict(path="synthetic-telescope.nc", sha256="a" * 64))
    guards, records = {}, []
    for i, array in enumerate(("a1100", "a1400", "a2000")):
        guards[array] = dict(accepted=True, status="accepted", complete=True,
                            accepted_parameters=[1000. + i, x[i], y[i], 4., 4.2, 0., 3.],
                            accepted_centroid_tangent_arcsec=[x[i], y[i]])
        records.append(dict(array=array, map=f"synthetic-{array}.fits", sha256=str(i + 1) * 64,
                            native_product_identity=f"synthetic:{array}",
                            fit=f"point-fit-{array}.json"))
    result = publisher(dest, observation, guards, records, metadata)
    assert result["status"] == "published", result
    path = dest / result["table"]
    table = Table.read(path, format="ascii.ecsv")
    assert list(table["array"]) == [0, 1, 2]
    assert table["array"].dtype.kind in "iu"
    assert str(table["amp"].unit) == str(table["amp_err"].unit) == "mJy / beam"
    assert str(table["x_t"].unit) == str(table["y_t"].unit) == "arcsec"
    np.testing.assert_array_equal(np.asarray(table["x_t"]), x)
    np.testing.assert_array_equal(np.asarray(table["y_t"]), y)
    np.testing.assert_array_equal(np.asarray(table["amp"]), [1000., 1001., 1002.])
    assert np.all(np.isnan(np.asarray(table["amp_err"])))
    assert table.meta["calibration_eligible"] is False
    assert table.meta["pointing_eligible"] is True
    return path, table


def run(args):
    bindings = []
    tcpath = "tolteca/reduce/engines/citlali.py"
    tcsource = source(args.tolteca, TOLTECA_REV, tcpath, bindings)
    tc = dict(np=np, u=u, Table=Table, get_logger=lambda: logging.getLogger("consumer-fixture"),
              pformat_yaml=str)
    extract(tcsource, {"_resolve_pointing_offsets", "load_data_objs"}, tc, f"{TOLTECA_REV}:{tcpath}")
    resolve = tc["_resolve_pointing_offsets"]
    tppath = "tolproj/steps/calibrate_flxscale.py"
    tpsource = source(args.tolproj, TOLPROJ_REV, tppath, bindings)
    errorsource = source(args.tolproj, TOLPROJ_REV, "tolproj/errors.py", bindings)
    tp = dict(Path=Path, Table=Table, math=math)
    extract(errorsource, {"TolprojUserError"}, tp, f"{TOLPROJ_REV}:tolproj/errors.py")
    extract(tpsource, {"ARRAY_INDEX_TO_NAME", "_find_pointing_product", "_metadata_unit",
                       "_array_name_map", "_read_recovered_fluxes"}, tp,
            f"{TOLPROJ_REV}:{tppath}")
    # This reader uses only membership, never frequency values. Names come from
    # the same pinned module's operational integer-to-array mapping.
    tp["DEFAULT_ARRAY_FREQUENCIES_GHZ"] = dict.fromkeys(tp["ARRAY_INDEX_TO_NAME"].values())
    sys.path.insert(0, str(ROOT / "tools/timestream_successor"))
    publish = importlib.import_module("configured_point_table").publish_pointing_table
    calibration = importlib.import_module("configured_point_config")._calibration
    sources = {}
    for name in ("configured_point_table.py", "configured_point_config.py"):
        path = ROOT / "tools/timestream_successor" / name
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    results, rejections, table_digests = {}, {}, {}
    with tempfile.TemporaryDirectory(prefix="citlali-point-consumers-") as scratch:
        project = Path(scratch)
        # Observation IDs use ordinary six-digit project directory conventions.
        before_path, before = published_fixture(project, 152389, 60000.123456,
            [3., 6., 9.], [-6., -3., 0.], (1.2, .3), (-1., -.5), publish)
        after_path, after = published_fixture(project, 152391, 60002.234567,
            [-9., -6., -3.], [6., 3., 0.], (2., .1), (2., .5), publish)
        # Use the actual CalObj directory discovery/load method before the
        # actual resolver, as normal project-relative cal_objs do after path
        # resolution. Parent observation folders contain one raw PPT each.
        loaded = []
        for path in (before_path, after_path):
            objects = tc["load_data_objs"](SimpleNamespace(path=path.parent.parent,
                                                         select="obsnum == 152390"))
            assert len(objects) == 1 and objects[0]["type"] == "ppt"
            loaded.append(objects[0]["data"])
        before, after = loaded
        science = telescope(152390, (.2, .1), (-.2, -.3))
        cases = dict(
            bracket=([before, after], science, [[-7.2, 4.2], [4., -6.],
                                                [60000.123456, 60002.234567]]),
            reversed=([after, before], science, [[-7.2, 4.2], [4., -6.],
                                                [60000.123456, 60002.234567]]),
            single_before=([before], science, [[-7.2], [4.], [60000.123456]]),
            single_after=([after], science, [[4.2], [-6.], [60002.234567]]),
            exact_observation=([before, after], telescope(152389, (.2, .1), (-.2, -.3)),
                               [[-7.2], [4.], [60000.123456]]))
        for name, (tables, tel, expected) in cases.items():
            item = json.loads(json.dumps(resolve(None, tables, tel)[0]))
            offsets = item["pointing_offsets"]
            actual = [offsets[0]["value_arcsec"], offsets[1]["value_arcsec"],
                      offsets[2]["modified_julian_date"]]
            np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-14)
            config = {"inputs": [{"cal_items": [{"type": "array_prop_table"}, item]}]}
            calibration(config)
            results[name] = dict(resolved=actual, configured_adapter="accepted")
        for path, table in ((before_path, before), (after_path, after)):
            observation = table.meta["obsnum"]
            assert tp["_find_pointing_product"](project, "redu00", observation) == path
            table_digests[str(observation)] = hashlib.sha256(path.read_bytes()).hexdigest()
            try:
                tp["_read_recovered_fluxes"](path, observation)
            except tp["TolprojUserError"] as exc:
                assert "invalid recovered flux for a1100" in str(exc), exc
                rejections[str(observation)] = str(exc).replace(str(project), "<synthetic-project>")
            else:
                raise AssertionError("unavailable amp_err was accepted for flux calibration")
    record = dict(scope="Synthetic table publication and actual pinned consumer functions only; no reduction, flux calibration or full TolTECA process", sources=bindings,
                  implementation_sha256=sources, synthetic=results,
                  published_table_sha256=table_digests, flux_calibration_rejections=rejections,
                  checks=dict(all_three_array_identity=True, amplitudes_and_units=True,
                              signed_centroids=True, unavailable_errors_rejected=True,
                              conventional_discovery=True, tolteca_calobj_discovery=True),
                  limits=["Conditional sigma_A is not used as amp_err.",
                          "Calibration eligibility metadata alone is not enforced by the legacy reader; nonfinite errors enforce the hold.",
                          "Pointing eligibility is a publication guard, not observed scientific acceptance.",
                          "Native interpolation/AST execution is covered separately by C++ tests."])
    if args.output:
        args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(json.dumps(record, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tolteca", type=Path, default=Path("/Users/gwilson/GitHub/tolteca"))
    parser.add_argument("--tolproj", type=Path, default=Path("/Users/gwilson/GitHub/tolproj"))
    parser.add_argument("--output", type=Path)
    run(parser.parse_args())
