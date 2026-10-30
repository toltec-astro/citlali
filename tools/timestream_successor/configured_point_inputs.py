"""Bind one configured POINT acquisition to the existing native stage inputs.

This module inventories inputs; it performs no scientific processing. Transport
and acquisition checks happen here once. The native canonical APT owner remains
responsible for full semantic admission of the bound bundle.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import re
from pathlib import Path

import netCDF4
import numpy as np
import yaml

from configured_point_types import FileBinding, InputBinding, PointingMetadata, PreparedPoint


ARRAYS = ("a1100", "a1400", "a2000")
DEVELOPMENT_REFERENCE = "configurable-point-v1"
NO_MASK_AUTHORITY = "owner-configurable-POINT-no-spatial-mask-2026-10-28"


def _require(condition, message):
    if not condition:
        raise ValueError("configured_point.inputs: " + message)


def _digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class _Bindings:
    """One digest per resolved file for this preparation's lifetime."""

    def __init__(self):
        self.records: dict[str, InputBinding] = {}

    def bind(self, path: Path, role: str) -> FileBinding:
        path = Path(path).resolve(strict=True)
        _require(path.is_file(), "input is not a regular file: " + str(path))
        key = str(path)
        if key not in self.records:
            before = path.stat()
            sha = _digest(path)
            after = path.stat()
            _require(
                (before.st_size, before.st_mtime_ns, before.st_ino)
                == (after.st_size, after.st_mtime_ns, after.st_ino),
                "input changed while hashing: " + key,
            )
            self.records[key] = InputBinding(path=key, sha256=sha, bytes=after.st_size, roles=[])
        record = self.records[key]
        if role not in record["roles"]:
            record["roles"].append(role)
        return {"path": record["path"], "sha256": record["sha256"]}


def _path(value, launch_dir):
    _require(isinstance(value, str) and bool(value), "nonempty filepath required")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else launch_dir / path).resolve(strict=True)


def _ecsv(path):
    lines = path.read_text().splitlines()
    _require(bool(lines) and lines[0] == "# %ECSV 1.0", "canonical ECSV required: " + str(path))
    header = yaml.safe_load("\n".join(line[2:] for line in lines[1:] if line.startswith("# ")))
    _require(isinstance(header, dict), "ECSV header missing: " + str(path))
    meta = header.get("meta", {}).get("canonical_apt_v2", {})
    _require(isinstance(meta, dict) and bool(meta), "canonical APT-v2 metadata missing")
    _require(header.get("delimiter", ",") == ",", "canonical APT-v2 comma delimiter required")
    data = "\n".join(line for line in lines if not line.startswith("#"))
    rows = list(csv.DictReader(io.StringIO(data)))
    _require(
        meta.get("row_count") == len(rows), "canonical component row count differs: " + str(path)
    )
    return meta, rows


def _scope(metadata):
    observation = metadata.get("observation", {})
    return tuple(observation.get(key) for key in ("obsnum", "subobsnum", "scannum"))


def _bundle(manifest, scope, bindings):
    root = bindings.bind(manifest, "canonical APT root manifest")
    meta, rows = _ecsv(manifest)
    _require(
        meta.get("schema_version") == "citlali-canonical-apt-root-manifest-v2"
        and meta.get("component_role") == "manifest"
        and meta.get("product_kind") == "observation-matched"
        and _scope(meta) == scope,
        "canonical matched APT observation/schema differs",
    )
    receipt = manifest.with_suffix(manifest.suffix + ".sha256")
    bindings.bind(receipt, "canonical APT root receipt")
    lines = receipt.read_text().splitlines()
    _require(
        len(lines) == 5 and lines[0] == "citlali-canonical-apt-root-receipt-v2",
        "canonical APT root receipt missing or malformed",
    )
    values = dict(line.split("=", 1) for line in lines[1:])
    _require(
        set(values) == {"scope", "envelope_sha256", "byte_sha256", "byte_count"}
        and values["scope"] == "citlali-canonical-apt-root-manifest-byte-sha256-v2"
        and values["envelope_sha256"] == meta.get("envelope_sha256")
        and values["byte_sha256"] == "sha256:" + root["sha256"]
        and int(values["byte_count"]) == bindings.records[root["path"]]["bytes"],
        "canonical APT root receipt differs from manifest",
    )
    components = {}
    for row in rows:
        role = row["role"]
        relative = Path(row["relative_path"])
        _require(
            role not in components and not relative.is_absolute() and ".." not in relative.parts,
            "duplicate component role or unsafe canonical component path",
        )
        path = (manifest.parent / relative).resolve(strict=True)
        _require(
            path.is_relative_to(manifest.parent), "canonical component leaves bundle directory"
        )
        bound = bindings.bind(path, "canonical APT component: " + role)
        _require(
            row["transport_sha256"] == "sha256:" + bound["sha256"]
            and int(row["byte_count"]) == bindings.records[bound["path"]]["bytes"],
            "canonical APT component transport differs: " + role,
        )
        components[role] = (path, row)
    tables = {}
    for role, schema in (
        ("apt", "citlali-canonical-matched-apt-v2"),
        ("sources", "citlali-canonical-apt-sources-v2"),
    ):
        _require(role in components, "canonical component missing: " + role)
        path, descriptor = components[role]
        child, table = _ecsv(path)
        _require(
            child.get("schema_version") == schema
            and descriptor["schema"] == schema
            and child.get("component_role") == role
            and _scope(child) == scope
            and child.get("product_kind") == meta["product_kind"]
            and child.get("occurrence") == meta.get("occurrence")
            and child.get("event_reference") == meta.get("event_reference")
            and descriptor["semantic_sha256"] == child.get("semantic_sha256")
            and descriptor["envelope_sha256"] == child.get("envelope_sha256")
            and int(descriptor["row_count"]) == len(table),
            "canonical component observation or descriptor differs: " + role,
        )
        tables[role] = table
    return tables["apt"], tables["sources"]


def _integer(nc, name):
    _require(name in nc.variables, "missing acquisition header: " + name)
    value = nc[name][...]
    _require(value.size == 1 and not np.ma.is_masked(value), "invalid acquisition header: " + name)
    scalar = value.item()
    _require(
        np.isfinite(scalar) and scalar == int(scalar), "noninteger acquisition header: " + name
    )
    return int(scalar)


def _text(nc, name):
    _require(name in nc.variables, "missing acquisition header: " + name)
    value = nc[name][...]
    if value.dtype.kind == "S":
        return value.tobytes().decode("utf-8").strip(" \x00")
    return "".join(str(item) for item in np.asarray(value).ravel()).strip(" \x00")


def _pointing_metadata(
    nc, observation: int, calibration: list, telescope: FileBinding
) -> PointingMetadata:
    """Observe OG PPT metadata while TEL is open; do not learn corrections.

    Missing/masked/nonfinite scientific metadata withholds only the handoff.
    I/O and software errors propagate. TelTime is the same Unix-second series
    used by native CAL and OG's mean-MJD publication, before any resampling.
    """
    result = PointingMetadata(
        status="unavailable",
        cause=None,
        meta={},
        applied_pointing_offsets={},
        telescope=telescope,
    )
    try:

        def scalar(name):
            _require(name in nc.variables, "missing pointing header: " + name)
            value = nc[name][...]
            _require(
                value.size == 1 and not np.ma.is_masked(value), "invalid pointing header: " + name
            )
            value = float(value.item())
            _require(np.isfinite(value), "nonfinite pointing header: " + name)
            return value

        source = _text(nc, "Header.Source.SourceName")
        _require(bool(source), "empty pointing source name")
        key = "Data.TelescopeBackend.TelTime"
        _require(key in nc.variables, "missing pointing telescope time")
        times = nc[key][...]
        _require(
            times.ndim == 1
            and times.size > 0
            and not np.ma.is_masked(times)
            and np.isfinite(times).all(),
            "invalid pointing telescope time",
        )
        mean_unix = float(np.asarray(times, dtype=float).mean())
        mjd = mean_unix / 86400.0 + 40587.0
        _require(np.isfinite(mjd) and mjd > 0, "invalid pointing mean MJD")
        # A UTC observation date, distinct from wall-clock product creation.
        date = datetime.fromtimestamp(mean_unix, timezone.utc).isoformat()
        meta = dict(obsnum=observation, source=source, date=date, mjd=mjd)
        for axis in ("Az", "El"):
            for kind in ("User", "Paddle"):
                key = f"Header.PointModel.{axis}{kind}Off"
                meta[key] = scalar(key)
        astrometry = [item for item in calibration if item.get("type") == "astrometry"]
        _require(len(astrometry) == 1, "one explicit astrometry item required for pointing handoff")
        axes = {}
        for entry in astrometry[0].get("pointing_offsets", []):
            if "axes_name" not in entry:
                continue
            axis, values = entry["axes_name"], entry.get("value_arcsec")
            _require(
                axis in ("az", "alt")
                and axis not in axes
                and isinstance(values, list)
                and len(values) in (1, 2)
                and all(type(v) in (int, float) and np.isfinite(v) for v in values),
                "invalid applied pointing offsets for handoff",
            )
            axes[axis] = list(values)
        _require(set(axes) == {"az", "alt"}, "missing applied pointing offset axes")
        result.update(status="available", meta=meta, applied_pointing_offsets=axes)
    except (ValueError, OverflowError) as exc:
        result["cause"] = str(exc)
    return result


def _write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def prepare(
    config: dict, out: Path, *, launch_dir: Path, rtc_profiles: dict[str, Path]
) -> PreparedPoint:
    """Prepare fresh native request files from an already validated YAML config.

    All input paths, including ``kids.solver.fitreportdir``, are relative to
    ``launch_dir``. The input configuration is never mutated. ``bindings`` holds
    compact cached transport identities; native owners recheck them at use.
    """
    out = Path(out).absolute()
    launch_dir = Path(launch_dir).resolve(strict=True)
    _require(not out.exists(), "fresh inputs directory required: " + str(out))
    effective = copy.deepcopy(config)
    _require(
        effective.get("runtime", {}).get("reduction_type") == "pointing",
        "one POINT reduction required",
    )
    acquisitions = effective.get("inputs", [])
    _require(isinstance(acquisitions, list) and len(acquisitions) == 1, "one acquisition required")
    acquisition = acquisitions[0]
    match = re.fullmatch(r"(\d+)_0_2", str(acquisition.get("meta", {}).get("name", "")))
    if match is None:
        raise ValueError(
            "configured_point.inputs: POINT acquisition name must identify observation_0_2"
        )
    obs, sub, scan = int(match[1]), 0, 2
    _require(obs > 0, "positive observation number required")
    scope = (obs, sub, scan)
    _require(acquisition["meta"]["name"] == f"{obs}_0_2", "noncanonical acquisition name")
    data = acquisition.get("data_items", [])
    calibration = acquisition.get("cal_items", [])
    interfaces = {}
    for item in data:
        name = item.get("meta", {}).get("interface")
        _require(
            isinstance(name, str) and name not in interfaces, "duplicate or missing input interface"
        )
        _require(
            name in ("lmt", "hwpr") or re.fullmatch(r"toltec(?:[0-9]|1[0-2])", name),
            "unsupported input interface: " + name,
        )
        path = _path(item.get("filepath"), launch_dir)
        interfaces[name] = path
        item["filepath"] = str(path)
    _require("lmt" in interfaces, "one telescope input required")
    manifests = [item for item in calibration if item.get("type") == "array_prop_table"]
    _require(len(manifests) == 1, "one canonical matched APT required")
    for item in calibration:
        if "filepath" in item:
            item["filepath"] = str(_path(item["filepath"], launch_dir))
    manifest = Path(manifests[0]["filepath"])
    fitreportdir = _path(
        effective.get("kids", {}).get("solver", {}).get("fitreportdir"), launch_dir
    )
    _require(fitreportdir.is_dir(), "kids.solver.fitreportdir must be a directory")
    effective["kids"]["solver"]["fitreportdir"] = str(fitreportdir)
    bindings = _Bindings()
    apt, sources = _bundle(manifest, scope, bindings)
    source_index = {}
    for source in sources:
        key = (source["role"], int(source["nw"]))
        _require(
            key not in source_index and key[0] in ("raw", "kmp"),
            "duplicate or unsupported APT source",
        )
        source_index[key] = source
    networks = sorted(int(name[6:]) for name in interfaces if name.startswith("toltec"))
    _require(bool(networks), "no raw networks supplied")
    _require(
        set(source_index) == {(role, nw) for role in ("raw", "kmp") for nw in networks},
        "APT source population differs from configured raw networks",
    )
    _require(
        {int(row["nw"]) for row in apt} == set(networks), "APT detector network population differs"
    )
    telescope = bindings.bind(interfaces["lmt"], "POINT telescope")
    with netCDF4.Dataset(interfaces["lmt"]) as nc:
        _require(
            tuple(_integer(nc, "Header.Dcs." + key) for key in ("ObsNum", "SubObsNum", "ScanNum"))
            == scope,
            "telescope acquisition identity differs",
        )
        _require(
            _text(nc, "Header.Dcs.ObsGoal") == "Pointing"
            or _text(nc, "Header.Dcs.ObsPgm") == "Pointing",
            "telescope is not a POINT acquisition",
        )
        pointing_metadata = _pointing_metadata(nc, obs, calibration, telescope)
    for name, path in interfaces.items():
        if name == "hwpr":
            bindings.bind(path, "listed HWPR input; inert with polarimetry disabled")
    timing, inventory, actual_arrays = [], [], set()
    for nw in networks:
        path = interfaces["toltec" + str(nw)]
        raw = bindings.bind(path, "raw detector input")
        with netCDF4.Dataset(path) as nc:
            _require(
                tuple(
                    _integer(nc, "Header.Toltec." + key)
                    for key in ("ObsNum", "SubObsNum", "ScanNum", "RoachIndex")
                )
                == (*scope, nw),
                "raw acquisition or network identity differs",
            )
            _require(
                "Data.Toltec.Is" in nc.variables and "Data.Toltec.Qs" in nc.variables,
                "paired I/Q data missing",
            )
            shape = nc["Data.Toltec.Is"].shape
            _require(
                len(shape) == 2 and min(shape) > 0 and nc["Data.Toltec.Qs"].shape == shape,
                "paired I/Q channel shape differs",
            )
            rows, channels = shape
            tune_scope = tuple(
                _integer(nc, "Header.Toltec.TargSweep" + key)
                for key in ("ObsNum", "SubObsNum", "ScanNum")
            )
        pattern = (
            f"toltec{nw}_{tune_scope[0]:06d}_{tune_scope[1]:03d}_{tune_scope[2]:04d}_*_tune.txt"
        )
        candidates = sorted(fitreportdir.glob(pattern))
        _require(len(candidates) == 1, "missing or ambiguous KMP association: " + pattern)
        tune = bindings.bind(candidates[0], "associated KMP tune")
        for role, bound, identity in (("raw", raw, scope), ("kmp", tune, tune_scope)):
            source = source_index[(role, nw)]
            _require(
                tuple(int(source[key]) for key in ("obsnum", "subobsnum", "scannum")) == identity
                and source["interface"] == "toltec" + str(nw)
                and int(source["channel_count"]) == channels
                and int(source["byte_count"]) == bindings.records[bound["path"]]["bytes"]
                and source["content_sha256"] == "sha256:" + bound["sha256"],
                role + " is not the exact canonical APT source",
            )
        members = sorted(
            (row for row in apt if int(row["nw"]) == nw), key=lambda row: int(row["kids_tone"])
        )
        array_index = 0 if nw < 7 else 1 if nw < 11 else 2
        _require(
            [int(row["kids_tone"]) for row in members] == list(range(channels))
            and all(int(row["array"]) == array_index for row in members),
            "APT channel or array correspondence differs",
        )
        array = ARRAYS[array_index]
        actual_arrays.add(array)
        _require(array in rtc_profiles, "missing RTC profile: " + array)
        profile = bindings.bind(
            _path(str(rtc_profiles[array]), launch_dir), "RTC profile: " + array
        )
        timing.append(dict(network=nw, **raw))
        inventory.append(
            dict(
                network=nw,
                array=array,
                raw=raw,
                tune=tune,
                rows=rows,
                channels=channels,
                profile=profile,
            )
        )
    _require(
        actual_arrays == set(ARRAYS), "POINT trial requires actual inputs for all three arrays"
    )
    for item in calibration:
        if "filepath" in item:
            bindings.bind(Path(item["filepath"]), "configured calibration input")
    # Native stages consume these exact files; ordinary output publication belongs
    # to the caller, not to the historical runtime.output_dir in this snapshot.
    out.mkdir(parents=True)
    ep = out / "effective-point.yaml"
    with ep.open("x") as stream:
        yaml.safe_dump(effective, stream, sort_keys=False)
    effective_binding = bindings.bind(ep, "effective POINT configuration")
    manifest_binding = bindings.bind(manifest, "selected canonical APT")
    ast = out / "ast-source-binding.json"
    _write(
        ast,
        dict(
            schema="citlali-ast-motion-source-binding-v1",
            authority_policy_id="wp7-ast-scan-motion-v2",
            observation=obs,
            subobservation=sub,
            scan=scan,
            telescope=telescope,
        ),
    )
    ast_binding = bindings.bind(ast, "AST source binding")
    processing = out / "processing-request.json"
    _write(
        processing,
        dict(
            schema="citlali-native-processing-request-v1",
            observation=obs,
            subobservation=sub,
            scan=scan,
            realized=False,
            method="preserved-gap-grid-forced-10s-context32",
            effective_config_sha256=effective_binding["sha256"],
            timing_inputs=timing,
            manifest=manifest_binding,
            telescope=telescope,
        ),
    )
    processing_binding = bindings.bind(processing, "native processing request")
    requests = []
    for row in inventory:
        path = out / f"network{row['network']}-input.json"
        request = dict(
            schema="rtc-multidetector-experiment-v1",
            development_reference=DEVELOPMENT_REFERENCE,
            mode="POINT",
            observation=obs,
            subobservation=sub,
            scan=scan,
            network=row["network"],
            raw=row["raw"],
            tune=row["tune"],
            manifest=manifest_binding,
            telescope=telescope,
            effective_config=effective_binding,
            ast_acceptance=ast_binding,
            native_source="exact-raw-kids-v1",
            terminal_request="rtc-only",
            prepare_output_grid=True,
            speed_support_comparison="both",
            source_protection_authority=NO_MASK_AUTHORITY,
            reassessment_selection=dict(
                intent="retain-development-candidate",
                authority="owner-configurable-POINT-2026-10-28",
                purpose="configured local POINT forward execution",
                positive_rationale="explicit current acquisition; existing native treatment; no feedback or spatial mask",
            ),
            decision_apply=dict(
                policy="existing-authorities-with-unavailable-isolated-admission-v1",
                timing_inputs=timing,
                processing_request=processing_binding,
            ),
            filter_plan=row["profile"],
            detectors=[dict(channel=i, filter="lowpass") for i in range(row["channels"])],
        )
        disturbance = (
            effective.get("successor", {}).get("rtc", {}).get("disturbance_exclusion", "disabled")
        )
        if disturbance != "disabled":
            request["disturbance_exclusion"] = disturbance
        _write(path, request)
        requests.append(
            dict(network=row["network"], input=bindings.bind(path, "native network request"))
        )
    observation_request = out / "observation.json"
    _write(
        observation_request,
        dict(
            schema="citlali-observation-networks-v1",
            development_reference=DEVELOPMENT_REFERENCE,
            mode="POINT",
            observation=obs,
            subobservation=sub,
            scan=scan,
            networks=requests,
        ),
    )
    bindings.bind(observation_request, "native observation request")
    return PreparedPoint(
        observation=obs,
        subobservation=sub,
        scan=scan,
        arrays=list(ARRAYS),
        networks=networks,
        observation_request=observation_request,
        effective_config=ep,
        bindings=list(bindings.records.values()),
        native_cells=sum(row["rows"] * row["channels"] for row in inventory),
        pointing_metadata=pointing_metadata,
    )
