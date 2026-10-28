"""Small transport/identity fixtures; native APT semantics have separate gates."""
import copy
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import netCDF4
import numpy as np
import pytest
import yaml

import configured_point_inputs as inputs


OBS = 240123


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ecsv(path, role, schema, rows, fields=None):
    metadata = dict(schema_version=schema, component_role=role, product_kind="observation-matched",
                    occurrence="fixture-occurrence", event_reference="fixture-event",
                    observation=dict(obsnum=OBS, subobsnum=0, scannum=2),
                    semantic_sha256="sha256:" + "1" * 64,
                    envelope_sha256="sha256:" + "2" * 64, row_count=len(rows))
    header = yaml.safe_dump(dict(meta=dict(canonical_apt_v2=metadata), delimiter=","), sort_keys=False)
    data = io.StringIO()
    writer = csv.DictWriter(data, fieldnames=fields or list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path.write_text("# %ECSV 1.0\n# ---\n" + "".join("# " + line + "\n" for line in header.splitlines()) + data.getvalue())
    return metadata


def scalar(nc, name, value):
    nc.createVariable(name, "i4", ())[...] = value


def text(nc, name, value):
    dimension = "string_" + name
    nc.createDimension(dimension, len(value))
    nc.createVariable(name, "S1", (dimension,))[:] = np.asarray(list(value), dtype="S1")


def fixture(tmp_path, networks=(0, 7, 11)):
    launch = tmp_path / "project" / "reduced"
    launch.mkdir(parents=True)
    rawdir, kmpdir, caldir = [launch.parent / name for name in ("raw", "kmp", "cal")]
    for path in (rawdir, kmpdir, caldir):
        path.mkdir()
    telescope = launch / "recomputed.nc"
    with netCDF4.Dataset(telescope, "w") as nc:
        for key, value in zip(("ObsNum", "SubObsNum", "ScanNum"), (OBS, 0, 2)):
            scalar(nc, "Header.Dcs." + key, value)
        text(nc, "Header.Dcs.ObsGoal", "Pointing")
        text(nc, "Header.Dcs.ObsPgm", "Lissajous")
    apt, sources, data = [], [], [dict(filepath="recomputed.nc", meta=dict(interface="lmt"))]
    profiles = {}
    for nw in networks:
        array = 0 if nw < 7 else 1 if nw < 11 else 2
        path = rawdir / f"toltec{nw}.nc"
        with netCDF4.Dataset(path, "w") as nc:
            for key, value in zip(("ObsNum", "SubObsNum", "ScanNum", "RoachIndex"), (OBS, 0, 2, nw)):
                scalar(nc, "Header.Toltec." + key, value)
            for key, value in zip(("ObsNum", "SubObsNum", "ScanNum"), (OBS - 1, 0, 1)):
                scalar(nc, "Header.Toltec.TargSweep" + key, value)
            nc.createDimension("rows", 4)
            nc.createDimension("channels", 2)
            for key in ("Is", "Qs"):
                nc.createVariable("Data.Toltec." + key, "f8", ("rows", "channels"))[:] = 1.0
        tune = kmpdir / f"toltec{nw}_{OBS-1:06d}_000_0001_fixture_tune.txt"
        tune.write_text("flag\n0\n0\n")
        for role, source_path, acquisition in (("raw", path, (OBS, 0, 2)), ("kmp", tune, (OBS - 1, 0, 1))):
            sources.append(dict(role=role, nw=nw, interface=f"toltec{nw}", channel_count=2,
                                obsnum=acquisition[0], subobsnum=acquisition[1], scannum=acquisition[2],
                                content_sha256="sha256:" + sha(source_path), byte_count=source_path.stat().st_size))
        apt.extend(dict(nw=nw, kids_tone=i, array=array) for i in range(2))
        data.append(dict(filepath=f"../raw/{path.name}", meta=dict(interface=f"toltec{nw}")))
        name = inputs.ARRAYS[array]
        profile = caldir / (name + ".json")
        profile.write_text(json.dumps(dict(array=name)))
        profiles[name] = profile
    descriptors = []
    for role, schema, rows in (("apt", "citlali-canonical-matched-apt-v2", apt),
                               ("sources", "citlali-canonical-apt-sources-v2", sources)):
        path = caldir / (role + ".ecsv")
        meta = ecsv(path, role, schema, rows)
        descriptors.append(dict(role=role, schema=schema, relative_path=path.name,
                                semantic_sha256=meta["semantic_sha256"], envelope_sha256=meta["envelope_sha256"],
                                transport_sha256="sha256:" + sha(path), byte_count=path.stat().st_size,
                                row_count=len(rows)))
    manifest = caldir / "manifest.ecsv"
    meta = ecsv(manifest, "manifest", "citlali-canonical-apt-root-manifest-v2", descriptors)
    manifest.with_suffix(".ecsv.sha256").write_text(
        "citlali-canonical-apt-root-receipt-v2\n"
        "scope=citlali-canonical-apt-root-manifest-byte-sha256-v2\n"
        f"envelope_sha256={meta['envelope_sha256']}\n"
        f"byte_sha256=sha256:{sha(manifest)}\nbyte_count={manifest.stat().st_size}\n")
    config = dict(runtime=dict(reduction_type="pointing", output_dir="preserved-output"),
                  kids=dict(solver=dict(fitreportdir="../kmp")),
                  inputs=[dict(meta=dict(name=f"{OBS}_0_2"), data_items=data,
                               cal_items=[dict(type="array_prop_table", filepath="../cal/manifest.ecsv")])])
    return config, launch, profiles


def run(config, launch, profiles, out=None):
    return inputs.prepare(config, out or launch / "new-inputs", launch_dir=launch, rtc_profiles=profiles)


def test_selected_disturbance_policy_reaches_every_native_network(tmp_path):
    from configured_point_config import DISTURBANCE_POLICY
    config, launch, profiles = fixture(tmp_path)
    config['successor'] = {'rtc': {'disturbance_exclusion': DISTURBANCE_POLICY}}
    result = run(config, launch, profiles)
    observation = json.loads(result['observation_request'].read_text())
    for entry in observation['networks']:
        request = json.loads(Path(entry['input']['path']).read_text())
        assert request['disturbance_exclusion'] == DISTURBANCE_POLICY
        assert 'point_transfer_runtime_profile' not in request


def test_nonhistorical_all_array_acquisition_and_separate_kmp_directory(tmp_path):
    config, launch, profiles = fixture(tmp_path)
    original = copy.deepcopy(config)
    result = run(config, launch, profiles)
    assert config == original
    assert result["observation"] == OBS
    assert result["networks"] == [0, 7, 11]
    assert result["arrays"] == list(inputs.ARRAYS)
    assert result["native_cells"] == 24
    effective = yaml.safe_load(result["effective_config"].read_text())
    assert effective["kids"]["solver"]["fitreportdir"] == str((launch.parent / "kmp").resolve())
    assert effective["runtime"]["output_dir"] == "preserved-output"
    observation = json.loads(result["observation_request"].read_text())
    assert observation["development_reference"] == "configurable-point-v1"
    for entry in observation["networks"]:
        request = json.loads(Path(entry["input"]["path"]).read_text())
        assert request["source_protection_authority"] == inputs.NO_MASK_AUTHORITY
        assert Path(request["tune"]["path"]).parent == launch.parent / "kmp"
        assert len(request["decision_apply"]["timing_inputs"]) == 3
        array = {0: "a1100", 7: "a1400", 11: "a2000"}[entry["network"]]
        assert request["filter_plan"]["path"] == str(profiles[array].resolve())


def test_hashes_shared_files_only_once(tmp_path, monkeypatch):
    config, launch, profiles = fixture(tmp_path)
    counts = Counter()
    original = inputs._digest

    def count(path):
        counts[path] += 1
        return original(path)

    monkeypatch.setattr(inputs, "_digest", count)
    result = run(config, launch, profiles)
    assert all(count == 1 for count in counts.values())
    assert len(result["bindings"]) == len(counts)


def test_shuffled_data_items_preserve_identity_and_sort_native_networks(tmp_path):
    config, launch, profiles = fixture(tmp_path, networks=(0, 1, 7, 11, 12))
    data = config["inputs"][0]["data_items"]
    config["inputs"][0]["data_items"] = [data[i] for i in (5, 2, 0, 4, 1, 3)]
    supplied_order = [item["meta"]["interface"] for item in config["inputs"][0]["data_items"]]
    result = run(config, launch, profiles)
    request = json.loads(result["observation_request"].read_text())
    assert [entry["network"] for entry in request["networks"]] == [0, 1, 7, 11, 12]
    effective = yaml.safe_load(result["effective_config"].read_text())
    assert [item["meta"]["interface"] for item in effective["inputs"][0]["data_items"]] == supplied_order
    for entry in request["networks"]:
        network = json.loads(Path(entry["input"]["path"]).read_text())
        assert network["network"] == entry["network"]
        assert Path(network["raw"]["path"]).name == f"toltec{entry['network']}.nc"
        assert [row["network"] for row in network["decision_apply"]["timing_inputs"]] == [0, 1, 7, 11, 12]


def test_foreign_telescope_rejected_before_output(tmp_path):
    config, launch, profiles = fixture(tmp_path)
    with netCDF4.Dataset(launch / "recomputed.nc", "a") as nc:
        nc["Header.Dcs.ObsNum"][...] = OBS + 1
    with pytest.raises(ValueError, match="telescope acquisition identity"):
        run(config, launch, profiles)
    assert not (launch / "new-inputs").exists()


def test_foreign_raw_network_rejected(tmp_path):
    config, launch, profiles = fixture(tmp_path)
    with netCDF4.Dataset(launch.parent / "raw/toltec7.nc", "a") as nc:
        nc["Header.Toltec.RoachIndex"][...] = 8
    with pytest.raises(ValueError, match="raw acquisition or network"):
        run(config, launch, profiles)


@pytest.mark.parametrize("role", ["raw", "kmp"])
def test_changed_acquisition_bytes_rejected(tmp_path, role):
    config, launch, profiles = fixture(tmp_path)
    if role == "raw":
        with netCDF4.Dataset(launch.parent / "raw/toltec0.nc", "a") as nc:
            nc["Data.Toltec.Is"][0, 0] = 7
    else:
        tune = next((launch.parent / "kmp").glob("toltec0_*"))
        tune.write_text(tune.read_text() + "changed\n")
    with pytest.raises(ValueError, match=role + " is not the exact canonical"):
        run(config, launch, profiles)


def test_duplicate_network_rejected(tmp_path):
    config, launch, profiles = fixture(tmp_path)
    config["inputs"][0]["data_items"].append(copy.deepcopy(config["inputs"][0]["data_items"][1]))
    with pytest.raises(ValueError, match="duplicate or missing input interface"):
        run(config, launch, profiles)


def test_missing_array_rejected(tmp_path):
    config, launch, profiles = fixture(tmp_path, networks=(0, 7))
    with pytest.raises(ValueError, match="all three arrays"):
        run(config, launch, profiles)


def test_ambiguous_tune_rejected(tmp_path):
    config, launch, profiles = fixture(tmp_path)
    tune = next((launch.parent / "kmp").glob("toltec0_*"))
    tune.with_name(tune.name.replace("fixture", "duplicate")).write_bytes(tune.read_bytes())
    with pytest.raises(ValueError, match="ambiguous KMP association"):
        run(config, launch, profiles)


def test_changed_manifest_receipt_rejected(tmp_path):
    config, launch, profiles = fixture(tmp_path)
    receipt = launch.parent / "cal/manifest.ecsv.sha256"
    receipt.write_text(receipt.read_text().replace("byte_count=", "byte_count=1"))
    with pytest.raises(ValueError, match="root receipt differs"):
        run(config, launch, profiles)


def test_changed_component_rejected(tmp_path):
    config, launch, profiles = fixture(tmp_path)
    component = launch.parent / "cal/apt.ecsv"
    component.write_text(component.read_text() + "0,2,0\n")
    with pytest.raises(ValueError, match="component transport differs"):
        run(config, launch, profiles)


def test_existing_destination_and_input_sources_preserved(tmp_path):
    config, launch, profiles = fixture(tmp_path)
    output = launch / "new-inputs"
    output.mkdir()
    marker = output / "existing"
    marker.write_text("preserve")
    with pytest.raises(ValueError, match="fresh inputs directory"):
        run(config, launch, profiles)
    assert marker.read_text() == "preserve"


@pytest.mark.parametrize("name", ["240123_1_2", "240123_0_1", "0240123_0_2"])
def test_unsupported_or_noncanonical_acquisition_rejected(tmp_path, name):
    config, launch, profiles = fixture(tmp_path)
    config["inputs"][0]["meta"]["name"] = name
    with pytest.raises(ValueError, match="acquisition name"):
        run(config, launch, profiles)
