"""Configuration behavior at the ordinary TolTECA-to-native POINT boundary."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from configured_point_config import ARRAYS, REPO, load_request


@pytest.fixture
def config():
    # Representative prepared low-level layout, including scientifically
    # different active legacy controls that the explicit migration retires.
    config = {
        "runtime": {"reduction_type": "pointing", "output_dir": "reduced/trial", "n_threads": 6,
                    "verbose": True, "parallel_policy": "omp", "use_subdir": True},
        "kids": {"fitter": {"modelspec": "gainlintrend"}, "solver": {"fitreportdir": "../data"}},
        "interface_sync_offset": [{f"toltec{n}": 0} for n in range(13)],
        "inputs": [{"meta": {"name": "152389_0_2"}, "data_items": [
            {"meta": {"interface": "lmt"}, "filepath": "reduced/recomputed.nc"},
            *({"meta": {"interface": f"toltec{n}"}, "filepath": f"../data/toltec{n}.nc"}
              for n in (0, 1, 5, 7, 9, 11, 12)),
        ], "cal_items": [
            {"type": "array_prop_table", "filepath": "../apts/manifest.ecsv"},
            {"type": "astrometry", "pointing_offsets": [
                {"axes_name": "az", "value_arcsec": [0]},
                {"axes_name": "alt", "value_arcsec": [0]},
            ]},
            {"type": "photometry", "beammap_source": {"fluxes": [{"array_name": "a1100", "value_mJy": 2000}]}},
        ]}],
        "timestream": {"type": "xs", "raw_time_chunk": {"downsample": {"factor": 2},
                         "despike": {"source_protection": {"enabled": True}}},
                       "fruit_loops": {"enabled": True, "max_iters": 10},
                       "learning": {"enabled": True},
                       "processed_time_chunk": {"clean": {"adaptive_selector": {"enabled": True}}}},
        "mapmaking": {"method": "jinc", "pixel_size_arcsec": 2.0, "coverage_cut": .5},
        "noise_maps": {"enabled": True, "products": {"apply_empirical_weights": True}},
        "post_processing": {"source_fitting": {"model": "gaussian"}, "source_finding": {"enabled": True}},
        "coadd": {"enabled": True}, "pointing": {"source_strategy": "legacy"},
        "source": {"map_regime": "old"}, "wiener_filter": {"lowpass_only": False},
        "beammap": {"iter_max": 5},
    }
    overlay = yaml.safe_load((REPO / "data/configured_point_v1.yaml").read_text())
    config["successor"] = overlay["successor"]
    config["runtime"].update(overlay["runtime"])
    return config


def test_real_overlay_resolves_job_relative_inputs_and_preserves_calibration(config, tmp_path):
    launch = tmp_path / "project" / "pointings-152389"
    original = deepcopy(config)
    request = load_request(config, launch)
    effective = request.effective_config
    assert config == original
    assert request.output_dir == launch / "reduced/trial"
    assert request.arrays == ARRAYS and request.workers == 6 and request.verbose is False
    assert request.fit_diagnostics is True and request.retain_intermediates is False
    assert effective["kids"]["solver"]["fitreportdir"] == str(launch.parent / "data")
    assert effective["kids"]["solver"]["parallel_policy"] == "seq" and effective["kids"]["solver"]["extra_output"] is False
    assert effective["inputs"][0]["data_items"][0]["filepath"] == str(launch / "reduced/recomputed.nc")
    assert effective["inputs"][0]["data_items"][1]["filepath"] == str(launch.parent / "data/toltec0.nc")
    assert effective["inputs"][0]["cal_items"][0]["filepath"] == str(launch.parent / "apts/manifest.ecsv")
    assert effective["inputs"][0]["cal_items"][1:] == original["inputs"][0]["cal_items"][1:]
    assert effective["interface_sync_offset"] == original["interface_sync_offset"]
    with pytest.raises(FrozenInstanceError):
        request.workers = 1


def test_independent_array_ranks_and_map_choices_reach_native_copy(config, tmp_path):
    config["successor"]["cleaning"]["ranks"] = {"a1100": 3, "a1400": 5, "a2000": 7}
    config["successor"]["map"].update(pixel_size_arcsec=1.25, coverage_cut=.25)
    request = load_request(config, tmp_path)
    native = request.effective_config["timestream"]["processed_time_chunk"]["clean"]
    assert native["grouping"] == ["nw"]
    assert native["standard_pca"]["n_eig_to_cut"] == {"a1100": [3], "a1400": [5], "a2000": [7]}
    assert request.cleaning_method == "observed-als"
    assert request.pixel_size_arcsec == 1.25 and request.coverage_cut == .25
    assert request.effective_config["mapmaking"]["pixel_size_arcsec"] == 1.25
    assert request.effective_config["mapmaking"]["coverage_cut"] == .25
    assert all(path.is_file() for path in request.rtc_profiles.values())
    assert [__import__("json").loads(request.rtc_profiles[a].read_text())["factor"] for a in ARRAYS] == [1, 1, 2]


def test_explicit_profile_replaces_active_legacy_controls_with_recorded_dispositions(config, tmp_path):
    request = load_request(config, tmp_path)
    effective = request.effective_config
    assert effective["timestream"]["fruit_loops"] == {"enabled": False}
    assert "learning" not in effective["timestream"]
    assert effective["timestream"]["raw_time_chunk"] == {
        "IIR_filter": {"enabled": False}, "filter": {"enabled": True, "n_terms": 32}}
    assert effective["timestream"]["chunking"] == {"chunk_mode": "duration", "value": 10, "force_chunking": True}
    assert effective["runtime"]["interp_over_gaps"] is True
    assert "adaptive_selector" not in effective["timestream"]["processed_time_chunk"]["clean"]
    assert effective["mapmaking"]["method"] == "containing-pixel"
    assert effective["noise_maps"] == {"enabled": False} and effective["coadd"] == {"enabled": False}
    assert "post_processing" not in effective and "wiener_filter" not in effective
    disposed = {r["path"]: r for r in request.migration_dispositions}
    assert set(disposed) == {"timestream", "mapmaking", "noise_maps", "post_processing", "coadd",
                             "pointing", "source", "wiener_filter", "beammap", "kids.solver",
                             "inputs[0].cal_items[2]"}
    assert all(r["disposition"] == "replaced" and r["replacement"] for path, r in disposed.items()
               if not path.startswith("inputs["))
    assert disposed["inputs[0].cal_items[2]"]["disposition"] == "metadata-only"


def test_profile_replacement_requires_explicit_migration(config, tmp_path):
    del config["successor"]["migration"]
    with pytest.raises(ValueError, match="migration"):
        load_request(config, tmp_path)


@pytest.mark.parametrize("section", [None, "cleaning", "rtc", "map", "feedback", "products"])
def test_unknown_successor_controls_fail_before_processing(config, tmp_path, section):
    node = config["successor"] if section is None else config["successor"][section]
    node["invented_option"] = True
    with pytest.raises(ValueError, match="unsupported successor"):
        load_request(config, tmp_path)


@pytest.mark.parametrize("path,value", [
    (("cleaning", "method"), "pairwise-covariance"),
    (("cleaning", "method"), "standard-pca"),
    (("rtc", "profile"), "automatic"),
    (("map", "method"), "jinc"),
    (("map", "frame"), "FK5"),
    (("feedback", "enabled"), True),
    (("products", "fits"), False),
])
def test_unimplemented_scientific_requests_are_not_silently_substituted(config, tmp_path, path, value):
    config["successor"][path[0]][path[1]] = value
    with pytest.raises(ValueError):
        load_request(config, tmp_path)


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "5", None, float("inf"), float("nan")])
def test_invalid_rank_is_not_an_integer_request(config, tmp_path, value):
    config["successor"]["cleaning"]["ranks"]["a1400"] = value
    with pytest.raises(ValueError, match="ranks.a1400"):
        load_request(config, tmp_path)


@pytest.mark.parametrize("key,value", [
    ("pixel_size_arcsec", True), ("pixel_size_arcsec", 0), ("pixel_size_arcsec", -1),
    ("pixel_size_arcsec", float("nan")), ("pixel_size_arcsec", float("inf")),
    ("coverage_cut", False), ("coverage_cut", -0.1), ("coverage_cut", 1.1),
    ("coverage_cut", float("nan")), ("coverage_cut", "0.1"),
])
def test_map_controls_must_be_finite_and_in_range(config, tmp_path, key, value):
    config["successor"]["map"][key] = value
    with pytest.raises(ValueError, match=key):
        load_request(config, tmp_path)


@pytest.mark.parametrize("value", [True, 0, 13, 1.5, "6"])
def test_worker_count_must_be_a_supported_integer(config, tmp_path, value):
    config["runtime"]["n_threads"] = value
    with pytest.raises(ValueError, match="n_threads"):
        load_request(config, tmp_path)


@pytest.mark.parametrize("key,value", [("verbose", "false"), ("verbose", 0)])
def test_verbose_requires_boolean(config, tmp_path, key, value):
    config["runtime"][key] = value
    with pytest.raises(ValueError, match="boolean"):
        load_request(config, tmp_path)


@pytest.mark.parametrize("arrays", [["a1100"], ["a1100", "a1400", "a1400"], ["a1100", "a1400", True]])
def test_all_arrays_are_required(config, tmp_path, arrays):
    config["successor"]["arrays"] = arrays
    with pytest.raises(ValueError, match="arrays"):
        load_request(config, tmp_path)


def test_quiet_and_verbose_change_only_reporting_request(config, tmp_path):
    quiet = load_request(config, tmp_path)
    config["runtime"]["verbose"] = True
    loud = load_request(config, tmp_path)
    expected = deepcopy(quiet.effective_config)
    expected["runtime"]["verbose"] = True
    assert loud.effective_config == expected
    assert (loud.ranks, loud.rtc_profiles, loud.pixel_size_arcsec, loud.coverage_cut) == (
        quiet.ranks, quiet.rtc_profiles, quiet.pixel_size_arcsec, quiet.coverage_cut)


def test_products_are_separately_optional_and_independent_of_verbosity(config, tmp_path):
    config["successor"]["products"].update(fit_diagnostics=False, retain_intermediates=True)
    request = load_request(config, tmp_path)
    assert request.fit_diagnostics is False and request.retain_intermediates is True and request.verbose is False


def test_absolute_input_paths_and_different_observation_are_not_rewritten(config, tmp_path):
    config["inputs"][0]["meta"]["name"] = "999999_0_2"
    path = str(tmp_path / "external" / "telescope.nc")
    config["inputs"][0]["data_items"][0]["filepath"] = path
    request = load_request(config, tmp_path / "working")
    assert request.effective_config["inputs"][0]["meta"]["name"] == "999999_0_2"
    assert request.effective_config["inputs"][0]["data_items"][0]["filepath"] == path


def test_unimplemented_acquisition_shape_is_rejected(config, tmp_path):
    config["inputs"][0]["meta"]["name"] = "152389_0_1"
    with pytest.raises(ValueError, match="subobservation 0, scan 2"):
        load_request(config, tmp_path)


@pytest.mark.parametrize("entries", [None, [{"toltec0": True}], [{"toltec0": 0.01}],
                                    [{"toltec0": float("nan")}], [{"toltec0": 0}, {"toltec0": 0}],
                                    [{"toltec13": 0}], [{"toltec01": 0}], []])
def test_invalid_missing_or_nonzero_native_timing_fails_before_processing(config, tmp_path, entries):
    config["interface_sync_offset"] = entries
    with pytest.raises(ValueError, match="interface_sync_offset"):
        load_request(config, tmp_path)


@pytest.mark.parametrize("name", ["0_0_2", "1000000_0_2"])
def test_observation_must_fit_existing_tolteca_product_discovery(config, tmp_path, name):
    config["inputs"][0]["meta"]["name"] = name
    with pytest.raises(ValueError, match="six-digit product discovery"):
        load_request(config, tmp_path)


def astrometry(config):
    return next(item for item in config["inputs"][0]["cal_items"] if item["type"] == "astrometry")


def test_actual_default_calibration_sentinel_is_adapted_without_mutating_request(config, tmp_path):
    defaults = yaml.safe_load((REPO / "data/config.yaml").read_text())["inputs"][0]["cal_items"]
    original_astrometry = deepcopy(next(item for item in defaults if item["type"] == "astrometry"))
    original_correction = deepcopy(next(item for item in defaults if item["type"] == "flxscale_correction"))
    assert original_astrometry["pointing_offsets"][-1] == {"modified_julian_date": [0.0]}
    config["inputs"][0]["cal_items"][1] = deepcopy(original_astrometry)
    config["inputs"][0]["cal_items"].append(deepcopy(original_correction))
    before = deepcopy(config)
    request = load_request(config, tmp_path)
    effective_offsets = astrometry(request.effective_config)["pointing_offsets"]
    assert effective_offsets == original_astrometry["pointing_offsets"][:-1]
    assert config == before
    assert any(row["disposition"] == "normalized" and row["original"] == [0.0]
               for row in request.migration_dispositions)
    assert any(row["disposition"] == "identity-only" for row in request.migration_dispositions)


@pytest.mark.parametrize("epochs", [[0], [0.0], [0, 0.0]])
def test_zero_epoch_sentinel_preserves_exact_single_axis_values(config, tmp_path, epochs):
    offsets = astrometry(config)["pointing_offsets"]
    offsets.append({"modified_julian_date": epochs})
    original_axes = deepcopy(offsets[:-1])
    request = load_request(config, tmp_path)
    assert astrometry(request.effective_config)["pointing_offsets"] == original_axes


@pytest.mark.parametrize("count", [1, 2])
def test_actual_epochs_and_offset_values_are_preserved_exactly(config, tmp_path, count):
    offsets = astrometry(config)["pointing_offsets"]
    for entry in offsets:
        entry["value_arcsec"] = [0.0] * count
    offsets.append({"modified_julian_date": [61090.125, 61090.875]})
    before = deepcopy(offsets)
    request = load_request(config, tmp_path)
    assert astrometry(request.effective_config)["pointing_offsets"] == before
    assert not any(row["disposition"] == "normalized" for row in request.migration_dispositions)


@pytest.mark.parametrize("axis_lengths", [(2, 2), (1, 2), (2, 1)])
def test_sentinel_never_invents_epochs_for_two_value_or_mixed_axes(config, tmp_path, axis_lengths):
    offsets = astrometry(config)["pointing_offsets"]
    for entry, length in zip(offsets, axis_lengths):
        entry["value_arcsec"] = [0.0] * length
    offsets.append({"modified_julian_date": [0.0]})
    with pytest.raises(ValueError, match="pointing_offsets"):
        load_request(config, tmp_path)


@pytest.mark.parametrize("epochs", [[False], [float("nan")], [float("inf")], ["0"], [],
                                    [0, 0, 0], [61090.5], [-1, 61091], [0, 61091], [61091, 61090]])
def test_only_recognized_zero_sentinel_or_real_epoch_pair_is_admitted(config, tmp_path, epochs):
    astrometry(config)["pointing_offsets"].append({"modified_julian_date": epochs})
    with pytest.raises(ValueError, match="modified_julian_date"):
        load_request(config, tmp_path)


@pytest.mark.parametrize("epochs", [[-1], [-1, 0], [-2, -1], [0, -1]])
def test_nonpositive_legacy_sentinels_never_become_actual_epoch_pairs(config, tmp_path, epochs):
    offsets = astrometry(config)["pointing_offsets"]
    original_axes = deepcopy(offsets)
    offsets.append({"modified_julian_date": epochs})
    request = load_request(config, tmp_path)
    assert astrometry(request.effective_config)["pointing_offsets"] == original_axes
    assert any(row["disposition"] == "normalized" and row["original"] == epochs
               for row in request.migration_dispositions)


def test_duplicate_epoch_records_are_not_silently_removed(config, tmp_path):
    astrometry(config)["pointing_offsets"].extend([
        {"modified_julian_date": [0.0]}, {"modified_julian_date": [61090, 61091]}])
    with pytest.raises(ValueError, match="duplicated"):
        load_request(config, tmp_path)


@pytest.mark.parametrize("value", [0, -1, 2, .5, True, False, "1", None, float("nan"), float("inf"), [1]])
def test_nonidentity_or_invalid_fluxscale_correction_fails_before_processing(config, tmp_path, value):
    config["inputs"][0]["cal_items"].append({"type": "flxscale_correction", "value": value})
    with pytest.raises(ValueError, match="flxscale_correction"):
        load_request(config, tmp_path)


@pytest.mark.parametrize("value", [1, 1.0])
def test_identity_fluxscale_correction_is_preserved_and_dispositioned(config, tmp_path, value):
    item = {"type": "flxscale_correction", "value": value}
    config["inputs"][0]["cal_items"].append(item)
    request = load_request(config, tmp_path)
    assert request.effective_config["inputs"][0]["cal_items"][-1] == item
    assert any(row["disposition"] == "identity-only" for row in request.migration_dispositions)


def test_duplicate_fluxscale_corrections_rejected_even_if_both_identity(config, tmp_path):
    config["inputs"][0]["cal_items"].extend([
        {"type": "flxscale_correction", "value": 1}, {"type": "flxscale_correction", "value": 1}])
    with pytest.raises(ValueError, match="duplicate.*flxscale_correction"):
        load_request(config, tmp_path)


def test_fluxscale_correction_requires_value(config, tmp_path):
    config["inputs"][0]["cal_items"].append({"type": "flxscale_correction"})
    with pytest.raises(ValueError, match="missing.*value"):
        load_request(config, tmp_path)


def test_unknown_calibration_type_is_not_silently_ignored(config, tmp_path):
    config["inputs"][0]["cal_items"].append({"type": "new_correction", "value": 1.2})
    with pytest.raises(ValueError, match="unsupported.*cal_items type"):
        load_request(config, tmp_path)
