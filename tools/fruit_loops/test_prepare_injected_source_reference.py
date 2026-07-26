from prepare_injected_source_reference import prepare_reference_config


def source_config() -> dict:
    return {
        "runtime": {
            "reduction_type": "pointing",
            "output_dir": "/old/reduced/",
        },
        "inputs": [{"path": "/data/obs133410.nc"}],
        "timestream": {
            "raw_time_chunk": {"kernel": {"enabled": False}},
            "fruit_loops": {
                "enabled": True,
                "path": "/old/maps/",
                "restart_path": "/old/redu08/",
                "max_iters": 5,
                "save_all_iters": False,
                "diagnostics_enabled": False,
                "injected_source_test": {"enabled": True},
            },
        },
    }


def test_prepares_fresh_uninterrupted_reference() -> None:
    source = source_config()
    prepared = prepare_reference_config(
        source, output_dir="/new/reference/reduced/", iterations=10
    )

    fruit = prepared["timestream"]["fruit_loops"]
    assert prepared["runtime"]["output_dir"] == "/new/reference/reduced/"
    assert prepared["timestream"]["raw_time_chunk"]["kernel"]["enabled"]
    assert fruit["path"] is None
    assert fruit["restart_path"] is None
    assert fruit["max_iters"] == 10
    assert fruit["save_all_iters"]
    assert fruit["diagnostics_enabled"]
    assert fruit["injected_source_test"] == {
        "enabled": False,
        "start_iteration": 1,
        "array_amplitude_mjy_beam": [0.0, 0.0, 0.0],
    }
    assert source["runtime"]["output_dir"] == "/old/reduced/"


def test_rejects_nonpointing_or_multiobservation_source() -> None:
    source = source_config()
    source["runtime"]["reduction_type"] = "science"
    try:
        prepare_reference_config(
            source, output_dir="/new/reference/reduced/", iterations=10
        )
    except ValueError as error:
        assert "pointing mode" in str(error)
    else:
        raise AssertionError("science source unexpectedly accepted")

    source = source_config()
    source["inputs"].append({"path": "/data/obs133411.nc"})
    try:
        prepare_reference_config(
            source, output_dir="/new/reference/reduced/", iterations=10
        )
    except ValueError as error:
        assert "exactly one observation" in str(error)
    else:
        raise AssertionError("multi-observation source unexpectedly accepted")
