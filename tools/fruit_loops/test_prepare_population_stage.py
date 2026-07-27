from pathlib import Path

import yaml

from prepare_population_stage import (
    prepare_observation_config,
    write_stage_package,
)


def source_config() -> dict:
    return {
        "inputs": [
            {
                "meta": {"name": "101_0_2"},
                "cal_items": [{
                    "type": "array_prop_table",
                    "filepath": "/project/apts/apt_101_matched.ecsv",
                }],
                "data_items": [{"filepath": "/data/101.nc"}],
            },
            {
                "meta": {"name": "102_0_2"},
                "cal_items": [{
                    "type": "array_prop_table",
                    "filepath": "/project/apts/apt_102_matched.ecsv",
                }],
                "data_items": [{"filepath": "/data/102.nc"}],
            },
        ],
        "kids": {"solver": {"fitreportdir": "../data"}},
        "runtime": {
            "reduction_type": "pointing",
            "output_dir": "/old/",
            "use_subdir": False,
        },
        "timestream": {
            "raw_time_chunk": {"kernel": {"enabled": False}},
            "fruit_loops": {
                "enabled": False,
                "path": "/old/maps",
                "max_iters": 5,
                "save_all_iters": False,
            },
        },
    }


def test_prepares_fresh_single_observation_config() -> None:
    source = source_config()
    prepared = prepare_observation_config(
        source,
        input_item=source["inputs"][1],
        output_dir="/new/obs102/reduced/",
        fitreport_dir="/project/data",
        iterations=10,
    )

    fruit = prepared["timestream"]["fruit_loops"]
    assert len(prepared["inputs"]) == 1
    assert prepared["inputs"][0]["meta"]["name"] == "102_0_2"
    assert prepared["runtime"]["output_dir"] == "/new/obs102/reduced/"
    assert prepared["runtime"]["use_subdir"]
    assert prepared["kids"]["solver"]["fitreportdir"] == "/project/data"
    assert prepared["timestream"]["raw_time_chunk"]["kernel"]["enabled"]
    assert fruit["enabled"]
    assert fruit["path"] is None
    assert fruit["restart_path"] is None
    assert fruit["max_iters"] == 10
    assert fruit["save_all_iters"]
    assert fruit["diagnostics_enabled"]
    assert not fruit["injected_source_test"]["enabled"]
    assert len(source["inputs"]) == 2
    assert source["runtime"]["output_dir"] == "/old/"


def test_writes_checksummed_package_and_scripts(tmp_path: Path) -> None:
    source_path = tmp_path / "source.yaml"
    source_path.write_text(yaml.safe_dump(source_config(), sort_keys=False))
    matrix = tmp_path / "matrix.csv"
    matrix.write_text(
        "obsnum,source,quality_rank,quality_stratum,phase,selection_reason\n"
        "102,Uranus,7,normal,stage_a,existing\n"
        "101,Neptune,2,normal,stage_a,anchor\n"
    )
    output = tmp_path / "setup"

    observations = write_stage_package(
        source_path=source_path,
        run_matrix_path=matrix,
        output_dir=output,
        runtime_output_root="/project/diagnostics/stage_a",
        fitreport_dir="/project/data",
        phase="stage_a",
        iterations=10,
    )

    assert [row["obsnum"] for row in observations] == [101, 102]
    assert (output / "config_checksums.sha256").is_file()
    assert (output / "stage_a_jobs.tsv").read_text().splitlines()[1].startswith(
        "1\t101\t"
    )
    for name in (
        "snapshot_binary.sh",
        "preflight_stage_a.sh",
        "run_stage_a_task.sh",
        "submit_stage_a.sh",
        "status_stage_a.sh",
    ):
        assert (output / name).stat().st_mode & 0o111
    manifest = yaml.safe_load((output / "manifest.yaml").read_text())
    assert manifest["observation_count"] == 2
    assert manifest["source_observation_count"] == 2
    assert manifest["policy"]["immutable_binary_snapshot_required"]


def test_rejects_missing_observation(tmp_path: Path) -> None:
    source_path = tmp_path / "source.yaml"
    source_path.write_text(yaml.safe_dump(source_config()))
    matrix = tmp_path / "matrix.csv"
    matrix.write_text(
        "obsnum,source,quality_rank,quality_stratum,phase,selection_reason\n"
        "999,3c273,1,stress,stage_a,missing\n"
    )
    try:
        write_stage_package(
            source_path=source_path,
            run_matrix_path=matrix,
            output_dir=tmp_path / "output",
            runtime_output_root="/project/output",
            fitreport_dir="/project/data",
            phase="stage_a",
            iterations=10,
        )
    except ValueError as error:
        assert "missing obsnum 999" in str(error)
    else:
        raise AssertionError("missing observation unexpectedly accepted")


def test_writes_stage_b_scripts_with_frozen_binary_gate(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.yaml"
    source_path.write_text(yaml.safe_dump(source_config(), sort_keys=False))
    matrix = tmp_path / "matrix.csv"
    matrix.write_text(
        "obsnum,source,quality_rank,quality_stratum,phase,selection_reason\n"
        "101,Neptune,2,normal,remaining,complete_population\n"
    )
    output = tmp_path / "setup"
    required_sha = "a" * 64
    frozen_source = "/project/stage_a/setup/bin/citlali-" + required_sha

    write_stage_package(
        source_path=source_path,
        run_matrix_path=matrix,
        output_dir=output,
        runtime_output_root="/project/diagnostics/stage_b",
        fitreport_dir="/project/data",
        phase="remaining",
        iterations=10,
        stage_name="stage_b",
        min_free_kib=367_001_600,
        binary_source=frozen_source,
        expected_binary_sha256=required_sha,
    )

    assert (output / "stage_b_jobs.tsv").is_file()
    submit = (output / "submit_stage_b.sh").read_text()
    assert '--array="1-1%${ARRAY_CONCURRENCY}"' in submit
    snapshot = (output / "snapshot_binary.sh").read_text()
    assert frozen_source in snapshot
    assert required_sha in snapshot
    run = (output / "run_stage_b_task.sh").read_text()
    assert 'chmod --reference="${SETUP_DIR}/${config}"' in run
    manifest = yaml.safe_load((output / "manifest.yaml").read_text())
    assert manifest["stage_name"] == "stage_b"
    assert manifest["policy"]["required_binary_sha256"] == required_sha
