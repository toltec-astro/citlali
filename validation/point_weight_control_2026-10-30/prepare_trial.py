#!/usr/bin/env python3
"""Prepare one paired local POINT trial; never reduce during preparation.

Default: inspect and print a plan. --prepare creates only new configurations,
TolTECA workdirs, frozen TolProj launchers, and check-only owner drivers.
Only the generated owner's explicit `bash 02_redu.sh --run` starts a reduction.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

PROJECT = Path(
    "/Users/gwilson/work_toltec/local_data/2026-refactor/projects/COM01-local-trial-20261027"
)
PREPARATION = PROJECT / "software/preparation-20261028"
PREPARATION_PYTHON = PREPARATION / "venv/bin/python"
PREPARATION_ENTRY = PREPARATION / "run_tolteca.py"
SOURCES = {
    152418: PROJECT / "pointings-weighted-152418-20261029",
}
NUMBERED = re.compile(r"^\d{2}_.+\.ya?ml$")
EXCLUDED = {"40_setup.yaml", "99_zz_tolproj_submission_runtime.yaml"}
ARRAYS = ["a1100", "a1400", "a2000"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def binding(path):
    return {"path": str(Path(path).absolute()), "sha256": sha(path)}


def save_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False, default=str) + "\n"
    )


def numbered(directory):
    return sorted(
        p for p in directory.iterdir() if p.is_file() and NUMBERED.fullmatch(p.name)
    )


def paths(obs):
    return (
        PROJECT / f"pointings-weight-control-{obs}-20261030",
        PROJECT / f"evidence/weight-control-{obs}-20261030",
    )


def selection(obs):
    return f'scannum == 2 & (obsnum in [{obs}]) & (interface != "hk")'


def environment(evidence):
    return dict(
        os.environ,
        MPLBACKEND="Agg",
        MPLCONFIGDIR=str(evidence / "cache/matplotlib"),
        XDG_CACHE_HOME=str(evidence / "cache"),
        TOLTECA_NO_MPI="1",
        PYTHONDONTWRITEBYTECODE="1",
    )


def indexed(value, index=0):
    if isinstance(value, list):
        return value[index]
    return value[index] if index in value else value[str(index)]


def inventory():
    result = {
        "project": str(PROJECT),
        "preparation_python": str(PREPARATION_PYTHON),
        "preparation_entry": str(PREPARATION_ENTRY),
        "trials": [],
    }
    require((PROJECT / "project.yaml").is_file(), "Existing local project is missing")
    require(
        PREPARATION_PYTHON.is_file() and PREPARATION_ENTRY.is_file(),
        "Existing preparation layer is missing",
    )
    for obs, source in SOURCES.items():
        directory, evidence = paths(obs)
        require(source.is_dir(), f"Configuration source missing: {source}")
        manifest = PROJECT / f"apts/v2/apt_{obs}_matched.apt-v2/manifest.ecsv"
        require(manifest.is_file(), f"Matched APT missing: {manifest}")
        raw = sorted(
            p.name
            for p in (PROJECT / "data").glob(f"toltec*_{obs}_000_0002_*.nc")
            if re.match(rf"toltec\d+_{obs}_000_0002_", p.name)
            and not p.name.endswith("_processed.nc")
        )
        networks = sorted(int(re.match(r"toltec(\d+)_", name).group(1)) for name in raw)
        require(
            len(networks) == len(set(networks)) and all(0 <= n <= 12 for n in networks),
            f"Duplicate or unknown network science files for {obs}: {networks}",
        )
        require(
            any(n <= 6 for n in networks)
            and any(7 <= n <= 10 for n in networks)
            and any(n >= 11 for n in networks),
            f"Existing science filenames do not cover all three arrays for {obs}: {networks}",
        )
        reductions = source / "reduced"
        result["trials"].append(
            dict(
                observation=obs,
                source=str(source),
                destination=str(directory),
                evidence=str(evidence),
                destination_exists=directory.exists(),
                evidence_exists=evidence.exists(),
                source_numbered=[binding(p) for p in numbered(source)],
                existing_reductions=[
                    p.name for p in sorted(reductions.iterdir()) if p.is_dir()
                ]
                if reductions.exists()
                else [],
                raw_filenames=raw,
                available_networks=networks,
                absent_networks=[n for n in range(13) if n not in networks],
                selected_apt_manifest=binding(manifest),
            )
        )
    return result


def validate_runtime(launcher):
    launcher = Path(launcher).expanduser().absolute()
    require(
        launcher.is_file() and os.access(launcher, os.X_OK),
        f"Runtime launcher unavailable: {launcher}",
    )
    require(
        launcher.name == "citlali" and launcher.parent.name == "bin",
        "Expected packaged bin/citlali launcher",
    )
    root = launcher.parent.parent
    manifest = root / "manifest.json"
    record = json.loads(manifest.read_text())
    require(
        record.get("schema") == "citlali-configured-point-runtime-v1",
        "Unexpected runtime manifest schema",
    )
    require(
        record["source"].get("tracked_clean") is True
        and re.fullmatch("[0-9a-f]{40}", record["source"]["revision"]),
        "Runtime must identify a clean committed source",
    )
    names = {entry["path"] for entry in record["files"]}
    require(
        {
            "data/configured_point_v2.yaml",
            "tools/timestream_successor/configured_point_weighted.py",
            "tools/timestream_successor/configured_point_feedback.py",
            "bin/citlali-native",
        }
        <= names,
        "Runtime does not contain the weighted POINT suite",
    )
    for entry in record["files"]:
        relative = Path(entry["path"])
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            "Unsafe runtime member",
        )
        path = root / relative
        require(
            path.resolve().is_relative_to(root.resolve())
            and path.is_file()
            and path.stat().st_size == entry["bytes"]
            and sha(path) == entry["sha256"],
            f"Runtime member changed: {relative}",
        )
    require(
        sha(record["python"]["path"]) == record["python"]["sha256"],
        "External runtime Python changed",
    )
    return {
        "launcher": binding(launcher),
        "manifest": binding(manifest),
        "source": record["source"],
        "native": record["native"],
        "python": record["python"],
    }


def merged_check(obs, directory, expected_launcher):
    from tolteca.reduce import PipelineRuntime

    config = PipelineRuntime(directory).config
    require(
        len(config.inputs) == 1 and config.inputs[0].select == selection(obs),
        "Final observation selection differs",
    )
    require(
        config.inputs[0].path.resolve() == (PROJECT / "data").resolve(),
        "Input root does not resolve inside this project",
    )
    require(
        config.jobkey == "reduced" and len(config.steps) == 1,
        "Unexpected output layout or pipeline steps",
    )
    step = config.steps[0]
    require(
        step.name == "citlali" and step.dry_run is False and step.version == ">=0",
        "Unexpected Citlali launch configuration",
    )
    require(Path(step.path) == expected_launcher, "Final executable selection differs")
    low = step.config.low_level
    require(
        low["runtime"]["n_threads"] == 3
        and low["runtime"]["verbose"] is False
        and low["runtime"]["output_dir"] == "."
        and low["runtime"]["use_subdir"] is True,
        "Expected three workers, quiet mode, and next unused reduction directory",
    )
    s = low["successor"]
    expected = dict(
        weight_comparison=True,
        schema="citlali-point-v2",
        migration="replace-legacy-point-controls-v2",
        arrays=ARRAYS,
        cleaning=dict(
            method="noise-weighted-observed-als-v1",
            ranks=dict(a1100=3, a1400=4, a2000=6),
        ),
        detector_response=dict(min_relative_response=0.10),
        rtc=dict(
            profile="existing-array-lowpass-v1",
            disturbance_exclusion="repeated-level-shifts-unassessable-remainder-v1",
        ),
        map=dict(
            method="containing-pixel",
            frame="delta-source-AltAz",
            pixel_size_arcsec=2.0,
            coverage_cut=0.15,
        ),
        feedback=dict(enabled=True, max_iters=10),
        products=dict(fits=True, fit_diagnostics=True, retain_intermediates=False),
    )
    require(
        s == expected,
        "Final weighted POINT scientific configuration differs from the prepared trial",
    )
    require(not step.config.cal_objs, "Unexpected legacy calibration-object references")
    manifests = []
    for item in step.config.cal_items:
        if item["type"] == "array_prop_table":
            path = Path(item["filepath"]).resolve()
            require(
                path.is_relative_to((PROJECT / "apts/v2").resolve())
                and path.name == "manifest.ecsv"
                and path.is_file(),
                f"APT reference leaves the intended shared matched-v2 inputs: {path}",
            )
            manifests.append(binding(path))
        else:
            require(
                item["type"] in ("photometry", "astrometry"),
                "Unexpected derived calibration consumer",
            )
    selected = PROJECT / f"apts/v2/apt_{obs}_matched.apt-v2/manifest.ecsv"
    require(
        str(selected) in {entry["path"] for entry in manifests},
        "Selected observation APT absent",
    )
    return {
        "observation": obs,
        "selection": selection(obs),
        "input_root": str(PROJECT / "data"),
        "jobkey": config.jobkey,
        "runtime": low["runtime"],
        "successor": s,
        "intentionally_shared_apt_manifests": manifests,
        "limitation": "TolTECA configuration check only; raw inputs and low-level observation files are prepared at owner launch.",
    }


def check_trial(obs, run=False):
    directory, evidence = paths(obs)
    os.environ.update(environment(evidence))
    ready = json.loads((evidence / "READY.json").read_text())
    require(
        ready["schema"] == "citlali-weight-control-local-preparation-v1"
        and ready["observation"] == obs,
        "Prepared trial identity differs",
    )
    current = {str(p) for p in numbered(directory)}
    require(
        current == {b["path"] for b in ready["numbered_configs"]},
        "Numbered configuration file set changed",
    )
    for item in (
        ready["numbered_configs"] + ready["support_files"] + ready["shared_inputs"]
    ):
        require(
            sha(item["path"]) == item["sha256"],
            f"Prepared file changed: {item['path']}",
        )
    runtime = validate_runtime(ready["runtime"]["launcher"]["path"])
    require(runtime == ready["runtime"], "Frozen runtime identity differs")
    executable = ready["executable"]
    require(
        sha(executable["snapshot"])
        == executable["sha256"]
        == runtime["launcher"]["sha256"],
        "TolProj executable snapshot changed",
    )
    merged_check(obs, directory, Path(executable["launcher"]))
    from tolproj.reduction_resources import validate_reduction_resources

    resources = validate_reduction_resources(directory)
    require(
        resources.allocated_cpus == resources.requested_threads == 3,
        "TolProj resource plan differs",
    )
    print(
        f"Prepared {obs}: POINT v2, all arrays, ranks 3/4/6, equal versus inverse-noise weights, shared CAL/health, 10% response cut, 10 feedback passes per arm; frozen inputs/configuration verified.",
        flush=True,
    )
    if not run:
        print(
            "CHECK ONLY: no reduction launched. Grant may run: bash 02_redu.sh --run",
            flush=True,
        )
        return
    env = dict(
        environment(evidence),
        TOLPROJ_CITLALI_SNAPSHOT=executable["snapshot"],
        TOLPROJ_CITLALI_SHA256=executable["sha256"],
        TMPDIR=str(evidence / "node-local"),
    )
    raise SystemExit(
        subprocess.call(
            [str(evidence / "tolteca"), "-d", str(directory), "reduce"],
            cwd=directory,
            env=env,
        )
    )


def prepare(launcher, initial_inventory):
    import yaml
    from tolproj.reduction_runtime import freeze_reduction_executable
    from tolproj.reduction_resources import validate_reduction_resources

    runtime = validate_runtime(launcher)
    launcher = Path(runtime["launcher"]["path"])
    # Validate all destinations before creating the trial; never overwrite or
    # clean an existing workdir, partial setup, snapshot, or reference reduction.
    for obs in SOURCES:
        for path in paths(obs):
            require(
                not path.exists(),
                f"Destination already exists; preserve it and choose a reviewed new trial: {path}",
            )
    profile = yaml.safe_load(
        (launcher.parent.parent / "data/configured_point_v2.yaml").read_text()
    )
    require(
        profile["successor"]["schema"] == "citlali-point-v2",
        "Runtime weighted profile differs",
    )
    profile["runtime"].update(
        n_threads=3, verbose=False, output_dir=".", use_subdir=True
    )
    profile["successor"]["cleaning"]["ranks"] = dict(a1100=3, a1400=4, a2000=6)
    profile["successor"]["map"].update(pixel_size_arcsec=2.0, coverage_cut=0.15)
    profile["successor"]["detector_response"]["min_relative_response"] = 0.10
    profile["successor"]["weight_comparison"] = True
    profile["successor"]["feedback"].update(enabled=True, max_iters=10)
    for obs, source in SOURCES.items():
        directory, evidence = paths(obs)
        directory.mkdir()
        evidence.mkdir()
        for name in ("cache/matplotlib", "node-local"):
            (evidence / name).mkdir(parents=True)
        for path in numbered(source):
            if path.name not in EXCLUDED:
                shutil.copy2(path, directory / path.name)
        kit = source / ".citlali_refactor_kit.yaml"
        if kit.is_file():
            shutil.copy2(kit, directory / kit.name)
        runtime_path = directory / "71_pointing_runtime.yaml"
        config = yaml.safe_load(runtime_path.read_text())
        step = indexed(config["reduce"]["steps"])
        step["path"] = str(launcher)
        step["config"]["low_level"]["runtime"].update(profile["runtime"])
        runtime_path.write_text(yaml.safe_dump(config, sort_keys=False))
        expert_path = directory / "99_pointing_expert_overrides.yaml"
        config = yaml.safe_load(expert_path.read_text())
        indexed(config["reduce"]["inputs"])["select"] = selection(obs)
        step = indexed(config["reduce"]["steps"])
        step["version"] = ">=0"
        step["dry_run"] = False
        step.pop("path", None)
        low = step["config"]["low_level"]
        low.setdefault("runtime", {}).update(profile["runtime"])
        low["successor"] = profile["successor"]
        expert_path.write_text(yaml.safe_dump(config, sort_keys=False))
        env = environment(evidence)
        wrapper = "#!/bin/sh\n" + "".join(
            f"export {name}={shlex.quote(env[name])}\n"
            for name in (
                "MPLBACKEND",
                "MPLCONFIGDIR",
                "XDG_CACHE_HOME",
                "TOLTECA_NO_MPI",
                "PYTHONDONTWRITEBYTECODE",
            )
        )
        wrapper += f'exec {shlex.quote(str(PREPARATION_PYTHON))} {shlex.quote(str(PREPARATION_ENTRY))} --no_banner -n "$@"\n'
        (evidence / "tolteca").write_text(wrapper)
        (evidence / "tolteca").chmod(0o755)
        with (evidence / "setup.log").open("w") as log:
            subprocess.run(
                [str(evidence / "tolteca"), "-d", str(directory), "setup", "-f"],
                cwd=directory,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        os.environ.update(env)
        merged_check(obs, directory, launcher)
        executable = freeze_reduction_executable(directory)
        executable_record = {k: str(v) for k, v in asdict(executable).items()}
        save_json(evidence / "executable.json", executable_record)
        frozen_driver = evidence / "prepare-weight-control.py"
        shutil.copy2(Path(__file__).resolve(), frozen_driver)
        script = (
            "#!/usr/bin/env bash\n#SBATCH --job-name=local-weight-control\n#SBATCH --cpus-per-task=3\n"
            "#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --mem=16G\n#SBATCH --time=04:00:00\n"
            "set -euo pipefail\n# Default is check-only. Grant alone supplies --run. Existing reductions are preserved.\n"
            f'exec {shlex.quote(str(PREPARATION_PYTHON))} -B {shlex.quote(str(frozen_driver))} --check-trial {obs} "$@"\n'
        )
        (directory / "02_redu.sh").write_text(script)
        (directory / "02_redu.sh").chmod(0o755)
        subprocess.run(["bash", "-n", str(directory / "02_redu.sh")], check=True)
        resources = validate_reduction_resources(directory)
        require(
            resources.allocated_cpus == resources.requested_threads == 3,
            "Resource allocation differs",
        )
        merged = merged_check(obs, directory, executable.launcher)
        save_json(evidence / "effective-configuration-check.json", merged)
        source_record = next(
            t for t in initial_inventory["trials"] if t["observation"] == obs
        )
        support = [
            directory / "02_redu.sh",
            evidence / "tolteca",
            frozen_driver,
            evidence / "executable.json",
            evidence / "effective-configuration-check.json",
            executable.launcher,
            PREPARATION_ENTRY,
            PREPARATION / "runtime-identity.json",
            PREPARATION / "data/finals2000A.all",
        ]
        if (directory / ".citlali_refactor_kit.yaml").is_file():
            support.append(directory / ".citlali_refactor_kit.yaml")
        ready = dict(
            schema="citlali-weight-control-local-preparation-v1",
            observation=obs,
            project=str(PROJECT),
            directory=str(directory),
            source_inventory=source_record,
            runtime=runtime,
            executable=executable_record,
            numbered_configs=[binding(p) for p in numbered(directory)],
            support_files=[binding(p) for p in support],
            shared_inputs=merged["intentionally_shared_apt_manifests"],
            preparation_only=True,
            reduction_launched=False,
            owner_check_command=f"bash {shlex.quote(str(directory / '02_redu.sh'))}",
            owner_run_command=f"bash {shlex.quote(str(directory / '02_redu.sh'))} --run",
        )
        save_json(evidence / "READY.json", ready)
        (directory / "README.md").write_text(
            f"# Prepared paired weight-control POINT trial {obs}\n\n"
            "Run `bash 02_redu.sh` to check the frozen preparation. Grant may launch with "
            "`bash 02_redu.sh --run`. No existing outputs are removed; subsequent runs reserve the next unused reduNN.\n\n"
            f"Source configuration: {source}\nEvidence: {evidence}\nRuntime source: {runtime['source']['revision']}\n\n"
            "All three arrays; ranks 3/4/6; 2 arcsec pixels; coverage 0.15; three workers; response cut 0.10; "
            "ten feedback passes per arm after each initial map; quiet mode; shared CAL/health; "
            "equal maps under reduNN/comparison/equal/152418/raw, noise-weighted maps under reduNN/152418/raw. "
            "weight-comparison.json must report matched_support_complete. "
            "The prepared checks do not assess maps or prove scientific acceptance.\n"
        )
        check_trial(obs)
        require(
            not (directory / "reduced").exists(),
            "Preparation unexpectedly created reduction outputs",
        )
    for trial in initial_inventory["trials"]:
        for item in trial["source_numbered"]:
            require(
                sha(item["path"]) == item["sha256"],
                f"Original reference configuration changed: {item['path']}",
            )
    print(
        "Paired local trial prepared and checked. No reduction was launched.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-launcher", type=Path)
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Create the one fresh paired workdir; still never launch reductions",
    )
    parser.add_argument(
        "--check-trial", type=int, choices=tuple(SOURCES), help=argparse.SUPPRESS
    )
    parser.add_argument("--run", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(
        not args.run or args.check_trial is not None,
        "--run is available only to a prepared owner driver",
    )
    require(
        not (args.prepare and args.check_trial is not None),
        "Preparation and owner execution are separate",
    )
    if args.prepare or args.check_trial is not None:
        require(PREPARATION_PYTHON.is_file(), "Existing preparation Python is missing")
        if Path(sys.prefix).resolve() != (PREPARATION / "venv").resolve():
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            os.execve(
                str(PREPARATION_PYTHON),
                [
                    str(PREPARATION_PYTHON),
                    "-B",
                    str(Path(__file__).resolve()),
                    *sys.argv[1:],
                ],
                env,
            )
    if args.check_trial is not None:
        check_trial(args.check_trial, run=args.run)
        return
    snapshot = inventory()
    if args.runtime_launcher is not None:
        snapshot["runtime"] = validate_runtime(args.runtime_launcher)
    if not args.prepare:
        print(json.dumps(snapshot, indent=2, allow_nan=False))
        print(
            "INVENTORY ONLY: no project files changed. Add --prepare with the committed runtime launcher to prepare the paired trials."
        )
        return
    require(
        args.runtime_launcher is not None,
        "--prepare requires --runtime-launcher from the reviewed committed bundle",
    )
    prepare(args.runtime_launcher, snapshot)


if __name__ == "__main__":
    try:
        main()
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as exc:
        raise SystemExit(f"Weighted POINT preparation/check failed: {exc}") from exc
