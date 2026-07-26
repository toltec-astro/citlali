#!/usr/bin/env python3
"""Prepare single-observation configs and Unity launch scripts for one stage."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
from pathlib import Path

import yaml


SCHEMA_VERSION = "citlali-fruit-loop-population-stage-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested(config: dict, *path: str) -> dict:
    node: object = config
    for key in path:
        if not isinstance(node, dict) or key not in node:
            raise ValueError(f"required config path is absent: {'.'.join(path)}")
        node = node[key]
    if not isinstance(node, dict):
        raise ValueError(f"config path is not a mapping: {'.'.join(path)}")
    return node


def input_obsnum(item: dict) -> int:
    try:
        name = str(item["meta"]["name"])
        return int(name.split("_", maxsplit=1)[0])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("input has no parseable meta.name obsnum") from error


def apt_path(item: dict) -> str:
    for cal_item in item.get("cal_items", []):
        if cal_item.get("type") != "array_prop_table":
            continue
        path = cal_item.get("filepath")
        if isinstance(path, str):
            return path
    raise ValueError(f"input {input_obsnum(item)} has no APT filepath")


def read_run_rows(path: Path, *, phase: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if row.get("phase") == phase
        ]
    if not rows:
        raise ValueError(f"run matrix has no rows for phase {phase!r}")
    obsnums = [int(row["obsnum"]) for row in rows]
    if len(obsnums) != len(set(obsnums)):
        raise ValueError(f"phase {phase!r} contains duplicate observations")
    return sorted(rows, key=lambda row: int(row["quality_rank"]))


def prepare_observation_config(
    source: dict,
    *,
    input_item: dict,
    output_dir: str,
    fitreport_dir: str,
    iterations: int,
) -> dict:
    if iterations < 2:
        raise ValueError("fruit-loop run requires at least two iterations")
    config = copy.deepcopy(source)
    runtime = nested(config, "runtime")
    if runtime.get("reduction_type") != "pointing":
        raise ValueError("population run requires pointing reduction mode")
    config["inputs"] = [copy.deepcopy(input_item)]
    runtime["output_dir"] = output_dir
    runtime["use_subdir"] = True
    nested(config, "kids", "solver")["fitreportdir"] = fitreport_dir
    nested(config, "timestream", "raw_time_chunk")["kernel"]["enabled"] = True
    fruit = nested(config, "timestream", "fruit_loops")
    fruit["enabled"] = True
    fruit["path"] = None
    fruit["restart_path"] = None
    fruit["max_iters"] = iterations
    fruit["save_all_iters"] = True
    fruit["diagnostics_enabled"] = True
    fruit["injected_source_test"] = {
        "enabled": False,
        "start_iteration": 1,
        "array_amplitude_mjy_beam": [0.0, 0.0, 0.0],
    }
    return config


def shell_scripts() -> dict[str, str]:
    return {
        "snapshot_binary.sh": """#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/work/toltec/commissioning2025-test/2026-ENG-hero-multiyear-pointings-v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/diagnostics/fruitloop_population_v1/stage_a}"
SETUP_DIR="${SETUP_DIR:-${OUTPUT_ROOT}/setup}"
CITLALI_SOURCE="${CITLALI_SOURCE:-/work/toltec/citlali_dev/citlali_refactor/build/bin/citlali}"
BIN_DIR="${SETUP_DIR}/bin"

test -x "${CITLALI_SOURCE}"
mkdir -p "${BIN_DIR}"
source_sha="$(sha256sum "${CITLALI_SOURCE}" | awk '{print $1}')"
snapshot="${BIN_DIR}/citlali-${source_sha}"
if test ! -e "${snapshot}"; then
    install -m 0755 "${CITLALI_SOURCE}" "${snapshot}"
fi
echo "${source_sha}  ${snapshot}" | sha256sum -c -
"${snapshot}" --version >"${BIN_DIR}/citlali-${source_sha}.version.txt" 2>&1 || true
printf 'CITLALI_SNAPSHOT=%s\\nCITLALI_SHA256=%s\\n' \
    "${snapshot}" "${source_sha}" >"${SETUP_DIR}/binary.env"
echo "Frozen Citlali ${source_sha} at ${snapshot}"
""",
        "preflight_stage_a.sh": """#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/work/toltec/commissioning2025-test/2026-ENG-hero-multiyear-pointings-v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/diagnostics/fruitloop_population_v1/stage_a}"
SETUP_DIR="${SETUP_DIR:-${OUTPUT_ROOT}/setup}"
JOBS="${SETUP_DIR}/stage_a_jobs.tsv"
MIN_FREE_KIB="${MIN_FREE_KIB:-31457280}"

test -f "${SETUP_DIR}/binary.env"
source "${SETUP_DIR}/binary.env"
test -x "${CITLALI_SNAPSHOT}"
echo "${CITLALI_SHA256}  ${CITLALI_SNAPSHOT}" | sha256sum -c -
(cd "${SETUP_DIR}" && sha256sum -c config_checksums.sha256)
test -d "${PROJECT_ROOT}/apts/hero_rc1"
test -d "${PROJECT_ROOT}/data"
free_kib="$(df -Pk "${OUTPUT_ROOT}" | awk 'NR == 2 {print $4}')"
if test -z "${free_kib}" || test "${free_kib}" -lt "${MIN_FREE_KIB}"; then
    echo "Refusing launch with less than ${MIN_FREE_KIB} KiB free at ${OUTPUT_ROOT}" >&2
    exit 1
fi

n_jobs=0
while IFS=$'\\t' read -r task obsnum config output apt rank stratum source; do
    test "${task}" = "$((n_jobs + 1))"
    test -f "${SETUP_DIR}/${config}"
    test -f "${apt}"
    while read -r input_path; do
        test -f "${input_path}"
    done < <(awk '/^[[:space:]]*- filepath: / {print $3}' "${SETUP_DIR}/${config}")
    if find "${output}" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
        echo "Refusing nonempty output for obs ${obsnum}: ${output}" >&2
        exit 1
    fi
    n_jobs=$((n_jobs + 1))
done < <(tail -n +2 "${JOBS}")

test "${n_jobs}" -eq 16
mkdir -p "${OUTPUT_ROOT}/logs"
echo "Stage A preflight passed for ${n_jobs} jobs; ${free_kib} KiB free."
""",
        "run_stage_a_task.sh": """#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/work/toltec/commissioning2025-test/2026-ENG-hero-multiyear-pointings-v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/diagnostics/fruitloop_population_v1/stage_a}"
SETUP_DIR="${SETUP_DIR:-${OUTPUT_ROOT}/setup}"
JOBS="${SETUP_DIR}/stage_a_jobs.tsv"
task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

source "${SETUP_DIR}/binary.env"
line="$(sed -n "$((task_id + 1))p" "${JOBS}")"
test -n "${line}"
IFS=$'\\t' read -r task obsnum config output apt rank stratum source <<<"${line}"
test "${task}" = "${task_id}"
test -f "${SETUP_DIR}/${config}"
test -f "${apt}"
echo "${CITLALI_SHA256}  ${CITLALI_SNAPSHOT}" | sha256sum -c -
if find "${output}" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
    echo "Refusing nonempty output for obs ${obsnum}: ${output}" >&2
    exit 1
fi
echo "Starting obs=${obsnum} rank=${rank} stratum=${stratum} binary=${CITLALI_SHA256}"
exec "${CITLALI_SNAPSHOT}" -l info "${SETUP_DIR}/${config}"
""",
        "submit_stage_a.sh": """#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/work/toltec/commissioning2025-test/2026-ENG-hero-multiyear-pointings-v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/diagnostics/fruitloop_population_v1/stage_a}"
SETUP_DIR="${SETUP_DIR:-${OUTPUT_ROOT}/setup}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-4}"

"${SETUP_DIR}/preflight_stage_a.sh"
sbatch \
    --job-name=flpop-a \
    --array="1-16%${ARRAY_CONCURRENCY}" \
    --output="${OUTPUT_ROOT}/logs/flpop-a-%A_%a.out" \
    --time=24:00:00 \
    --mem=64G \
    --cpus-per-task=6 \
    --partition=toltec-cpu \
    --chdir="${PROJECT_ROOT}" \
    "${SETUP_DIR}/run_stage_a_task.sh"
""",
        "status_stage_a.sh": """#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/work/toltec/commissioning2025-test/2026-ENG-hero-multiyear-pointings-v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/diagnostics/fruitloop_population_v1/stage_a}"
SETUP_DIR="${SETUP_DIR:-${OUTPUT_ROOT}/setup}"
JOBS="${SETUP_DIR}/stage_a_jobs.tsv"

squeue -u "${USER}" -n flpop-a || true
printf 'obsnum\\trank\\tstratum\\titerations\\tstate\\n'
while IFS=$'\\t' read -r task obsnum config output apt rank stratum source; do
    iterations=0
    if test -d "${output}"; then
        iterations="$(find "${output}" -mindepth 1 -maxdepth 1 -type d -name 'redu??' | wc -l | tr -d ' ')"
    fi
    state=not_started
    if test "${iterations}" -eq 10; then
        state=products_present
    elif test "${iterations}" -gt 0; then
        state=partial
    fi
    printf '%s\\t%s\\t%s\\t%s\\t%s\\n' \
        "${obsnum}" "${rank}" "${stratum}" "${iterations}" "${state}"
done < <(tail -n +2 "${JOBS}")

echo
echo "Potential error-level log lines:"
grep -EHi '(^|[^[:alpha:]])(error|fatal)([^[:alpha:]]|$)' \
    "${OUTPUT_ROOT}"/logs/flpop-a-*.out 2>/dev/null || true
""",
    }


def write_stage_package(
    *,
    source_path: Path,
    run_matrix_path: Path,
    output_dir: Path,
    runtime_output_root: str,
    fitreport_dir: str,
    phase: str,
    iterations: int,
) -> list[dict]:
    source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError("source config must contain one YAML mapping")
    inputs = source.get("inputs")
    if not isinstance(inputs, list):
        raise ValueError("source config inputs must be a list")
    input_lookup = {input_obsnum(item): item for item in inputs}
    if len(input_lookup) != len(inputs):
        raise ValueError("source config contains duplicate input obsnums")

    rows = read_run_rows(run_matrix_path, phase=phase)
    root = runtime_output_root.rstrip("/")
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    checksums = []
    observations = []
    for task, row in enumerate(rows, start=1):
        obsnum = int(row["obsnum"])
        if obsnum not in input_lookup:
            raise ValueError(f"source config is missing obsnum {obsnum}")
        item = input_lookup[obsnum]
        apt = apt_path(item)
        expected_apt = f"/apt_{obsnum}_matched.ecsv"
        if not apt.endswith(expected_apt):
            raise ValueError(
                f"obsnum {obsnum} has unexpected APT path {apt!r}"
            )
        runtime_output_dir = f"{root}/obs{obsnum}/reduced/"
        config = prepare_observation_config(
            source,
            input_item=item,
            output_dir=runtime_output_dir,
            fitreport_dir=fitreport_dir,
            iterations=iterations,
        )
        filename = (
            f"citlali_rc1_fruitloops{iterations}_o{obsnum}.yaml"
        )
        path = output_dir / filename
        path.write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )
        digest = sha256(path)
        checksums.append(f"{digest}  {filename}")
        job = {
            "task": task,
            "obsnum": obsnum,
            "config": filename,
            "output": runtime_output_dir,
            "apt": apt,
            "rank": int(row["quality_rank"]),
            "stratum": row["quality_stratum"],
            "source": row["source"],
        }
        jobs.append(job)
        observations.append(
            {
                **job,
                "selection_reason": row["selection_reason"],
                "config_sha256": digest,
            }
        )

    fields = ("task", "obsnum", "config", "output", "apt", "rank",
              "stratum", "source")
    with (output_dir / "stage_a_jobs.tsv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(jobs)
    (output_dir / "config_checksums.sha256").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8"
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_config": str(source_path.resolve()),
        "source_config_sha256": sha256(source_path),
        "source_observation_count": len(inputs),
        "run_matrix": str(run_matrix_path.resolve()),
        "run_matrix_sha256": sha256(run_matrix_path),
        "phase": phase,
        "observation_count": len(observations),
        "iterations": iterations,
        "fitreport_dir": fitreport_dir,
        "runtime_output_root": root,
        "policy": {
            "one_observation_per_process": True,
            "save_all_iterations": True,
            "processed_kernel_output": True,
            "diagnostics_enabled": True,
            "restart_path": None,
            "injected_source_test_enabled": False,
            "immutable_binary_snapshot_required": True,
        },
        "observations": observations,
    }
    (output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    for name, content in shell_scripts().items():
        path = output_dir / name
        path.write_text(content, encoding="utf-8")
        path.chmod(0o755)
    return observations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--run-matrix", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--runtime-output-root", required=True)
    parser.add_argument("--fitreport-dir", required=True)
    parser.add_argument("--phase", default="sentinel_extension_first")
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    observations = write_stage_package(
        source_path=args.input,
        run_matrix_path=args.run_matrix,
        output_dir=args.output_dir,
        runtime_output_root=args.runtime_output_root,
        fitreport_dir=args.fitreport_dir,
        phase=args.phase,
        iterations=args.iterations,
    )
    print(
        f"wrote {len(observations)} single-observation jobs "
        f"to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
