#!/usr/bin/env python3
"""Freeze and orchestrate the SCI-ALIGN-001 pointing-fit Unity campaign.

The campaign has a deliberately staged lifecycle:

1. authenticate the S/N-selected, owner-reviewed source-quality cohort;
2. freeze exact Unity PTC/PPT identities;
3. make the map-space prerequisite products in an isolated Slurm array;
4. run one known fit gate, then the remaining fit-gate array; and
5. stop for owner review before any expensive ``resume-observation`` work.

This tool writes diagnostic manifests and job scripts only.  It does not
submit jobs or alter Citlali products.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any, Iterable

from astropy.table import Table

import analyze_sci_align_001_lissajous_pointing as map_space


PILOT_OBSNUM = 150818


class ContractError(RuntimeError):
    """A frozen input or campaign lifecycle contract was violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_checksums(root: Path, names: Iterable[str], manifest: str) -> None:
    lines = [f"{sha256_file(root / name)}  {name}" for name in sorted(names)]
    (root / manifest).write_text("\n".join(lines) + "\n")


def verify_manifest(root: Path, name: str = "SHA256SUMS") -> None:
    manifest = root / name
    if not manifest.is_file():
        raise ContractError(f"checksum manifest is missing: {manifest}")
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = root / relative.strip()
        if sha256_file(path) != expected:
            raise ContractError(f"checksum mismatch: {path}")


def git_commit(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        capture_output=True, text=True,
    )
    return completed.stdout.strip()


def quality_rows(package: Path) -> tuple[list[dict[str, Any]], dict[int, Any]]:
    verify_manifest(package)
    verify_manifest(package / "frozen")
    verify_manifest(package / "result")
    human = json.loads((package / "human_morphology_review.json").read_text())
    if human.get("schema") != "sci-align-001-pointing-human-morphology-review-v1":
        raise ContractError("unsupported human morphology review schema")
    table = Table.read(package / "result" / "snr_selected_pointings.ecsv")
    selected = []
    for source in table:
        row = {name: source[name].item() if hasattr(source[name], "item") else source[name]
               for name in table.colnames}
        row["obsnum"] = int(row["obsnum"])
        if not bool(row["snr_pass"]):
            raise ContractError(f"obs {row['obsnum']}: selected row fails S/N gate")
        selected.append(row)
    if int(human["accepted_observation_count"]) != len(selected):
        raise ContractError("human review count does not match selected table")
    frozen = json.loads((package / "frozen" / "frozen_input.json").read_text())
    identities = {int(row["obsnum"]): row for row in frozen["rows"]}
    if {row["obsnum"] for row in selected} - identities.keys():
        raise ContractError("selected observation is absent from quality identities")
    return selected, identities


def schema_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = {int(row["obsnum"]): row for row in csv.DictReader(handle)}
    if not rows:
        raise ContractError("schema audit is empty")
    return rows


def campaign_protocol(
    template_path: Path,
    selection_path: Path,
    selection_sha256s_path: Path,
    source_package: Path,
    count: int,
    repository_commit: str,
) -> dict[str, Any]:
    protocol = json.loads(template_path.read_text())
    if protocol.get("schema") != "sci-align-001-lissajous-timestream-protocol-v1":
        raise ContractError("unsupported timestream protocol template")
    protocol["campaign"] = {
        "schema": "sci-align-001-pointing-fit-campaign-v1",
        "repository_commit": repository_commit,
        "lifecycle": (
            "map prerequisites; pilot fit gate; remaining fit gates; owner "
            "review; explicitly approved checkpointed resumes"
        ),
        "fit_gate_maximum_wall_seconds": 1800,
        "resume_maximum_wall_seconds": 144000,
    }
    protocol["scope"].update({
        "pointing_count": count,
        "bracket_group_count": 0,
        "selection_change_prohibited": True,
        "high_snr_classification_authority": (
            "checksum-bound a1100 S/N>=60 source-quality audit plus owner "
            "morphology disposition"
        ),
        "unity_access": "owner-run only",
        "production_code_change": False,
    })
    protocol["input_authority"] = {
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": sha256_file(selection_path),
        "selection_sha256s_sha256": sha256_file(selection_sha256s_path),
        "source_quality_package": str(source_package),
        "source_quality_sha256s_sha256": sha256_file(
            source_package / "SHA256SUMS"
        ),
        "map_per_observation_sha256s_required": True,
        "map_corpus_sha256s_required_before_fit_gate": True,
    }
    protocol["corpus"] = {
        "primary_sets": ["all_snr_selected_66"],
        "aberration_sensitivity": (
            "compare the full cohort with the strongest recorded aberration "
            "tail removed; aberration metrics are covariates, not gates"
        ),
        "observation_resampling_unit": "one complete pointing observation",
        "corpus_inference_status": "not part of fit-gate campaign",
    }
    return protocol


def shell_join(parts: Iterable[str | Path]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def sbatch_header(
    name: str, *, array: str | None, output: Path, error: Path,
    time: str, memory: str, cpus: int = 1,
) -> list[str]:
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={name}",
        "#SBATCH --partition=toltec-cpu",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={memory}",
        f"#SBATCH --time={time}",
        f"#SBATCH --output={output}",
        f"#SBATCH --error={error}",
    ]
    if array is not None:
        lines.append(f"#SBATCH --array={array}")
    lines.extend(["set -euo pipefail", ""])
    return lines


def write_campaign_scripts(
    root: Path, repo: Path, python: str, obsnums: list[int], concurrency: int,
) -> list[str]:
    frozen = root / "frozen"
    jobs = root / "jobs"
    logs = jobs / "logs"
    map_root = root / "map_results"
    fit_root = root / "fit_results"
    jobs.mkdir(parents=True)
    logs.mkdir(parents=True)
    runtime_cache = root / "_runtime_cache"
    map_commands = []
    gate_commands = []
    resume_commands = []
    for obsnum in obsnums:
        map_commands.append(shell_join([
            python, repo / "tools/diagnostics/analyze_sci_align_001_lissajous_pointing.py",
            "run-one", "--selection-dir", frozen, "--output-root", map_root,
            "--obsnum", str(obsnum),
        ]))
        fit_output = fit_root / f"o{obsnum}"
        gate_commands.append(shell_join([
            python, repo / "tools/diagnostics/analyze_sci_align_001_lissajous_timestream.py",
            "fit-gate", "--protocol", frozen / "timestream_protocol.json",
            "--selection", frozen / "selected_pointings.json",
            "--map-root", map_root, "--obsnum", str(obsnum),
            "--output", fit_output, "--maximum-wall-seconds", "1800",
        ]))
        resume_commands.append(shell_join([
            python, repo / "tools/diagnostics/analyze_sci_align_001_lissajous_timestream.py",
            "resume-observation", "--protocol", frozen / "timestream_protocol.json",
            "--selection", frozen / "selected_pointings.json",
            "--map-root", map_root, "--obsnum", str(obsnum),
            "--output", fit_output, "--maximum-wall-seconds", "144000",
            "--owner-review-approved",
        ]))
    files = {
        "map.commands.txt": map_commands,
        "fit_gate.commands.txt": gate_commands,
        "resume.commands.txt": resume_commands,
    }
    for name, lines in files.items():
        (jobs / name).write_text("\n".join(lines) + "\n")

    def array_script(name: str, table: str, count: int, time: str, memory: str) -> str:
        header = sbatch_header(
            name, array=f"0-{count - 1}%{concurrency}",
            output=logs / "%x_%A_%a.out", error=logs / "%x_%A_%a.err",
            time=time, memory=memory,
        )
        body = [
            f"mkdir -p {shlex.quote(str(runtime_cache / 'matplotlib'))} "
            f"{shlex.quote(str(runtime_cache / 'xdg'))}",
            "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
            "export MPLBACKEND=Agg",
            f"export MPLCONFIGDIR={shlex.quote(str(runtime_cache / 'matplotlib'))}",
            f"export XDG_CACHE_HOME={shlex.quote(str(runtime_cache / 'xdg'))}",
            f"command_table={shlex.quote(str(jobs / table))}",
            'command=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$command_table")',
            'test -n "$command"',
            'bash -c "$command"',
        ]
        return "\n".join([*header, *body]) + "\n"

    (jobs / "run_map_array.sbatch").write_text(array_script(
        "sci-align-map", "map.commands.txt", len(obsnums), "04:00:00", "32G"
    ))
    aggregate = sbatch_header(
        "sci-align-map-agg", array=None,
        output=logs / "%x_%j.out", error=logs / "%x_%j.err",
        time="01:00:00", memory="16G",
    )
    aggregate.extend([
        f"mkdir -p {shlex.quote(str(runtime_cache / 'matplotlib'))} "
        f"{shlex.quote(str(runtime_cache / 'xdg'))}",
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
        "export MPLBACKEND=Agg",
        f"export MPLCONFIGDIR={shlex.quote(str(runtime_cache / 'matplotlib'))}",
        f"export XDG_CACHE_HOME={shlex.quote(str(runtime_cache / 'xdg'))}",
    ])
    aggregate.append(shell_join([
        python, repo / "tools/diagnostics/analyze_sci_align_001_lissajous_pointing.py",
        "aggregate", "--selection-dir", frozen, "--output", map_root,
        "--existing-observation-root", map_root,
    ]))
    (jobs / "run_map_aggregate.sbatch").write_text("\n".join(aggregate) + "\n")

    pilot_index = obsnums.index(PILOT_OBSNUM)
    pilot = sbatch_header(
        "sci-align-fit-pilot", array=None,
        output=logs / "%x_%j.out", error=logs / "%x_%j.err",
        time="00:45:00", memory="32G",
    )
    pilot.extend([
        f"mkdir -p {shlex.quote(str(runtime_cache / 'matplotlib'))} "
        f"{shlex.quote(str(runtime_cache / 'xdg'))}",
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
        "export MPLBACKEND=Agg",
        f"export MPLCONFIGDIR={shlex.quote(str(runtime_cache / 'matplotlib'))}",
        f"export XDG_CACHE_HOME={shlex.quote(str(runtime_cache / 'xdg'))}",
    ])
    pilot.append(gate_commands[pilot_index])
    (jobs / "run_fit_gate_pilot.sbatch").write_text("\n".join(pilot) + "\n")

    remaining = [command for index, command in enumerate(gate_commands)
                 if index != pilot_index]
    (jobs / "fit_gate_remaining.commands.txt").write_text(
        "\n".join(remaining) + "\n"
    )
    (jobs / "run_fit_gate_remaining_array.sbatch").write_text(array_script(
        "sci-align-fit-gate", "fit_gate_remaining.commands.txt", len(remaining),
        "00:45:00", "32G",
    ))
    (jobs / "run_resume_array.sbatch").write_text(array_script(
        "sci-align-resume", "resume.commands.txt", len(obsnums), "48:00:00", "32G"
    ))
    names = [*files, "fit_gate_remaining.commands.txt", "run_map_array.sbatch",
             "run_map_aggregate.sbatch", "run_fit_gate_pilot.sbatch",
             "run_fit_gate_remaining_array.sbatch", "run_resume_array.sbatch"]
    write_checksums(jobs, names, "JOB_SHA256SUMS")
    return [f"jobs/{name}" for name in [*names, "JOB_SHA256SUMS"]]


def freeze(args: argparse.Namespace) -> None:
    output = args.output_root.resolve()
    if output.exists():
        raise ContractError(f"output already exists: {output}")
    frozen = output / "frozen"
    frozen.mkdir(parents=True)
    selected, quality_identities = quality_rows(args.quality_package.resolve())
    audited = schema_rows(args.schema_audit.resolve())
    rows = []
    for quality in sorted(selected, key=lambda row: row["obsnum"]):
        obsnum = quality["obsnum"]
        audit = audited.get(obsnum)
        if audit is None or audit.get("status") != "ready":
            raise ContractError(f"obs {obsnum}: Unity schema audit is not ready")
        ptc = Path(audit["ptc_path"]).resolve()
        ppt = Path(audit["ppt_path"]).resolve()
        if not ptc.is_file() or not ppt.is_file():
            raise ContractError(f"obs {obsnum}: Unity PTC/PPT product is missing")
        if int(audit["size_bytes"]) != ptc.stat().st_size:
            raise ContractError(f"obs {obsnum}: PTC size changed after schema audit")
        identity = quality_identities[obsnum]
        if sha256_file(ppt) != identity["ppt_sha256"]:
            raise ContractError(f"obs {obsnum}: PPT differs from quality audit")
        row: dict[str, Any] = {
            "beammap_obsnum": 0,
            "pointing_obsnum": obsnum,
            "selection_role": "snr_selected_owner_morphology_pass",
            "brightness_stratum": "snr_ge_60",
            "ptc_path": str(ptc),
            "ptc_sha256": sha256_file(ptc),
            "ppt_path": str(ppt),
            "ppt_sha256": identity["ppt_sha256"],
            "a1100_map_sha256": quality["a1100_map_sha256"],
            "strongest_abs_smoothed_residual_fraction_peak": float(
                quality["strongest_abs_smoothed_residual_fraction_peak"]
            ),
            "strongest_positive_secondary_peak_fraction": float(
                quality["strongest_positive_secondary_peak_fraction"]
            ),
            "coherent_residual_component_count": int(
                quality["coherent_residual_component_count"]
            ),
            "morphology_disposition": "pass_with_recorded_aberration_structure",
        }
        row.update(map_space.ppt_a1100(ppt))
        row.update(map_space.ptc_summary(ptc))
        rows.append(row)
    if PILOT_OBSNUM not in {row["pointing_obsnum"] for row in rows}:
        raise ContractError(f"required pilot observation {PILOT_OBSNUM} is absent")
    selection = {
        "schema": "sci-align-001-lissajous-pointing-selection-v1",
        "selection_rule": (
            "a1100 PPT sig2noise >= 60 from the checksum-bound source-quality "
            "audit; all rows retained after owner morphology review"
        ),
        "selection_is_independent_of_timing_results": True,
        "morphology_is_not_an_automatic_gate": True,
        "row_count": len(rows),
        "rows": rows,
    }
    write_json(frozen / "selected_pointings.json", selection)
    with (frozen / "selected_pointings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    map_protocol = json.loads(args.map_protocol_template.read_text())
    write_json(frozen / "frozen_protocol.json", map_protocol)
    write_checksums(
        frozen,
        ["frozen_protocol.json", "selected_pointings.csv", "selected_pointings.json"],
        "SELECTION_SHA256SUMS",
    )
    # The map-space loader expects this conventional name and tolerates the
    # later addition of the timestream protocol to the same frozen directory.
    (frozen / "SHA256SUMS").write_text(
        (frozen / "SELECTION_SHA256SUMS").read_text()
    )
    repository_commit = git_commit(args.repo_root.resolve())
    protocol = campaign_protocol(
        args.timestream_protocol_template.resolve(),
        frozen / "selected_pointings.json",
        frozen / "SELECTION_SHA256SUMS",
        args.quality_package.resolve(), len(rows), repository_commit,
    )
    write_json(frozen / "timestream_protocol.json", protocol)
    job_names = write_campaign_scripts(
        output, args.repo_root.resolve(), args.python, [
            row["pointing_obsnum"] for row in rows
        ], args.array_concurrency,
    )
    preparation = {
        "schema": "sci-align-001-pointing-fit-campaign-preparation-v1",
        "repository_commit": repository_commit,
        "observation_count": len(rows),
        "pilot_obsnum": PILOT_OBSNUM,
        "selection_sha256": sha256_file(frozen / "selected_pointings.json"),
        "timestream_protocol_sha256": sha256_file(
            frozen / "timestream_protocol.json"
        ),
        "source_quality_package_sha256s_sha256": sha256_file(
            args.quality_package.resolve() / "SHA256SUMS"
        ),
        "lifecycle_stop": (
            "fit gates only; run_resume_array.sbatch requires a separate, "
            "explicit owner decision after reviewing all gate artifacts"
        ),
        "job_files": job_names,
    }
    write_json(output / "campaign_preparation.json", preparation)
    write_checksums(
        output,
        ["campaign_preparation.json", "frozen/SELECTION_SHA256SUMS",
         "frozen/timestream_protocol.json", "jobs/JOB_SHA256SUMS"],
        "PREPARATION_SHA256SUMS",
    )
    print(
        f"campaign frozen: pointings={len(rows)} pilot={PILOT_OBSNUM} "
        f"output={output}"
    )


def audit_gates(args: argparse.Namespace) -> None:
    selection = json.loads(args.selection.read_text())
    rows = []
    for selected in selection["rows"]:
        obsnum = int(selected["pointing_obsnum"])
        root = args.fit_root / f"o{obsnum}"
        result: dict[str, Any] = {
            "obsnum": obsnum,
            "status": "missing",
            "automatic_structural_status": "unknown",
        }
        try:
            verify_manifest(root, "FIT_GATE_SHA256SUMS")
            gate = json.loads((root / "fit_gate.json").read_text())
            result["status"] = "fit_gate_complete"
            result["automatic_structural_status"] = gate["quality_gate"][
                "automatic_structural_status"
            ]
            for model in ("constant", "lag", "hysteresis", "joint"):
                fit = gate["point_model_results"][model]
                result[f"{model}_status"] = fit["status"]
                result[f"{model}_tau_ms"] = float(fit["tau_ms"])
                parameters = fit.get("parameters", {})
                result[f"{model}_h_az_arcsec"] = float(
                    parameters.get("h_az_arcsec", float("nan"))
                )
                result[f"{model}_h_el_arcsec"] = float(
                    parameters.get("h_el_arcsec", float("nan"))
                )
        except (OSError, KeyError, ValueError, ContractError) as error:
            result["status"] = "invalid" if root.exists() else "missing"
            result["error"] = str(error)
        rows.append(result)
    output = args.output.resolve()
    if output.exists():
        raise ContractError(f"output already exists: {output}")
    output.mkdir(parents=True)
    Table(rows=rows).write(output / "fit_gate_status.ecsv", format="ascii.ecsv")
    summary = {
        "schema": "sci-align-001-pointing-fit-gate-campaign-audit-v1",
        "selection_sha256": sha256_file(args.selection),
        "expected_count": len(rows),
        "complete_count": sum(row["status"] == "fit_gate_complete" for row in rows),
        "missing_count": sum(row["status"] == "missing" for row in rows),
        "invalid_count": sum(row["status"] == "invalid" for row in rows),
        "owner_review_status": "required_before_resume",
    }
    write_json(output / "manifest.json", summary)
    write_checksums(
        output, ["fit_gate_status.ecsv", "manifest.json"], "SHA256SUMS"
    )
    print(
        f"gate audit: complete={summary['complete_count']} "
        f"missing={summary['missing_count']} invalid={summary['invalid_count']} "
        f"output={output}"
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--schema-audit", type=Path, required=True)
    freeze_parser.add_argument("--quality-package", type=Path, required=True)
    freeze_parser.add_argument("--map-protocol-template", type=Path, required=True)
    freeze_parser.add_argument(
        "--timestream-protocol-template", type=Path, required=True
    )
    freeze_parser.add_argument("--repo-root", type=Path, required=True)
    freeze_parser.add_argument("--output-root", type=Path, required=True)
    freeze_parser.add_argument("--python", default="python")
    freeze_parser.add_argument("--array-concurrency", type=int, default=4)
    audit = sub.add_parser("audit-gates")
    audit.add_argument("--selection", type=Path, required=True)
    audit.add_argument("--fit-root", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "freeze":
            if args.array_concurrency < 1:
                raise ContractError("array concurrency must be positive")
            freeze(args)
        elif args.command == "audit-gates":
            audit_gates(args)
        else:  # pragma: no cover
            raise ContractError(f"unsupported command: {args.command}")
    except (ContractError, OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
