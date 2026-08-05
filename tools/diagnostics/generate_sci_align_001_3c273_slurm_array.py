#!/usr/bin/env python3
"""Generate a configurable Slurm array for selected SCI-ALIGN-001 maps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence


SELECTED_MANIFEST_SCHEMA = "sci-align-001-3c273-selected-manifest-v2"
COMMAND_TABLE_SCHEMA = "sci-align-001-3c273-command-table-v2"
SBATCH_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
DEFAULT_SBATCH_OPTIONS: tuple[tuple[str, str], ...] = (
    ("time", "48:00:00"),
    ("mem", "64G"),
    ("cpus-per-task", "6"),
    ("nodes", "1"),
    ("ntasks", "1"),
    ("partition", "toltec-cpu"),
    ("parsable", ""),
)
DEFAULT_PROTOCOL = (
    Path(__file__).resolve().parents[2]
    / "validation/sci_align_001_3c273_corpus_tooling_2026-08-03"
    / "frozen_analysis_protocol.json"
)


class SchedulerError(ValueError):
    """Invalid selected manifest or scheduler configuration."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest_object(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def absolute_preserving_symlink(path: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else Path.cwd() / expanded


def resolve_python_executable(value: Path) -> Path:
    """Resolve an explicit interpreter path or a bare command through ``PATH``.

    A bare ``python`` is the documented Unity invocation.  Treating it as a
    relative path would instead bind it to the generator's current directory.
    ``shutil.which`` resolves that command at generation time, so the rendered
    scripts contain one explicit interpreter identity rather than depending on
    a later shell's ``PATH``.
    """
    text = str(value)
    if value.parent == Path("."):
        located = shutil.which(text)
        if located is None:
            raise SchedulerError(f"--python command is not available on PATH: {text}")
        path = absolute_preserving_symlink(Path(located))
    else:
        path = absolute_preserving_symlink(value)
    if not path.is_file():
        raise SchedulerError(f"--python executable is not a regular file: {path}")
    return path


def paths_overlap(first: Path, second: Path) -> bool:
    first = resolved(first)
    second = resolved(second)
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


def load_selected_manifest(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SchedulerError(f"cannot read selected manifest {path}: {error}") from error
    if not isinstance(document, dict) or document.get("schema_version") != SELECTED_MANIFEST_SCHEMA:
        raise SchedulerError("unsupported selected-manifest schema")
    for field in ("source_inventory_sha256", "owner_selection_sha256", "obsnum_allowlist_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(document.get(field) or "")):
            raise SchedulerError(f"selected manifest lacks a valid {field}")
    if document.get("owner_selection_format") not in {"csv", "json"}:
        raise SchedulerError("selected manifest has invalid owner_selection_format")
    if document.get("obsnum_allowlist_schema_version") != "sci-align-001-3c273-obsnum-allowlist-v1":
        raise SchedulerError("selected manifest has unsupported obsnum allowlist schema")
    allowlist_name = str(document.get("obsnum_allowlist_filename") or "")
    if Path(allowlist_name).name != allowlist_name or not allowlist_name.endswith(".json"):
        raise SchedulerError("selected manifest has invalid obsnum allowlist filename")
    allowlist_path = path.parent / allowlist_name
    if not allowlist_path.is_file() or sha256_file(allowlist_path) != document["obsnum_allowlist_sha256"]:
        raise SchedulerError("selected manifest obsnum allowlist copy/digest is invalid")
    recorded = document.get("manifest_sha256")
    base = {key: value for key, value in document.items() if key != "manifest_sha256"}
    measured = digest_object(base)
    if recorded != measured:
        raise SchedulerError(
            f"selected-manifest digest mismatch: recorded={recorded!r} measured={measured}"
        )
    rows = document.get("rows")
    if not isinstance(rows, list) or not rows:
        raise SchedulerError("selected manifest has no rows")
    candidate_ids: set[str] = set()
    roles_by_observation: dict[int, list[str]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SchedulerError(f"selected row {index} must be an object")
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id or row.get("map_id") != candidate_id:
            raise SchedulerError(f"selected row {index} has invalid candidate/map identity")
        if candidate_id in candidate_ids:
            raise SchedulerError(f"selected row {index} duplicates candidate {candidate_id}")
        candidate_ids.add(candidate_id)
        obsnum = row.get("observation_number")
        if isinstance(obsnum, bool) or not isinstance(obsnum, int) or row.get("obsnum") != obsnum:
            raise SchedulerError(f"selected row {index} has invalid observation identity")
        role = str(row.get("analysis_role") or "")
        if role not in {"primary", "sensitivity"}:
            raise SchedulerError(
                f"selected row {index} has invalid analysis_role {role!r}"
            )
        roles_by_observation.setdefault(obsnum, []).append(role)
        if row.get("core_eligible") is not True:
            raise SchedulerError(f"selected candidate is not core eligible: {candidate_id}")
    for obsnum, roles in sorted(roles_by_observation.items()):
        if roles.count("primary") != 1:
            raise SchedulerError(
                f"observation {obsnum} must contain exactly one primary reduction"
            )
    document["rows"] = sorted(
        rows, key=lambda row: (int(row["observation_number"]), str(row["candidate_id"]))
    )
    return document


def validate_output_isolation(
    rows: list[dict[str, Any]],
    targets: Sequence[Path],
    controls: Sequence[Path] = (),
) -> None:
    source_paths: list[Path] = []
    for row in rows:
        for key in (
            "reduction_path",
            "reduction_run_path",
        ):
            if row.get(key):
                source_paths.append(Path(str(row[key])))
        for key in (
            "detector_tod_path",
            "telescope_path",
            "output_apt_path",
            "provenance_path",
            "config_path",
        ):
            if row.get(key):
                source_paths.append(Path(str(row[key])).parent)
        for raw in row.get("raw_files", []) or []:
            if isinstance(raw, dict) and raw.get("path"):
                source_paths.append(Path(str(raw["path"])).parent)
    for target in targets:
        for source in source_paths:
            if paths_overlap(target, source):
                raise SchedulerError(
                    f"scheduler/output path overlaps selected source: {target} and {source}"
                )
        for control in controls:
            if paths_overlap(target, control):
                raise SchedulerError(
                    f"scheduler/output path overlaps control input: {target} and {control}"
                )
    for index, target in enumerate(targets):
        for other in targets[index + 1 :]:
            if paths_overlap(target, other):
                raise SchedulerError(
                    f"scheduler output targets overlap: {target} and {other}"
                )


def command_rows(
    manifest: dict[str, Any],
    *,
    python: Path,
    runner: Path,
    protocol: Path,
    selected_manifest: Path,
    output_root: Path,
    resume: bool,
) -> list[dict[str, Any]]:
    result = []
    for array_index, row in enumerate(manifest["rows"]):
        argv = [
            str(python),
            str(runner),
            "--manifest",
            str(selected_manifest),
            "--protocol",
            str(protocol),
            "--candidate-id",
            str(row["candidate_id"]),
            "--output-root",
            str(output_root),
        ]
        if resume:
            argv.append("--resume")
        result.append(
            {
                "schema_version": COMMAND_TABLE_SCHEMA,
                "array_index": array_index,
                "candidate_id": row["candidate_id"],
                "map_id": row["map_id"],
                "observation_number": row["observation_number"],
                "analysis_role": row["analysis_role"],
                "duplicate_group_id": row.get("duplicate_group_id"),
                "session_id": row.get("session_id"),
                "argv_json": canonical_json(argv),
                "display_command": shlex.join(argv),
            }
        )
    return result


def parse_sbatch_options(values: Sequence[str]) -> list[tuple[str, str]]:
    options: dict[str, str] = dict(DEFAULT_SBATCH_OPTIONS)
    reserved = {"array", "job-name", "account", "parsable"}
    for raw in values:
        if "=" not in raw:
            raise SchedulerError(f"--sbatch-option requires KEY=VALUE: {raw!r}")
        key, value = raw.split("=", 1)
        if not SBATCH_KEY_RE.fullmatch(key) or key in reserved:
            raise SchedulerError(f"invalid or reserved Slurm option key: {key!r}")
        if not value or any(character in value for character in "\r\n"):
            raise SchedulerError(f"invalid Slurm option value for {key!r}")
        if key == "account":
            raise SchedulerError("no Slurm account directive is authorized for this workflow")
        options[key] = value
    return sorted(options.items())


def render_script(
    *,
    command_table: Path,
    python: Path,
    row_count: int,
    job_name: str,
    array_concurrency: int | None,
    sbatch_options: Sequence[tuple[str, str]],
    command_table_sha256: str,
    selected_manifest_sha256: str,
    obsnum_allowlist: Path,
    obsnum_allowlist_sha256: str,
    protocol_sha256: str,
) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", job_name):
        raise SchedulerError("--job-name contains unsupported characters")
    if not sbatch_options:
        sbatch_options = DEFAULT_SBATCH_OPTIONS
    array = f"0-{row_count - 1}"
    if array_concurrency is not None:
        if array_concurrency <= 0:
            raise SchedulerError("--array-concurrency must be positive")
        array += f"%{array_concurrency}"
    lines = [
        "#!/usr/bin/env bash",
        "# Owner-editable scheduler policy: update the directives below before submission.",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --array={array}",
    ]
    lines.extend(
        f"#SBATCH --{key}" if value == "" else f"#SBATCH --{key}={value}"
        for key, value in sbatch_options
    )
    lines.extend(
        [
            "",
            # Slurm stops parsing directives after the first non-comment shell
            # command, so strict shell mode belongs below every #SBATCH line.
            "set -euo pipefail",
            "export OMP_NUM_THREADS=1",
            "export OPENBLAS_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            "",
            f"command_table={shlex.quote(str(command_table))}",
            f"python_exec={shlex.quote(str(python))}",
            f"expected_command_table_sha256={command_table_sha256}",
            f"expected_selected_manifest_sha256={selected_manifest_sha256}",
            f"obsnum_allowlist={shlex.quote(str(obsnum_allowlist))}",
            f"expected_obsnum_allowlist_sha256={obsnum_allowlist_sha256}",
            f"expected_protocol_sha256={protocol_sha256}",
            'array_index="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"',
            '"${python_exec}" - "${command_table}" "${array_index}" "${expected_command_table_sha256}" "${expected_selected_manifest_sha256}" "${obsnum_allowlist}" "${expected_obsnum_allowlist_sha256}" "${expected_protocol_sha256}" <<\'PY\'',
            "import csv",
            "import hashlib",
            "import json",
            "import subprocess",
            "import sys",
            "",
            "table_path = sys.argv[1]",
            "array_index = int(sys.argv[2])",
            "expected_table_sha = sys.argv[3]",
            "expected_manifest_sha = sys.argv[4]",
            "allowlist_path = sys.argv[5]",
            "expected_allowlist_sha = sys.argv[6]",
            "expected_protocol_sha = sys.argv[7]",
            "def sha256(path):",
            "    digest = hashlib.sha256()",
            "    with open(path, 'rb') as stream:",
            "        for block in iter(lambda: stream.read(1024 * 1024), b''):",
            "            digest.update(block)",
            "    return digest.hexdigest()",
            "if sha256(table_path) != expected_table_sha:",
            "    raise SystemExit('command-table SHA-256 changed after array generation')",
            "with open(table_path, encoding=\"utf-8\", newline=\"\") as stream:",
            "    matches = [",
            "        row for row in csv.DictReader(stream)",
            "        if int(row[\"array_index\"]) == array_index",
            "    ]",
            "if len(matches) != 1:",
            "    raise SystemExit(f\"array index {array_index} resolved to {len(matches)} commands\")",
            "argv = json.loads(matches[0][\"argv_json\"])",
            "if not isinstance(argv, list) or not all(isinstance(value, str) for value in argv):",
            "    raise SystemExit(\"invalid argv_json in command table\")",
            "manifest_path = argv[argv.index('--manifest') + 1]",
            "protocol_path = argv[argv.index('--protocol') + 1]",
            "if sha256(manifest_path) != expected_manifest_sha:",
            "    raise SystemExit('selected-manifest SHA-256 changed after array generation')",
            "if sha256(allowlist_path) != expected_allowlist_sha:",
            "    raise SystemExit('ObsNum allowlist SHA-256 changed after array generation')",
            "if sha256(protocol_path) != expected_protocol_sha:",
            "    raise SystemExit('analysis-protocol SHA-256 changed after array generation')",
            "subprocess.run(argv, check=True)",
            "PY",
            "",
        ]
    )
    return "\n".join(lines)


def render_command_table(rows: list[dict[str, Any]]) -> str:
    fields = [
        "schema_version",
        "array_index",
        "candidate_id",
        "map_id",
        "observation_number",
        "analysis_role",
        "duplicate_group_id",
        "session_id",
        "argv_json",
        "display_command",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def render_serial_script(
    *,
    command_table: Path,
    python: Path,
    command_table_sha256: str,
    selected_manifest_sha256: str,
    obsnum_allowlist: Path,
    obsnum_allowlist_sha256: str,
    protocol_sha256: str,
) -> str:
    """Render a checksum-bound owner serial runner for the same command table."""

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "export OMP_NUM_THREADS=1",
        "export OPENBLAS_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
        "",
        f"command_table={shlex.quote(str(command_table))}",
        f"python_exec={shlex.quote(str(python))}",
        f"expected_command_table_sha256={command_table_sha256}",
        f"expected_selected_manifest_sha256={selected_manifest_sha256}",
        f"obsnum_allowlist={shlex.quote(str(obsnum_allowlist))}",
        f"expected_obsnum_allowlist_sha256={obsnum_allowlist_sha256}",
        f"expected_protocol_sha256={protocol_sha256}",
        '"${python_exec}" - "${command_table}" "${expected_command_table_sha256}" "${expected_selected_manifest_sha256}" "${obsnum_allowlist}" "${expected_obsnum_allowlist_sha256}" "${expected_protocol_sha256}" <<\'PY\'',
        "import csv",
        "import hashlib",
        "import json",
        "import subprocess",
        "import sys",
        "",
        "table_path, expected_table_sha, expected_manifest_sha, allowlist_path, expected_allowlist_sha, expected_protocol_sha = sys.argv[1:]",
        "def sha256(path):",
        "    digest = hashlib.sha256()",
        "    with open(path, 'rb') as stream:",
        "        for block in iter(lambda: stream.read(1024 * 1024), b''):",
        "            digest.update(block)",
        "    return digest.hexdigest()",
        "if sha256(table_path) != expected_table_sha:",
        "    raise SystemExit('command-table SHA-256 changed after serial generation')",
        "if sha256(allowlist_path) != expected_allowlist_sha:",
        "    raise SystemExit('ObsNum allowlist SHA-256 changed after serial generation')",
        "with open(table_path, encoding='utf-8', newline='') as stream:",
        "    rows = list(csv.DictReader(stream))",
        "for expected_index, row in enumerate(rows):",
        "    if int(row['array_index']) != expected_index:",
        "        raise SystemExit('command table array indices are not deterministic')",
        "    argv = json.loads(row['argv_json'])",
        "    manifest_path = argv[argv.index('--manifest') + 1]",
        "    protocol_path = argv[argv.index('--protocol') + 1]",
        "    if sha256(manifest_path) != expected_manifest_sha:",
        "        raise SystemExit('selected-manifest SHA-256 changed after serial generation')",
        "    if sha256(protocol_path) != expected_protocol_sha:",
        "        raise SystemExit('analysis-protocol SHA-256 changed after serial generation')",
        "    subprocess.run(argv, check=True)",
        "PY",
        "",
    ]
    return "\n".join(lines)


def write_command_table(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--output-script", type=Path, required=True)
    parser.add_argument(
        "--serial-script", type=Path,
        help="Also generate a checksum-bound serial execution script.",
    )
    parser.add_argument("--command-table", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path(__file__).resolve().parent / "run_sci_align_001_3c273_beammap.py",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--job-name", default="sci-align-001-3c273")
    parser.add_argument("--array-concurrency", type=int, default=8)
    parser.add_argument(
        "--sbatch-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Optional override of the standard time/memory/CPU/node/task/partition "
            "directives. An account directive is intentionally not accepted."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = resolved(args.selected_manifest)
    script_path = resolved(args.output_script)
    serial_script = resolved(args.serial_script) if args.serial_script else None
    command_table = resolved(
        args.command_table
        if args.command_table is not None
        else script_path.with_suffix(".commands.csv")
    )
    output_root = resolved(args.output_root)
    # Preserve an explicitly selected virtual-environment interpreter instead
    # of resolving its symlink to a host-specific system Python path.
    runner = absolute_preserving_symlink(args.runner)
    python = resolve_python_executable(args.python)
    protocol = resolved(args.protocol)
    if not protocol.is_file():
        raise SchedulerError(f"analysis protocol does not exist: {protocol}")
    manifest = load_selected_manifest(manifest_path)
    validate_output_isolation(
        manifest["rows"],
        [script_path, command_table, output_root, *([serial_script] if serial_script else [])],
        [manifest_path, protocol, resolved(runner), resolved(python)],
    )
    options = parse_sbatch_options(args.sbatch_option)
    rows = command_rows(
        manifest,
        python=python,
        runner=runner,
        protocol=protocol,
        selected_manifest=manifest_path,
        output_root=output_root,
        resume=args.resume,
    )
    command_text = render_command_table(rows)
    command_sha = hashlib.sha256(command_text.encode("utf-8")).hexdigest()
    script = render_script(
        command_table=command_table,
        python=python,
        row_count=len(rows),
        job_name=args.job_name,
        array_concurrency=args.array_concurrency,
        sbatch_options=options,
        command_table_sha256=command_sha,
        selected_manifest_sha256=sha256_file(manifest_path),
        obsnum_allowlist=manifest_path.parent / str(manifest["obsnum_allowlist_filename"]),
        obsnum_allowlist_sha256=str(manifest["obsnum_allowlist_sha256"]),
        protocol_sha256=sha256_file(protocol),
    )
    serial = render_serial_script(
        command_table=command_table,
        python=python,
        command_table_sha256=command_sha,
        selected_manifest_sha256=sha256_file(manifest_path),
        obsnum_allowlist=manifest_path.parent / str(manifest["obsnum_allowlist_filename"]),
        obsnum_allowlist_sha256=str(manifest["obsnum_allowlist_sha256"]),
        protocol_sha256=sha256_file(protocol),
    ) if serial_script else None
    if args.dry_run:
        print(script, end="")
        print("# command table")
        for row in rows:
            print(f"# {row['array_index']}: {row['display_command']}")
        return 0
    script_path.parent.mkdir(parents=True, exist_ok=True)
    write_command_table(command_table, command_text)
    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(0o755)
    if serial_script is not None and serial is not None:
        serial_script.parent.mkdir(parents=True, exist_ok=True)
        serial_script.write_text(serial, encoding="utf-8")
        serial_script.chmod(0o755)
    print(
        f"Slurm array generated: rows={len(rows)} script={script_path} "
        f"commands={command_table}"
        + (f" serial={serial_script}" if serial_script else "")
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SchedulerError as error:
        print(f"scheduler generation failed: {error}", file=sys.stderr)
        raise SystemExit(2)
