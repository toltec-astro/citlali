#!/usr/bin/env python3
"""Create a compact JSON manifest for Citlali reduction outputs.

The manifest is intended for behavior-preserving refactor validation. It
records stable file metadata, SHA-256 checksums, and best-effort structured
summaries for FITS, netCDF, table, and log products.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fnmatch
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - depends on validation environment
    np = None  # type: ignore[assignment]


SCHEMA_VERSION = "citlali-output-manifest-v1"
HASH_BLOCK_SIZE = 1024 * 1024

FITS_HEADER_EXACT_KEYS = {
    "EXTNAME",
    "BUNIT",
    "OBJECT",
    "SOURCE",
    "PROJID",
    "OBSGOAL",
    "OBSID",
    "OBSNUM",
    "DATE-OBS",
    "TELESCOP",
    "INSTRUME",
    "RADECSYS",
    "RADESYS",
    "EQUINOX",
}
FITS_HEADER_PREFIXES = (
    "NAXIS",
    "CTYPE",
    "CUNIT",
    "CRPIX",
    "CRVAL",
    "CDELT",
    "CROTA",
    "CD",
    "PC",
)
NETCDF_ATTR_KEYS = {
    "units",
    "long_name",
    "standard_name",
    "description",
    "axis",
    "calendar",
    "_FillValue",
    "missing_value",
    "scale_factor",
    "add_offset",
    "valid_min",
    "valid_max",
}
LOG_WARNING_RE = re.compile(r"\bwarn(?:ing)?\b", re.IGNORECASE)
LOG_ERROR_RE = re.compile(r"\b(?:error|failed|failure|traceback)\b", re.IGNORECASE)
LOG_CRITICAL_RE = re.compile(r"\b(?:critical|fatal|segmentation fault|abort)\b", re.IGNORECASE)


def utc_now() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(HASH_BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def json_scalar(value: Any) -> Any:
    """Convert common scientific scalar types to strict JSON values."""
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if np is not None:
        if isinstance(value, np.generic):
            return json_scalar(value.item())
    try:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
    except Exception:
        pass
    return str(value)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return json_scalar(value)


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    return result if math.isfinite(result) else None


def classify_file(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name.endswith((".fits", ".fit", ".fits.gz", ".fit.gz")):
        return "fits"
    if suffix in {".nc", ".nc4", ".cdf"}:
        return "netcdf"
    if suffix == ".ecsv":
        return "ecsv"
    if suffix == ".csv":
        return "csv"
    if suffix in {".log", ".out", ".err"} or "log" in name:
        return "log"
    if suffix in {".yaml", ".yml", ".json", ".txt", ".md"}:
        return "text"
    return "other"


def should_exclude(path: Path, root: Path, patterns: list[str]) -> bool:
    rel = path.relative_to(root).as_posix()
    return any(fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(path.name, pattern) for pattern in patterns)


def file_record(path: Path, root: Path, include_hash: bool) -> dict[str, Any]:
    stat = path.stat()
    record: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "kind": classify_file(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_hash:
        record["sha256"] = sha256_file(path)
    return record


def array_digest(array: Any) -> str | None:
    if np is None:
        return None
    try:
        arr = np.ascontiguousarray(np.asarray(array))
        return hashlib.sha256(arr.view(np.uint8)).hexdigest()
    except Exception:
        return None


def summarize_numeric_array(array: Any, max_elements: int) -> dict[str, Any]:
    if np is None:
        return {"summary_warning": "numpy is not available"}

    try:
        arr = np.asanyarray(array)
    except Exception as exc:
        return {"summary_error": str(exc)}

    result: dict[str, Any] = {
        "shape": [int(v) for v in arr.shape],
        "dtype": str(arr.dtype),
        "n_elements": int(arr.size),
    }

    if arr.size == 0:
        result["empty"] = True
        return result

    if max_elements > 0 and arr.size > max_elements:
        result["summary_skipped"] = (
            f"array has {arr.size} elements, above --max-array-elements={max_elements}"
        )
        return result

    if arr.dtype.kind not in "biufc?":
        result["summary_skipped"] = f"non-numeric dtype {arr.dtype}"
        return result

    mask_count = 0
    if np.ma.isMaskedArray(arr):
        mask_count = int(np.ma.count_masked(arr))
        fill_value = np.nan if arr.dtype.kind in "fc" else 0
        arr = np.ma.filled(arr, fill_value=fill_value)
    result["masked_count"] = mask_count

    result["data_sha256"] = array_digest(arr)

    if arr.dtype.kind == "?":
        bool_values = np.asarray(arr, dtype=bool)
        result["true_count"] = int(np.count_nonzero(bool_values))
        result["false_count"] = int(bool_values.size - np.count_nonzero(bool_values))
        return result

    if arr.dtype.kind == "c":
        values = np.abs(np.asarray(arr))
        result["complex_summary"] = "absolute_value"
    else:
        values = np.asarray(arr, dtype=float)

    finite = np.isfinite(values)
    result["finite_count"] = int(np.count_nonzero(finite))
    result["nan_count"] = int(np.count_nonzero(np.isnan(values)))
    result["posinf_count"] = int(np.count_nonzero(np.isposinf(values)))
    result["neginf_count"] = int(np.count_nonzero(np.isneginf(values)))

    if result["finite_count"] == 0:
        return result

    finite_values = values[finite]
    median = float(np.median(finite_values))
    result.update(
        {
            "min": float(np.min(finite_values)),
            "max": float(np.max(finite_values)),
            "sum": float(np.sum(finite_values, dtype=np.float64)),
            "mean": float(np.mean(finite_values, dtype=np.float64)),
            "std": float(np.std(finite_values, dtype=np.float64)),
            "median": median,
            "mad": float(np.median(np.abs(finite_values - median))),
        }
    )
    return json_ready(result)


def selected_fits_header(header: Any) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for key in header.keys():
        key_str = str(key)
        if key_str in FITS_HEADER_EXACT_KEYS or key_str.startswith(FITS_HEADER_PREFIXES):
            selected[key_str] = json_scalar(header.get(key))
    return selected


def summarize_fits(path: Path, max_array_elements: int) -> dict[str, Any]:
    try:
        from astropy.io import fits
    except Exception as exc:  # pragma: no cover - depends on validation environment
        return {"summary_warning": f"astropy.io.fits is not available: {exc}"}

    try:
        with fits.open(path, memmap=False) as hdul:
            hdus: list[dict[str, Any]] = []
            for index, hdu in enumerate(hdul):
                hdu_record: dict[str, Any] = {
                    "index": index,
                    "name": str(getattr(hdu, "name", "")),
                    "class": type(hdu).__name__,
                    "header": selected_fits_header(hdu.header),
                }
                data = getattr(hdu, "data", None)
                if data is not None:
                    hdu_record["data"] = summarize_numeric_array(data, max_array_elements)
                hdus.append(json_ready(hdu_record))
            return {"format": "fits", "hdu_count": len(hdus), "hdus": hdus}
    except Exception as exc:
        return {"summary_error": str(exc)}


def netcdf_attr_value(obj: Any, name: str) -> Any:
    try:
        return json_scalar(getattr(obj, name))
    except Exception:
        try:
            return json_scalar(obj.getncattr(name))
        except Exception:
            return None


def summarize_netcdf(path: Path, max_array_elements: int) -> dict[str, Any]:
    try:
        import netCDF4
    except Exception as exc:  # pragma: no cover - depends on validation environment
        return {"summary_warning": f"netCDF4 is not available: {exc}"}

    try:
        with netCDF4.Dataset(path) as ds:
            dimensions = {
                name: {
                    "size": int(len(dim)),
                    "unlimited": bool(dim.isunlimited()),
                }
                for name, dim in sorted(ds.dimensions.items())
            }
            global_attrs = {
                name: netcdf_attr_value(ds, name)
                for name in sorted(ds.ncattrs())
                if name in NETCDF_ATTR_KEYS or name.upper() in FITS_HEADER_EXACT_KEYS
            }
            variables: list[dict[str, Any]] = []
            for name in sorted(ds.variables):
                var = ds.variables[name]
                var_record: dict[str, Any] = {
                    "name": name,
                    "dimensions": [str(dim) for dim in var.dimensions],
                    "shape": [int(v) for v in var.shape],
                    "dtype": str(var.dtype),
                    "attrs": {
                        attr: netcdf_attr_value(var, attr)
                        for attr in sorted(var.ncattrs())
                        if attr in NETCDF_ATTR_KEYS
                    },
                }
                dtype_kind = getattr(var.dtype, "kind", "")
                if dtype_kind in "biufc?":
                    n_elements = math.prod(var.shape) if var.shape else 1
                    if max_array_elements <= 0 or n_elements <= max_array_elements:
                        var_record["data"] = summarize_numeric_array(var[:], max_array_elements)
                    else:
                        var_record["data"] = {
                            "shape": [int(v) for v in var.shape],
                            "dtype": str(var.dtype),
                            "n_elements": int(n_elements),
                            "summary_skipped": (
                                f"variable has {n_elements} elements, above "
                                f"--max-array-elements={max_array_elements}"
                            ),
                        }
                variables.append(json_ready(var_record))
            return {
                "format": "netcdf",
                "dimensions": dimensions,
                "global_attrs": json_ready(global_attrs),
                "variable_count": len(variables),
                "variables": variables,
            }
    except Exception as exc:
        return {"summary_error": str(exc)}


def summarize_astropy_table(path: Path, max_table_rows: int, max_array_elements: int) -> dict[str, Any] | None:
    try:
        from astropy.table import Table
    except Exception:
        return None

    try:
        table = Table.read(path, format="ascii.ecsv")
    except Exception:
        return None

    result: dict[str, Any] = {
        "format": "ecsv",
        "row_count": int(len(table)),
        "column_names": [str(name) for name in table.colnames],
        "columns": [],
    }
    if max_table_rows > 0 and len(table) > max_table_rows:
        result["summary_skipped"] = (
            f"table has {len(table)} rows, above --max-table-rows={max_table_rows}"
        )
        return json_ready(result)

    for name in table.colnames:
        column = table[name]
        col_record: dict[str, Any] = {
            "name": str(name),
            "dtype": str(getattr(column, "dtype", "")),
        }
        if np is not None and getattr(column, "dtype", None) is not None and column.dtype.kind in "biufc?":
            col_record["data"] = summarize_numeric_array(column, max_array_elements)
        result["columns"].append(json_ready(col_record))
    return json_ready(result)


def summarize_csv(path: Path, max_table_rows: int) -> dict[str, Any]:
    row_count = 0
    columns: dict[str, list[float]] = {}
    nonnumeric: set[str] = set()
    fieldnames: list[str] = []
    truncated = False

    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = [str(name) for name in (reader.fieldnames or [])]
            columns = {name: [] for name in fieldnames}
            for row in reader:
                row_count += 1
                if max_table_rows > 0 and row_count > max_table_rows:
                    truncated = True
                    break
                for name in fieldnames:
                    value = (row.get(name) or "").strip()
                    if not value:
                        continue
                    parsed = safe_float(value)
                    if parsed is None:
                        nonnumeric.add(name)
                    elif name not in nonnumeric:
                        columns[name].append(parsed)
    except Exception as exc:
        return {"summary_error": str(exc)}

    column_records: list[dict[str, Any]] = []
    for name in fieldnames:
        values = columns.get(name, [])
        record: dict[str, Any] = {"name": name}
        if name not in nonnumeric and values:
            record["numeric"] = numeric_list_summary(values)
        column_records.append(json_ready(record))

    result: dict[str, Any] = {
        "format": "csv",
        "row_count": row_count,
        "column_names": fieldnames,
        "columns": column_records,
    }
    if truncated:
        result["summary_skipped"] = (
            f"table exceeded --max-table-rows={max_table_rows}; numeric stats use first {max_table_rows} rows"
        )
    return json_ready(result)


def numeric_list_summary(values: list[float]) -> dict[str, Any]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return {"finite_count": 0, "n_values": len(values)}
    finite_sorted = sorted(finite)
    n = len(finite_sorted)
    median = finite_sorted[n // 2] if n % 2 else 0.5 * (finite_sorted[n // 2 - 1] + finite_sorted[n // 2])
    mean = sum(finite_sorted) / n
    variance = sum((v - mean) ** 2 for v in finite_sorted) / n
    return {
        "n_values": len(values),
        "finite_count": n,
        "min": finite_sorted[0],
        "max": finite_sorted[-1],
        "sum": sum(finite_sorted),
        "mean": mean,
        "std": math.sqrt(variance),
        "median": median,
        "mad": numeric_median([abs(v - median) for v in finite_sorted]),
    }


def numeric_median(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    n = len(values)
    return values[n // 2] if n % 2 else 0.5 * (values[n // 2 - 1] + values[n // 2])


def summarize_table(path: Path, kind: str, max_table_rows: int, max_array_elements: int) -> dict[str, Any]:
    if kind == "ecsv":
        astropy_summary = summarize_astropy_table(path, max_table_rows, max_array_elements)
        if astropy_summary is not None:
            return astropy_summary
        summary = summarize_text_or_log(path, "text")
        summary["format"] = "ecsv"
        summary["summary_warning"] = (
            "astropy.table is not available or could not parse this ECSV; "
            "structured table columns were not summarized"
        )
        return summary
    return summarize_csv(path, max_table_rows)


def summarize_text_or_log(path: Path, kind: str) -> dict[str, Any]:
    line_count = 0
    warning_count = 0
    error_count = 0
    critical_count = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line_count += 1
                if LOG_WARNING_RE.search(line):
                    warning_count += 1
                if LOG_ERROR_RE.search(line):
                    error_count += 1
                if LOG_CRITICAL_RE.search(line):
                    critical_count += 1
    except Exception as exc:
        return {"summary_error": str(exc)}

    result: dict[str, Any] = {
        "format": kind,
        "line_count": line_count,
    }
    if kind == "log":
        result.update(
            {
                "warning_count": warning_count,
                "error_count": error_count,
                "critical_count": critical_count,
            }
        )
    return result


def summarize_file(path: Path, kind: str, max_array_elements: int, max_table_rows: int) -> dict[str, Any]:
    if kind == "fits":
        return summarize_fits(path, max_array_elements)
    if kind == "netcdf":
        return summarize_netcdf(path, max_array_elements)
    if kind in {"csv", "ecsv"}:
        return summarize_table(path, kind, max_table_rows, max_array_elements)
    if kind in {"log", "text"}:
        return summarize_text_or_log(path, kind)
    return {}


def summarize_config_file(path: Path, include_hash: bool) -> dict[str, Any]:
    expanded = path.expanduser()
    record: dict[str, Any] = {
        "path": str(expanded),
        "exists": expanded.is_file(),
    }
    if expanded.is_file():
        stat = expanded.stat()
        record["size_bytes"] = stat.st_size
        if include_hash:
            record["sha256"] = sha256_file(expanded)
    return record


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    files: list[dict[str, Any]] = []
    warnings: list[str] = []

    paths = sorted(path for path in root.rglob("*") if path.is_file())
    for path in paths:
        if should_exclude(path, root, args.exclude):
            continue
        record = file_record(path, root, include_hash=not args.skip_content_hash)
        summary = summarize_file(path, record["kind"], args.max_array_elements, args.max_table_rows)
        if "summary_warning" in summary:
            warnings.append(f"{record['path']}: {summary['summary_warning']}")
        if "summary_error" in summary:
            warnings.append(f"{record['path']}: {summary['summary_error']}")
        if summary:
            record["summary"] = summary
        files.append(json_ready(record))

    by_kind = Counter(record["kind"] for record in files)
    total_size = sum(int(record["size_bytes"]) for record in files)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": utc_now(),
        "run": {
            "case": args.case,
            "description": args.description,
            "output_dir": str(root),
            "git_sha": args.git_sha,
            "branch": args.branch,
            "command": args.command,
            "n_threads": args.n_threads,
            "parallel_policy": args.parallel_policy,
            "wall_time_sec": args.wall_time_sec,
            "peak_rss_kb": args.peak_rss_kb,
            "environment": args.environment_note,
        },
        "inputs": {
            "config_files": [
                summarize_config_file(Path(config_file), include_hash=not args.skip_content_hash)
                for config_file in args.config_file
            ],
        },
        "aggregate": {
            "file_count": len(files),
            "total_size_bytes": total_size,
            "by_kind": dict(sorted(by_kind.items())),
        },
        "files": files,
        "tool_warnings": warnings,
    }
    return json_ready(manifest)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, help="Short baseline case name.")
    parser.add_argument("--output-dir", required=True, help="Reduction output directory to summarize.")
    parser.add_argument("--manifest-out", required=True, help="Path to write JSON manifest.")
    parser.add_argument("--description", default="", help="Human-readable case description.")
    parser.add_argument("--git-sha", default="", help="Citlali git SHA used for the reduction.")
    parser.add_argument("--branch", default="", help="Citlali branch used for the reduction.")
    parser.add_argument("--command", default="", help="Exact reduction command.")
    parser.add_argument("--config-file", action="append", default=[], help="Config file used by the reduction.")
    parser.add_argument("--environment-note", default="", help="Module/host/environment note.")
    parser.add_argument("--n-threads", type=int, default=None, help="Number of Citlali threads.")
    parser.add_argument("--parallel-policy", default="", help="Citlali parallel policy.")
    parser.add_argument("--wall-time-sec", type=float, default=None, help="Measured wall time in seconds.")
    parser.add_argument("--peak-rss-kb", type=int, default=None, help="Measured peak RSS in KB.")
    parser.add_argument(
        "--max-array-elements",
        type=int,
        default=5_000_000,
        help="Maximum array elements to summarize per variable/HDU; 0 means no limit.",
    )
    parser.add_argument(
        "--max-table-rows",
        type=int,
        default=500_000,
        help="Maximum table rows to scan for numeric table stats; 0 means no limit.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob pattern to exclude, matched against relative path or basename.",
    )
    parser.add_argument(
        "--skip-content-hash",
        action="store_true",
        help="Skip SHA-256 content hashes for faster metadata-only manifests.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    manifest = build_manifest(args)
    out_path = Path(args.manifest_out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    warning_count = len(manifest.get("tool_warnings", []))
    print(
        f"wrote {out_path} with {manifest['aggregate']['file_count']} files "
        f"({warning_count} tool warnings)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
