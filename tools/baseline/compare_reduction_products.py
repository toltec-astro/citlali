#!/usr/bin/env python3
"""Compare numeric products from two Citlali reduction directories.

By default this is a validation triage tool. It finds matching FITS, netCDF,
CSV, and ECSV products, reports product-set differences, and ranks numeric
array/column differences by size. ``--strict`` turns the same comparison into
an acceptance gate that fails on missing, extra, changed, or skipped items.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from astropy.io import fits
    from astropy.table import Table
except Exception:  # pragma: no cover - depends on validation environment
    fits = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]

try:
    import netCDF4
except Exception:  # pragma: no cover - depends on validation environment
    netCDF4 = None  # type: ignore[assignment]


REDU_RE = re.compile(r"^redu(\d+)$")
PRODUCT_SUFFIXES = {".fits", ".fit", ".nc", ".nc4", ".cdf", ".csv", ".ecsv"}
NUMERIC_KINDS = {"b", "i", "u", "f", "c", "?"}
TIMESTREAM_PATTERNS = ("*_timestream.nc", "*_rtc_timestream.nc", "*_ptc_timestream.nc")
DEFAULT_EXCLUDE = (
    "*.log",
    "*.log.gz",
    "*.out",
    "*.err",
    "*.yaml",
    "*.yml",
    "*.json",
    "learning_iter_*.csv",
    "*/logs/*",
    "index.yaml",
)
FITS_HEADER_KEYS = {
    "EXTNAME",
    "BUNIT",
    "OBJECT",
    "SOURCE",
    "PROJID",
    "OBSID",
    "OBSNUM",
    "DATE-OBS",
}


@dataclass(frozen=True)
class Product:
    relpath: str
    path: Path
    kind: str
    size_bytes: int


def classify_product(path: Path) -> str:
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
    return "other"


def is_numeric_kind(kind: str) -> bool:
    return kind in NUMERIC_KINDS


def redu_number(path: Path) -> int | None:
    match = REDU_RE.match(path.name)
    return int(match.group(1)) if match else None


def find_latest_redu(reduced_root: Path) -> Path:
    candidates = [
        child
        for child in reduced_root.iterdir()
        if child.is_dir() and redu_number(child) is not None
    ]
    if not candidates:
        raise FileNotFoundError(f"no reduNN directories under {reduced_root}")
    return max(candidates, key=lambda path: redu_number(path) or -1)


def contains_comparable_products(path: Path) -> bool:
    for child in path.rglob("*"):
        if child.is_file() and classify_product(child) != "other":
            return True
    return False


def resolve_redu_path(path: Path, redu: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise NotADirectoryError(path)
    if redu_number(path) is not None:
        return path
    if redu == "latest":
        try:
            return find_latest_redu(path)
        except FileNotFoundError:
            if contains_comparable_products(path):
                return path
            raise
    child = path / redu
    if not child.is_dir():
        raise NotADirectoryError(child)
    if redu_number(child) is None:
        raise ValueError(f"{child} is not named reduNN")
    return child.resolve()


def resolve_from_base_root(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.mode == "auto":
        raise ValueError("--mode is required with --base-root")
    root = Path(args.base_root).expanduser().resolve()
    baseline_root = root / args.mode / args.baseline_label / "reduced"
    candidate_root = root / args.mode / args.candidate_label / "reduced"
    return (
        resolve_redu_path(baseline_root, args.baseline_redu),
        resolve_redu_path(candidate_root, args.candidate_redu),
    )


def path_matches(path: str, patterns: list[str] | tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(Path(path).name, pattern) for pattern in patterns)


def collect_products(root: Path, args: argparse.Namespace) -> dict[str, Product]:
    products: dict[str, Product] = {}
    include = list(args.include)
    exclude = list(DEFAULT_EXCLUDE) + list(args.exclude)
    if not args.include_timestream:
        exclude.extend(TIMESTREAM_PATTERNS)

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relpath = path.relative_to(root).as_posix()
        if include and not path_matches(relpath, include):
            continue
        if exclude and path_matches(relpath, exclude):
            continue
        kind = classify_product(path)
        if kind == "other":
            continue
        products[relpath] = Product(
            relpath=relpath,
            path=path,
            kind=kind,
            size_bytes=path.stat().st_size,
        )
    return products


def finite_array(values: Any) -> np.ndarray:
    array = np.asanyarray(values)
    if np.ma.isMaskedArray(array):
        fill = np.nan if array.dtype.kind in "fc" else 0
        array = np.ma.filled(array, fill_value=fill)
    if array.dtype.kind == "c":
        array = np.abs(array)
    elif array.dtype.kind == "?":
        array = array.astype(np.int8)
    else:
        array = array.astype(np.float64, copy=False)
    return np.asarray(array)


def numeric_diff(
    baseline: Any,
    candidate: Any,
    *,
    product: str,
    item: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    base = finite_array(baseline)
    cand = finite_array(candidate)
    result: dict[str, Any] = {
        "product": product,
        "item": item,
        "shape_baseline": list(base.shape),
        "shape_candidate": list(cand.shape),
    }
    if base.shape != cand.shape:
        result["status"] = "shape_changed"
        return result
    n_elements = int(base.size)
    result["n_elements"] = n_elements
    if args.max_array_elements > 0 and n_elements > args.max_array_elements:
        result["status"] = "skipped_large_array"
        return result

    base_finite = np.isfinite(base)
    cand_finite = np.isfinite(cand)
    both = base_finite & cand_finite
    result["finite_baseline"] = int(np.count_nonzero(base_finite))
    result["finite_candidate"] = int(np.count_nonzero(cand_finite))
    result["finite_both"] = int(np.count_nonzero(both))
    result["finite_mismatch"] = int(np.count_nonzero(base_finite ^ cand_finite))
    if result["finite_both"] == 0:
        result["status"] = (
            "within_tolerance"
            if result["finite_baseline"] == result["finite_candidate"] and result["finite_mismatch"] == 0
            else "no_common_finite_values"
        )
        return result

    delta = cand[both] - base[both]
    abs_delta = np.abs(delta)
    scale = np.maximum(np.abs(base[both]), args.frac_floor)
    frac = abs_delta / scale
    tol = args.atol + args.rtol * np.abs(base[both])
    result.update(
        {
            "status": "different" if bool(np.any(abs_delta > tol)) else "within_tolerance",
            "max_abs_diff": float(np.max(abs_delta)),
            "median_abs_diff": float(np.median(abs_delta)),
            "mean_abs_diff": float(np.mean(abs_delta, dtype=np.float64)),
            "rms_abs_diff": float(np.sqrt(np.mean(delta * delta, dtype=np.float64))),
            "max_frac_diff": float(np.max(frac)),
            "median_frac_diff": float(np.median(frac)),
        }
    )
    return result


def selected_header(header: Any) -> dict[str, Any]:
    return {key: header.get(key) for key in FITS_HEADER_KEYS if key in header}


def compare_fits(product: str, baseline: Path, candidate: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    if fits is None:
        return [{"product": product, "item": "FITS", "status": "skipped_missing_astropy"}]
    results: list[dict[str, Any]] = []
    try:
        with fits.open(baseline, memmap=True) as base_hdus, fits.open(candidate, memmap=True) as cand_hdus:
            if len(base_hdus) != len(cand_hdus):
                results.append(
                    {
                        "product": product,
                        "item": "HDU count",
                        "status": "hdu_count_changed",
                        "baseline": len(base_hdus),
                        "candidate": len(cand_hdus),
                    }
                )
            for index, (base_hdu, cand_hdu) in enumerate(zip(base_hdus, cand_hdus)):
                extname = str(base_hdu.header.get("EXTNAME", index))
                item = f"HDU {index}:{extname}"
                base_header = selected_header(base_hdu.header)
                cand_header = selected_header(cand_hdu.header)
                if base_header != cand_header:
                    results.append(
                        {
                            "product": product,
                            "item": f"{item} header",
                            "status": "header_changed",
                            "baseline": base_header,
                            "candidate": cand_header,
                        }
                    )
                if base_hdu.data is None and cand_hdu.data is None:
                    continue
                if base_hdu.data is None or cand_hdu.data is None:
                    results.append({"product": product, "item": item, "status": "data_presence_changed"})
                    continue
                if (
                    not is_numeric_kind(np.asanyarray(base_hdu.data).dtype.kind)
                    or not is_numeric_kind(np.asanyarray(cand_hdu.data).dtype.kind)
                ):
                    continue
                results.append(
                    numeric_diff(base_hdu.data, cand_hdu.data, product=product, item=item, args=args)
                )
    except Exception as exc:
        results.append({"product": product, "item": "FITS", "status": "error", "error": str(exc)})
    return results


def compare_netcdf(product: str, baseline: Path, candidate: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    if netCDF4 is None:
        return [{"product": product, "item": "netCDF", "status": "skipped_missing_netcdf4"}]
    results: list[dict[str, Any]] = []
    try:
        with netCDF4.Dataset(baseline) as base_ds, netCDF4.Dataset(candidate) as cand_ds:
            base_dims = {name: len(dim) for name, dim in base_ds.dimensions.items()}
            cand_dims = {name: len(dim) for name, dim in cand_ds.dimensions.items()}
            if base_dims != cand_dims:
                results.append(
                    {
                        "product": product,
                        "item": "dimensions",
                        "status": "dimensions_changed",
                        "baseline": base_dims,
                        "candidate": cand_dims,
                    }
                )
            base_vars = set(base_ds.variables)
            cand_vars = set(cand_ds.variables)
            for name in sorted(base_vars - cand_vars):
                results.append({"product": product, "item": name, "status": "missing_variable"})
            for name in sorted(cand_vars - base_vars):
                results.append({"product": product, "item": name, "status": "extra_variable"})
            for name in sorted(base_vars & cand_vars):
                base_var = base_ds.variables[name]
                cand_var = cand_ds.variables[name]
                if (
                    not is_numeric_kind(getattr(base_var.dtype, "kind", ""))
                    or not is_numeric_kind(getattr(cand_var.dtype, "kind", ""))
                ):
                    continue
                size = math.prod(base_var.shape) if base_var.shape else 1
                if args.max_array_elements > 0 and size > args.max_array_elements:
                    results.append(
                        {
                            "product": product,
                            "item": name,
                            "status": "skipped_large_array",
                            "shape_baseline": list(base_var.shape),
                            "shape_candidate": list(cand_var.shape),
                            "n_elements": int(size),
                        }
                    )
                    continue
                results.append(numeric_diff(base_var[:], cand_var[:], product=product, item=name, args=args))
    except Exception as exc:
        results.append({"product": product, "item": "netCDF", "status": "error", "error": str(exc)})
    return results


def table_numeric_columns(table: Any) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    for name in table.colnames:
        col = table[name]
        dtype = getattr(col, "dtype", None)
        if dtype is not None and is_numeric_kind(dtype.kind):
            columns[str(name)] = col
    return columns


def compare_table(product: str, baseline: Path, candidate: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    if Table is None:
        return [{"product": product, "item": "table", "status": "skipped_missing_astropy"}]
    results: list[dict[str, Any]] = []
    try:
        table_format = "ascii.ecsv" if baseline.suffix.lower() == ".ecsv" else "ascii.csv"
        base_table = Table.read(baseline, format=table_format)
        cand_table = Table.read(candidate, format=table_format)
        if len(base_table) != len(cand_table):
            results.append(
                {
                    "product": product,
                    "item": "rows",
                    "status": "row_count_changed",
                    "baseline": len(base_table),
                    "candidate": len(cand_table),
                }
            )
        base_cols = set(base_table.colnames)
        cand_cols = set(cand_table.colnames)
        for name in sorted(base_cols - cand_cols):
            results.append({"product": product, "item": name, "status": "missing_column"})
        for name in sorted(cand_cols - base_cols):
            results.append({"product": product, "item": name, "status": "extra_column"})
        base_numeric = table_numeric_columns(base_table)
        cand_numeric = table_numeric_columns(cand_table)
        for name in sorted(set(base_numeric) & set(cand_numeric)):
            results.append(numeric_diff(base_numeric[name], cand_numeric[name], product=product, item=name, args=args))
    except Exception as exc:
        results.append({"product": product, "item": "table", "status": "error", "error": str(exc)})
    return results


def compare_product(product: str, baseline: Product, candidate: Product, args: argparse.Namespace) -> list[dict[str, Any]]:
    if baseline.kind != candidate.kind:
        return [
            {
                "product": product,
                "item": "kind",
                "status": "kind_changed",
                "baseline": baseline.kind,
                "candidate": candidate.kind,
            }
        ]
    if baseline.kind == "fits":
        return compare_fits(product, baseline.path, candidate.path, args)
    if baseline.kind == "netcdf":
        return compare_netcdf(product, baseline.path, candidate.path, args)
    if baseline.kind in {"csv", "ecsv"}:
        return compare_table(product, baseline.path, candidate.path, args)
    return []


def diff_rank(record: dict[str, Any]) -> tuple[float, float, int]:
    return (
        float(record.get("max_frac_diff", 0.0) or 0.0),
        float(record.get("max_abs_diff", 0.0) or 0.0),
        int(record.get("finite_mismatch", 0) or 0),
    )


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "product",
        "item",
        "status",
        "n_elements",
        "finite_baseline",
        "finite_candidate",
        "finite_both",
        "finite_mismatch",
        "max_abs_diff",
        "median_abs_diff",
        "rms_abs_diff",
        "max_frac_diff",
        "median_frac_diff",
        "shape_baseline",
        "shape_candidate",
        "error",
    ]
    return {key: record[key] for key in keys if key in record}


def product_counts(products: dict[str, Product]) -> dict[str, int]:
    return dict(sorted(Counter(product.kind for product in products.values()).items()))


def build_comparison(args: argparse.Namespace) -> dict[str, Any]:
    if args.base_root:
        baseline_root, candidate_root = resolve_from_base_root(args)
    else:
        if not args.baseline or not args.candidate:
            raise ValueError("provide BASELINE and CANDIDATE paths or use --base-root")
        baseline_root = resolve_redu_path(Path(args.baseline), args.baseline_redu)
        candidate_root = resolve_redu_path(Path(args.candidate), args.candidate_redu)

    baseline_products = collect_products(baseline_root, args)
    candidate_products = collect_products(candidate_root, args)
    base_paths = set(baseline_products)
    cand_paths = set(candidate_products)

    records: list[dict[str, Any]] = []
    for relpath in sorted(base_paths & cand_paths):
        records.extend(compare_product(relpath, baseline_products[relpath], candidate_products[relpath], args))

    numeric_records = [
        record for record in records
        if "max_abs_diff" in record or record.get("status") == "shape_changed"
    ]
    changed_records = [
        record for record in records
        if record.get("status") not in {"within_tolerance", "skipped_large_array"}
    ]
    skipped_records = [record for record in records if str(record.get("status", "")).startswith("skipped")]
    top_numeric = sorted(numeric_records, key=diff_rank, reverse=True)[: args.top]

    return {
        "strict": bool(args.strict),
        "mode": args.mode,
        "baseline_root": str(baseline_root),
        "candidate_root": str(candidate_root),
        "baseline_counts": product_counts(baseline_products),
        "candidate_counts": product_counts(candidate_products),
        "common_product_count": len(base_paths & cand_paths),
        "missing_products": sorted(base_paths - cand_paths),
        "extra_products": sorted(cand_paths - base_paths),
        "record_count": len(records),
        "changed_record_count": len(changed_records),
        "skipped_record_count": len(skipped_records),
        "top_numeric": [compact_record(record) for record in top_numeric],
        "changed_records": [compact_record(record) for record in changed_records[: args.max_records]],
        "skipped_records": [compact_record(record) for record in skipped_records[: args.max_records]],
    }


def strict_exit_code(result: dict[str, Any]) -> int:
    if result["missing_products"] or result["extra_products"]:
        return 2
    if result["skipped_record_count"]:
        return 3
    if result["changed_record_count"]:
        return 4
    return 0


def fmt_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6g}"
    except Exception:
        return str(value)


def markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    out = []
    for idx, row in enumerate(rows):
        out.append("| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |")
        if idx == 0:
            out.append("| " + " | ".join("-" * widths[i] for i in range(len(row))) + " |")
    return "\n".join(out)


def render_markdown(result: dict[str, Any], top: int) -> str:
    lines = [
        "# Citlali Reduction Product Comparison",
        "",
        f"- Mode: `{result['mode']}`",
        f"- Baseline: `{result['baseline_root']}`",
        f"- Candidate: `{result['candidate_root']}`",
        f"- Strict gate: `{result['strict']}`",
        f"- Common products: {result['common_product_count']}",
        f"- Changed records: {result['changed_record_count']}",
        f"- Skipped records: {result['skipped_record_count']}",
        "",
        "## Product Counts",
        "",
        markdown_table(
            [
                ["kind", "baseline", "candidate"],
                *[
                    [
                        kind,
                        str(result["baseline_counts"].get(kind, 0)),
                        str(result["candidate_counts"].get(kind, 0)),
                    ]
                    for kind in sorted(set(result["baseline_counts"]) | set(result["candidate_counts"]))
                ],
            ]
        ),
        "",
        "## Product Set Differences",
        "",
    ]
    missing = result["missing_products"]
    extra = result["extra_products"]
    if not missing and not extra:
        lines.append("No missing or extra comparable products.")
    else:
        if missing:
            lines.append("Missing from candidate:")
            lines.extend(f"- `{path}`" for path in missing[:top])
            if len(missing) > top:
                lines.append(f"- ... {len(missing) - top} more")
        if extra:
            lines.append("Extra in candidate:")
            lines.extend(f"- `{path}`" for path in extra[:top])
            if len(extra) > top:
                lines.append(f"- ... {len(extra) - top} more")

    lines.extend(["", "## Largest Numeric Differences", ""])
    numeric_rows = [["product", "item", "status", "finite", "max abs", "med abs", "max frac"]]
    for record in result["top_numeric"][:top]:
        numeric_rows.append(
            [
                str(record.get("product", "")),
                str(record.get("item", "")),
                str(record.get("status", "")),
                str(record.get("finite_both", "")),
                fmt_float(record.get("max_abs_diff")),
                fmt_float(record.get("median_abs_diff")),
                fmt_float(record.get("max_frac_diff")),
            ]
        )
    if len(numeric_rows) == 1:
        lines.append("No comparable numeric arrays or columns were found.")
    else:
        lines.append(markdown_table(numeric_rows))

    if result["changed_records"]:
        lines.extend(["", "## Non-Tolerance Changes", ""])
        change_rows = [["product", "item", "status", "detail"]]
        for record in result["changed_records"][:top]:
            detail = ""
            if "error" in record:
                detail = str(record["error"])
            elif "shape_baseline" in record or "shape_candidate" in record:
                detail = f"{record.get('shape_baseline')} -> {record.get('shape_candidate')}"
            elif "finite_mismatch" in record:
                detail = f"finite mismatch {record.get('finite_mismatch')}"
            change_rows.append(
                [
                    str(record.get("product", "")),
                    str(record.get("item", "")),
                    str(record.get("status", "")),
                    detail,
                ]
            )
        lines.append(markdown_table(change_rows))

    if result["skipped_records"]:
        lines.extend(["", "## Skipped Numeric Items", ""])
        lines.extend(
            f"- `{record.get('product')}` `{record.get('item')}`: {record.get('status')}"
            for record in result["skipped_records"][:top]
        )
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", nargs="?", help="Baseline reduNN directory or reduced root.")
    parser.add_argument("candidate", nargs="?", help="Candidate reduNN directory or reduced root.")
    parser.add_argument(
        "--base-root",
        default="",
        help="Root containing <mode>/<label>/reduced trees, e.g. .../2026-refactor.",
    )
    parser.add_argument("--mode", default="auto", choices=["auto", "point", "beammap", "science"])
    parser.add_argument("--baseline-label", default="citlali")
    parser.add_argument("--candidate-label", default="refactor")
    parser.add_argument("--baseline-redu", default="latest", help="'latest' or an explicit reduNN.")
    parser.add_argument("--candidate-redu", default="latest", help="'latest' or an explicit reduNN.")
    parser.add_argument("--include", action="append", default=[], help="Comparable product glob to include.")
    parser.add_argument("--exclude", action="append", default=[], help="Product glob to exclude.")
    parser.add_argument("--include-timestream", action="store_true", help="Include *_timestream.nc products.")
    parser.add_argument("--max-array-elements", type=int, default=10_000_000)
    parser.add_argument("--frac-floor", type=float, default=1.0e-12)
    parser.add_argument("--atol", type=float, default=2.0e-8)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--max-records", type=int, default=200)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing/extra products and any changed or skipped comparison item.",
    )
    parser.add_argument("--json-out", default="", help="Optional path for machine-readable JSON.")
    parser.add_argument("--report-out", default="", help="Optional path for Markdown report.")
    return parser.parse_args(argv)


def write_text(path: str, text: str) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    result = build_comparison(args)
    report = render_markdown(result, args.top)
    if args.json_out:
        out = Path(args.json_out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    if args.report_out:
        write_text(args.report_out, report)
    print(report, end="")
    return strict_exit_code(result) if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
