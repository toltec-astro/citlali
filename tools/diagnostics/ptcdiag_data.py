#!/usr/bin/env python3
"""Shared ptcdiag loading helpers for engineering plots and reports."""

from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np


ARRAY_NAME_TO_ID = {"a1100": 0, "a1400": 1, "a2000": 2}


def _parse_int_list(spec: str) -> list[int]:
    values: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return values


def _parse_networks(spec: str) -> list[int] | None:
    spec = spec.strip().lower()
    if spec == "all":
        return None
    return sorted(set(_parse_int_list(spec)))


def filled(var: netCDF4.Variable, fill: float | int | None = None) -> np.ndarray:
    data = var[:]
    if np.ma.isMaskedArray(data):
        if fill is None:
            dtype = np.asarray(data).dtype
            if np.issubdtype(dtype, np.floating):
                fill = float("nan")
            else:
                fill = np.iinfo(np.int32).min
        data = np.ma.filled(data, fill_value=fill)
    return np.asarray(data)


def _find_ptcdiag(raw_dir: Path) -> Path | None:
    files = sorted(raw_dir.rglob("*_ptcdiag.nc"))
    return files[0] if files else None


def collect_products(redu_dir: Path) -> list[tuple[str, Path]]:
    products: list[tuple[str, Path]] = []
    for obsdir in sorted(redu_dir.iterdir()):
        if not obsdir.is_dir() or not obsdir.name.isdigit():
            continue
        raw_dir = obsdir / "raw"
        if not raw_dir.is_dir():
            continue
        product = _find_ptcdiag(raw_dir)
        if product is not None:
            products.append((obsdir.name, product))
    return products


def _network_ids(ds: netCDF4.Dataset) -> np.ndarray:
    for name in (
        "ptc_second_pass_network_ids",
        "corr_nw_network_ids",
        "weight_corr_penalty_network_ids",
        "weight_busy_row_suppression_network_ids",
        "adaptive_pca_network_ids",
    ):
        if name in ds.variables:
            return filled(ds.variables[name], fill=-2147483647).astype(int)
    return np.asarray([], dtype=int)


def _get_2d(ds: netCDF4.Dataset, name: str, fill: float | int) -> np.ndarray | None:
    var = ds.variables.get(name)
    if var is None:
        return None
    return filled(var, fill=fill)


def _nanmax(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return float("nan")
    return float(np.nanmax(arr))


def _nanmin(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return float("nan")
    return float(np.nanmin(arr))


def _nanmedian(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return float("nan")
    return float(np.nanmedian(arr))


def _ptc_severity(
    newly_flagged_fraction: float,
    corr_penalty_factor: float,
    busy_row_factor: float,
    top_event_score: float,
    proposed_flagged_fraction: float,
) -> float:
    terms: list[float] = []
    if np.isfinite(newly_flagged_fraction):
        terms.append(newly_flagged_fraction / 0.01)
    if np.isfinite(proposed_flagged_fraction):
        terms.append(proposed_flagged_fraction / 0.02)
    if np.isfinite(corr_penalty_factor):
        terms.append(max(0.0, 1.0 - corr_penalty_factor) / 0.25)
    if np.isfinite(busy_row_factor):
        terms.append(max(0.0, 1.0 - busy_row_factor) / 0.50)
    if np.isfinite(top_event_score):
        terms.append(top_event_score / 6.0)
    if not terms:
        return float("nan")
    return float(max(terms))


def load_reduction_tables(
    redu_dir: Path,
    array: str,
    networks_spec: str = "all",
    obsnums_spec: str = "all",
) -> dict[str, object]:
    redu_dir = Path(redu_dir).expanduser().resolve()
    if not redu_dir.is_dir():
        raise NotADirectoryError(redu_dir)

    array_id = ARRAY_NAME_TO_ID[array]
    obsnum_filter = None if obsnums_spec == "all" else set(_parse_int_list(obsnums_spec))
    requested_networks = _parse_networks(networks_spec)
    products = collect_products(redu_dir)
    if obsnum_filter is not None:
        products = [(obs, path) for obs, path in products if int(obs) in obsnum_filter]
    if not products:
        raise FileNotFoundError(f"no ptcdiag products found under {redu_dir}")

    obs_rows: list[dict[str, object]] = []
    scan_network_rows: list[dict[str, object]] = []
    discovered_networks: set[int] = set()
    product_paths: dict[str, str] = {}

    for obsnum, nc_file in products:
        with netCDF4.Dataset(nc_file) as ds:
            det_array = filled(ds.variables["ptc_diag_array"], fill=-2147483647).astype(int)
            det_network = filled(ds.variables["ptc_diag_network"], fill=-2147483647).astype(int)
            if not np.any(det_array == array_id):
                continue

            output_scan = filled(ds.variables["output_scan_index"], fill=-2147483647).astype(int)
            network_ids = _network_ids(ds)
            if network_ids.size == 0:
                continue
            array_networks = sorted(
                {
                    int(nw)
                    for nw in det_network[np.where(det_array == array_id)[0]].tolist()
                    if int(nw) >= 0
                }
            )
            if not array_networks:
                continue
            selected_networks = requested_networks
            if selected_networks is None:
                selected_networks = array_networks
            else:
                selected_networks = [int(nw) for nw in selected_networks if int(nw) in set(array_networks)]
            if not selected_networks:
                continue

            rows_before = len(scan_network_rows)
            newly_flagged = _get_2d(ds, "ptc_second_pass_newly_flagged_fraction", float("nan"))
            proposed_flagged = _get_2d(ds, "ptc_second_pass_proposed_flagged_fraction", float("nan"))
            top_event_score = _get_2d(ds, "ptc_second_pass_top_event_score", float("nan"))
            corr_penalty_factor = _get_2d(ds, "weight_corr_penalty_factor", float("nan"))
            busy_row_factor = _get_2d(ds, "weight_busy_row_suppression_factor", float("nan"))
            adaptive_chosen_k = _get_2d(ds, "adaptive_pca_chosen_k", -2147483647)
            adaptive_margin = _get_2d(ds, "adaptive_pca_score_margin", float("nan"))
            max_resid_z = _get_2d(ds, "ptc_second_pass_max_unflagged_residual_z", float("nan"))

            for nw_idx, nw in enumerate(network_ids.tolist()):
                nw = int(nw)
                if nw < 0 or nw not in selected_networks:
                    continue
                discovered_networks.add(nw)
                for scan_idx, output_scan_index in enumerate(output_scan.tolist()):
                    row_new = float(newly_flagged[scan_idx, nw_idx]) if newly_flagged is not None else float("nan")
                    row_prop = float(proposed_flagged[scan_idx, nw_idx]) if proposed_flagged is not None else float("nan")
                    row_event = float(top_event_score[scan_idx, nw_idx]) if top_event_score is not None else float("nan")
                    row_corr = float(corr_penalty_factor[scan_idx, nw_idx]) if corr_penalty_factor is not None else float("nan")
                    row_busy = float(busy_row_factor[scan_idx, nw_idx]) if busy_row_factor is not None else float("nan")
                    row_k = int(adaptive_chosen_k[scan_idx, nw_idx]) if adaptive_chosen_k is not None else -2147483647
                    row_margin = float(adaptive_margin[scan_idx, nw_idx]) if adaptive_margin is not None else float("nan")
                    row_resid = float(max_resid_z[scan_idx, nw_idx]) if max_resid_z is not None else float("nan")
                    scan_network_rows.append(
                        {
                            "obsnum": obsnum,
                            "source_file": str(nc_file),
                            "array": array,
                            "output_scan_index": int(output_scan_index),
                            "network": nw,
                            "ptc_severity": _ptc_severity(row_new, row_corr, row_busy, row_event, row_prop),
                            "newly_flagged_fraction": row_new,
                            "proposed_flagged_fraction": row_prop,
                            "top_event_score": row_event,
                            "corr_penalty_factor": row_corr,
                            "busy_row_factor": row_busy,
                            "adaptive_chosen_k": row_k,
                            "adaptive_score_margin": row_margin,
                            "max_unflagged_residual_z": row_resid,
                        }
                    )

            obs_view = scan_network_rows[rows_before:]
            if not obs_view:
                continue
            product_paths[obsnum] = str(nc_file)
            obs_rows.append(
                {
                    "obsnum": obsnum,
                    "array": array,
                    "source_file": str(nc_file),
                    "n_scan_network_rows": len(obs_view),
                    "n_rows_with_new_flags": int(sum(float(row["newly_flagged_fraction"]) > 0.0 for row in obs_view)),
                    "max_ptc_severity": _nanmax([float(row["ptc_severity"]) for row in obs_view]),
                    "max_newly_flagged_fraction": _nanmax([float(row["newly_flagged_fraction"]) for row in obs_view]),
                    "max_proposed_flagged_fraction": _nanmax([float(row["proposed_flagged_fraction"]) for row in obs_view]),
                    "max_top_event_score": _nanmax([float(row["top_event_score"]) for row in obs_view]),
                    "max_unflagged_residual_z": _nanmax([float(row["max_unflagged_residual_z"]) for row in obs_view]),
                    "min_corr_penalty_factor": _nanmin([float(row["corr_penalty_factor"]) for row in obs_view]),
                    "min_busy_row_factor": _nanmin([float(row["busy_row_factor"]) for row in obs_view]),
                    "max_adaptive_chosen_k": _nanmax([float(row["adaptive_chosen_k"]) for row in obs_view]),
                }
            )

    if not scan_network_rows:
        raise FileNotFoundError(f"no usable ptcdiag rows found for array={array} under {redu_dir}")

    selected_networks = sorted(discovered_networks if requested_networks is None else set(requested_networks))
    by_network_rows: list[dict[str, object]] = []
    for nw in selected_networks:
        rr = [row for row in scan_network_rows if int(row["network"]) == nw]
        if not rr:
            continue
        worst_row = max(rr, key=lambda row: float(row["ptc_severity"]) if np.isfinite(float(row["ptc_severity"])) else -np.inf)
        by_network_rows.append(
            {
                "network": nw,
                "n_rows": len(rr),
                "n_obsnums": len(set(str(row["obsnum"]) for row in rr)),
                "n_rows_with_new_flags": int(sum(float(row["newly_flagged_fraction"]) > 0.0 for row in rr)),
                "max_ptc_severity": _nanmax([float(row["ptc_severity"]) for row in rr]),
                "median_newly_flagged_fraction": _nanmedian([float(row["newly_flagged_fraction"]) for row in rr]),
                "max_newly_flagged_fraction": _nanmax([float(row["newly_flagged_fraction"]) for row in rr]),
                "min_corr_penalty_factor": _nanmin([float(row["corr_penalty_factor"]) for row in rr]),
                "min_busy_row_factor": _nanmin([float(row["busy_row_factor"]) for row in rr]),
                "max_top_event_score": _nanmax([float(row["top_event_score"]) for row in rr]),
                "max_unflagged_residual_z": _nanmax([float(row["max_unflagged_residual_z"]) for row in rr]),
                "worst_obsnum": worst_row["obsnum"],
            }
        )

    return {
        "array": array,
        "n_ptcdiag": len(product_paths),
        "selected_networks": selected_networks,
        "obs_rows": obs_rows,
        "scan_network_rows": scan_network_rows,
        "by_network_rows": by_network_rows,
        "product_paths": product_paths,
    }


def load_detector_rows(
    nc_file: str | Path,
    array: str,
    output_scan_index: int,
) -> list[dict[str, object]]:
    array_id = ARRAY_NAME_TO_ID[array]
    nc_file = Path(nc_file).expanduser().resolve()
    rows: list[dict[str, object]] = []
    with netCDF4.Dataset(nc_file) as ds:
        scans = filled(ds.variables["output_scan_index"], fill=-2147483647).astype(int)
        scan_matches = np.where(scans == int(output_scan_index))[0]
        if scan_matches.size == 0:
            return rows
        scan_idx = int(scan_matches[0])
        det_array = filled(ds.variables["ptc_diag_array"], fill=-2147483647).astype(int)
        det_network = filled(ds.variables["ptc_diag_network"], fill=-2147483647).astype(int)
        det_uid = filled(ds.variables["ptc_diag_uid"], fill=-2147483647).astype(int)
        weights = filled(ds.variables["ptc_detector_weight"], fill=float("nan"))[scan_idx]
        flagged = filled(ds.variables["ptc_detector_flagged_fraction"], fill=float("nan"))[scan_idx]
        rms = filled(ds.variables["ptc_detector_rms"], fill=float("nan"))[scan_idx]
        for det_idx in np.where(det_array == array_id)[0].tolist():
            rows.append(
                {
                    "output_scan_index": int(output_scan_index),
                    "uid": int(det_uid[det_idx]),
                    "network": int(det_network[det_idx]),
                    "weight": float(weights[det_idx]),
                    "flagged_fraction": float(flagged[det_idx]),
                    "rms": float(rms[det_idx]),
                }
            )
    return rows
