#!/usr/bin/env python
"""Extract TOD samples whose detector pointing falls near the source center."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


RAD_TO_ASEC = 206264.80624709636


def _var(ds: Dataset, name: str) -> np.ndarray | None:
    if name not in ds.variables:
        return None
    return np.asarray(ds.variables[name][:])


def _required(ds: Dataset, name: str) -> np.ndarray:
    value = _var(ds, name)
    if value is None:
        raise KeyError(f"missing required NetCDF variable {name!r}")
    return value


def _det_offsets(ds: Dataset, frame: str) -> tuple[np.ndarray, np.ndarray]:
    if frame == "raw":
        x_name = "apt_x_t_raw"
        y_name = "apt_y_t_raw"
    elif frame == "catalog":
        x_name = "apt_x_t"
        y_name = "apt_y_t"
    else:
        x_name = "apt_x_t_derot"
        y_name = "apt_y_t_derot"
    x = _required(ds, x_name).astype(float)
    y = _required(ds, y_name).astype(float)
    return x, y


def _scan_rows(ds: Dataset) -> list[tuple[int, int, int]]:
    scan_indices = _required(ds, "scan_indices").astype(int)
    output_scan_index = _required(ds, "output_scan_index").astype(int)
    rows: list[tuple[int, int, int]] = []
    for row, (start, stop) in enumerate(scan_indices):
        if start < 0 or stop < start:
            continue
        rows.append((int(output_scan_index[row]), int(start), int(stop) + 1))
    return rows


def _network_values(ds: Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    uid = _required(ds, "apt_uid").astype(int)
    array = _required(ds, "apt_array").astype(int)
    nw = _required(ds, "apt_nw").astype(int)
    return uid, array, nw


def _selected_detector_indices(
    uid: np.ndarray,
    array: np.ndarray,
    nw: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    mask = np.ones(uid.size, dtype=bool)
    if args.array is not None:
        mask &= array == args.array
    if args.network is not None:
        mask &= nw == args.network
    if args.uid:
        mask &= np.isin(uid, np.asarray(args.uid, dtype=int))
    return np.flatnonzero(mask)


def extract(args: argparse.Namespace) -> None:
    tod_path = Path(args.tod).expanduser()
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Dataset(tod_path) as ds, out_path.open("w", newline="") as fo:
        uid, array, nw = _network_values(ds)
        det_indices = _selected_detector_indices(uid, array, nw, args)
        if det_indices.size == 0:
            raise RuntimeError("no detectors match the requested selection")

        x_off, y_off = _det_offsets(ds, args.offset_frame)
        x_off = x_off[det_indices]
        y_off = y_off[det_indices]
        det_uid = uid[det_indices]
        det_array = array[det_indices]
        det_nw = nw[det_indices]

        tel_el = _required(ds, "TelElAct").astype(float)
        az_phys = _required(ds, "az_phys").astype(float) * RAD_TO_ASEC
        alt_phys = _required(ds, "alt_phys").astype(float) * RAD_TO_ASEC
        pointing_az = _var(ds, "pointing_offset_az")
        pointing_alt = _var(ds, "pointing_offset_alt")
        if pointing_az is None:
            pointing_az = np.zeros_like(az_phys)
        if pointing_alt is None:
            pointing_alt = np.zeros_like(alt_phys)
        pointing_az = pointing_az.astype(float)
        pointing_alt = pointing_alt.astype(float)

        signal_v = ds.variables["signal"]
        flags_v = ds.variables["flags"]
        weights_v = ds.variables.get("ptc_detector_weight")

        writer = csv.writer(fo)
        writer.writerow(
            [
                "stream",
                "output_scan",
                "sample",
                "uid",
                "array",
                "nw",
                "radius_arcsec",
                "az_arcsec",
                "alt_arcsec",
                "signal",
                "flag",
                "weight",
            ]
        )

        n_rows = 0
        radius = float(args.radius_arcsec)
        stream = tod_path.stem
        for scan_row, (output_scan, start, stop) in enumerate(_scan_rows(ds)):
            sl = slice(start, stop)
            elev = tel_el[sl]
            cos_el = np.cos(elev)[:, None]
            sin_el = np.sin(elev)[:, None]
            rot_az = cos_el * x_off[None, :] - sin_el * y_off[None, :] + pointing_az[sl, None]
            rot_alt = cos_el * y_off[None, :] + sin_el * x_off[None, :] + pointing_alt[sl, None]
            az = az_phys[sl, None] + rot_az
            alt = alt_phys[sl, None] + rot_alt
            rad = np.hypot(az, alt)
            hit_sample, hit_det = np.nonzero(rad <= radius)
            if hit_sample.size == 0:
                continue

            sig = np.asarray(signal_v[sl, det_indices])
            flags = np.asarray(flags_v[sl, det_indices])
            weights = None
            if weights_v is not None and weights_v.ndim == 2:
                weights = np.asarray(weights_v[scan_row, det_indices])

            for s_i, d_i in zip(hit_sample, hit_det):
                sample = start + int(s_i)
                weight = "" if weights is None else float(weights[d_i])
                writer.writerow(
                    [
                        stream,
                        output_scan,
                        sample,
                        int(det_uid[d_i]),
                        int(det_array[d_i]),
                        int(det_nw[d_i]),
                        float(rad[s_i, d_i]),
                        float(az[s_i, d_i]),
                        float(alt[s_i, d_i]),
                        float(sig[s_i, d_i]),
                        float(flags[s_i, d_i]),
                        weight,
                    ]
                )
                n_rows += 1
                if args.max_rows and n_rows >= args.max_rows:
                    return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tod", required=True, help="Citlali RTC/PTC TOD NetCDF file.")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument("--radius-arcsec", type=float, default=10.0)
    parser.add_argument(
        "--offset-frame",
        choices=["raw", "catalog", "derot"],
        default="raw",
        help="Detector offset columns to use when reconstructing detector pointing.",
    )
    parser.add_argument("--array", type=int, choices=[0, 1, 2], help="Optional array id filter.")
    parser.add_argument("--network", type=int, help="Optional network id filter.")
    parser.add_argument("--uid", type=int, action="append", help="Optional detector uid filter; repeatable.")
    parser.add_argument("--max-rows", type=int, default=0, help="Debug limit on output rows.")
    return parser.parse_args()


def main() -> None:
    extract(parse_args())


if __name__ == "__main__":
    main()
