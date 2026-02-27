#!/usr/bin/env python3
"""Build soft within-network slot priors from historical measured APT files.

This produces quantile-based "slots" along each network footprint in an
array-centered frame. The priors are intentionally soft (inflated sigmas with
minimum floors) so they can guide source identification without hard locking.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Dict, Iterable, List, Tuple

import numpy as np
from astropy.table import Table


ARRAY_NAME = {
    0: "a1100",
    1: "a1400",
    2: "a2000",
}


def robust_sigma(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    return float(1.4826 * mad)


def parse_obsnum(filepath: str) -> int:
    m = re.search(r"beammap_(\d+)_", os.path.basename(filepath))
    if m is None:
        raise ValueError(f"could not parse obsnum from {filepath}")
    return int(m.group(1))


def build_obs_network_points(
    files: Iterable[str],
    min_good_per_array_obs: int,
    min_points_per_network_obs: int,
) -> Dict[Tuple[int, int, int], np.ndarray]:
    """Return mapping (obsnum, array, nw) -> Nx2 array of centered x/y points."""
    out: Dict[Tuple[int, int, int], np.ndarray] = {}
    for fp in files:
        obsnum = parse_obsnum(fp)
        apt = Table.read(fp, format="ascii.ecsv")
        arr = np.asarray(apt["array"]).astype(int)
        nw = np.asarray(apt["nw"]).astype(int)
        flag = np.asarray(apt["flag"]).astype(int)
        x_t = np.asarray(apt["x_t"], dtype=float)
        y_t = np.asarray(apt["y_t"], dtype=float)

        for array in sorted(set(arr.tolist())):
            good = (
                (arr == array)
                & (flag == 0)
                & np.isfinite(x_t)
                & np.isfinite(y_t)
            )
            if int(good.sum()) < min_good_per_array_obs:
                continue
            x0 = float(np.median(x_t[good]))
            y0 = float(np.median(y_t[good]))

            for nw_id in sorted(set(nw[good].tolist())):
                m = good & (nw == nw_id)
                if int(m.sum()) < min_points_per_network_obs:
                    continue
                x = x_t[m] - x0
                y = y_t[m] - y0
                pts = np.column_stack([x, y])
                # Sort by y so quantile slots follow the long axis ordering.
                order = np.argsort(pts[:, 1], kind="mergesort")
                out[(obsnum, int(array), int(nw_id))] = pts[order]
    return out


def summarize_soft_slots(
    obs_pts: Dict[Tuple[int, int, int], np.ndarray],
    n_slots: int,
    min_obs_per_network: int,
    sigma_inflate: float,
    x_sigma_floor_arcsec: float,
    y_sigma_floor_arcsec: float,
) -> Table:
    # Group keys by (array, nw)
    by_network: Dict[Tuple[int, int], List[Tuple[int, np.ndarray]]] = {}
    for (obsnum, array, nw), pts in obs_pts.items():
        by_network.setdefault((array, nw), []).append((obsnum, pts))

    rows = []
    for (array, nw), items in sorted(by_network.items()):
        obsnums = sorted({obs for obs, _ in items})
        if len(obsnums) < min_obs_per_network:
            continue

        # Precompute slot positions for each observation by quantile in sorted-y list.
        slot_xy_per_obs: List[np.ndarray] = []
        for _, pts in items:
            n = pts.shape[0]
            q = np.linspace(0.0, 1.0, n_slots)
            idx = np.clip(np.rint(q * (n - 1)).astype(int), 0, n - 1)
            slot_xy_per_obs.append(pts[idx, :])  # shape (n_slots, 2)

        stack = np.stack(slot_xy_per_obs, axis=0)  # (n_obs, n_slots, 2)
        # Per-slot aggregate across observations.
        for slot in range(n_slots):
            x = stack[:, slot, 0]
            y = stack[:, slot, 1]
            sx = robust_sigma(x)
            sy = robust_sigma(y)
            sx_soft = max(x_sigma_floor_arcsec, sigma_inflate * sx) if np.isfinite(sx) else x_sigma_floor_arcsec
            sy_soft = max(y_sigma_floor_arcsec, sigma_inflate * sy) if np.isfinite(sy) else y_sigma_floor_arcsec
            rows.append(
                {
                    "prior_level": "slot_soft",
                    "array": int(array),
                    "array_name": ARRAY_NAME.get(int(array), f"array{array}"),
                    "nw": int(nw),
                    "slot_index": int(slot),
                    "slot_frac": float(slot / (n_slots - 1) if n_slots > 1 else 0.0),
                    "n_obs": int(len(obsnums)),
                    "x_rel_med_arcsec": float(np.median(x)),
                    "y_rel_med_arcsec": float(np.median(y)),
                    "x_rel_sigma_arcsec": float(sx),
                    "y_rel_sigma_arcsec": float(sy),
                    "x_rel_sigma_soft_arcsec": float(sx_soft),
                    "y_rel_sigma_soft_arcsec": float(sy_soft),
                    "x_rel_q05_arcsec": float(np.quantile(x, 0.05)),
                    "x_rel_q95_arcsec": float(np.quantile(x, 0.95)),
                    "y_rel_q05_arcsec": float(np.quantile(y, 0.05)),
                    "y_rel_q95_arcsec": float(np.quantile(y, 0.95)),
                }
            )

    return Table(rows=rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-glob",
        required=True,
        help="glob for measured APT files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="output ECSV file path",
    )
    parser.add_argument(
        "--n-slots",
        type=int,
        default=64,
        help="number of quantile slots per network (default: 64)",
    )
    parser.add_argument(
        "--min-good-per-array-obs",
        type=int,
        default=100,
        help="minimum good-detector count to use an obs+array for centering",
    )
    parser.add_argument(
        "--min-points-per-network-obs",
        type=int,
        default=60,
        help="minimum points required in an obs+network to contribute slot samples",
    )
    parser.add_argument(
        "--min-obs-per-network",
        type=int,
        default=8,
        help="minimum observation count required to emit a network slot prior",
    )
    parser.add_argument(
        "--sigma-inflate",
        type=float,
        default=1.8,
        help="inflate robust per-slot sigma by this factor for soft gating",
    )
    parser.add_argument(
        "--x-sigma-floor-arcsec",
        type=float,
        default=6.0,
        help="minimum soft sigma in x for each slot",
    )
    parser.add_argument(
        "--y-sigma-floor-arcsec",
        type=float,
        default=8.0,
        help="minimum soft sigma in y for each slot",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if len(files) == 0:
        raise SystemExit(f"no files matched {args.input_glob}")

    obs_pts = build_obs_network_points(
        files,
        min_good_per_array_obs=args.min_good_per_array_obs,
        min_points_per_network_obs=args.min_points_per_network_obs,
    )
    if len(obs_pts) == 0:
        raise SystemExit("no valid observation/network samples collected")

    priors = summarize_soft_slots(
        obs_pts=obs_pts,
        n_slots=args.n_slots,
        min_obs_per_network=args.min_obs_per_network,
        sigma_inflate=args.sigma_inflate,
        x_sigma_floor_arcsec=args.x_sigma_floor_arcsec,
        y_sigma_floor_arcsec=args.y_sigma_floor_arcsec,
    )
    if len(priors) == 0:
        raise SystemExit("no slot priors generated; loosen thresholds")

    priors.meta["prior_version"] = "beammap_slot_soft_v1"
    priors.meta["prior_frame"] = "array-centered-derotated"
    priors.meta["input_file_count"] = int(len(files))
    priors.meta["input_files"] = [os.path.basename(f) for f in files]
    priors.meta["n_slots"] = int(args.n_slots)
    priors.meta["min_good_per_array_obs"] = int(args.min_good_per_array_obs)
    priors.meta["min_points_per_network_obs"] = int(args.min_points_per_network_obs)
    priors.meta["min_obs_per_network"] = int(args.min_obs_per_network)
    priors.meta["sigma_inflate"] = float(args.sigma_inflate)
    priors.meta["x_sigma_floor_arcsec"] = float(args.x_sigma_floor_arcsec)
    priors.meta["y_sigma_floor_arcsec"] = float(args.y_sigma_floor_arcsec)
    priors.meta["notes"] = [
        "Slots are quantile samples of sorted-y detector positions per observation+network.",
        "x_rel/y_rel are in array-centered coordinates derived from x_t/y_t.",
        "soft sigmas are inflated and floored to avoid over-constraining assignment.",
    ]

    outdir = os.path.dirname(os.path.abspath(args.output))
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    priors.write(args.output, format="ascii.ecsv", overwrite=True)
    print(f"wrote {len(priors)} soft slot prior rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

