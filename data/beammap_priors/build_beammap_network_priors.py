#!/usr/bin/env python3
"""Build beammap network priors from historical measured APT ECSV files.

The priors are built in an array-centered frame to remove per-observation
boresight shifts:
  x_rel = x_t - median(x_t good dets in same obs+array)
  y_rel = y_t - median(y_t good dets in same obs+array)

The output is one row per (array, network) with robust location/shape
statistics that can be used to seed and gate beammap source identification.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from astropy.table import Table


ARRAY_NAME = {
    0: "a1100",
    1: "a1400",
    2: "a2000",
}


def robust_sigma(values: np.ndarray) -> float:
    """Return robust sigma estimate from MAD."""
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


@dataclass
class Sample:
    obsnum: int
    array: int
    nw: int
    uid: int
    x_rel: float
    y_rel: float
    x_raw_rel: float
    y_raw_rel: float


def collect_samples(files: Iterable[str], min_good_per_array_obs: int) -> List[Sample]:
    samples: List[Sample] = []
    for fp in files:
        obsnum = parse_obsnum(fp)
        apt = Table.read(fp, format="ascii.ecsv")
        arr = np.asarray(apt["array"]).astype(int)
        nw = np.asarray(apt["nw"]).astype(int)
        uid = np.asarray(apt["uid"]).astype(int)
        flag = np.asarray(apt["flag"]).astype(int)
        x_t = np.asarray(apt["x_t"], dtype=float)
        y_t = np.asarray(apt["y_t"], dtype=float)
        x_raw = np.asarray(apt["x_t_raw"], dtype=float)
        y_raw = np.asarray(apt["y_t_raw"], dtype=float)

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

            good_raw = good & np.isfinite(x_raw) & np.isfinite(y_raw)
            if int(good_raw.sum()) > 0:
                x0_raw = float(np.median(x_raw[good_raw]))
                y0_raw = float(np.median(y_raw[good_raw]))
            else:
                x0_raw = float("nan")
                y0_raw = float("nan")

            idxs = np.where(good)[0]
            for i in idxs:
                xr = x_t[i] - x0
                yr = y_t[i] - y0
                if np.isfinite(x_raw[i]) and np.isfinite(y_raw[i]) and np.isfinite(x0_raw) and np.isfinite(y0_raw):
                    xr_raw = x_raw[i] - x0_raw
                    yr_raw = y_raw[i] - y0_raw
                else:
                    xr_raw = float("nan")
                    yr_raw = float("nan")
                samples.append(
                    Sample(
                        obsnum=obsnum,
                        array=int(array),
                        nw=int(nw[i]),
                        uid=int(uid[i]),
                        x_rel=float(xr),
                        y_rel=float(yr),
                        x_raw_rel=float(xr_raw),
                        y_raw_rel=float(yr_raw),
                    )
                )
    return samples


def summarize_network(samples: List[Sample], min_samples_per_network: int) -> Table:
    # group indices by (array, nw)
    groups: Dict[Tuple[int, int], List[int]] = {}
    for i, s in enumerate(samples):
        groups.setdefault((s.array, s.nw), []).append(i)

    rows = []
    for (array, nw), idxs in sorted(groups.items()):
        n_samples = len(idxs)
        if n_samples < min_samples_per_network:
            continue

        obsnums = sorted({samples[i].obsnum for i in idxs})
        uids = sorted({samples[i].uid for i in idxs})
        x = np.array([samples[i].x_rel for i in idxs], dtype=float)
        y = np.array([samples[i].y_rel for i in idxs], dtype=float)
        r = np.hypot(x, y)

        valid_raw = [
            i for i in idxs
            if np.isfinite(samples[i].x_raw_rel) and np.isfinite(samples[i].y_raw_rel)
        ]
        x_raw = np.array([samples[i].x_raw_rel for i in valid_raw], dtype=float)
        y_raw = np.array([samples[i].y_raw_rel for i in valid_raw], dtype=float)

        cov = np.cov(np.vstack([x, y]), ddof=1) if n_samples > 1 else np.full((2, 2), np.nan)
        eigvals, eigvecs = np.linalg.eigh(cov) if np.all(np.isfinite(cov)) else (np.array([np.nan, np.nan]), np.full((2, 2), np.nan))
        major = float(np.sqrt(eigvals[1])) if np.all(np.isfinite(eigvals)) and eigvals[1] >= 0 else float("nan")
        minor = float(np.sqrt(eigvals[0])) if np.all(np.isfinite(eigvals)) and eigvals[0] >= 0 else float("nan")
        theta = float(np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))) if np.all(np.isfinite(eigvecs)) else float("nan")

        row = {
            "prior_level": "network",
            "array": int(array),
            "array_name": ARRAY_NAME.get(int(array), f"array{array}"),
            "nw": int(nw),
            "n_obs": int(len(obsnums)),
            "n_samples": int(n_samples),
            "n_uids": int(len(uids)),
            "x_rel_med_arcsec": float(np.median(x)),
            "y_rel_med_arcsec": float(np.median(y)),
            "x_rel_sigma_arcsec": robust_sigma(x),
            "y_rel_sigma_arcsec": robust_sigma(y),
            "x_rel_q05_arcsec": float(np.quantile(x, 0.05)),
            "x_rel_q95_arcsec": float(np.quantile(x, 0.95)),
            "y_rel_q05_arcsec": float(np.quantile(y, 0.05)),
            "y_rel_q95_arcsec": float(np.quantile(y, 0.95)),
            "r_rel_q95_arcsec": float(np.quantile(r, 0.95)),
            "cov_xx_arcsec2": float(cov[0, 0]),
            "cov_xy_arcsec2": float(cov[0, 1]),
            "cov_yy_arcsec2": float(cov[1, 1]),
            "pca_major_sigma_arcsec": major,
            "pca_minor_sigma_arcsec": minor,
            "pca_major_theta_deg": theta,
            "x_raw_rel_med_arcsec": float(np.median(x_raw)) if x_raw.size else float("nan"),
            "y_raw_rel_med_arcsec": float(np.median(y_raw)) if y_raw.size else float("nan"),
            "x_raw_rel_sigma_arcsec": robust_sigma(x_raw) if x_raw.size else float("nan"),
            "y_raw_rel_sigma_arcsec": robust_sigma(y_raw) if y_raw.size else float("nan"),
            "n_raw_samples": int(x_raw.size),
        }
        rows.append(row)

    out = Table(rows=rows)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-glob",
        required=True,
        help="glob for historical measured APT files (e.g. apt_commissioning_beammap_*_citlali.ecsv)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="output ECSV path",
    )
    parser.add_argument(
        "--min-good-per-array-obs",
        type=int,
        default=100,
        help="minimum count of good detectors required to use an obs+array for centering",
    )
    parser.add_argument(
        "--min-samples-per-network",
        type=int,
        default=500,
        help="minimum number of centered detector samples required to emit a network prior row",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if len(files) == 0:
        raise SystemExit(f"no files matched {args.input_glob}")

    samples = collect_samples(files, min_good_per_array_obs=args.min_good_per_array_obs)
    if len(samples) == 0:
        raise SystemExit("no valid samples collected from inputs")

    priors = summarize_network(samples, min_samples_per_network=args.min_samples_per_network)
    if len(priors) == 0:
        raise SystemExit("no network priors produced; lower --min-samples-per-network")

    priors.meta["prior_version"] = "beammap_network_v1"
    priors.meta["prior_frame"] = "array-centered-derotated"
    priors.meta["input_file_count"] = int(len(files))
    priors.meta["input_files"] = [os.path.basename(f) for f in files]
    priors.meta["min_good_per_array_obs"] = int(args.min_good_per_array_obs)
    priors.meta["min_samples_per_network"] = int(args.min_samples_per_network)
    priors.meta["notes"] = [
        "x_rel/y_rel are computed from x_t/y_t after subtracting per-observation array medians from good detectors.",
        "x_raw_rel/y_raw_rel are analogous centered raw-coordinate values from x_t_raw/y_t_raw when available.",
        "These priors are intended for initialization/gating, not hard detector ID assignment.",
    ]

    outdir = os.path.dirname(os.path.abspath(args.output))
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    priors.write(args.output, format="ascii.ecsv", overwrite=True)
    print(f"wrote {len(priors)} network prior rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

