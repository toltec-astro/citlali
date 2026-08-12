#!/usr/bin/env python3
"""Freeze and audit pointing-source quality before SCI-ALIGN timing fits.

The audit deliberately separates two decisions:

* the automatic first cut is only the a1100 PPT signal-to-noise threshold;
* map width and ellipticity are descriptive, while residual structure and
  secondary peaks are surfaced for human review rather than used as automatic
  exclusions.

No PTC signal samples or timing-fit results are read by this program.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.table import Table  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402
from scipy.ndimage import gaussian_filter, label, maximum_filter  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402


FWHM_PER_SIGMA = 2.3548200450309493
PDF_METADATA = {
    "Creator": "SCI-ALIGN-001 pointing source-quality audit",
    "CreationDate": datetime.datetime(2026, 8, 12, tzinfo=datetime.timezone.utc),
    "ModDate": datetime.datetime(2026, 8, 12, tzinfo=datetime.timezone.utc),
}


class ContractError(RuntimeError):
    """An input violates the bounded diagnostic contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_checksums(root: Path, names: Iterable[str]) -> None:
    lines = [f"{sha256_file(root / name)}  {name}" for name in sorted(names)]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def verify_checksums(root: Path) -> None:
    manifest = root / "SHA256SUMS"
    if not manifest.is_file():
        raise ContractError(f"missing checksum manifest: {manifest}")
    for raw in manifest.read_text().splitlines():
        expected, name = raw.split(None, 1)
        path = root / name.strip()
        actual = sha256_file(path)
        if actual != expected:
            raise ContractError(f"checksum mismatch for {path}")


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    if protocol.get("schema") != "sci-align-001-pointing-source-quality-protocol-v1":
        raise ContractError("unsupported source-quality protocol schema")
    if protocol.get("automatic_selection", {}).get("field") != "a1100.sig2noise":
        raise ContractError("automatic selection must use only a1100.sig2noise")
    if protocol.get("morphology", {}).get("automatic_exclusion") is not False:
        raise ContractError("morphology must not automatically exclude observations")
    return protocol


def map_local_path(path: str, unity_root: Path, local_root: Path) -> Path:
    source = Path(path)
    try:
        relative = source.relative_to(unity_root)
    except ValueError as error:
        raise ContractError(f"path is outside declared Unity root: {source}") from error
    return (local_root / relative).resolve()


def freeze(args: argparse.Namespace) -> None:
    audit_path = args.schema_audit.resolve()
    protocol_path = args.protocol.resolve()
    output = args.output.resolve()
    if output.exists():
        raise ContractError(f"output already exists: {output}")
    protocol = load_protocol(protocol_path)
    rows = list(csv.DictReader(audit_path.open()))
    if not rows:
        raise ContractError("schema audit is empty")
    obsnums = [int(row["obsnum"]) for row in rows]
    if len(set(obsnums)) != len(obsnums):
        raise ContractError("schema audit contains duplicate observations")

    frozen_rows: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda value: int(value["obsnum"])):
        obsnum = int(row["obsnum"])
        if row["status"] != "ready":
            raise ContractError(f"obs {obsnum}: schema status is {row['status']!r}")
        ppt = map_local_path(row["ppt_path"], args.unity_root, args.local_root)
        ptc = map_local_path(row["ptc_path"], args.unity_root, args.local_root)
        pointing_map = ppt.parent / (
            f"toltec_commissioning_a1100_pointing_{obsnum}_citlali.fits"
        )
        for kind, path in (("PPT", ppt), ("PTC", ptc), ("a1100 map", pointing_map)):
            if not path.is_file():
                raise ContractError(f"obs {obsnum}: missing {kind}: {path}")
        if ptc.stat().st_size != int(row["size_bytes"]):
            raise ContractError(
                f"obs {obsnum}: local PTC size differs from verified inventory"
            )
        frozen_rows.append({
            "obsnum": obsnum,
            "ppt_path": str(ppt),
            "ppt_size_bytes": ppt.stat().st_size,
            "ppt_sha256": sha256_file(ppt),
            "a1100_map_path": str(pointing_map),
            "a1100_map_size_bytes": pointing_map.stat().st_size,
            "a1100_map_sha256": sha256_file(pointing_map),
            # The PTC is not read or hashed here.  Its inventory identity is
            # retained solely so the accepted observation can later be joined
            # to a separately frozen timing analysis.
            "ptc_path": str(ptc),
            "ptc_size_bytes": int(row["size_bytes"]),
            "ptc_modified": row["modified"],
        })

    output.mkdir(parents=True)
    Table(rows=frozen_rows).write(
        output / "frozen_inputs.ecsv", format="ascii.ecsv"
    )
    frozen = {
        "schema": "sci-align-001-pointing-source-quality-input-v1",
        "schema_audit_path": str(audit_path),
        "schema_audit_sha256": sha256_file(audit_path),
        "protocol": protocol,
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "observation_count": len(frozen_rows),
        "rows": frozen_rows,
        "ptc_identity_note": (
            "PTC content is not used or hashed by this source-quality audit; "
            "path, size, and modification time are retained from the verified "
            "schema inventory for later independent timing-input freezing."
        ),
    }
    write_json(output / "frozen_input.json", frozen)
    write_checksums(output, ("frozen_input.json", "frozen_inputs.ecsv"))
    print(f"frozen source-quality inputs: observations={len(frozen_rows)} output={output}")


def spatial_axes(header: fits.Header, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    if header.get("CTYPE1") != "AZOFFSET" or header.get("CTYPE2") != "ELOFFSET":
        raise ContractError("pointing map does not use AZOFFSET/ELOFFSET axes")
    nx, ny = shape[1], shape[0]
    x = (
        (np.arange(nx, dtype=float) + 1.0 - float(header["CRPIX1"]))
        * float(header["CDELT1"])
        + float(header["CRVAL1"])
    )
    y = (
        (np.arange(ny, dtype=float) + 1.0 - float(header["CRPIX2"]))
        * float(header["CDELT2"])
        + float(header["CRVAL2"])
    )
    return x, y


def ppt_a1100(path: Path) -> dict[str, float]:
    table = Table.read(path, format="ascii.ecsv")
    required = {
        "array", "amp", "x_t", "y_t", "a_fwhm", "b_fwhm", "angle",
        "sig2noise",
    }
    missing = sorted(required - set(table.colnames))
    if missing:
        raise ContractError(f"{path}: missing PPT columns {missing}")
    selected = table[np.asarray(table["array"], dtype=int) == 0]
    if len(selected) != 1:
        raise ContractError(f"{path}: expected exactly one a1100 row")
    row = selected[0]
    result = {name: float(row[name]) for name in required - {"array"}}
    if not all(math.isfinite(value) for value in result.values()):
        raise ContractError(f"{path}: non-finite a1100 PPT value")
    if result["amp"] <= 0.0 or min(result["a_fwhm"], result["b_fwhm"]) <= 0.0:
        raise ContractError(f"{path}: non-positive a1100 fit scale")
    return result


def load_pointing_map(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, fits.Header]:
    with fits.open(path, memmap=True) as hdus:
        try:
            signal_hdu = hdus["signal_I"]
            weight_hdu = hdus["weight_I"]
        except (KeyError, IndexError) as error:
            raise ContractError(f"{path}: missing signal_I or weight_I") from error
        signal = np.asarray(signal_hdu.data, dtype=float).squeeze()
        weight = np.asarray(weight_hdu.data, dtype=float).squeeze()
        header = signal_hdu.header.copy()
    if signal.ndim != 2 or weight.shape != signal.shape:
        raise ContractError(f"{path}: invalid map geometry")
    x, y = spatial_axes(header, signal.shape)
    return signal, weight, x, y, header


def robust_sigma(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 20:
        return math.nan
    median = float(np.median(finite))
    sigma = 1.482602218505602 * float(np.median(np.abs(finite - median)))
    return sigma


def gaussian_plane(
    params: np.ndarray, xx: np.ndarray, yy: np.ndarray, scale_xy: float
) -> np.ndarray:
    amp, x0, y0, sx, sy, theta, b0, bx, by = params
    ct, st = math.cos(theta), math.sin(theta)
    dx, dy = xx - x0, yy - y0
    u = ct * dx + st * dy
    v = -st * dx + ct * dy
    return (
        amp * np.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2))
        + b0 + bx * (xx - x0) / scale_xy + by * (yy - y0) / scale_xy
    )


def fit_source(
    image: np.ndarray,
    weight: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    ppt: dict[str, float],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    xx, yy = np.meshgrid(x, y)
    px, py = ppt["x_t"], ppt["y_t"]
    major = max(ppt["a_fwhm"], ppt["b_fwhm"])
    fit_radius = min(
        float(protocol["morphology"]["maximum_fit_radius_arcsec"]),
        max(float(protocol["morphology"]["minimum_fit_radius_arcsec"]), 2.5 * major),
    )
    radius = np.hypot(xx - px, yy - py)
    valid = np.isfinite(image) & np.isfinite(weight) & (weight > 0.0)
    fit_mask = valid & (radius <= fit_radius)
    if np.count_nonzero(fit_mask) < 100:
        raise ContractError("insufficient finite source-map support")

    xv, yv, zv = xx[fit_mask], yy[fit_mask], image[fit_mask]
    background = float(np.median(zv[radius[fit_mask] >= 0.75 * fit_radius]))
    scale = float(np.nanmax(zv) - background)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ContractError("pointing map has non-positive source contrast")
    angle = float(ppt["angle"])
    if abs(angle) > 2.0 * math.pi:
        angle = math.radians(angle)
    p0 = np.asarray([
        max(ppt["amp"], scale), px, py,
        ppt["a_fwhm"] / FWHM_PER_SIGMA,
        ppt["b_fwhm"] / FWHM_PER_SIGMA,
        angle, background, 0.0, 0.0,
    ])
    center_limit = max(6.0, major)
    lower = np.asarray([
        0.0, px - center_limit, py - center_limit, 0.5, 0.5, -math.pi,
        -math.inf, -math.inf, -math.inf,
    ])
    upper = np.asarray([
        math.inf, px + center_limit, py + center_limit, 20.0, 20.0, math.pi,
        math.inf, math.inf, math.inf,
    ])

    def residual(params: np.ndarray) -> np.ndarray:
        return (gaussian_plane(params, xv, yv, fit_radius) - zv) / scale

    result = least_squares(
        residual, p0, bounds=(lower, upper), loss="soft_l1", f_scale=0.03,
        max_nfev=3000,
    )
    if not result.success or np.any(~np.isfinite(result.x)):
        raise ContractError(f"source fit failed: {result.message}")
    params = result.x
    model = gaussian_plane(params, xx, yy, fit_radius)
    return {
        "params": params,
        "model": model,
        "valid": valid,
        "fit_radius_arcsec": fit_radius,
        "fit_residual_rms_fraction_peak": float(np.sqrt(np.mean(result.fun ** 2))),
    }


def source_quality_metrics(
    image: np.ndarray,
    weight: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    ppt: dict[str, float],
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    fit = fit_source(image, weight, x, y, ppt, protocol)
    params = fit["params"]
    amp, x0, y0, sx, sy, theta = params[:6]
    model = fit["model"]
    residual = image - model
    xx, yy = np.meshgrid(x, y)
    ct, st = math.cos(theta), math.sin(theta)
    dx, dy = xx - x0, yy - y0
    u = ct * dx + st * dy
    v = -st * dx + ct * dy
    q = np.sqrt((u / (FWHM_PER_SIGMA * sx)) ** 2 + (v / (FWHM_PER_SIGMA * sy)) ** 2)
    radial = np.hypot(dx, dy)
    crop_half = float(protocol["morphology"]["display_half_width_arcsec"])
    valid = fit["valid"] & (radial <= crop_half)
    review = valid & (q >= float(protocol["morphology"]["review_core_radius_fwhm"]))
    noise_region = valid & (q >= float(protocol["morphology"]["noise_inner_radius_fwhm"]))
    noise = robust_sigma(residual[noise_region])
    if not math.isfinite(noise) or noise <= 0.0:
        raise ContractError("cannot estimate positive residual noise")

    pixel = math.sqrt(abs(float(np.median(np.diff(x))) * float(np.median(np.diff(y)))))
    smooth_sigma = float(protocol["morphology"]["residual_smoothing_arcsec"]) / pixel
    filled = np.where(fit["valid"], residual, 0.0)
    support = gaussian_filter(fit["valid"].astype(float), smooth_sigma, mode="constant")
    smooth = gaussian_filter(filled, smooth_sigma, mode="constant")
    smooth = np.divide(smooth, support, out=np.full_like(smooth, np.nan), where=support > 0.5)
    threshold = max(
        float(protocol["morphology"]["component_sigma_threshold"]) * noise,
        float(protocol["morphology"]["component_peak_fraction_threshold"]) * amp,
    )
    component_mask = review & np.isfinite(smooth) & (np.abs(smooth) >= threshold)
    labels, count = label(component_mask, structure=np.ones((3, 3), dtype=int))
    beam_area_pixels = (
        2.0 * math.pi * sx * sy / max(pixel * pixel, np.finfo(float).eps)
    )
    minimum_area = max(
        2,
        int(math.ceil(
            float(protocol["morphology"]["minimum_component_beam_fraction"])
            * beam_area_pixels
        )),
    )
    coherent_count = 0
    for index in range(1, count + 1):
        if np.count_nonzero(labels == index) >= minimum_area:
            coherent_count += 1

    separation = max(3, int(round(0.5 * FWHM_PER_SIGMA * min(sx, sy) / pixel)))
    local_max = smooth == maximum_filter(smooth, size=2 * separation + 1, mode="constant", cval=-np.inf)
    secondary_threshold = max(
        float(protocol["morphology"]["secondary_peak_sigma_threshold"]) * noise,
        float(protocol["morphology"]["secondary_peak_fraction_threshold"]) * amp,
    )
    peaks = review & local_max & np.isfinite(smooth) & (smooth >= secondary_threshold)
    peak_values = smooth[peaks]
    strongest_secondary = float(np.max(peak_values)) if peak_values.size else 0.0
    strongest_abs = float(np.max(np.abs(smooth[review])))
    rms = float(np.sqrt(np.mean(residual[review] ** 2)))
    metrics = {
        "snr_a1100": ppt["sig2noise"],
        "ppt_x_arcsec": ppt["x_t"],
        "ppt_y_arcsec": ppt["y_t"],
        "ppt_a_fwhm_arcsec": ppt["a_fwhm"],
        "ppt_b_fwhm_arcsec": ppt["b_fwhm"],
        "ppt_axis_ratio": max(ppt["a_fwhm"], ppt["b_fwhm"]) / min(ppt["a_fwhm"], ppt["b_fwhm"]),
        "fit_x_arcsec": float(x0),
        "fit_y_arcsec": float(y0),
        "fit_centroid_shift_from_ppt_arcsec": float(math.hypot(x0 - ppt["x_t"], y0 - ppt["y_t"])),
        "fit_major_fwhm_arcsec": float(FWHM_PER_SIGMA * max(sx, sy)),
        "fit_minor_fwhm_arcsec": float(FWHM_PER_SIGMA * min(sx, sy)),
        "fit_axis_ratio": float(max(sx, sy) / min(sx, sy)),
        "fit_angle_rad": float(theta),
        "fit_amplitude_native": float(amp),
        "fit_residual_rms_fraction_peak": fit["fit_residual_rms_fraction_peak"],
        "residual_noise_native": noise,
        "off_core_residual_rms_over_noise": rms / noise,
        "strongest_abs_smoothed_residual_fraction_peak": strongest_abs / amp,
        "strongest_positive_secondary_peak_fraction": strongest_secondary / amp,
        "positive_secondary_peak_count": int(np.count_nonzero(peaks)),
        "coherent_residual_component_count": coherent_count,
        "coherent_component_minimum_pixels": minimum_area,
    }
    diagnostic = {
        "image": image,
        "residual": residual,
        "smooth_residual": smooth,
        "x": x,
        "y": y,
        "params": params,
        "peaks": peaks,
    }
    return metrics, diagnostic


def crop_for_display(
    image: np.ndarray, x: np.ndarray, y: np.ndarray, cx: float, cy: float,
    half_width: float,
) -> tuple[np.ndarray, tuple[float, float, float, float], np.ndarray, np.ndarray]:
    xs = np.flatnonzero((x >= cx - half_width) & (x <= cx + half_width))
    ys = np.flatnonzero((y >= cy - half_width) & (y <= cy + half_width))
    if xs.size < 2 or ys.size < 2:
        raise ContractError("map does not cover requested display crop")
    view = image[np.ix_(ys, xs)]
    dx = float(np.median(np.diff(x)))
    dy = float(np.median(np.diff(y)))
    extent = (x[xs[0]] - 0.5 * dx, x[xs[-1]] + 0.5 * dx,
              y[ys[0]] - 0.5 * dy, y[ys[-1]] + 0.5 * dy)
    return view, extent, xs, ys


def render_contact_sheet(
    path: Path,
    selected: list[tuple[dict[str, Any], dict[str, Any]]],
    protocol: dict[str, Any],
) -> None:
    half_width = float(protocol["morphology"]["display_half_width_arcsec"])
    with PdfPages(path, metadata=PDF_METADATA) as pdf:
        for offset in range(0, len(selected), 3):
            page = selected[offset:offset + 3]
            fig, axes = plt.subplots(2, 3, figsize=(15, 9), squeeze=False)
            for column, (row, diagnostic) in enumerate(page):
                params = diagnostic["params"]
                amp, x0, y0, sx, sy, theta = params[:6]
                raw, extent, xs, ys = crop_for_display(
                    diagnostic["image"], diagnostic["x"], diagnostic["y"],
                    x0, y0, half_width,
                )
                residual = diagnostic["residual"][np.ix_(ys, xs)]
                peak_rows, peak_cols = np.nonzero(
                    diagnostic["peaks"][np.ix_(ys, xs)]
                )
                peak_x = diagnostic["x"][xs[peak_cols]]
                peak_y = diagnostic["y"][ys[peak_rows]]
                raw_ax, residual_ax = axes[0, column], axes[1, column]
                raw_ax.imshow(
                    raw, origin="lower", extent=extent, interpolation="nearest",
                    cmap="viridis", vmin=-0.2 * amp, vmax=1.05 * amp,
                )
                residual_limit = max(0.15 * amp, np.nanpercentile(np.abs(residual), 98.0))
                residual_ax.imshow(
                    residual, origin="lower", extent=extent, interpolation="nearest",
                    cmap="RdBu_r", vmin=-residual_limit, vmax=residual_limit,
                )
                for axis in (raw_ax, residual_ax):
                    axis.plot(x0, y0, marker="+", color="white", markersize=9, mew=1.5)
                    axis.add_patch(Ellipse(
                        (x0, y0), FWHM_PER_SIGMA * sx,
                        FWHM_PER_SIGMA * sy,
                        angle=math.degrees(theta), fill=False, edgecolor="white",
                        linewidth=1.0, alpha=0.9,
                    ))
                    axis.set_xlabel("Az offset (arcsec)")
                    axis.set_aspect("equal")
                if peak_x.size:
                    raw_ax.plot(
                        peak_x, peak_y, linestyle="none", marker="x",
                        color="magenta", markersize=7, mew=1.4,
                    )
                    residual_ax.plot(
                        peak_x, peak_y, linestyle="none", marker="x",
                        color="yellow", markersize=7, mew=1.4,
                    )
                raw_ax.set_ylabel("El offset (arcsec)")
                residual_ax.set_ylabel("El offset (arcsec)")
                raw_ax.set_title(
                    f"Obs {row['obsnum']}  S/N={row['snr_a1100']:.1f}\n"
                    f"standard a1100 map"
                )
                residual_ax.set_title(
                    "Gaussian+plane residual\n"
                    f"secondary={row['strongest_positive_secondary_peak_fraction']:.2f} "
                    f"components={row['coherent_residual_component_count']}"
                )
            for column in range(len(page), 3):
                axes[0, column].axis("off")
                axes[1, column].axis("off")
            fig.suptitle(
                "SCI-ALIGN-001 pointing source-quality review — S/N-selected observations\n"
                "Width and ellipticity are descriptive; morphology is not automatically excluded",
                fontsize=14,
            )
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
            pdf.savefig(fig)
            plt.close(fig)


def run(args: argparse.Namespace) -> None:
    frozen_path = args.frozen_input.resolve()
    frozen_root = frozen_path.parent
    verify_checksums(frozen_root)
    frozen = json.loads(frozen_path.read_text())
    if frozen.get("schema") != "sci-align-001-pointing-source-quality-input-v1":
        raise ContractError("unsupported frozen-input schema")
    protocol = frozen["protocol"]
    load_protocol(Path(frozen["protocol_path"]))
    if sha256_file(Path(frozen["protocol_path"])) != frozen["protocol_sha256"]:
        raise ContractError("protocol identity changed after freezing")
    output = args.output.resolve()
    if output.exists():
        raise ContractError(f"output already exists: {output}")
    output.mkdir(parents=True)

    rows: list[dict[str, Any]] = []
    selected_diagnostics: list[tuple[dict[str, Any], dict[str, Any]]] = []
    snr_min = float(protocol["automatic_selection"]["minimum_inclusive"])
    for frozen_row in frozen["rows"]:
        obsnum = int(frozen_row["obsnum"])
        ppt_path = Path(frozen_row["ppt_path"])
        map_path = Path(frozen_row["a1100_map_path"])
        if sha256_file(ppt_path) != frozen_row["ppt_sha256"]:
            raise ContractError(f"obs {obsnum}: PPT identity changed")
        if sha256_file(map_path) != frozen_row["a1100_map_sha256"]:
            raise ContractError(f"obs {obsnum}: map identity changed")
        ppt = ppt_a1100(ppt_path)
        image, weight, x, y, _ = load_pointing_map(map_path)
        metrics, diagnostic = source_quality_metrics(
            image, weight, x, y, ppt, protocol
        )
        row = {
            "obsnum": obsnum,
            "snr_pass": bool(metrics["snr_a1100"] >= snr_min),
            "morphology_disposition": "human_review_required",
            **metrics,
            "ppt_sha256": frozen_row["ppt_sha256"],
            "a1100_map_sha256": frozen_row["a1100_map_sha256"],
        }
        rows.append(row)
        if row["snr_pass"]:
            selected_diagnostics.append((row, diagnostic))

    rows.sort(key=lambda value: int(value["obsnum"]))
    selected_diagnostics.sort(
        key=lambda item: (
            -float(item[0]["strongest_abs_smoothed_residual_fraction_peak"]),
            int(item[0]["obsnum"]),
        )
    )
    Table(rows=rows).write(output / "pointing_source_quality.ecsv", format="ascii.ecsv")
    selected_rows = [row for row in rows if row["snr_pass"]]
    Table(rows=selected_rows).write(output / "snr_selected_pointings.ecsv", format="ascii.ecsv")
    pdf_name = "pointing_source_quality_contact_sheet.pdf"
    render_contact_sheet(output / pdf_name, selected_diagnostics, protocol)
    summary = {
        "schema": "sci-align-001-pointing-source-quality-result-v1",
        "frozen_input_path": str(frozen_path),
        "frozen_input_sha256": sha256_file(frozen_path),
        "protocol_sha256": frozen["protocol_sha256"],
        "observation_count": len(rows),
        "snr_threshold_inclusive": snr_min,
        "snr_pass_count": len(selected_rows),
        "snr_fail_count": len(rows) - len(selected_rows),
        "automatic_morphology_exclusion": False,
        "contact_sheet_order": (
            "descending strongest absolute smoothed off-core residual fraction; "
            "order is review prioritization, not an exclusion ranking"
        ),
        "timing_results_used": False,
        "ptc_signal_samples_read": False,
    }
    write_json(output / "result.json", summary)
    write_checksums(output, (
        "pointing_source_quality.ecsv", "snr_selected_pointings.ecsv",
        pdf_name, "result.json",
    ))
    print(
        f"pointing source-quality audit complete: total={len(rows)} "
        f"snr_pass={len(selected_rows)} output={output}"
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--schema-audit", type=Path, required=True)
    freeze_parser.add_argument("--protocol", type=Path, required=True)
    freeze_parser.add_argument("--unity-root", type=Path, required=True)
    freeze_parser.add_argument("--local-root", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    freeze_parser.set_defaults(func=freeze)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--frozen-input", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.set_defaults(func=run)
    return root


def main() -> None:
    args = parser().parse_args()
    try:
        args.func(args)
    except ContractError as error:
        raise SystemExit(f"ERROR: {error}") from error


if __name__ == "__main__":
    main()
