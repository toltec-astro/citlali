#!/usr/bin/env python3
"""Test split-direction Beammap shifts against retained kernel products.

This read-only diagnostic consumes one completed ``beammap.direction_mode:
all`` reduction and an independently selected detector table.  It compares
the standard/left/right signal maps with the matching retained kernel maps,
then measures common-morphology translations in a nuclear-core mask and a
standard-only vertical-jet mask.  It never launches Citlali or modifies a
reduction product.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import textwrap
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.table import Table  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy.ndimage import binary_dilation, map_coordinates  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

import render_sci_align_001_split_direction_beammaps as base  # noqa: E402


MODES = base.MODES


class ContractError(RuntimeError):
    """A retained product does not satisfy this diagnostic's contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def kernel_plane(
    product: base.FitsProduct,
    map_index: int,
    flag: int,
) -> tuple[np.ndarray, Any]:
    name = f"kernel_det_{map_index}_I"
    candidates = []
    if flag in product.by_flag:
        candidates.append(product.by_flag[flag])
    if None in product.by_flag:
        candidates.append(product.by_flag[None])
    candidates.extend(
        hdus for key, hdus in product.by_flag.items()
        if key not in {flag, None}
    )
    for hdus in candidates:
        try:
            hdu = hdus[name]
        except (KeyError, IndexError):
            continue
        image = np.asarray(hdu.data, dtype=float).squeeze()
        if image.ndim != 2:
            raise ContractError(f"invalid kernel plane geometry for {name}")
        return image, hdu.header.copy()
    raise ContractError(f"required retained kernel extension is missing: {name}")


def robust_background(image: np.ndarray, radius: np.ndarray, inner: float) -> float:
    values = image[np.isfinite(image) & (radius >= inner)]
    if values.size < 20:
        values = image[np.isfinite(image)]
    if values.size == 0:
        raise ContractError("map crop has no finite values")
    return float(np.median(values))


def fit_gaussian_core(
    image: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    cx: float,
    cy: float,
    half_width: float,
) -> dict[str, Any]:
    xx, yy = np.meshgrid(x, y)
    radius = np.hypot(xx - cx, yy - cy)
    finite = np.isfinite(image) & (radius <= half_width)
    if np.count_nonzero(finite) < 50:
        return {"status": "insufficient_finite_support"}
    background = robust_background(image, radius, 0.75 * half_width)
    work = image - background
    peak_flat = np.nanargmax(np.where(finite, work, np.nan))
    peak_row, peak_col = np.unravel_index(peak_flat, work.shape)
    peak = float(work[peak_row, peak_col])
    if not math.isfinite(peak) or peak <= 0.0:
        return {"status": "nonpositive_peak"}
    scale = peak
    z = work[finite] / scale
    xv, yv = xx[finite], yy[finite]
    xmid, ymid = float(x[peak_col]), float(y[peak_row])
    span = max(float(np.ptp(x)), float(np.ptp(y)))
    p0 = np.asarray([1.0, xmid, ymid, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0])
    lower = np.asarray([
        0.0, cx - half_width, cy - half_width, 0.4, 0.4, -math.pi,
        -math.inf, -math.inf, -math.inf,
    ])
    upper = np.asarray([
        math.inf, cx + half_width, cy + half_width,
        max(half_width, 1.0), max(half_width, 1.0), math.pi,
        math.inf, math.inf, math.inf,
    ])

    def residual(p: np.ndarray) -> np.ndarray:
        amp, x0, y0, sx, sy, theta, b0, bx, by = p
        ct, st = math.cos(theta), math.sin(theta)
        dx, dy = xv - x0, yv - y0
        u = ct * dx + st * dy
        v = -st * dx + ct * dy
        model = (
            amp * np.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2))
            + b0 + bx * (xv - cx) / span + by * (yv - cy) / span
        )
        return model - z

    result = least_squares(
        residual, p0, bounds=(lower, upper), loss="soft_l1",
        f_scale=0.03, max_nfev=3000,
    )
    if not result.success or np.any(~np.isfinite(result.x)):
        return {"status": "fit_failed", "message": result.message}
    amp, x0, y0, sx, sy, theta, *_ = result.x
    return {
        "status": "success",
        "x_arcsec": float(x0),
        "y_arcsec": float(y0),
        "major_fwhm_arcsec": float(2.354820045 * max(sx, sy)),
        "minor_fwhm_arcsec": float(2.354820045 * min(sx, sy)),
        "angle_rad": float(theta),
        "amplitude_native": float(amp * scale),
        "residual_rms_fraction_peak": float(np.sqrt(np.mean(result.fun ** 2))),
        "n_pixels": int(result.fun.size),
    }


def sample_local_grid(
    image: np.ndarray,
    wcs: Any,
    cx: float,
    cy: float,
    offsets: np.ndarray,
) -> np.ndarray:
    gx, gy = np.meshgrid(offsets, offsets)
    px, py = wcs.world_to_pixel_values(cx + gx, cy + gy)
    return base.bilinear(image, np.asarray(px).ravel(), np.asarray(py).ravel()).reshape(gx.shape)


def normalize_stack_member(image: np.ndarray, scale: float) -> np.ndarray:
    if not math.isfinite(scale) or scale <= 0.0:
        raise ContractError("invalid positive stack normalization")
    return image / scale


def nanmedian_stack(members: Sequence[np.ndarray]) -> np.ndarray:
    if not members:
        raise ContractError("stack has no detector members")
    with np.errstate(all="ignore"):
        result = np.nanmedian(np.stack(members), axis=0)
    if np.count_nonzero(np.isfinite(result)) < 100:
        raise ContractError("stack has insufficient finite support")
    return result


def shifted_reference(
    reference: np.ndarray,
    dx_arcsec: float,
    dy_arcsec: float,
    pixel_arcsec: float,
) -> np.ndarray:
    rows, cols = np.indices(reference.shape, dtype=float)
    # target(x,y) = reference(x-dx,y-dy) places a feature at +dx,+dy.
    coords = np.asarray([
        rows - dy_arcsec / pixel_arcsec,
        cols - dx_arcsec / pixel_arcsec,
    ])
    return map_coordinates(
        reference, coords, order=1, mode="constant", cval=np.nan,
        prefilter=False,
    )


def fit_common_translation(
    reference: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    offsets: np.ndarray,
) -> dict[str, Any]:
    pixel = float(np.median(np.diff(offsets)))
    yy, xx = np.meshgrid(offsets, offsets, indexing="ij")
    # Keep the residual vector's membership fixed throughout optimization.
    # The interior guard also guarantees that every allowed translation stays
    # inside the sampled reference grid.
    guard = int(math.ceil(6.0 / abs(pixel))) + 2
    interior = np.zeros(reference.shape, dtype=bool)
    if 2 * guard < min(reference.shape):
        interior[guard:-guard, guard:-guard] = True
    fixed = mask & interior & np.isfinite(reference) & np.isfinite(target)
    if np.count_nonzero(fixed) < 20:
        return {"status": "insufficient_mask_support"}
    scale = float(np.nanmax(reference[fixed]) - np.nanmedian(reference[fixed]))
    if not math.isfinite(scale) or scale <= 0.0:
        return {"status": "nonpositive_reference_contrast"}

    def residual(p: np.ndarray) -> np.ndarray:
        dx, dy, amp, b0, bx, by = p
        shifted = shifted_reference(reference, dx, dy, pixel)
        model = (
            amp * shifted + b0 + bx * xx / max(abs(offsets[0]), 1.0)
            + by * yy / max(abs(offsets[0]), 1.0)
        )
        values = (model[fixed] - target[fixed]) / scale
        # Retain a constant-length residual even if an isolated NaN exists in
        # the reference stack inside the otherwise valid fitting support.
        return np.where(np.isfinite(values), values, 1.0e3)

    fit = least_squares(
        residual, np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        bounds=(
            np.asarray([-6.0, -6.0, 0.2, -math.inf, -math.inf, -math.inf]),
            np.asarray([6.0, 6.0, 2.5, math.inf, math.inf, math.inf]),
        ),
        loss="soft_l1", f_scale=0.05, max_nfev=2000,
    )
    if not fit.success or np.any(~np.isfinite(fit.x)):
        return {"status": "fit_failed", "message": fit.message}
    dx, dy, amp, b0, bx, by = fit.x
    return {
        "status": "success",
        "dx_arcsec": float(dx), "dy_arcsec": float(dy),
        "amplitude": float(amp), "background": float(b0),
        "background_x": float(bx), "background_y": float(by),
        "robust_cost": float(fit.cost), "n_pixels": int(fit.fun.size),
        "residual_rms_fraction_contrast": float(np.sqrt(np.mean(fit.fun ** 2))),
    }


def mask_definitions(
    standard: np.ndarray,
    offsets: np.ndarray,
    fwhm: float,
) -> dict[str, np.ndarray]:
    yy, xx = np.meshgrid(offsets, offsets, indexing="ij")
    radius = np.hypot(xx, yy)
    outer = radius >= 2.0 * fwhm
    background = float(np.nanmedian(standard[outer]))
    noise = 1.4826 * float(np.nanmedian(np.abs(standard[outer] - background)))
    contrast = standard - background
    threshold = max(0.025 * float(np.nanmax(contrast)), 3.0 * noise)
    core = radius <= 0.85 * fwhm
    vertical_support = (
        (np.abs(xx) <= 0.85 * fwhm)
        & (radius > 0.85 * fwhm)
        & (np.abs(yy) <= 2.5 * fwhm)
        & (contrast >= threshold)
    )
    # A one-pixel dilation preserves a standard-only definition while giving
    # the translated directional morphology room to move inside the mask.
    expanded = binary_dilation(
        vertical_support,
        structure=np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
        iterations=1,
    )
    return {
        "nuclear_core": core,
        "vertical_jet": expanded,
        "core_plus_vertical_jet": core | expanded,
    }


def registration_rows(
    stacks: dict[str, dict[str, np.ndarray]],
    masks: dict[str, np.ndarray],
    offsets: np.ndarray,
    scan: base.ScanSummary,
) -> list[dict[str, Any]]:
    rows = []
    denominator = scan.right_rate_arcsec_s - scan.left_rate_arcsec_s
    for family in ("signal", "kernel"):
        regions = masks if family == "signal" else {"kernel_core": masks["nuclear_core"]}
        for region, mask in regions.items():
            left = fit_common_translation(
                stacks[family]["standard"], stacks[family]["left"], mask, offsets,
            )
            right = fit_common_translation(
                stacks[family]["standard"], stacks[family]["right"], mask, offsets,
            )
            row: dict[str, Any] = {
                "family": family, "region": region,
                "mask_pixel_count": int(np.count_nonzero(mask)),
                "left_status": left["status"], "right_status": right["status"],
            }
            for mode, fit in (("left", left), ("right", right)):
                for key, value in fit.items():
                    if key not in {"status", "message"}:
                        row[f"{mode}_{key}"] = value
            if left["status"] == right["status"] == "success":
                dx = right["dx_arcsec"] - left["dx_arcsec"]
                dy = right["dy_arcsec"] - left["dy_arcsec"]
                parallel = dx * scan.axis_x + dy * scan.axis_y
                perpendicular = dx * scan.cross_x + dy * scan.cross_y
                row.update({
                    "status": "success",
                    "delta_x_right_minus_left_arcsec": dx,
                    "delta_y_right_minus_left_arcsec": dy,
                    "delta_parallel_right_minus_left_arcsec": parallel,
                    "delta_perpendicular_right_minus_left_arcsec": perpendicular,
                    "timing_equivalent_ms": 1000.0 * parallel / denominator,
                })
            else:
                row["status"] = "fit_failed"
            rows.append(row)
    return rows


def classify_evidence(
    registrations: Sequence[dict[str, Any]],
    pixel_arcsec: float,
) -> dict[str, Any]:
    rows = {(row["family"], row["region"]): row for row in registrations}
    nuclear = rows[("signal", "nuclear_core")]
    combined = rows[("signal", "core_plus_vertical_jet")]
    kernel = rows[("kernel", "kernel_core")]
    tolerance = max(0.5 * abs(pixel_arcsec), 0.25)
    result: dict[str, Any] = {
        "schema": "sci-align-001-split-direction-transfer-decision-v1",
        "scope": (
            "synthetic kernel generated in RTC after calibration/extinction and "
            "propagated through subsequent RTC filtering, PTC cleaning, and mapmaking"
        ),
        "out_of_scope": (
            "raw detector-data/timestamp association and operations before synthetic "
            "kernel generation"
        ),
        "translation_tolerance_arcsec": tolerance,
        "tolerance_basis": "one-half of the diagnostic stack pixel, with a 0.25-arcsec floor",
        "statistical_scope": (
            "resolution-bounded descriptive classification; detector-level values are "
            "retained but no independent-pixel likelihood or formal confidence interval is claimed"
        ),
    }
    if any(row["status"] != "success" for row in (nuclear, combined, kernel)):
        result.update({
            "classification": "inconclusive_fit_failure",
            "downstream_filtering_artifact_disposition": "inconclusive",
        })
        return result
    signal = float(nuclear["delta_parallel_right_minus_left_arcsec"])
    signal_combined = float(combined["delta_parallel_right_minus_left_arcsec"])
    kernel_shift = float(kernel["delta_parallel_right_minus_left_arcsec"])
    signal_resolved = abs(signal) >= 2.0 * tolerance
    morphology_stable = abs(signal_combined - signal) <= tolerance
    kernel_centered = abs(kernel_shift) <= tolerance
    kernel_comoving = abs(kernel_shift - signal) <= tolerance
    result.update({
        "signal_nuclear_right_minus_left_arcsec": signal,
        "signal_core_plus_jet_right_minus_left_arcsec": signal_combined,
        "kernel_right_minus_left_arcsec": kernel_shift,
        "signal_resolved": signal_resolved,
        "signal_morphology_stable": morphology_stable,
        "kernel_centered": kernel_centered,
        "kernel_comoving_with_signal": kernel_comoving,
    })
    if not signal_resolved:
        classification = "no_resolved_signal_translation"
        disposition = "inconclusive"
    elif not morphology_stable:
        classification = "signal_translation_morphology_sensitive"
        disposition = "inconclusive"
    elif kernel_centered and not kernel_comoving:
        classification = "signal_shift_with_centered_downstream_transfer_kernel"
        disposition = "strongly_disfavored_within_kernel_scope"
    elif kernel_comoving and not kernel_centered:
        classification = "signal_and_downstream_transfer_kernel_comove"
        disposition = "favored_within_kernel_scope"
    else:
        classification = "mixed_kernel_signal_translation"
        disposition = "inconclusive"
    result.update({
        "classification": classification,
        "downstream_filtering_artifact_disposition": disposition,
    })
    return result


def draw_stack_row(
    axes: Sequence[Any],
    images: dict[str, np.ndarray],
    offsets: np.ndarray,
    title_prefix: str,
) -> None:
    finite = np.concatenate([image[np.isfinite(image)] for image in images.values()])
    low, high = np.percentile(finite, [2.0, 99.5])
    extent = (offsets[0], offsets[-1], offsets[0], offsets[-1])
    for ax, mode in zip(axes, MODES):
        ax.imshow(images[mode], origin="lower", extent=extent, cmap="viridis",
                  vmin=low, vmax=high, interpolation="nearest", aspect="equal")
        ax.set_title(f"{title_prefix} {mode}", fontsize=10)
        ax.set_xlabel("Along-scan offset (arcsec)", fontsize=8)
        ax.set_ylabel("Cross-scan offset (arcsec)", fontsize=8)
        ax.tick_params(labelsize=7)


def make_summary_pdf(
    path: Path,
    stacks: dict[str, dict[str, np.ndarray]],
    masks: dict[str, np.ndarray],
    registrations: list[dict[str, Any]],
    offsets: np.ndarray,
    observation: int,
    array_name: str,
    n_detectors: int,
    decision: dict[str, Any],
) -> None:
    extent = (offsets[0], offsets[-1], offsets[0], offsets[-1])
    rows_by_key = {(r["family"], r["region"]): r for r in registrations}
    with PdfPages(path) as pdf:
        fig, axes = plt.subplots(2, 3, figsize=(11, 8.5), constrained_layout=True)
        draw_stack_row(axes[0], stacks["signal"], offsets, "signal")
        difference = stacks["signal"]["left"] - stacks["signal"]["right"]
        limit = float(np.nanpercentile(np.abs(difference), 99.0))
        axes[1, 0].imshow(difference, origin="lower", extent=extent, cmap="coolwarm",
                          vmin=-limit, vmax=limit, interpolation="nearest", aspect="equal")
        axes[1, 0].set_title("signal left - right", fontsize=10)
        colors = {"nuclear_core": "white", "vertical_jet": "cyan"}
        for name, color in colors.items():
            axes[1, 0].contour(offsets, offsets, masks[name].astype(float),
                               levels=[0.5], colors=[color], linewidths=1.0)
        axes[1, 0].set_xlabel("Along-scan offset (arcsec)", fontsize=8)
        axes[1, 0].set_ylabel("Cross-scan offset (arcsec)", fontsize=8)
        axes[1, 0].tick_params(labelsize=7)

        center = len(offsets) // 2
        for mode, color in zip(MODES, ("0.2", "tab:blue", "tab:orange")):
            profile = base.normalized_profile(stacks["signal"][mode][center, :])
            axes[1, 1].plot(offsets, profile, color=color, label=mode)
        axes[1, 1].set_title("signal along-scan stack", fontsize=10)
        axes[1, 1].set_xlabel("Along-scan offset (arcsec)", fontsize=8)
        axes[1, 1].set_ylabel("Normalized signal", fontsize=8)
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(alpha=0.2)

        axes[1, 2].axis("off")
        lines = [
            f"Obs {observation} {array_name}",
            f"clean stack detectors: {n_detectors}", "",
            "Common-morphology right-left results:",
        ]
        for region in ("nuclear_core", "vertical_jet", "core_plus_vertical_jet"):
            row = rows_by_key[("signal", region)]
            if row["status"] == "success":
                lines.append(
                    f"{region}: {row['delta_parallel_right_minus_left_arcsec']:+.3f} arcsec, "
                    f"{row['timing_equivalent_ms']:+.2f} ms"
                )
            else:
                lines.append(f"{region}: {row['status']}")
        axes[1, 2].text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=9)
        fig.suptitle("SCI-ALIGN-001 signal morphology registration", fontsize=14)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(11, 8.5), constrained_layout=True)
        draw_stack_row(axes[0], stacks["kernel"], offsets, "kernel")
        difference = stacks["kernel"]["left"] - stacks["kernel"]["right"]
        limit = max(float(np.nanpercentile(np.abs(difference), 99.0)), np.finfo(float).eps)
        axes[1, 0].imshow(difference, origin="lower", extent=extent, cmap="coolwarm",
                          vmin=-limit, vmax=limit, interpolation="nearest", aspect="equal")
        axes[1, 0].set_title("kernel left - right", fontsize=10)
        axes[1, 0].set_xlabel("Along-scan offset (arcsec)", fontsize=8)
        axes[1, 0].set_ylabel("Cross-scan offset (arcsec)", fontsize=8)
        axes[1, 0].tick_params(labelsize=7)
        center = len(offsets) // 2
        for mode, color in zip(MODES, ("0.2", "tab:blue", "tab:orange")):
            profile = base.normalized_profile(stacks["kernel"][mode][center, :])
            axes[1, 1].plot(offsets, profile, color=color, label=mode)
        axes[1, 1].set_title("kernel along-scan stack", fontsize=10)
        axes[1, 1].set_xlabel("Along-scan offset (arcsec)", fontsize=8)
        axes[1, 1].set_ylabel("Normalized kernel", fontsize=8)
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(alpha=0.2)
        axes[1, 2].axis("off")
        row = rows_by_key[("kernel", "kernel_core")]
        lines = ["Kernel common-morphology result:"]
        if row["status"] == "success":
            lines.extend([
                f"right-left parallel: {row['delta_parallel_right_minus_left_arcsec']:+.4f} arcsec",
                f"right-left perpendicular: {row['delta_perpendicular_right_minus_left_arcsec']:+.4f} arcsec",
                f"timing equivalent: {row['timing_equivalent_ms']:+.3f} ms",
            ])
        else:
            lines.append(row["status"])
        lines.extend([
            "", "Classification:",
        ])
        lines.extend(textwrap.wrap(str(decision["classification"]), width=38))
        lines.extend([
            "", "Interpretation:",
            "kernel ~ signal shift:",
            "  downstream transfer artifact favored",
            "kernel centered; signal shifts:",
            "  downstream filtering strongly disfavored",
            "raw timestamp association is out of scope",
        ])
        axes[1, 2].text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=8)
        fig.suptitle("SCI-ALIGN-001 retained kernel transfer test", fontsize=14)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)


def run(args: argparse.Namespace) -> None:
    raw_dir = base.discover_raw_dir(args.reduction_root)
    standard_apt = base.require_unique(
        [path for path in raw_dir.glob("apt_*.ecsv") if base.is_standard_apt(path)],
        "standard Beammap APT",
    )
    apt_paths = {mode: base.product_apt_path(standard_apt, mode) for mode in MODES}
    tables = {mode: Table.read(path, format="ascii.ecsv") for mode, path in apt_paths.items()}
    for mode, table in tables.items():
        base.require_columns(table, base.REQUIRED_APT_COLUMNS, f"{mode} APT")
    indices = {mode: base.uid_index(table, f"{mode} APT") for mode, table in tables.items()}
    selection_path = args.selection.resolve()
    if not selection_path.is_file():
        raise ContractError(f"selection table is missing: {selection_path}")
    selection = Table.read(selection_path, format="ascii.ecsv")
    if "uid" not in selection.colnames or len(selection) == 0:
        raise ContractError("selection table lacks a nonempty uid column")
    selected_uids = [base.int_scalar(value) for value in selection["uid"]]
    if len(set(selected_uids)) != len(selected_uids) or any(uid < 0 for uid in selected_uids):
        raise ContractError("selection table contains invalid or duplicate UIDs")
    for uid in selected_uids:
        if any(uid not in indices[mode] for mode in MODES):
            raise ContractError(f"selected uid={uid} is absent from a directional APT")
    observation = base.int_scalar(tables["standard"].meta.get("obsnum"))
    if observation < 0:
        raise ContractError("standard APT lacks obsnum metadata")
    registry = base.require_unique(
        raw_dir.parent.rglob("beammap_direction_scan_registry_all.csv"),
        "all-mode direction scan registry",
    )
    scan = base.scan_summary(registry)
    fits_paths = {mode: base.discover_fits(raw_dir, args.array, mode) for mode in MODES}
    output = args.output.resolve()
    if output.exists():
        raise ContractError(f"refusing existing output directory: {output}")

    offsets = np.arange(
        -args.half_width_arcsec,
        args.half_width_arcsec + 0.5 * args.pixel_arcsec,
        args.pixel_arcsec,
    )
    stack_members = {
        family: {mode: [] for mode in MODES}
        for family in ("signal", "kernel")
    }
    kernel_rows = []
    stack_uids = []
    clean_count = 0
    input_paths = [selection_path, registry, *apt_paths.values()]
    for values in fits_paths.values():
        input_paths.extend(values)

    with contextlib.ExitStack() as stack:
        products = {mode: base.FitsProduct(paths, stack) for mode, paths in fits_paths.items()}
        for uid in selected_uids:
            apt = {
                mode: base.row_values(tables[mode], indices[mode][uid])
                for mode in MODES
            }
            clean = all(
                int(apt[mode].get("flag", -1)) == 0
                and int(apt[mode].get("flag2", 0)) == 0
                for mode in MODES
            )
            mode_maps = {}
            mode_kernel_fits = {}
            for mode in MODES:
                map_index = indices[mode][uid]
                signal, weight, header = products[mode].planes(
                    map_index, int(apt[mode]["flag"]),
                )
                kernel, kernel_header = kernel_plane(
                    products[mode], map_index, int(apt[mode]["flag"]),
                )
                if signal.shape != kernel.shape or base.wcs_signature(header) != base.wcs_signature(kernel_header):
                    raise ContractError(
                        f"uid={uid} mode={mode} signal/kernel WCS or shape differs"
                    )
                wcs = base.spatial_wcs(header)
                x, y = base.image_coordinates(wcs, signal.shape)
                cx = float(apt["standard"]["x_t_raw"])
                cy = float(apt["standard"]["y_t_raw"])
                ys, xs = base.crop_mask(x, y, cx, cy, args.half_width_arcsec)
                signal_masked = base.masked_signal(signal, weight)
                kernel_masked = np.where(np.isfinite(kernel) & (weight > 0.0), kernel, np.nan)
                mode_kernel_fits[mode] = fit_gaussian_core(
                    kernel_masked[ys, xs], x[xs], y[ys], cx, cy,
                    min(args.kernel_fit_half_width_arcsec, args.half_width_arcsec),
                )
                mode_maps[mode] = (signal_masked, kernel_masked, wcs)

            row: dict[str, Any] = {
                "uid": uid,
                "array": int(apt["standard"]["array"]),
                "nw": int(apt["standard"]["nw"]),
                "directional_clean": clean,
            }
            signal_dx = apt["right"]["x_t_raw"] - apt["left"]["x_t_raw"]
            signal_dy = apt["right"]["y_t_raw"] - apt["left"]["y_t_raw"]
            signal_parallel = signal_dx * scan.axis_x + signal_dy * scan.axis_y
            row.update({
                "signal_delta_parallel_right_minus_left_arcsec": signal_parallel,
                "signal_delta_perpendicular_right_minus_left_arcsec": (
                    signal_dx * scan.cross_x + signal_dy * scan.cross_y
                ),
                "signal_timing_equivalent_ms": (
                    1000.0 * signal_parallel
                    / (scan.right_rate_arcsec_s - scan.left_rate_arcsec_s)
                ),
            })
            for mode in MODES:
                fit = mode_kernel_fits[mode]
                row[f"{mode}_kernel_fit_status"] = fit["status"]
                for key, value in fit.items():
                    if key not in {"status", "message"}:
                        row[f"{mode}_kernel_{key}"] = value
            if all(mode_kernel_fits[mode]["status"] == "success" for mode in MODES):
                left, right = mode_kernel_fits["left"], mode_kernel_fits["right"]
                dx = right["x_arcsec"] - left["x_arcsec"]
                dy = right["y_arcsec"] - left["y_arcsec"]
                parallel = dx * scan.axis_x + dy * scan.axis_y
                perpendicular = dx * scan.cross_x + dy * scan.cross_y
                row.update({
                    "kernel_fit_status": "success",
                    "kernel_delta_parallel_right_minus_left_arcsec": parallel,
                    "kernel_delta_perpendicular_right_minus_left_arcsec": perpendicular,
                    "kernel_timing_equivalent_ms": (
                        1000.0 * parallel
                        / (scan.right_rate_arcsec_s - scan.left_rate_arcsec_s)
                    ),
                    "kernel_minus_signal_parallel_arcsec": parallel - signal_parallel,
                })
            else:
                row["kernel_fit_status"] = "fit_failed"
            kernel_rows.append(row)

            if not clean or row["kernel_fit_status"] != "success":
                continue
            clean_count += 1
            stack_uids.append(uid)
            standard_scale = float(apt["standard"]["amp"])
            standard_kernel_peak = float(mode_kernel_fits["standard"]["amplitude_native"])
            cx = float(apt["standard"]["x_t_raw"])
            cy = float(apt["standard"]["y_t_raw"])
            for mode in MODES:
                signal, kernel, wcs = mode_maps[mode]
                stack_members["signal"][mode].append(normalize_stack_member(
                    sample_local_grid(signal, wcs, cx, cy, offsets), standard_scale,
                ))
                stack_members["kernel"][mode].append(normalize_stack_member(
                    sample_local_grid(kernel, wcs, cx, cy, offsets), standard_kernel_peak,
                ))

    if clean_count < args.minimum_clean_detectors:
        raise ContractError(
            f"only {clean_count} clean kernel-fit detectors; "
            f"minimum is {args.minimum_clean_detectors}"
        )
    stacks = {
        family: {
            mode: nanmedian_stack(stack_members[family][mode])
            for mode in MODES
        }
        for family in ("signal", "kernel")
    }
    standard_widths = []
    for uid in stack_uids:
        values = base.row_values(tables["standard"], indices["standard"][uid])
        standard_widths.append(math.sqrt(values["a_fwhm"] * values["b_fwhm"]))
    fwhm = float(np.median(standard_widths))
    masks = mask_definitions(stacks["signal"]["standard"], offsets, fwhm)
    registrations = registration_rows(stacks, masks, offsets, scan)
    decision = classify_evidence(registrations, args.pixel_arcsec)

    output.mkdir(parents=True)
    kernel_metrics = output / "detector_kernel_metrics.ecsv"
    registration_path = output / "stack_registration.ecsv"
    decision_path = output / "diagnostic_decision.json"
    Table(rows=kernel_rows).write(kernel_metrics, format="ascii.ecsv")
    Table(rows=registrations).write(registration_path, format="ascii.ecsv")
    write_json(decision_path, decision)
    stack_path = output / "stacked_maps.npz"
    np.savez_compressed(
        stack_path, offsets_arcsec=offsets,
        signal_standard=stacks["signal"]["standard"],
        signal_left=stacks["signal"]["left"], signal_right=stacks["signal"]["right"],
        kernel_standard=stacks["kernel"]["standard"],
        kernel_left=stacks["kernel"]["left"], kernel_right=stacks["kernel"]["right"],
        mask_nuclear_core=masks["nuclear_core"],
        mask_vertical_jet=masks["vertical_jet"],
        mask_core_plus_vertical_jet=masks["core_plus_vertical_jet"],
    )
    pdf_path = output / f"split_direction_transfer_o{observation}_{args.array}.pdf"
    make_summary_pdf(
        pdf_path, stacks, masks, registrations, offsets, observation,
        args.array, clean_count, decision,
    )
    manifest = {
        "schema": "sci-align-001-split-direction-transfer-v1",
        "tool": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "observation_number": observation,
        "array": args.array,
        "selection": {
            "path": str(selection_path), "sha256": sha256_file(selection_path),
            "selected_count": len(selected_uids),
            "clean_kernel_stack_count": clean_count,
            "uses_kernel_or_directional_result_for_membership": False,
        },
        "position_frame": "raw_altaz_detector_map",
        "scan": scan.__dict__,
        "stack": {
            "estimator": "equal-detector nanmedian after standard-amplitude normalization",
            "pixel_arcsec": args.pixel_arcsec,
            "half_width_arcsec": args.half_width_arcsec,
            "standard_geometric_fwhm_arcsec": fwhm,
            "jet_mask_authority": "standard signal stack only; vertical significant support outside nuclear core",
        },
        "kernel_fit": {
            "model": "elliptical Gaussian plus affine background",
            "fit_half_width_arcsec": args.kernel_fit_half_width_arcsec,
        },
        "registration": {
            "model": "shared standard morphology translated into left/right with amplitude and affine background",
            "interpretation": {
                "kernel_shift_similar_to_signal_shift": "pipeline transfer or mapmaking artifact favored",
                "kernel_centered_signal_shifted": "ordinary filtering artifact strongly disfavored",
            },
        },
        "decision": decision,
        "inputs": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(set(input_paths))
        ],
    }
    write_json(output / "manifest.json", manifest)
    outputs = [
        kernel_metrics, registration_path, decision_path, stack_path, pdf_path,
        output / "manifest.json",
    ]
    (output / "SHA256SUMS").write_text("".join(
        f"{sha256_file(path)}  {path.name}\n" for path in sorted(outputs)
    ))
    print(
        f"transfer diagnostic complete: obs={observation} array={args.array} "
        f"selected={len(selected_uids)} clean_stack={clean_count} output={output}"
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--reduction-root", type=Path, required=True)
    result.add_argument("--selection", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--array", choices=sorted(base.ARRAY_IDS), default="a1100")
    result.add_argument("--half-width-arcsec", type=float, default=20.0)
    result.add_argument("--pixel-arcsec", type=float, default=1.0)
    result.add_argument("--kernel-fit-half-width-arcsec", type=float, default=12.0)
    result.add_argument("--minimum-clean-detectors", type=int, default=30)
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        if args.half_width_arcsec <= 0.0 or args.pixel_arcsec <= 0.0:
            raise ContractError("positive half-width and pixel scale are required")
        if args.kernel_fit_half_width_arcsec <= 0.0:
            raise ContractError("positive kernel fit half-width is required")
        run(args)
    except (ContractError, base.ContractError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
