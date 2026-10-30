"""Saved-map inspection and analytic sampling arithmetic; never fit or reduce."""

import argparse
import hashlib
import json
from pathlib import Path

from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm
import numpy as np
import yaml

ROOT = Path(
    "/Users/gwilson/work_toltec/local_data/2026-refactor/projects/COM01-local-trial-20261027/pointings-weight-control-152418-20261030/reduced/redu00"
)


def load(p):
    return json.loads(p.read_text())


def bind(p):
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def xy(h, shape):
    row, col = np.indices(shape)
    return (
        (col + 1 - h["CRPIX1"]) * h["CDELT1"] + h["CRVAL1"],
        (row + 1 - h["CRPIX2"]) * h["CDELT2"] + h["CRVAL2"],
    )


def gaussian(p, x, y):
    dx, dy = x - p[1], y - p[2]
    u = np.cos(p[5]) * dx + np.sin(p[5]) * dy
    v = -np.sin(p[5]) * dx + np.cos(p[5]) * dy
    return p[0] * np.exp(-0.5 * ((u / p[3]) ** 2 + (v / p[4]) ** 2))


def sampling(p, step, sub):
    # Uniform area quadrature inside every pixel of a fixed, ample source domain.
    # Native pixels are centered at integer multiples of the step (offset origin).
    centers = np.arange(-40, 40 + step / 2, step)
    xx, yy = np.meshgrid(centers, centers)
    held = gaussian(p, xx, yy)
    energy = error = signed = absolute = total = maximum = 0.0
    offsets = (np.arange(sub) + 0.5) / sub - 0.5
    for oy in offsets:
        for ox in offsets:
            g = gaussian(p, xx + ox * step, yy + oy * step)
            d = held - g
            energy += np.sum(g * g)
            error += np.sum(d * d)
            signed += np.sum(d)
            absolute += np.sum(abs(d))
            total += np.sum(g)
            maximum = max(maximum, float(np.max(abs(d))))
    return dict(
        step_arcsec=step,
        subpixels_per_axis=sub,
        relative_L2_error=float(np.sqrt(error / energy)),
        signed_brightness_area_error=float(signed / total),
        absolute_brightness_area_error=float(absolute / total),
        max_pointwise_error_over_peak=maximum / p[0],
    )


def inspect(output, figures):
    run = load(ROOT / "RUN.json")
    assert run["status"] == "completed" and run["final_cycle"] == 10
    result = dict(
        scope="saved noise-weighted evidence; no reduction or source refit",
        run=bind(ROOT / "RUN.json"),
        arrays={},
    )
    fig, ax = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    sf, sa = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for row, array in enumerate(("a1100", "a1400")):
        cycles = []
        last_mask = None
        for cycle in range(10):
            stage = ROOT / ".point-work" / f"weighted-{cycle:02d}"
            path = stage / f"feedback-{array}"
            feedback = load(path / "feedback.json")
            ref = run["feedback"]["cycles"][cycle]["source_selection"][array]
            assert feedback == ref and feedback["status"] == "ready"
            assert (
                feedback["threshold"] == 3
                and feedback["gain"] == 1
                and feedback["replacement"]
            )
            model = Path(feedback["model_path"])
            mp = Path(feedback["map_path"])
            assert bind(model)["sha256"] == feedback["model_sha256"]
            assert bind(mp)["sha256"] == feedback["map_sha256"]
            request = load(
                ROOT / ".point-work" / f"weighted-{cycle + 1:02d}-request.json"
            )
            for key, pth in [
                ("candidate", model),
                ("feedback", path / "feedback.json"),
                ("bundle", mp.parent / "bundle.yaml"),
            ]:
                assert bind(pth) == request["models"][array][key]
            nextplan = yaml.safe_load(
                (
                    ROOT
                    / ".point-work"
                    / f"weighted-{cycle + 1:02d}/products/{array}/plan.yaml"
                ).read_text()
            )
            nextref = run["feedback"]["cycles"][cycle + 1]["native"]
            nextpath = Path(nextref["path"])
            assert bind(nextpath) == nextref
            nextcycle = yaml.safe_load(nextpath.read_text())
            planpath = nextpath.parent / array / "plan.yaml"
            assert (
                bind(planpath)["sha256"]
                == nextcycle["arrays"][array]["files"]["plan.yaml"]
            )
            requestpath = (
                ROOT / ".point-work" / f"weighted-{cycle + 1:02d}-request.json"
            )
            assert bind(requestpath)["sha256"] == nextplan["request_sha256"]
            assert nextplan["source_model"] == "sha256:" + feedback["model_sha256"]
            with fits.open(model) as h, fits.open(mp) as m:
                assert all(
                    hdu.verify_checksum() == 1 and hdu.verify_datasum() == 1
                    for hdu in h
                )
                z = h[0].data.copy()
                mask = h["ADMITTED"].data.astype(bool)
                valid = h["VALID"].data.astype(bool)
                header = h[0].header.copy()
                assert header["CDELT1"] == -2 and header["CDELT2"] == 2
                assert header["CUNIT1"] == header["CUNIT2"] == "arcsec"
                x, y = xy(header, z.shape)
                mh = m[0].header
                cols = np.rint(
                    (x - mh["CRVAL1"]) / mh["CDELT1"] + mh["CRPIX1"] - 1
                ).astype(int)
                rows = np.rint(
                    (y - mh["CRVAL2"]) / mh["CDELT2"] + mh["CRPIX2"] - 1
                ).astype(int)
                assert np.array_equal(
                    x, (cols + 1 - mh["CRPIX1"]) * mh["CDELT1"] + mh["CRVAL1"]
                )
                assert np.array_equal(
                    y, (rows + 1 - mh["CRPIX2"]) * mh["CDELT2"] + mh["CRVAL2"]
                )
                signal = m[0].data[rows, cols].copy()
                assert np.all(m["SCI_SUPPORT"].data[rows[valid], cols[valid]])
            p = np.array(feedback["guard"]["accepted_parameters"])
            relative = signal - p[6]
            g = gaussian(p, x, y)
            domain = np.hypot(x, y) <= 40
            assert np.array_equal(valid, domain)
            assert np.all(z[valid & ~mask] == 0)
            assert np.array_equal(z[mask], relative[mask])
            assert not feedback["candidate_summary"]["boundary_contact"]
            assert not feedback["candidate_summary"]["unsupported_contact"]
            radius = np.hypot(x - p[1], y - p[2])
            profiles = []
            for low in range(0, 24, 2):
                region = valid & (radius >= low) & (radius < low + 2)
                outside = region & ~mask
                profiles.append(
                    dict(
                        radius_mid_arcsec=low + 1,
                        pixels=int(region.sum()),
                        selected_pixels=int((region & mask).sum()),
                        map_mean=float(relative[region].mean()),
                        feedback_mean=float(z[region].mean()),
                        gaussian_mean=float(g[region].mean()),
                        excluded_mean=float((relative[region] - z[region]).mean()),
                        unselected_positive_pixels=int(
                            (outside & (relative > 0)).sum()
                        ),
                        unselected_negative_pixels=int(
                            (outside & (relative < 0)).sum()
                        ),
                    )
                )
            sectors = []
            angle = np.arctan2(y - p[2], x - p[1])
            for lo, hi in [
                (-np.pi, -np.pi / 2),
                (-np.pi / 2, 0),
                (0, np.pi / 2),
                (np.pi / 2, np.pi),
            ]:
                region = (
                    valid
                    & ~mask
                    & (radius >= 8)
                    & (radius < 16)
                    & (angle >= lo)
                    & (angle < hi)
                )
                sectors.append(
                    dict(
                        angle_degrees=[float(np.rad2deg(lo)), float(np.rad2deg(hi))],
                        pixels=int(region.sum()),
                        mean=float(relative[region].mean()),
                    )
                )
            entry = dict(
                cycle=cycle,
                model=bind(model),
                feedback=bind(path / "feedback.json"),
                map=bind(mp),
                parameters=p.tolist(),
                selection=feedback["candidate_summary"],
                gaussian_fraction_outside_mask=float(g[~mask].sum() / g.sum()),
                changed_selection_pixels=None
                if last_mask is None
                else int((mask != last_mask).sum()),
                profiles=profiles,
                excluded_8_16_arcsec_sectors=sectors,
            )
            cycles.append(entry)
            last_mask = mask.copy()
            if cycle == 0:
                firstmask = mask.copy()
        # M9's published source is what M10 actually consumed.
        arr = dict(cycles=cycles, sampling={})
        for step in (2.0, 1.0):
            a = sampling(p, step, 16)
            b = sampling(p, step, 32)
            assert abs(a["relative_L2_error"] - b["relative_L2_error"]) < 0.001
            arr["sampling"][str(step)] = dict(coarse=a, refined=b)
        result["arrays"][array] = arr
        for col, (values, title) in enumerate(
            [
                (relative, "Saved M9, fitted background removed"),
                (z, "Actual feedback to M10"),
            ]
        ):
            im = ax[row, col].pcolormesh(
                x,
                y,
                np.where(valid, values, np.nan),
                shading="nearest",
                cmap="RdBu_r",
                norm=AsinhNorm(linear_width=5, vmin=-520, vmax=520),
            )
            ax[row, col].contour(
                x, y, firstmask, levels=[0.5], colors=["#f5a623"], linewidths=1.3
            )
            ax[row, col].contour(
                x, y, mask, levels=[0.5], colors=["black"], linewidths=1.2
            )
            ax[row, col].plot(p[1], p[2], "+", color="lime", markersize=8)
            ax[row, col].set(
                title=f"{array}: {title}",
                xlim=(p[1] - 20, p[1] + 20),
                ylim=(p[2] - 20, p[2] + 20),
                aspect="equal",
                xlabel="Azimuth offset (arcsec)",
                ylabel="Elevation offset (arcsec)",
            )
            bar = fig.colorbar(
                im, ax=ax[row, col], shrink=0.75, label="mJy / nominal beam"
            )
            bar.set_ticks([-100, -20, 0, 20, 100, 500])
        profiles = cycles[-1]["profiles"]
        rr = [v["radius_mid_arcsec"] for v in profiles]
        for key, label in [
            ("map_mean", "Map − fitted constant"),
            ("feedback_mean", "Feedback"),
            ("gaussian_mean", "Existing Gaussian fit"),
        ]:
            ax[row, 2].plot(rr, [v[key] for v in profiles], ".-", label=label)
        ax[row, 2].set(
            title=f"{array}: circular mean profile",
            xlabel="Radius from fitted center (arcsec)",
            ylabel="mJy / nominal beam",
            ylim=(-15, 100),
            xlim=(3, 23),
        )
        ax[row, 2].axhline(0, color="grey", lw=0.7)
        ax[row, 2].grid(alpha=0.2)
        ax[row, 2].legend(fontsize=8)
        xx = np.linspace(p[1] - 14, p[1] + 14, 2000)
        yy = np.full_like(xx, p[2])
        sa[row, 0].plot(
            xx - p[1],
            gaussian(p, xx, yy),
            label="Continuous saved Gaussian",
            color="black",
        )
        for step in (2.0, 1.0):
            # For decreasing native x, the exact tie convention differs only at
            # cell boundaries, which have zero measure in this illustration.
            held = gaussian(
                p, np.floor(xx / step + 0.5) * step, np.floor(yy / step + 0.5) * step
            )
            sa[row, 0].plot(
                xx - p[1], held, label=f"{step:g}″ containing-pixel values", alpha=0.8
            )
        sa[row, 0].set(
            title=f"{array}: illustrative cut through centroid",
            xlabel="Azimuth relative to centroid (arcsec)",
            ylabel="mJy / nominal beam",
        )
        sa[row, 0].legend(fontsize=8)
        sa[row, 1].plot(
            range(10),
            [c["selection"]["pixels"] for c in cycles],
            ".-",
            label="Admitted source pixels",
        )
        sa[row, 1].set(
            title=f"{array}: mask growth",
            xlabel="Map used to construct feedback (M0–M9)",
            ylabel="2″ pixels selected",
        )
        sa[row, 1].grid(alpha=0.2)
    fig.suptitle(
        "Noise weighting retained • Orange: initial mask; black: last feedback mask\nOutside the black contour, valid feedback is exactly zero. Profiles are descriptive, not error estimates."
    )
    sf.suptitle(
        "Sampling illustration from saved fitted shapes; not a reduction or a predicted flux bias"
    )
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "masks-profiles.png", dpi=160)
    plt.close(fig)
    sf.savefig(figures / "sampling.png", dpi=160)
    plt.close(sf)
    result["figures"] = {p.name: bind(p) for p in sorted(figures.glob("*.png"))}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    inspect(args.output, args.figures)
