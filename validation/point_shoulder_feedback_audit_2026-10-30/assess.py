#!/usr/bin/env python3
"""Inspect saved masks/maps. No fitting, selection replay, or native execution."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import yaml

OUT = Path(__file__).resolve().parent
TRIAL = Path(
    "/Users/gwilson/work_toltec/local_data/2026-refactor/projects/COM01-local-trial-20261027/pointings-source-recovery-152418-20261030"
)
ROOT = TRIAL / "run-001"
checked = {}


def bind(path, expected=None):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None:
        assert digest == expected, str(path)
    checked[str(path)] = digest
    return digest


def js(path, expected=None):
    bind(path, expected)
    return json.loads(Path(path).read_text())


def fy(path, expected=None):
    bind(path, expected)
    return yaml.safe_load(Path(path).read_text())


def planes(path, expected):
    bind(path, expected)
    with fits.open(path, memmap=False) as hdus:
        assert all(h.verify_checksum() == 1 and h.verify_datasum() == 1 for h in hdus)
        return {h.name: np.array(h.data) for h in hdus}


manifest = js(
    TRIAL / "trial.json",
    "229ec59da325db0a7949e08a4fec75d150a3f834ddd2f30756f40fbc403abead",
)
run = js(
    ROOT / "RUN.json",
    "c143873de7181443fc96181d5449df55307100cde1a8f9c821f1da03bd38b4ca",
)
metrics = {}
fig, axes = plt.subplots(3, 4, figsize=(14, 10), layout="constrained")
prof, pax = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
for ai, array in enumerate(("a1100", "a1400", "a2000")):
    geo = manifest["geometry"][array]
    bind(geo["path"], geo["sha256"])
    with np.load(geo["path"]) as g:
        x, y = g["x"].copy(), g["y"].copy()
        center = g["parameters"][1:3].copy()
    row, col = np.argwhere((x == 0) & (y == 0))[0]
    sl = np.s_[row - 20 : row + 21, col - 20 : col + 21]
    x, y = x[sl], y[sl]
    disk = np.hypot(x, y) <= 40
    assert disk.sum() == 1257 and x[20, 21] == -2 and y[21, 20] == 2
    raster = manifest["rasters"]["SHOULDER"][array]
    truth = np.asarray(js(raster["path"], raster["sha256"])["values"]).reshape(41, 41)
    assert np.all(truth >= 0) and np.all(truth[~disk] == 0)
    power = float(np.sum(truth**2))
    brightness = float(np.sum(truth))
    maps, masks, models, feedback = {}, {}, {}, {}
    for role in ("REF", "SHOULDER"):
        maps[role], masks[role], models[role], feedback[role] = [], [], [], []
        for cycle in range(11):
            rec = run["cases"][role]["cycles"][cycle]
            native = fy(rec["native"]["path"], rec["native"]["sha256"])
            stage = ROOT / role / f"cycle-{cycle:02d}"
            bundle = native["arrays"][array]
            assert fy(stage / "products" / array / "bundle.yaml") == bundle
            plan = fy(
                stage / "products" / array / "plan.yaml", bundle["files"]["plan.yaml"]
            )
            assert plan["weighting_control"] == "inverse-noise"
            assert plan["WCS"]["CDELT1"] == -2 and plan["WCS"]["CDELT2"] == 2
            if cycle:
                assert (
                    plan["source_model"]
                    == "sha256:" + feedback[role][-1]["model_sha256"]
                )
            p = planes(
                stage / "products" / array / "map.fits", bundle["files"]["map.fits"]
            )
            signal = p["PRIMARY"][sl]
            assert (
                np.all(p["SCI_SUPPORT"][sl][disk]) and np.isfinite(signal[disk]).all()
            )
            maps[role].append(signal)
            if cycle == 10:
                continue
            fb = js(stage / f"feedback-{array}" / "feedback.json")
            assert fb == rec["selection"][array]
            assert fb["map_sha256"] == bundle["files"]["map.fits"]
            assert (
                fb["input_bundle_sha256"]
                == checked[str(stage / "products" / array / "bundle.yaml")]
            )
            q = planes(fb["model_path"], fb["model_sha256"])
            assert set(q) == {"CANDIDATE", "ADMITTED", "VALID"}
            mask, valid = q["ADMITTED"].astype(bool), q["VALID"].astype(bool)
            assert np.array_equal(valid, disk) and not np.any(mask & ~valid)
            b = signal - fb["guard"]["accepted_parameters"][6]
            np.testing.assert_array_equal(q["CANDIDATE"][mask], b[mask])
            assert np.all(q["CANDIDATE"][valid & ~mask] == 0)
            assert np.isnan(q["CANDIDATE"][~valid]).all()
            summary = fb["candidate_summary"]
            assert summary["pixels"] == int(mask.sum())
            assert (
                not summary["boundary_contact"] and not summary["unsupported_contact"]
            )
            masks[role].append(mask)
            models[role].append(np.where(valid, q["CANDIDATE"], 0))
            feedback[role].append(fb)
    series = []
    for m in range(10):
        rm, sm = masks["REF"][m], masks["SHOULDER"][m]
        rf, sf = models["REF"][m], models["SHOULDER"][m]
        common, added, removed = rm & sm, sm & ~rm, rm & ~sm
        terms = {
            "common": np.where(common, sf - rf, 0),
            "added": np.where(added, sf, 0),
            "removed": np.where(removed, -rf, 0),
        }
        f = sf - rf
        np.testing.assert_allclose(sum(terms.values()), f, rtol=0, atol=0)
        d = maps["SHOULDER"][m + 1] - maps["REF"][m + 1]
        error = d - truth
        outside = disk & ~sm
        k = feedback["SHOULDER"][m]["candidate_summary"][
            "unassociated_above_threshold_pixels"
        ]
        # Exact count but unknown positions: upper bound by K largest omitted truth values.
        top = np.sort(truth[outside])[-k:] if k else np.array([])

        def zone(mask):
            return dict(
                pixels=int(mask.sum()),
                truth_brightness_fraction=float(truth[mask].sum() / brightness),
                truth_power_fraction=float(np.sum(truth[mask] ** 2) / power),
                error_power_fraction=float(
                    np.sum(error[mask] ** 2) / np.sum(error[disk] ** 2)
                ),
                signed_error_brightness_area=float(4 * error[mask].sum()),
                negative_error_area=float(4 * np.minimum(error[mask], 0).sum()),
                positive_error_area=float(4 * np.maximum(error[mask], 0).sum()),
                local_projection_gain=float(
                    np.sum(d[mask] * truth[mask]) / np.sum(truth[mask] ** 2)
                ),
            )

        item = dict(
            candidate=m,
            output=m + 1,
            ref_pixels=int(rm.sum()),
            shoulder_pixels=int(sm.sum()),
            added_pixels=int(added.sum()),
            removed_pixels=int(removed.sum()),
            mask_change_previous=None
            if m == 0
            else int(np.count_nonzero(sm != masks["SHOULDER"][m - 1])),
            feedback_gain=float(np.sum(f * truth) / power),
            output_gain=float(np.sum(d[disk] * truth[disk]) / power),
            eta=float(np.sqrt(np.sum(error[disk] ** 2) / power)),
            terms={
                name: dict(
                    gain=float(np.sum(v * truth) / power),
                    brightness_area=float(4 * v.sum()),
                )
                for name, v in terms.items()
            },
            inside=zone(sm),
            outside=zone(outside),
            disconnected_above_threshold_pixels=k,
            below_threshold_omitted_brightness_fraction_lower_bound=float(
                (truth[outside].sum() - top.sum()) / brightness
            ),
            omitted_map_delta_projection=float(
                np.sum((maps["SHOULDER"][m] - maps["REF"][m])[outside] * truth[outside])
                / np.sum(truth[outside] ** 2)
            ),
            background_constant_difference=float(
                feedback["SHOULDER"][m]["guard"]["accepted_parameters"][6]
                - feedback["REF"][m]["guard"]["accepted_parameters"][6]
            ),
        )
        series.append(item)
    late_common = masks["SHOULDER"][4] & masks["SHOULDER"][9]
    late_added = masks["SHOULDER"][9] & ~masks["SHOULDER"][4]
    late_removed = masks["SHOULDER"][4] & ~masks["SHOULDER"][9]
    f4 = models["SHOULDER"][4] - models["REF"][4]
    f9 = models["SHOULDER"][9] - models["REF"][9]
    # Here REF masks are unchanged; split the late contrast change geometrically.
    assert np.array_equal(masks["REF"][4], masks["REF"][9])
    late_terms = [
        np.where(late_common, f9 - f4, 0),
        np.where(late_added, f9, 0),
        np.where(late_removed, -f4, 0),
    ]
    np.testing.assert_allclose(sum(late_terms), f9 - f4, rtol=0, atol=0)
    series[-1]["late_feedback_gain_change_terms"] = {
        name: float(np.sum(v * truth) / power)
        for name, v in zip(("common", "added", "removed"), late_terms)
    }
    series[-1]["late_added_pixels"] = int(late_added.sum())
    metrics[array] = series
    # Final applied m9 masks over M10 products. Common brightness scale in first 3 panels.
    f = models["SHOULDER"][9] - models["REF"][9]
    d = maps["SHOULDER"][10] - maps["REF"][10]
    e = d - truth
    for j, (values, title) in enumerate(
        (
            (truth, "Known shoulder"),
            (f, "Applied feedback difference m9"),
            (d, "Recovered difference M10"),
            (e, "Recovery error M10"),
        )
    ):
        ax = axes[ai, j]
        v = np.where(disk, values, np.nan)
        limit = (
            float(np.max(np.abs(e[disk])))
            if j == 3
            else max(float(np.max(v[disk])) for v in (truth, f, d))
        )
        im = ax.imshow(
            v,
            origin="lower",
            extent=[41, -41, -41, 41],
            cmap="RdBu_r" if j == 3 else "viridis",
            vmin=-limit if j == 3 else 0,
            vmax=limit,
        )
        ax.contour(
            x, y, masks["REF"][9], levels=[0.5], colors=["#f5a623"], linewidths=0.8
        )
        ax.contour(
            x,
            y,
            masks["SHOULDER"][9],
            levels=[0.5],
            colors=["white" if j < 3 else "black"],
            linewidths=0.9,
        )
        ax.set(
            xlim=(30, -20),
            ylim=(-25, 25),
            title=f"{array}: {title}",
            xlabel="Azimuth offset (arcsec)",
            ylabel="Elevation offset (arcsec)",
        )
        fig.colorbar(im, ax=ax, shrink=0.7, label="mJy / nominal beam")
    radius = np.hypot(x - center[0], y - center[1])
    rings = [(radius >= lo) & (radius < lo + 2) & disk for lo in range(0, 32, 2)]
    rr = np.arange(1, 32, 2)
    for val, label, style in [
        (truth, "Known shoulder", "k-"),
        (maps["SHOULDER"][5] - maps["REF"][5], "Recovered M5", "C1--"),
        (d, "Recovered M10", "C0-"),
        (f, "Feedback m9", "C2:"),
    ]:
        pax[ai].plot(rr, [float(val[z].mean()) for z in rings], style, label=label)
    pax[ai].axhline(0, color="gray", lw=0.5)
    pax[ai].set(
        title=array,
        xlabel="Radius from frozen REF M0 center (arcsec)",
        ylabel="Mean brightness (mJy / nominal beam)",
    )
    pax[ai].legend(fontsize=8)
fig.suptitle(
    "152418: broad-source feedback and recovery | orange: REF mask; white/black: SHOULDER mask"
)
fig.savefig(OUT / "feedback_residuals.png", dpi=160)
prof.savefig(OUT / "profiles.png", dpi=160)
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
(OUT / "INPUTS.json").write_text(
    json.dumps(dict(files=checked, count=len(checked)), indent=2) + "\n"
)
print(
    json.dumps(
        {
            a: [r for r in series if r["candidate"] in [4, 9]]
            for a, series in metrics.items()
        },
        indent=2,
    )
)
