#!/usr/bin/env python3
"""Read saved products only. No reduction, restoration, source fit, or learned state."""

import hashlib
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
TRIAL = Path(
    "/Users/gwilson/work_toltec/local_data/2026-refactor/projects/COM01-local-trial-20261027/pointings-source-recovery-152418-20261030"
)
ROOT = TRIAL / "run-001"
ARRAYS = ("a1100", "a1400", "a2000")
ROLES = ("REF", "CORE+", "CORE-", "SHOULDER", "SEED")
manifest_expected = "229ec59da325db0a7949e08a4fec75d150a3f834ddd2f30756f40fbc403abead"
checked = {}


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def bind(path, expected=None):
    path = Path(path).resolve()
    key = str(path)
    if key not in checked:
        checked[key] = sha(path)
    if expected is not None:
        assert checked[key] == expected, (key, "digest changed")
    return dict(path=key, sha256=checked[key])


def js(path):
    bind(path)
    return json.loads(Path(path).read_text())


def ys(path):
    bind(path)
    return yaml.load(Path(path).read_text(), Loader=yaml.CSafeLoader)


def refs(value):
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            bind(value["path"], value["sha256"])
        for v in value.values():
            refs(v)
    elif isinstance(value, list):
        for v in value:
            refs(v)


def planes(path):
    bind(path)
    with fits.open(path, memmap=False) as hdus:
        for h in hdus:
            assert h.verify_checksum() == 1 and h.verify_datasum() == 1
        return {h.name: np.array(h.data) for h in hdus}, hdus[0].header.copy()


def norm(x):
    return float(np.sqrt(np.sum(np.square(x))))


bind(TRIAL / "trial.json", manifest_expected)
m = js(TRIAL / "trial.json")
refs(m["bindings"])
run = js(ROOT / "RUN.json")
refs(run)
assert run["status"] == "completed" and run["invocations"] == 54
assert all(
    run["cases"][r]["status"] == "completed" and len(run["cases"][r]["cycles"]) == 11
    for r in ROLES
)
receipts = [c["receipt"] for case in run["cases"].values() for c in case["cycles"]]
assert len({c["request"]["path"] for c in receipts}) == 54
assert all(c["return_code"] == 0 and not c["unexpected_error_lines"] for c in receipts)
rest = js(ROOT / "RESTORATION.json")
refs(rest)
assert (
    rest["status"] == "exact_saved_payloads_verified"
    and rest["upstream_invocations"] == 1
)
payloads = 0
for n in m["restoration"]["networks"]:
    root = Path(rest["shadow_roots"][n["original_root"]])
    for name, digest in n["payloads"].items():
        bind(root / name, digest)
        payloads += 1
    for name in n["metadata"]:
        assert sha(root / name) == sha(Path(n["original_root"]) / name)
assert payloads == rest["payload_files"] == 16565

metrics = dict(
    run=bind(ROOT / "RUN.json"),
    trial=bind(TRIAL / "trial.json"),
    runtime=m["runtime_manifest"],
    source=m["source_revision"],
    duration_seconds=run["seconds"],
    arrays={},
)
counts = dict(
    ref_exact_planes=0,
    reference_checks=0,
    failed_segments=0,
    changed_original_support=0,
    changed_downstream_support=0,
    plane_support_changes=0,
    lost_benchmark_pixels=0,
    verified_parent_trajectories=0,
    seed_parent_equalities=0,
    restored_payloads=payloads,
)
max_weight_error = 0.0
support_changes = []
fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=False)
mapfig, mapaxes = plt.subplots(3, 4, figsize=(14, 10))
for ai, array in enumerate(ARRAYS):
    with np.load(m["geometry"][array]["path"]) as npz:
        g = {k: npz[k] for k in npz.files}
    disk = g["disk"]
    assert int(disk.sum()) == 1257
    yy, xx = np.indices((41, 41))
    small_disk = np.hypot(xx - 20, yy - 20) <= 20
    row, col = np.argwhere((g["x"] == 0) & (g["y"] == 0))[0]
    sl = np.s_[row - 20 : row + 21, col - 20 : col + 21]
    truth = {}
    for role in ("CORE+", "CORE-", "SHOULDER"):
        q = np.asarray(js(m["rasters"][role][array]["path"])["values"]).reshape(41, 41)
        t = np.zeros(disk.shape)
        t[sl] = q
        truth[role] = t
    assert np.array_equal(truth["CORE-"], -truth["CORE+"])
    truth["SEED"] = truth["SHOULDER"]
    data = {}
    supports = {}
    plans = {}
    measure = {}
    results = {}
    for role in ROLES:
        data[role] = []
        supports[role] = []
        plans[role] = []
        measure[role] = []
        series = []
        for cycle in range(11):
            actual = "REF" if role == "SEED" and cycle == 0 else role
            stage = ROOT / actual / f"cycle-{cycle:02d}"
            p = stage / "products" / array
            bundle = ys(p / "bundle.yaml")
            for name, digest in bundle["files"].items():
                bind(p / name, digest)
            d, header = planes(p / "map.fits")
            v = d["SCI_SUPPORT"].astype(bool)
            assert (
                header["CDELT1"] == -2
                and header["CDELT2"] == 2
                and header["ARRAY"] == array
            )
            assert (
                bundle["cycle_index"] == cycle
                and bundle["frozen_health_identity"]
                == "sha256:" + js(m["base_request"]["path"])["health"]["sha256"]
            )
            plan = ys(p / "plan.yaml")
            request_path = stage.parent / f"cycle-{cycle:02d}-request.json"
            request = js(request_path)
            assert request["recovery"] == m["cases"][actual]
            assert (
                request["weighting_control"] == "inverse-noise"
                and request["cycle_index"] == cycle
            )
            assert plan["request_sha256"] == bind(request_path)["sha256"]
            assert plan["diagnostic_recovery"]["case"] == actual
            assert (
                plan["diagnostic_recovery"]["identity"]
                == "sha256:" + m["cases"][actual]["sha256"]
            )
            assert all(s["converged"] for s in plan["segments"])
            if role == "REF":
                old, _ = planes(
                    Path(m["reference_work"])
                    / f"weighted-{cycle:02d}/products"
                    / array
                    / "map.fits"
                )
                assert d.keys() == old.keys()
                for name in d:
                    assert np.array_equal(d[name], old[name], equal_nan=True)
                    counts["ref_exact_planes"] += 1
                for z in plan["diagnostic_recovery"]["weighted_references"]:
                    max_weight_error = max(
                        max_weight_error, float(z["max_absolute_error"])
                    )
                    counts["reference_checks"] += 1
            else:
                baseline = plans["REF"][cycle]["segments"]
                assert len(plan["segments"]) == len(baseline)
                for a, b in zip(plan["segments"], baseline, strict=True):
                    for key in (
                        "network",
                        "scan",
                        "rank",
                        "input_support_sha256",
                        "source_free_sha256",
                        "ordered_domain_sha256",
                    ):
                        assert a[key] == b[key], key
                    for key in (
                        "residual_support_sha256",
                        "cleaned_support_sha256",
                        "mapped_support_sha256",
                    ):
                        counts["changed_downstream_support"] += int(a[key] != b[key])
                    if role == "SEED":
                        assert (
                            a["recovery_parent_sha256"] == b["recovery_parent_sha256"]
                        )
                        counts["seed_parent_equalities"] += 1
                changed = not np.array_equal(v, supports["REF"][cycle])
                counts["plane_support_changes"] += int(changed)
                if changed:
                    support_changes.append(
                        dict(
                            array=array,
                            case=role,
                            cycle=cycle,
                            gained=int(np.sum(v & ~supports["REF"][cycle])),
                            lost=int(np.sum(supports["REF"][cycle] & ~v)),
                        )
                    )
            if cycle:
                first = plans[role][0]["segments"]
                assert [s["recovery_parent_sha256"] for s in plan["segments"]] == [
                    s["recovery_parent_sha256"] for s in first
                ]
                counts["verified_parent_trajectories"] += 1
            if role == "SEED" and cycle:
                assert plan["diagnostic_recovery"]["seed_applied"] == (cycle == 1)
            elif role != "SEED":
                assert not plan["diagnostic_recovery"]["seed_applied"]
            measurement = js(stage / "measurements.json")[array]
            params = measurement["total"]["fit"]["parameters"]
            assert measurement["total"]["fit"]["guard"]["accepted"]
            data[role].append(d["PRIMARY"])
            supports[role].append(v)
            plans[role].append(plan)
            measure[role].append(measurement)
            item = dict(
                cycle=cycle,
                peak=params[0],
                centroid=params[1:3],
                constant=params[6],
                major_fwhm=measurement["total"]["fit"]["major_fwhm"],
                minor_fwhm=measurement["total"]["fit"]["minor_fwhm"],
                background=measurement["total"]["background"],
            )
            if role != "REF":
                t = truth[role][disk]
                delta = (d["PRIMARY"] - data["REF"][cycle])[disk]
                valid = v[disk] & supports["REF"][cycle][disk]
                assert valid.all()
                denom = float(np.sum(t * t))
                gain = float(np.sum(delta * t) / denom)
                eta = norm(delta - t) / np.sqrt(denom)
                rho = norm(delta - gain * t) / np.sqrt(denom)
                item.update(
                    gain=gain,
                    distortion=eta,
                    orthogonal_distortion=rho,
                    difference_norm_fraction=norm(delta) / np.sqrt(denom),
                    difference_max_mJy=float(np.abs(delta).max()),
                )
                assert np.isclose(eta**2, (gain - 1) ** 2 + rho**2, atol=1e-12)
                if not (role == "SEED" and cycle == 0):
                    recorded = measurement["comparison"]
                    for key in ("gain", "distortion", "orthogonal_distortion"):
                        assert np.isclose(
                            item[key], recorded[key], rtol=1e-12, atol=1e-12
                        ), (role, cycle, array, key)
                    counts["lost_benchmark_pixels"] += recorded["lost_benchmark_pixels"]
            series.append(item)
        a, b = measure[role][5]["total"], measure[role][10]["total"]
        p5 = np.array(a["fit"]["parameters"])
        p10 = np.array(b["fit"]["parameters"])
        delta = (data[role][10] - data[role][5])[disk]
        drift = dict(
            peak_mJy=float(p10[0] - p5[0]),
            peak_percent=float(100 * (p10[0] - p5[0]) / p5[0]),
            centroid_arcsec=float(np.linalg.norm(p10[1:3] - p5[1:3])),
            constant_mJy=float(p10[6] - p5[6]),
            major_fwhm_arcsec=b["fit"]["major_fwhm"] - a["fit"]["major_fwhm"],
            minor_fwhm_arcsec=b["fit"]["minor_fwhm"] - a["fit"]["minor_fwhm"],
            max_pixel_mJy=float(np.abs(delta).max()),
            rms_pixel_mJy=float(np.sqrt(np.mean(delta**2))),
            aperture_raw=[
                y["value_mJy_per_beam_arcsec2"] - x["value_mJy_per_beam_arcsec2"]
                for x, y in zip(a["apertures"], b["apertures"], strict=True)
            ],
            aperture_background_subtracted=[
                y["background_subtracted"] - x["background_subtracted"]
                for x, y in zip(a["apertures"], b["apertures"], strict=True)
            ],
        )
        if role != "REF":
            drift["gain_percentage_points"] = 100 * (
                series[10]["gain"] - series[5]["gain"]
            )
        candidates = []
        for cycle in range(10):
            if role == "SEED" and cycle == 0:
                cp = ROOT / "SEED/seed" / (array + ".fits")
            else:
                cp = (
                    ROOT
                    / role
                    / f"cycle-{cycle:02d}"
                    / f"feedback-{array}/candidate.fits"
                )
            c, _ = planes(cp)
            ref, _ = planes(
                ROOT / "REF" / f"cycle-{cycle:02d}" / f"feedback-{array}/candidate.fits"
            )
            if role != "REF":
                t = truth[role][sl][small_disk]
                cd = (c["CANDIDATE"] - ref["CANDIDATE"])[small_disk]
                candidates.append(
                    dict(
                        cycle=cycle,
                        projection=float(np.sum(cd * t) / np.sum(t * t)),
                        norm_fraction=norm(cd) / norm(t),
                        added_pixels=int(
                            np.sum(
                                c["ADMITTED"].astype(bool)
                                & ~ref["ADMITTED"].astype(bool)
                            )
                        ),
                        removed_pixels=int(
                            np.sum(
                                ref["ADMITTED"].astype(bool)
                                & ~c["ADMITTED"].astype(bool)
                            )
                        ),
                    )
                )
        applied5, _ = planes(
            ROOT / role / "cycle-04" / f"feedback-{array}/candidate.fits"
        )
        applied10, _ = planes(
            ROOT / role / "cycle-09" / f"feedback-{array}/candidate.fits"
        )
        drift["applied_candidate_mask_changed_pixels"] = int(
            np.sum(applied5["ADMITTED"] != applied10["ADMITTED"])
        )
        source_free = plans[role][0]["segments"]
        results[role] = dict(
            series=series,
            drift_M5_M10=drift,
            candidates=candidates,
            source_free_projected_count=sum(
                s["recovery_source_free_nonzero"] for s in source_free
            ),
            source_free_squared_signal=sum(
                float(s["recovery_source_free_squared_signal"]) for s in source_free
            ),
        )
        if role in ("CORE+", "CORE-", "SHOULDER"):
            results[role]["ideal_comparison"] = js(
                ROOT / role / "IDEAL_COMPARISON.json"
            )[array]
    # Save directly recomputed sign asymmetry/even residual, not a source fit.
    even = (data["CORE+"][10] + data["CORE-"][10] - 2 * data["REF"][10]) / 2
    results["compact_even_M10"] = dict(
        norm_fraction=norm(even[disk]) / norm(truth["CORE+"][disk]),
        max_mJy=float(np.abs(even[disk]).max()),
        gain_asymmetry=results["CORE+"]["series"][10]["gain"]
        - results["CORE-"]["series"][10]["gain"],
    )
    metrics["arrays"][array] = results
    for role in ("CORE+", "CORE-", "SHOULDER"):
        axes[ai, 0].plot(
            range(11), [100 * s["gain"] for s in results[role]["series"]], label=role
        )
    axes[ai, 0].axhline(100, color="gray", ls=":")
    axes[ai, 0].set_ylabel(array + " response (%)")
    axes[ai, 0].set_ylim(0, 110)
    seed = results["SEED"]["series"]
    axes[ai, 1].semilogy(
        range(1, 11),
        [100 * s["difference_norm_fraction"] for s in seed[1:]],
        "o-",
        color="purple",
    )
    axes[ai, 1].set_ylabel("Seed residual norm (%)")
    for role in ROLES:
        p = np.array([s["peak"] for s in results[role]["series"]])
        axes[ai, 2].plot(range(5, 11), 100 * (p[5:] / p[10] - 1), label=role)
    axes[ai, 2].set_ylabel("Peak relative to M10 (%)")
    for ax in axes[ai]:
        ax.axvline(5, color="black", ls="--", alpha=0.4)
        ax.grid(alpha=0.2)
        ax.set_xlabel("M cycle")
    t = truth["SHOULDER"]
    d5 = data["SHOULDER"][5] - data["REF"][5]
    d10 = data["SHOULDER"][10] - data["REF"][10]
    images = [t, d10, d10 - t, d10 - d5]
    titles = [
        "Known shoulder",
        "Recovered at M10",
        "Recovery error M10",
        "Recovery change M5→M10",
    ]
    for j, (img, title) in enumerate(zip(images, titles, strict=True)):
        v = np.where(small_disk, img[sl], np.nan)
        limit = np.nanmax(np.abs(v))
        im = mapaxes[ai, j].imshow(
            v,
            origin="lower",
            extent=[41, -41, -41, 41],
            vmin=-limit,
            vmax=limit,
            cmap="RdBu_r",
        )
        mapaxes[ai, j].set_title(array + " " + title, fontsize=9)
        mapaxes[ai, j].set_xlabel("Azimuth offset (arcsec)")
        mapaxes[ai, j].set_ylabel("Elevation offset (arcsec)")
        mapfig.colorbar(im, ax=mapaxes[ai, j], shrink=0.75, label="mJy / nominal beam")
for ax in axes[:, 2]:
    ax.set_xlim(4.8, 10.2)
axes[0, 0].legend()
axes[0, 2].legend(ncol=2, fontsize=8)
axes[0, 0].set_title("Signed known-source response")
axes[0, 1].set_title("Feedback-only memory / injected norm")
axes[0, 2].set_title("Late fitted-peak drift")
fig.tight_layout()
fig.savefig(OUT / "trajectories.png", dpi=160)
plt.close(fig)
mapfig.suptitle(
    "Fixed 40 arcsec benchmark; each panel has its own symmetric color scale",
    fontsize=12,
)
mapfig.tight_layout(rect=(0, 0, 1, 0.97))
mapfig.savefig(OUT / "shoulder_maps.png", dpi=160)
plt.close(mapfig)
metrics["support_changes"] = support_changes
metrics["validation"] = dict(
    **counts,
    max_recorded_weighted_reference_error_mJy=max_weight_error,
    authenticated_files=len(checked),
)
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
ledger = json.dumps(sorted(checked.items()), separators=(",", ":")).encode()
(OUT / "INPUTS.json").write_text(
    json.dumps(
        dict(
            trial=bind(TRIAL / "trial.json"),
            run=bind(ROOT / "RUN.json"),
            restoration=bind(ROOT / "RESTORATION.json"),
            summary=bind(ROOT / "SUMMARY.json"),
            ideal_controls=bind(ROOT / "IDEAL_CONTROLS.json"),
            authenticated_files=len(checked),
            authenticated_ledger_sha256=hashlib.sha256(ledger).hexdigest(),
            replay="assess.py reauthenticates saved manifest references, CAL payloads, bundles, FITS and measurements; no duplicate payload inventory retained",
        ),
        indent=2,
    )
    + "\n"
)
print(json.dumps(metrics["validation"], indent=2))
