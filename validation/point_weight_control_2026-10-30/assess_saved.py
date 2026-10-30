"""Read-only assessment of saved 152418 control outputs; never fit or reduce."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm
import numpy as np
from astropy.io import fits
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/timestream_successor"))
sys.path.insert(0, str(REPO / "validation/point_calibration_audit_2026-10-30"))
from configured_point_cli import bind  # noqa: E402
from configured_point_comparison import compare_support  # noqa: E402
from audit_saved import measurements, read_map  # noqa: E402

PROJECT = Path(
    "/Users/gwilson/work_toltec/local_data/2026-refactor/projects/COM01-local-trial-20261027"
)
RUN = PROJECT / "pointings-weight-control-152418-20261030/reduced/redu00"
ARRAYS = ("a1100", "a1400", "a2000")


def read(path):
    return json.loads(path.read_text())


def stats(values):
    med = np.median(values)
    return dict(
        pixels=len(values),
        median=float(med),
        std=float(np.std(values)),
        mad_sigma=float(1.4826022185 * np.median(abs(values - med))),
    )


def common_measurements(path, parameters, center, common):
    """Reuse saved-map aperture arithmetic with a common sky-pixel selection."""
    result = measurements(path, parameters, center)
    z, valid, x, y, *_ = read_map(path)
    radius = np.hypot(x - center[0], y - center[1])
    # All aperture footprints must be fully inside the common science domain.
    assert np.all(common[radius < 47])
    sky = common & (radius >= 45) & (radius <= 75)
    background = float(np.median(z[sky]))
    correction = result["annulus_median"] - background
    for aperture in result["apertures"]:
        area = (
            aperture["valid_fraction_of_circle"]
            * np.pi
            * aperture["radius_arcsec"] ** 2
        )
        aperture["annulus_subtracted_brightness_area"] += correction * area
    result["annulus_median"] = background
    result["annulus_common_pixels"] = int(sky.sum())
    return result


def assess(output, figures):
    roots = {"noise": RUN, "equal": RUN / "comparison/equal"}
    work = {"noise": RUN / ".point-work", "equal": RUN / ".point-work/equal-control"}
    records = {arm: read(root / "RUN.json") for arm, root in roots.items()}
    for record in records.values():
        assert record["status"] == "completed" and record["final_cycle"] == 10
        assert record["calibration_eligible"] is False
        assert all(
            s["return_code"] == 0 and not s["unexpected_error_lines"]
            for s in record["stages"]
        )
        for key in ("executable", "received_config", "frozen_health"):
            assert bind(Path(record[key]["path"])) == record[key]
    checks = compare_support(records["noise"], records["equal"], bind)
    assert read(RUN / "weight-comparison.json")["checks"] == checks
    for cycle in range(11):
        requests = [read(work[a] / f"weighted-{cycle:02d}-request.json") for a in roots]
        assert requests[0]["networks"] == requests[1]["networks"]
        assert requests[0]["effective_config"] == requests[1]["effective_config"]
        assert requests[0]["health"] == requests[1]["health"]
    result = dict(
        scope="saved products only; no fitting/reduction",
        run_bindings={a: bind(p / "RUN.json") for a, p in roots.items()},
        comparison_binding=bind(RUN / "weight-comparison.json"),
        evidence_bindings={
            name: bind(RUN / name)
            for name in ("detector-health.json", "detector-drops.json")
        },
        runtime_bindings={
            name: records["noise"][name]
            for name in ("executable", "received_config", "frozen_health")
        },
        support_checks=len(checks),
        segment_pairs=sum(c["segments"] for c in checks),
        health=read(RUN / "detector-health.json")["arrays"],
        disturbance=read(RUN / "detector-drops.json")["arrays"],
        seconds=records["noise"]["seconds"],
        arrays={},
    )
    maps_fig, map_axes = plt.subplots(3, 3, figsize=(12, 11), layout="constrained")
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), layout="constrained")
    for ai, array in enumerate(ARRAYS):
        paths = {
            a: [
                work[a] / f"weighted-{i:02d}/products/{array}/map.fits"
                for i in range(11)
            ]
            for a in roots
        }
        maps = {a: [read_map(p) for p in ps] for a, ps in paths.items()}
        all_maps = [m for values in maps.values() for m in values]
        x, y = all_maps[0][2:4]
        assert all(
            np.array_equal(m[2], x) and np.array_equal(m[3], y) for m in all_maps
        )
        common = np.logical_and.reduce([m[1] for m in all_maps])
        final_fits = {
            a: read(root / "152418/raw" / f"point-fit-{array}.json")
            for a, root in roots.items()
        }
        center = final_fits["noise"]["guard"]["accepted_parameters"][1:3]
        radius = np.hypot(x - center[0], y - center[1])
        bg = common & (radius >= 30) & (radius <= 75)
        core = common & (radius <= 20)
        arr = {
            "background_annulus_arcsec": [30, 75],
            "center_arcsec": center,
            "background_scope": "same supported pixels across both arms and all cycles; structure plus noise",
            "arms": {},
            "coverage": [],
        }
        for i in range(11):
            with fits.open(paths["noise"][i]) as a, fits.open(paths["equal"][i]) as b:
                assert all(
                    h.verify_checksum() == 1 and h.verify_datasum() == 1
                    for h in (*a, *b)
                )
                assert np.array_equal(a["HITS"].data, b["HITS"].data)
                norm_diff = int(
                    np.sum(a["NORM_SUPPORT"].data != b["NORM_SUPPORT"].data)
                )
            arr["coverage"].append(
                dict(
                    cycle=i,
                    hits_equal=True,
                    normalization_support_different_pixels=norm_diff,
                    nearest_science_support_difference_arcsec=float(
                        radius[maps["noise"][i][1] != maps["equal"][i][1]].min()
                    ),
                    science_support_different_pixels=int(
                        np.sum(maps["noise"][i][1] != maps["equal"][i][1])
                    ),
                )
            )
        for arm, root in roots.items():
            fit = final_fits[arm]
            assert (
                bind(root / "152418/raw" / fit["map"])["sha256"]
                == fit["map_sha256"]
                == bind(paths[arm][-1])["sha256"]
            )
            final = fit["guard"]["accepted_parameters"]
            cycles = []
            for i in range(11):
                entry = records[arm]["feedback"]["cycles"][i]
                guard = (
                    fit["guard"]
                    if i == 10
                    else entry["source_selection"][array]["guard"]
                )
                assert guard["accepted"] and guard["diagnostic"]["finite"]
                assert not any(guard["diagnostic"]["active_mask"])
                par = guard["accepted_parameters"]
                native = paths[arm][i].parent
                bundle = yaml.safe_load((native / "bundle.yaml").read_text())
                assert bind(paths[arm][i])["sha256"] == bundle["files"]["map.fits"]
                plan = yaml.safe_load((native / "plan.yaml").read_text())
                row = dict(
                    cycle=i,
                    parameters=par,
                    background=stats(maps[arm][i][0][bg]),
                    map_sha256=bundle["files"]["map.fits"],
                    contributors=entry["arrays"][array]["contributing_detectors"],
                    samples=entry["arrays"][array]["contributing_samples"],
                    missing_weight_detector_segments=sum(
                        s["health_retained_without_map_weight"]
                        for s in plan["segments"]
                    ),
                    weight_min=min(s["map_weight_min"] for s in plan["segments"]),
                    weight_max=max(s["map_weight_max"] for s in plan["segments"]),
                    unavailable_details=[
                        dict(network=s["network"], scan=s["scan"], **d)
                        for s in plan["segments"]
                        for d in s.get("detector_details", [])
                    ],
                )
                if i:
                    delta = maps[arm][i][0] - maps[arm][i - 1][0]
                    row["previous_core_map_change_rms"] = float(
                        np.sqrt(np.mean(delta[core] ** 2))
                    )
                    row["previous_common_map_change_rms"] = float(
                        np.sqrt(np.mean(delta[common] ** 2))
                    )
                cycles.append(row)
            arr["arms"][arm] = dict(
                fit_binding=bind(root / "152418/raw" / f"point-fit-{array}.json"),
                final=common_measurements(paths[arm][-1], final, center, common),
                cycles=cycles,
                conditional_sigma_A=fit["guard"]["sigma_A"],
                uncertainty_scope=fit["uncertainty_scope"],
            )
            axes[ai, 0].plot(
                range(11), [c["parameters"][0] for c in cycles], ".-", label=arm
            )
            ap = arr["arms"][arm]["final"]["apertures"]
            axes[ai, 1].plot(
                [v["radius_arcsec"] for v in ap],
                [v["annulus_subtracted_brightness_area"] for v in ap],
                ".-",
                label=arm,
            )
        a, b = (arr["arms"][k]["final"] for k in ("noise", "equal"))
        arr["comparison"] = dict(
            peak_ratio=a["parameters"][0] / b["parameters"][0],
            gaussian_area_ratio=a["gaussian_brightness_area"]
            / b["gaussian_brightness_area"],
            centroid_separation_arcsec=float(
                np.linalg.norm(np.array(a["parameters"][1:3]) - b["parameters"][1:3])
            ),
            background_std_ratio=arr["arms"]["noise"]["cycles"][-1]["background"]["std"]
            / arr["arms"]["equal"]["cycles"][-1]["background"]["std"],
        )
        for ci, (z, label) in enumerate(
            [
                (maps["equal"][-1][0], "Equal M10"),
                (maps["noise"][-1][0], "Noise M10"),
                (maps["noise"][-1][0] - maps["equal"][-1][0], "Noise − equal"),
            ]
        ):
            ax = map_axes[ai, ci]
            im = ax.pcolormesh(
                x,
                y,
                np.where(common, z, np.nan),
                shading="nearest",
                cmap="RdBu_r",
                norm=AsinhNorm(linear_width=5, vmin=-700, vmax=700),
            )
            ax.set(
                xlim=(-65, 65),
                ylim=(-65, 65),
                aspect="equal",
                title=f"{array}: {label}",
                xlabel="Azimuth offset (arcsec)",
                ylabel="Elevation offset (arcsec)",
            )
            bar = maps_fig.colorbar(im, ax=ax, shrink=0.8)
            bar.set_ticks([-500, -100, -20, 0, 20, 100, 500])
        for col, title in enumerate(
            ["Saved fitted peak", "Aperture brightness-area; 45–75″ sky"]
        ):
            axes[ai, col].set(
                title=f"{array}: {title}",
                xlabel="FRUIT application" if col == 0 else "Radius (arcsec)",
                ylabel="mJy / nominal beam"
                if col == 0
                else "mJy / nominal beam × arcsec²",
            )
            axes[ai, col].legend()
            axes[ai, col].grid(alpha=0.2)
        result["arrays"][array] = arr
        assert all(
            c["contributors"] == arr["arms"]["noise"]["cycles"][0]["contributors"]
            and c["samples"] == arr["arms"]["noise"]["cycles"][0]["samples"]
            for arm in arr["arms"].values()
            for c in arm["cycles"]
        )
    maps_fig.suptitle(
        "152418: matched detector/sample support; common supported pixels\nSame nonlinear color scale throughout (mJy / nominal beam)"
    )
    maps_fig.savefig(figures / "maps.png", dpi=150)
    plt.close(maps_fig)
    fig.savefig(figures / "fruit-apertures.png", dpi=150)
    plt.close(fig)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    args = parser.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    assess(args.output, args.figures)
