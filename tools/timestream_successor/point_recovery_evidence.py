"""Compact saved-map recovery accounting. No reduction or feedback selection."""

import json
from pathlib import Path

import numpy as np
from astropy.io import fits
import yaml

from configured_point_cli import bind, write_json
from point_recovery_products import ARRAYS, comparison, measurements, response


def load_map(path):
    with fits.open(path, memmap=False) as h:
        return np.array(h[0].data), np.array(h["SCI_SUPPORT"].data, bool)


def segment_check(products, previous, reference):
    """Freeze original support/parent; record downstream support changes as outcomes."""
    result = {}
    for array in ARRAYS:
        plan = yaml.safe_load((products / array / "plan.yaml").read_text())
        rows = plan["segments"]
        ref = yaml.safe_load((reference / array / "plan.yaml").read_text())["segments"]
        if len(rows) != len(ref):
            raise ValueError("segment domain changed")
        prior = previous.get(array) if previous else None
        if prior is not None and len(prior) != len(rows):
            raise ValueError("parent segment count changed")
        for index, (row, baseline) in enumerate(zip(rows, ref, strict=True)):
            for key in (
                "network",
                "scan",
                "rank",
                "input_support_sha256",
                "source_free_sha256",
                "ordered_domain_sha256",
            ):
                if row[key] != baseline[key]:
                    raise ValueError("original segment association changed: " + key)
            if not row["converged"]:
                raise ValueError("cleaner failed: " + array + " " + str(index))
            if (
                prior is not None
                and row["recovery_parent_sha256"]
                != prior[index]["recovery_parent_sha256"]
            ):
                raise ValueError("injected original parent changed between cycles")
        result[array] = rows
    return result


def publish_seed(models, raster_refs, destination):
    """Publish the actual one-use m0+shoulder operand, separately from BASE."""
    destination.mkdir()
    record = dict(
        schema="citlali-point-recovery-seed-v1", applied_only_to_cycle=1, arrays={}
    )
    for array in ARRAYS:
        source = Path(models[array]["candidate"]["path"])
        raster = np.array(
            json.loads(Path(raster_refs[array]["path"]).read_text())["values"]
        ).reshape(41, 41)
        with fits.open(source, memmap=False) as original:
            hdus = fits.HDUList([h.copy() for h in original])
        values, valid = hdus["CANDIDATE"].data, hdus["VALID"].data.astype(bool)
        if values.shape != raster.shape or np.any((raster != 0) & ~valid):
            raise ValueError("seed projection lacks original model support")
        hdus["CANDIDATE"].data = np.where(valid, values + raster, np.nan)
        hdus["ADMITTED"].data = (
            hdus["ADMITTED"].data.astype(bool) | (raster != 0)
        ).astype("u1")
        hdus[0].header["DIAGONLY"] = True
        path = destination / (array + ".fits")
        hdus.writeto(path, checksum=True)
        record["arrays"][array] = dict(
            base=models[array],
            shoulder=raster_refs[array],
            operand=bind(path),
            policy="one-use additive admission exception; ordinary source guard already passed",
        )
    write_json(destination / "seed.json", record)
    return bind(destination / "seed.json")


def record_cycle(
    manifest, role, cycle, stage, reference_stage, guards, geometries, truths
):
    record = {}
    difference_arrays = {}
    for array in ARRAYS:
        data, valid = load_map(stage / "products" / array / "map.fits")
        ref, support = load_map(reference_stage / "products" / array / "map.fits")
        g = geometries[array]
        row = dict(total=measurements(data, valid, g, guards[array]))
        if role != "REF":
            row["comparison"], delta = comparison(
                data, valid, ref, support, truths[role][array], g
            )
            difference_arrays[array + "_difference"] = delta
            difference_arrays[array + "_support"] = valid & support
            # Candidates m0..m9 only. m0 seed is separately published above.
            if cycle < 10:
                cp = stage / f"feedback-{array}/candidate.fits"
                rp = reference_stage / f"feedback-{array}/candidate.fits"
                with fits.open(cp) as c, fits.open(rp) as r:
                    d = c["CANDIDATE"].data - r["CANDIDATE"].data
                    v = c["VALID"].data.astype(bool) & r["VALID"].data.astype(bool)
                    yy, xx = np.indices((41, 41))
                    disk = np.hypot(xx - 20, yy - 20) <= 20
                    truth = np.array(
                        json.loads(
                            Path(
                                manifest["rasters"][
                                    role if role != "SEED" else "SHOULDER"
                                ][array]["path"]
                            ).read_text()
                        )["values"]
                    ).reshape(41, 41)
                    row["candidate_response"] = response(d[disk], truth[disk], v[disk])
                    ca = c["ADMITTED"].data.astype(bool)
                    ra = r["ADMITTED"].data.astype(bool)
                    row["candidate_support"] = dict(
                        added=int(np.sum(ca & ~ra)),
                        removed=int(np.sum(ra & ~ca)),
                        signed_brightness_area=float(np.sum(d[v]) * 4),
                    )
        record[array] = row
    write_json(stage / "measurements.json", record)
    if difference_arrays:
        np.savez_compressed(stage / "differences.npz", **difference_arrays)
    return record


def final_accounting(root, rows, geometries, truths):
    """No fits: signed asymmetry and final temporal drift from saved maps."""
    out = dict(
        scientific_acceptance="unqualified: owner criteria not yet set", arrays={}
    )
    for array in ARRAYS:
        g = geometries[array]
        disk = g["disk"]
        a = {}
        for role, row in rows.items():
            if row["status"] != "completed":
                continue
            last, support = load_map(
                root / role / "cycle-10/products" / array / "map.fits"
            )
            prior, pv = load_map(root / role / "cycle-09/products" / array / "map.fits")
            a[role] = dict(final_drift=measurements(last - prior, support & pv, g))
        if all(
            rows.get(r, {}).get("status") == "completed"
            for r in ("REF", "CORE+", "CORE-")
        ):
            a["compact_even"] = []
            for cycle in range(11):
                plus, pv = load_map(
                    root / "CORE+" / f"cycle-{cycle:02d}/products" / array / "map.fits"
                )
                minus, mv = load_map(
                    root / "CORE-" / f"cycle-{cycle:02d}/products" / array / "map.fits"
                )
                ref, rv = load_map(
                    root / "REF" / f"cycle-{cycle:02d}/products" / array / "map.fits"
                )
                even = (plus + minus - 2 * ref) / 2
                pos = response(
                    (plus - ref)[disk], truths["CORE+"][array][disk], (pv & rv)[disk]
                )
                neg = response(
                    (minus - ref)[disk], truths["CORE-"][array][disk], (mv & rv)[disk]
                )
                a["compact_even"].append(
                    dict(
                        cycle=cycle,
                        measurement=measurements(even, pv & mv & rv, g),
                        response=response(
                            even[disk],
                            truths["CORE+"][array][disk],
                            (pv & mv & rv)[disk],
                        ),
                        signed_gain_asymmetry=pos["gain"] - neg["gain"]
                        if pos["available"] and neg["available"]
                        else None,
                    )
                )
                np.savez_compressed(
                    root / (array + f"-compact-even-{cycle:02d}.npz"),
                    even=even,
                    support=pv & mv & rv,
                )
        out["arrays"][array] = a
    write_json(root / "SUMMARY.json", out)


def increments(actual, ideal, reference):
    """Absolute increments and errors precede ratios; no ratio threshold invented."""

    def scalars(m):
        p = m["fit"].get("parameters")
        result = {
            k: m[k]
            for k in (
                "template_raw",
                "template_constant",
                "template_background_subtracted",
            )
        }
        if p is not None:
            result.update(
                zip(
                    ("peak", "cx", "cy", "sigma_x", "sigma_y", "angle", "constant"),
                    p,
                    strict=True,
                )
            )
            # Angles/sigma labels are gauge dependent; compare ellipse instead.
            for k in ("sigma_x", "sigma_y", "angle"):
                result.pop(k)
            result["major_fwhm"] = m["fit"]["major_fwhm"]
            result["minor_fwhm"] = m["fit"]["minor_fwhm"]
            for i in range(2):
                for j in range(2):
                    result[f"ellipse_{i}{j}"] = m["fit"]["ellipse_arcsec2"][i][j]
        for i, a in enumerate(m["apertures"]):
            for k in (
                "value_mJy_per_beam_arcsec2",
                "constant_contribution",
                "background_subtracted",
            ):
                result[f"aperture_{i}_{k}"] = a[k]
        return result

    a, b, r = scalars(actual), scalars(ideal), scalars(reference)
    result = {}
    for key in sorted(a.keys() | b.keys() | r.keys()):
        if any(t.get(key) is None for t in (a, b, r)):
            result[key] = dict(available=False)
            continue
        actual_increment = a[key] - r[key]
        ideal_increment = b[key] - r[key]
        result[key] = dict(
            available=True,
            actual_increment=actual_increment,
            ideal_increment=ideal_increment,
            signed_error=actual_increment - ideal_increment,
            absolute_error=abs(actual_increment - ideal_increment),
            ratio=None,
            ratio_reason="absolute-only diagnostic; no denominator convention added",
        )
    return result


def seed_initial(root, work, geometries, truths, models):
    """Logical SEED M0 shares REF bytes; its separately published m0 adds truth once."""
    record = {}
    for array in ARRAYS:
        path = root / "REF/cycle-00/products" / array / "map.fits"
        data, valid = load_map(path)
        g = geometries[array]
        row, _ = comparison(data, valid, data, valid, truths["SEED"][array], g)
        row["shared_map"] = bind(path)
        row["ordinary_model"] = models[array]
        with (
            fits.open(work / "seed" / (array + ".fits")) as seeded,
            fits.open(models[array]["candidate"]["path"]) as ordinary,
        ):
            yy, xx = np.indices((41, 41))
            disk = np.hypot(xx - 20, yy - 20) <= 20
            delta = seeded["CANDIDATE"].data - ordinary["CANDIDATE"].data
            # Independent full-grid truth crop at the nominal origin.
            full = truths["SEED"][array]
            origin = np.argwhere((g["x"] == 0) & (g["y"] == 0))[0]
            y, x = origin
            truth = full[y - 20 : y + 21, x - 20 : x + 21]
            row["seed_candidate_response"] = response(
                delta[disk], truth[disk], seeded["VALID"].data.astype(bool)[disk]
            )
            row["added_candidate_pixels"] = int(
                np.sum(
                    seeded["ADMITTED"].data.astype(bool)
                    & ~ordinary["ADMITTED"].data.astype(bool)
                )
            )
        record[array] = row
    write_json(work / "M0-SHARED.json", record)
    return bind(work / "M0-SHARED.json")
