"""Publish native POINT maps and qualified fits through TolTECA discovery.

The filenames are discoverable by the existing project interface. The six MAP
planes retain their successor meanings; neither legacy image-analysis nor PPT
calibration compatibility is implied by discovery.
"""

from pathlib import Path
import re
import shutil

import numpy as np
from astropy.io import fits

from inspect_forward_map import digest, read_yaml, require
from point_additive_diagnostics import save_json


ARRAYS = ("a1100", "a1400", "a2000")
PLANES = ("PRIMARY", "NUMERATOR", "NORMALIZATION", "HITS", "NORM_SUPPORT", "SCI_SUPPORT")
UNITS = ("mJy/beam", "mJy/beam", "1", "count", "1", "1")
UNIFORM_FAMILY = "SCI-PTC:uniform_constant@draft-0.1"
EMPIRICAL_FAMILY = "OWNER-PTC:empirical-inverse-noise@2026-10-29-v1"
EMPIRICAL_UNITS = ("mJy/beam", "beam/mJy", "beam2/mJy2", "count", "1", "1")
EMPIRICAL_INTERPRETATION = "empirical-analysis-coefficient-not-map-precision"
FRAME = dict(
    ASTFRAME="delta-source-AltAz",
    CTYPE1="AZOFFSET",
    CTYPE2="ELOFFSET",
    CUNIT1="arcsec",
    CUNIT2="arcsec",
)
GRID_KEYS = tuple(FRAME) + ("CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CDELT1", "CDELT2", "NOMBFWHM")


def _inspect_map(root, observation, array):
    """Check the native publication, without replaying expensive upstream data."""
    path = root / "map.fits"
    bundle_path = root / "bundle.yaml"
    bundle = read_yaml(bundle_path)
    require(
        bundle["schema"] == "citlali-forward-map-bundle-v1"
        and bundle["role"] == "SCI-MAP:base_observation_map@1"
        and bundle["outcome"] == "success",
        "required MAP publication not successful",
    )
    match = re.fullmatch(r"obs:(\d+)_(\d+)_(\d+)", bundle["observation"])
    require(
        match is not None and int(match.group(1)) == observation,
        "MAP observation differs from requested observation",
    )
    require(
        (int(match.group(2)), int(match.group(3))) == (0, 2),
        "MAP subobservation/scan differs from supported acquisition",
    )
    require(bundle["array"] == array, "MAP array differs from requested array")
    sha = digest(path)
    require(sha == bundle["files"]["map.fits"], "MAP FITS digest mismatch")
    require(
        bundle["signal_unit"] == "mJy/beam"
        and bundle["measured_beam_or_full_response_inferred"] is False,
        "MAP nominal-beam quantity differs",
    )
    # Older uniform bundles did not always name their family. That historical
    # schema remains readable, while new empirical units require their own
    # explicit family and cannot be relabelled as uniform or map precision.
    family = bundle.get("coefficient_family", UNIFORM_FAMILY)
    require(family in (UNIFORM_FAMILY, EMPIRICAL_FAMILY), "unsupported MAP coefficient family")
    empirical = family == EMPIRICAL_FAMILY
    units = EMPIRICAL_UNITS if empirical else UNITS
    extra = {}
    if empirical:
        require(
            bundle.get("normalization_interpretation") == EMPIRICAL_INTERPRETATION,
            "empirical MAP normalization interpretation differs",
        )
        cycle = bundle.get("cycle_index")
        health = bundle.get("frozen_health_identity")
        require(
            type(cycle) is int
            and cycle >= 0
            and isinstance(health, str)
            and re.fullmatch(r"sha256:[0-9a-f]{64}", health) is not None,
            "empirical MAP cycle or frozen health identity invalid",
        )
        plan_path = root / "plan.yaml"
        plan_sha = digest(plan_path)
        require(bundle["files"].get("plan.yaml") == plan_sha, "empirical MAP plan digest mismatch")
        plan = read_yaml(plan_path)
        require(
            plan["schema"] == "citlali-weighted-point-map-plan-v1"
            and plan["array"] == array
            and plan["observation"] == bundle["observation"]
            and type(plan["cycle_index"]) is int
            and plan["cycle_index"] == cycle
            and plan["frozen_health_identity"] == health
            and plan["WCS"] == bundle["WCS"],
            "empirical MAP plan associations differ",
        )
        extra = dict(
            coefficient_family=family,
            coefficient_interpretation=EMPIRICAL_INTERPRETATION,
            plane_units=dict(zip(PLANES, units)),
            native_cycle_index=cycle,
            frozen_health_identity=health,
            native_plan_sha256=plan_sha,
        )
    with fits.open(path, memmap=False) as hdus:
        require(
            tuple(hdu.name for hdu in hdus) == PLANES,
            "MAP requires its six genuine signal/support planes",
        )
        header = hdus[0].header
        require(
            all(header.get(key) == value for key, value in FRAME.items()),
            "MAP requires source-offset AltAz with arcsecond axes",
        )
        require(
            all(np.isfinite(header[key]) for key in GRID_KEYS if key not in FRAME),
            "MAP grid/nominal beam must be finite",
        )
        require(
            header["CRVAL1"] == header["CRVAL2"] == 0.0
            and header["CDELT1"] < 0.0 < header["CDELT2"]
            and header["NOMBFWHM"] > 0.0,
            "MAP offset origin, axes or beam differs",
        )
        require(
            header["NOMBFWHM"] == bundle["nominal_beam_fwhm_arcsec"],
            "MAP nominal beam differs from native bundle",
        )
        require(
            all(
                bundle["WCS"].get(key) == header[key] for key in tuple(FRAME) + ("CRVAL1", "CRVAL2")
            ),
            "MAP frame differs from native bundle",
        )
        shape = hdus[0].data.shape
        require(len(shape) == 2 and min(shape) > 0, "MAP must be a nonempty image")
        if empirical:
            require(
                all(
                    bundle["WCS"].get(key) == header[key] for key in GRID_KEYS if key != "NOMBFWHM"
                ),
                "empirical MAP grid differs from native bundle",
            )
        for hdu, unit in zip(hdus, units):
            require(hdu.data is not None and hdu.data.shape == shape, "MAP plane shapes differ")
            require(
                hdu.header.get("BUNIT") == unit and hdu.header.get("ARRAY") == array,
                "MAP plane units or array differ",
            )
            require(
                all(hdu.header.get(key) == header[key] for key in GRID_KEYS),
                "MAP plane frames differ",
            )
            require(
                hdu.verify_checksum() == 1 and hdu.verify_datasum() == 1,
                "MAP FITS checksum missing or invalid",
            )
        science = hdus["SCI_SUPPORT"].data
        normalization_support = hdus["NORM_SUPPORT"].data
        require(
            np.isin(science, (0, 1)).all() and np.isin(normalization_support, (0, 1)).all(),
            "MAP support planes are not boolean",
        )
        science = science.astype(bool)
        require(
            science.any() and not (science & ~normalization_support.astype(bool)).any(),
            "MAP scientific support is empty or exceeds normalization support",
        )
        signal = hdus[0].data
        require(
            np.isfinite(signal[science]).all() and np.isnan(signal[~science]).all(),
            "MAP missing signal must remain unavailable, not zero",
        )
        normalization = hdus["NORMALIZATION"].data
        hits = hdus["HITS"].data
        require(
            np.isfinite(hdus["NUMERATOR"].data).all()
            and np.isfinite(normalization).all()
            and (normalization >= 0).all()
            and (normalization[science] > 0).all(),
            "invalid MAP numerator/normalization",
        )
        require(
            np.issubdtype(hits.dtype, np.integer) and (hits >= 0).all(), "invalid MAP hit counts"
        )
    return dict(
        source=path,
        sha256=sha,
        native_bundle_sha256=digest(bundle_path),
        native_product_identity=bundle["product_identity"],
        nominal_beam_fwhm_arcsec=bundle["nominal_beam_fwhm_arcsec"],
        **extra,
    )


def publish_maps(
    map_products: Path,
    destination: Path,
    observation: int,
    arrays: tuple[str, ...],
    *,
    fit_diagnostics: bool,
    verbose: bool,
) -> list[dict]:
    """Publish to a fresh reduNN/<obs>/raw directory; never replace a reduction.

    Missing/malformed maps and software/I/O failures propagate. A completed
    qualified fit diagnostic may truthfully report failed or unavailable; it
    does not convert an otherwise valid map into a missing required product.
    """
    require(type(observation) is int and observation >= 0, "invalid observation")
    require(
        bool(arrays)
        and len(set(arrays)) == len(arrays)
        and all(array in ARRAYS for array in arrays),
        "invalid requested arrays",
    )
    map_products, destination = Path(map_products), Path(destination)
    require(not destination.exists(), "POINT publication destination must be fresh")
    # Validate every required array before making any discoverable result.
    sources = [_inspect_map(map_products / array, observation, array) for array in arrays]
    if any(source.get("coefficient_family") == EMPIRICAL_FAMILY for source in sources):
        require(
            set(arrays) == set(ARRAYS) and len(arrays) == 3,
            "empirical POINT requires all three arrays",
        )
        require(
            all(source.get("coefficient_family") == EMPIRICAL_FAMILY for source in sources),
            "POINT arrays cannot mix empirical and uniform coefficient families",
        )
        require(
            len({source["native_cycle_index"] for source in sources}) == 1
            and len({source["frozen_health_identity"] for source in sources}) == 1,
            "empirical POINT arrays have different cycle or frozen health identities",
        )
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for array, source in zip(arrays, sources):
        name = f"toltec_point_{observation:06d}_{array}_successor.fits"
        output = destination / name
        shutil.copyfile(source["source"], output)
        require(digest(output) == source["sha256"], "required MAP copy differs")
        record = dict(
            schema="citlali-configured-point-product-v1",
            observation=observation,
            array=array,
            map=name,
            **{key: value for key, value in source.items() if key != "source"},
            frame="delta-source-AltAz",
            axis_units="arcsec",
            signal_units="mJy/nominal beam",
            planes=list(PLANES),
            normalization="dimensionless gridding sum; not inverse variance",
            calibration_eligible=False,
            legacy_analysis_compatible=False,
            fit_status="not_requested",
        )
        if source.get("coefficient_family") == EMPIRICAL_FAMILY:
            record["normalization"] = (
                "sum of empirical inverse-noise analysis coefficients in beam2/mJy2; "
                "not map precision or significance"
            )
        if fit_diagnostics:
            from point_reference import measure_forward

            guard = measure_forward(
                output, details=destination / f"point-fit-{array}-details" if verbose else None
            )
            fit_name = f"point-fit-{array}.json"
            save_json(
                destination / fit_name,
                dict(
                    schema="citlali-configured-point-fit-diagnostic-v1",
                    observation=observation,
                    array=array,
                    map=name,
                    map_sha256=source["sha256"],
                    frame="delta-source-AltAz",
                    centroid_axes=["azimuth_offset", "elevation_offset"],
                    parameter_names=[
                        "amplitude",
                        "cx",
                        "cy",
                        "sigma_x",
                        "sigma_y",
                        "angle",
                        "constant",
                    ],
                    parameter_units=[
                        "mJy/nominal beam",
                        "arcsec",
                        "arcsec",
                        "arcsec",
                        "arcsec",
                        "rad",
                        "mJy/nominal beam",
                    ],
                    sigma_A_units="mJy/nominal beam",
                    uncertainty_scope="conditional full frozen C0 reference-scale sigma_A; "
                    "not endpoint empirical uncertainty or full adaptive error",
                    calibration_eligible=False,
                    pointing_correction=None,
                    guard=guard,
                ),
            )
            record.update(fit_status=guard["status"], fit=fit_name)
        save_json(destination / f"point-product-{array}.json", record)
        records.append(record)
    return records
