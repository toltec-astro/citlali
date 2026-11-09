"""Required celestial SCIENCE publication; no pointing-table consumer handoff."""

from pathlib import Path
import shutil

from configured_point_products import ARRAYS, PLANES, EMPIRICAL_FAMILY, _inspect_map
from inspect_forward_map import digest, require
from point_additive_diagnostics import save_json


def publish_maps(map_products, destination, observation, arrays, **kwargs):
    require(type(observation) is int and observation == 152389, "SCIENCE pilot observation differs")
    require(not kwargs.get("fit_diagnostics", False), "SCIENCE does not publish POINT fits")
    require(len(arrays) == 3 and set(arrays) == set(ARRAYS), "SCIENCE requires all three arrays")
    destination = Path(destination)
    require(not destination.exists(), "SCIENCE publication destination must be fresh")
    sources = [_inspect_map(Path(map_products) / a, observation, a, celestial=True) for a in arrays]
    require(
        all(s.get("coefficient_family") == EMPIRICAL_FAMILY for s in sources),
        "SCIENCE requires its declared empirical coefficient family",
    )
    require(
        len({s["native_cycle_index"] for s in sources}) == 1
        and len({s["frozen_health_identity"] for s in sources}) == 1,
        "SCIENCE arrays have different cycle or frozen health identities",
    )
    require(
        len({(s["WCS"]["CRVAL1"], s["WCS"]["CRVAL2"]) for s in sources}) == 1,
        "SCIENCE arrays have different celestial reference centers",
    )
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for array, source in zip(arrays, sources, strict=True):
        name = f"toltec_science_{observation:06d}_{array}_successor.fits"
        output = destination / name
        shutil.copyfile(source["source"], output)
        require(digest(output) == source["sha256"], "required SCIENCE map copy differs")
        record = dict(
            schema="citlali-configured-science-product-v1",
            observation=observation,
            array=array,
            map=name,
            **{k: v for k, v in source.items() if k != "source"},
            frame="J2000-radec-gnomonic-tangent",
            axis_units="deg",
            signal_units="mJy/nominal beam",
            planes=list(PLANES),
            normalization="sum of empirical analysis coefficients; not map precision or significance",
            profile="single-observation compact-source pilot",
            coadd=False,
            calibration_eligible=False,
            legacy_analysis_compatible=False,
        )
        save_json(destination / f"science-product-{array}.json", record)
        records.append(record)
    return records
