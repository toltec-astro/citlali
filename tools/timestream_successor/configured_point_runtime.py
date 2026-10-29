"""Explicit installed application surface and resource lookup.

The same members serve source checkouts, immutable local bundles and wheels.
Historical dependencies are retained explicitly until their services are extracted.
"""

from pathlib import Path
import sys

RUNTIME_MODULES = (
    "configured_point_cli",
    "configured_point_config",
    "configured_point_drops",
    "configured_point_feedback",
    "configured_point_inputs",
    "configured_point_products",
    "configured_point_runtime",
    "configured_point_types",
    "configured_point_weighted",
    "inspect_forward_map",
    "inspect_point_source",
    "local_data_purge",
    "point_additive",
    "point_additive_diagnostics",
    "point_admission",
    "point_archive_compaction",
    "point_auxiliary",
    "point_corr_admission",
    "point_empirical",
    "point_empirical_io",
    "point_expected",
    "point_fruit_diagnostics",
    "point_fruit_trajectory",
    "point_guarded_feedback",
    "point_reference",
    "point_response_cd",
    "point_source_evolution",
    "point_timing",
)


def data_root(module_path: Path = Path(__file__), prefix: Path = Path(sys.prefix)) -> Path:
    """Locate versioned resources in a checkout/bundle or installed environment.

    No current-directory lookup or environment override can change the selected
    scientific defaults. The wheel installs resources under its environment.
    """
    source_data = module_path.resolve().parents[2] / "data"
    if (source_data / "configured_point_v1.yaml").is_file():
        return source_data
    installed_data = prefix / "share" / "citlali-point" / "data"
    if (installed_data / "configured_point_v1.yaml").is_file():
        return installed_data
    raise FileNotFoundError("configured POINT application data is unavailable")
