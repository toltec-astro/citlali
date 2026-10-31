"""Transport and preparation boundaries; scientific validity remains explicit.

Bindings identify exact file bytes. Paths are absolute after binding. Prepared
observation/network IDs retain their native acquisition meaning; cell counts
are storage counts, not scientific admission or availability measurements.
"""

from pathlib import Path
from typing import TypedDict


class FileBinding(TypedDict):
    path: str
    sha256: str


class InputBinding(FileBinding):
    bytes: int
    roles: list[str]


class PointingMetadata(TypedDict):
    """Small observed metadata for the one-way OG pointing-table adapter.

    Header angular values are radians; MJD is the mean telescope Unix time.
    Missing metadata withholds the table, never invents zero corrections.
    """

    status: str
    cause: str | None
    meta: dict
    applied_pointing_offsets: dict[str, list[float]]
    telescope: FileBinding


class PreparedPoint(TypedDict):
    observation: int
    subobservation: int
    scan: int
    arrays: list[str]
    networks: list[int]
    observation_request: Path
    effective_config: Path
    bindings: list[InputBinding]
    native_cells: int
    pointing_metadata: PointingMetadata
