"""HEASARC release missing from the Spack 1.2.2 builtin recipe snapshot."""

from spack.package import depends_on, version
from spack_repo.builtin.packages.cfitsio.package import Cfitsio as BuiltinCfitsio


class Cfitsio(BuiltinCfitsio):
    """Use the builtin build and variants with the 4.7.0 release archive."""

    version(
        "4.7.0",
        url="https://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio-4.7.0.tar.gz",
        sha256="ce573bbea8e75b429f8c3d3e86498741ba3dc9628a1530d2f65268397ad059e8",
    )
    depends_on("zlib-api", when="@4.7.0", type=("build", "link"))
