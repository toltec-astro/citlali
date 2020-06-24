#include <utils/logging.h>
#include <cstdlib>
#include <mpi.h>
#include <utils/grppiex.h>
#include <utils/eigen.h>
#include <utils/container.h>
#include <utils/formatter/matrix.h>

auto whoami() {
    int size, rank, namelen, verlen;
    char name[MPI_MAX_PROCESSOR_NAME];
    char ver[MPI_MAX_LIBRARY_VERSION_STRING];
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(name, &namelen);
    MPI_Get_library_version(ver, &verlen);
    return std::make_tuple(
        size, rank, std::string(name, namelen), std::string(ver, verlen));
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char *argv[]) {
    MPI_Init(NULL, NULL);
    auto [mpisize, mpirank, mpiproc, mpiver] = whoami();
    if (0 == mpirank) {
        SPDLOG_TRACE("MPI version: {}", mpiver);
    }
    SPDLOG_TRACE("MPI context: rank {}/{} proc {}",
              mpirank, mpisize, mpiproc);
    // run a grppi reduce
    auto data =  container_utils::index(10);
    eigen_utils::asvec(data).array() += mpirank;
    SPDLOG_TRACE("rank {}: reduce data{}", mpirank, data);
    auto sum = grppi::reduce(
        grppiex::dyn_ex(), data, 0.,
        [](auto x, auto y) { return x+y;}
        );
    SPDLOG_TRACE("rank {}: result {}", mpirank, sum);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
