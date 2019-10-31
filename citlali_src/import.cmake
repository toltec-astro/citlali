## this file is intended to be included from the citlali repo

# dependencies

include(Yaml)

# version header
include(gitversion)
GenVersionHeader(${CMAKE_CURRENT_BINARY_DIR} citlali)

# citlali
add_library(citlali_core STATIC)
target_sources(citlali_core
    PRIVATE
        src/citlali/citlali.cpp
        )

target_link_libraries(citlali_core
    PUBLIC
        cmake_utils::spdlog_and_fmt
        NetCDF::NetCDFCXX4
        utils::gitversion
    )
target_include_directories(citlali_core
    PUBLIC
        src
    )
option(CITLALI_BUILD_MPI "Build mpi exec" ON)
if (CITLALI_BUILD_MPI)
    include(MXX)
    print_target_properties(mxx::mxx)
    # if (TARGET MPI::MPI)
    #    print_target_properties(MPI::MPI)
    # else()
    #    message(FATAL_ERROR "No MPI found")
    # endif()
    add_executable(citlali_mpi)
    target_sources(citlali_mpi
        PRIVATE
            "src/citlali/mpi_main.cpp"
            )
    target_link_libraries(citlali_mpi PRIVATE
        citlali_core
        mxx::mxx
        grppi::grppi
        Eigen3::Eigen
        yaml::yaml
        )
endif()
# optional targets
option(CITLALI_BUILD_TESTS "Build tests" ON)
if (CITLALI_BUILD_TESTS)
    include(GBenchAndGTest)
    add_subdirectory(tests)
endif()
option(CITLALI_BUILD_DOCS "Build docs" ON)
if (CITLALI_BUILD_DOCS)
    include(DoxygenDocTarget)
endif()
