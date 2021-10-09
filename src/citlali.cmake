include(PrintProperties)

## version header
include(gitversion)
GenVersionHeader(citlali)

## dependencies

# import from kids_cpp the kids_core lib only
set(KIDS_BUILD_CLI OFF CACHE BOOL "" FORCE)
set(KIDS_BUILD_GUI OFF CACHE BOOL "" FORCE)
set(KIDS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(KIDS_BUILD_DOCS OFF CACHE BOOL "" FORCE)
include("kidscpp_src/import.cmake")

# we use GrPPI with OpenMP so disable the Eigen3 parallelization.
find_package(OpenMP REQUIRED)
option(USE_EIGEN3_MULTITHREADING "Enable multithreading inside Eigen3" OFF)

# other citlali deps
include(CCfits)
include(NetCDFCXX4)
include(Eigen3)
include(Ceres)
include(Spectra)
include(Yaml)
include(FileSystem)
include(SpdlogAndFmt)
include(Grppi)

if (VERBOSE)
    message("----- Summary of Dependencies -----")
    print_target_properties(cmake_utils::gitversion)
    print_target_properties(Spectra::Spectra)
    message("-----------------------------------")
endif()

## targets

add_library(citlali_core STATIC)
target_sources(citlali_core
    PRIVATE
        "src/citlali/dummy.cpp"
    )
target_include_directories(citlali_core PUBLIC "src")
target_link_libraries(citlali_core
    PUBLIC
        cmake_utils::spdlog_and_fmt
        cmake_utils::gitversion
        cmake_utils::ccfits
        NetCDF::NetCDFCXX4
        Eigen3::Eigen
        ceres::ceres
        grppi::grppi
        yaml-cpp::yaml-cpp
        Spectra::Spectra
    )

## optional targets
option(CITLALI_BUILD_CLI "Build CLI" ON)
if (CITLALI_BUILD_CLI)
    include(Clipp)
    add_executable(citlali)
    target_sources(citlali
        PRIVATE
            "src/citlali/main.cpp"
            "src/citlali/gaussmodels.cpp"
            )
    target_link_libraries(citlali
        PRIVATE
            kids_core
            citlali_core
            clipp::clipp
        )
endif()
option(CITLALI_BUILD_MPI "Build MPI CLI" OFF)
if (CITLALI_BUILD_MPI)
    include(MXX)
    if (VERBOSE)
        print_target_properties(mxx::mxx)
    endif()
    add_executable(citlali_mpi)
    target_sources(citlali_mpi
        PRIVATE
            "src/citlali/mpi_main.cpp"
            )
    target_link_libraries(citlali_mpi PRIVATE
        kids_core
        citlali_core
        mxx::mxx
        )
endif()
option(CITLALI_BUILD_TESTS "Build tests" OFF)
if (CITLALI_BUILD_TESTS)
    include(GBenchAndGTest)
    add_subdirectory(tests)
endif()
option(CITLALI_BUILD_DOCS "Build docs" OFF)
if (CITLALI_BUILD_DOCS)
    include(DoxygenDocTarget)
endif()
