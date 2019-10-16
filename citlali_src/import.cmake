# version header
include(gitversion)
GenVersionHeader(${CMAKE_CURRENT_BINARY_DIR}/citlali)

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
    )

add_executable(citlali_mpi)
target_sources(citlali_mpi
    PRIVATE
        "src/citlali/mpi_main.cpp"
        )
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
