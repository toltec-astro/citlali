project (utils_test_mpi LANGUAGES CXX C)
add_executable(test_mpi EXCLUDE_FROM_ALL)
find_package(MPI REQUIRED)
set_target_properties(test_mpi
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
target_sources(test_mpi
    PRIVATE
        main.cpp
        )
target_include_directories(test_mpi
    PRIVATE
        ${NETCDF_INCLUDE_DIRS}
        )
target_link_libraries(test_mpi
    PRIVATE
        common_utils
        MPI::MPI_C
        ${NETCDF_LIBRARIES})
add_dependencies(test_progs test_mpi)
