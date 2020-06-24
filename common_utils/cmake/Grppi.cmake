include_guard(GLOBAL)
include(FetchContentHelper)
FetchContentHelper(grppi GIT "https://github.com/Jerry-Ma/grppi.git" cpp20
    ADD_SUBDIR CONFIG_SUBDIR
        GRPPI_UNIT_TEST_ENABLE=OFF
        GRPPI_TBB_ENABLE=OFF
        )
# grppi with OpenMP backend
find_package(OpenMP REQUIRED)
add_library(grppi INTERFACE)
target_include_directories(grppi SYSTEM INTERFACE ${grppi_SOURCE_DIR}/include)
target_link_libraries(grppi INTERFACE OpenMP::OpenMP_CXX)
target_compile_definitions(grppi INTERFACE "-DGRPPI_OMP")
add_library(grppi::grppi ALIAS grppi)
