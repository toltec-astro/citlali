cmake_minimum_required(VERSION 3.25)

project(TulaNetcdfCxx4 VERSION 4.3.1 LANGUAGES CXX)

find_package(TulaCMake 3 CONFIG REQUIRED)
find_package(netCDF CONFIG REQUIRED)

find_path(NETCDF_CXX4_INCLUDE_DIR NAMES netcdf REQUIRED)
find_library(NETCDF_CXX4_LIBRARY NAMES netcdf-cxx4 REQUIRED)

add_library(netcdf_cxx4_vendor SHARED IMPORTED)
set_target_properties(
    netcdf_cxx4_vendor
    PROPERTIES
        IMPORTED_LOCATION "${NETCDF_CXX4_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${NETCDF_CXX4_INCLUDE_DIR}"
)

add_library(tula_netcdf_cxx4 INTERFACE)
add_library(tula_deps::netcdf_cxx4 ALIAS tula_netcdf_cxx4)
set_target_properties(
    tula_netcdf_cxx4
    PROPERTIES EXPORT_NAME netcdf_cxx4
)
target_link_libraries(
    tula_netcdf_cxx4
    INTERFACE netcdf_cxx4_vendor netCDF::netcdf
)

tula_cmake_install_package(
    PACKAGE TulaNetcdfCxx4
    EXPORT TulaNetcdfCxx4Targets
    NAMESPACE tula_deps::
    VERSION "${PROJECT_VERSION}"
    TARGETS tula_netcdf_cxx4
    CONFIG_TEMPLATE
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TulaNetcdfCxx4Config.cmake.in"
)
