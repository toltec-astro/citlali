include_guard(GLOBAL)
option(USE_INSTALLED_NETCDFCXX4 "Use installed netCDFcxx4" OFF)
include(PrintProperties)
if (USE_INSTALLED_NETCDFCXX4)
    set(NETCDFCXX4_DIR ${USE_INSTALLED_NETCDFCXX4}/lib/cmake)
    if (EXISTS ${NETCDFCXX4_DIR})
        message("Use netCDFcxx4 from ${USE_INSTALLED_NETCDFCXX4}")
    else()
        message("Use netCDFcxx4 from default location")
        unset(NETCDFCXX4_DIR)
    endif()
    find_package(NetCDF REQUIRED COMPONENTS CXX4)
    message("Found netCDFcxx4 in ${NETCDF_INCLUDE_DIRS}")
    add_library(netcdf-cxx4 INTERFACE)
    target_include_directories(netcdf-cxx4 INTERFACE ${NETCDF_INCLUDE_DIRS})
    target_link_libraries(netcdf-cxx4 INTERFACE ${NETCDF_CXX4_LIBRARIES})
else()
    include(FetchContentHelper)
    FetchContentHelper(netcdfcxx4 GIT https://github.com/Unidata/netcdf-cxx4.git "master"
        ADD_SUBDIR CONFIG_SUBDIR
            BUILD_SHARED_LIBS=OFF
            NCXX_ENABLE_TESTS=OFF
        PATCH_SUBDIR
            ${FCH_PATCH_DIR}/patch.sh "netcdf_cxx4_fixcmake.patch"
        )
endif()
add_library(NetCDF::NetCDFCXX4 ALIAS netcdf-cxx4)
print_target_properties(NetCDF::NetCDFCXX4)
