include_guard(GLOBAL)
option(USE_INSTALLED_EIGEN3 "Use installed Eigen3" OFF)
option(USE_EIGEN3_WITH_MKL "Use intel mkl library if installed" ON)
option(USE_EIGEN3_WITH_OMP "Use openmp library if installed" ON)
include(PrintProperties)
# performance libs
set(perfdefs "")
set(perflibs "")
find_package(MKL)
if (NOT MKL_FOUND)
    message("No intel mkl library found")
    set(USE_EIGEN3_WITH_MKL OFF)
else()
    set(perfdefs ${perfdefs} EIGEN_USE_MKL_ALL)
    set(perflibs ${perflibs} MKL::MKL)
    print_target_properties(MKL::MKL)
endif()
find_package(OpenMP)
if (NOT OpenMP_FOUND)
    message("No openmp library found")
    set(USE_EIGEN3_WITH_OMP OFF)
else()
    set(perflibs ${perflibs} OpenMP::OpenMP_CXX)
    print_target_properties(OpenMP::OpenMP_CXX)
endif()

if (USE_INSTALLED_EIGEN3)
    set(Eigen3_DIR ${USE_INSTALLED_EIGEN3}/share/eigen3/cmake)
    if (EXISTS ${Eigen3_DIR})
        message("Use Eigen3 from ${USE_INSTALLED_EIGEN3}")
    else()
        message("Use Eigen3 from default location")
        unset(Eigen3_DIR)
    endif()
    find_package(Eigen3 REQUIRED
        NO_MODULE
        NO_CMAKE_PACKAGE_REGISTRY
        NO_CMAKE_BUILDS_PATH)
    message("Found Eigen3 in ${EIGEN3_INCLUDE_DIR}")
else()
    include(FetchContentHelper)
    FetchContentHelper(eigen GIT "https://gitlab.com/libeigen/eigen.git" "master"
        ADD_SUBDIR CONFIG_SUBDIR
            BUILD_TESTING=OFF
        PATCH_SUBDIR
            ${FCH_PATCH_DIR}/patch.sh "eigen3*.patch"
        REGISTER_PACKAGE
            Eigen3 INCLUDE_CONTENT
                "set(EIGEN3_FOUND TRUE)\nset(EIGEN3_INCLUDE_DIR \${eigen_SOURCE_DIR})"
        )
endif()
if (TARGET eigen)
    set(eigen_target "eigen")
else()
    set(eigen_target "Eigen3::Eigen")
endif()
set_property(
    TARGET ${eigen_target}
    APPEND PROPERTY
    INTERFACE_COMPILE_OPTIONS
    $<$<COMPILE_LANGUAGE:CXX>:-march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native --expt-relaxed-constexpr --expt-extended-lambda -Xcudafe --display_error_number>
)
if (USE_EIGEN3_WITH_MKL)
    message("Enable mkl libraries for Eigen3::Eigen")
    set_property(
        TARGET ${eigen_target}
        APPEND PROPERTY
        INTERFACE_COMPILE_DEFINITIONS EIGEN_USE_MKL_ALL
    )
    set_property(
        TARGET ${eigen_target}
        APPEND PROPERTY
        INTERFACE_LINK_LIBRARIES MKL::MKL
    )
endif()
if (USE_EIGEN3_WITH_OMP)
    message("Enable omp libraries for Eigen3::Eigen")
    set_property(
        TARGET ${eigen_target}
        APPEND PROPERTY
        INTERFACE_LINK_LIBRARIES OpenMP::OpenMP_CXX
    )
endif()
print_target_properties(Eigen3::Eigen)
message("Create Eigen target cmake_utils::Eigen with performance libs")
add_library(eigen_with_perflibs INTERFACE)
target_compile_definitions(eigen_with_perflibs
    INTERFACE
        ${perfdefs}
    )
target_link_libraries(eigen_with_perflibs
    INTERFACE
    	Eigen3::Eigen
        ${perflibs}
    )
add_library(cmake_utils::EigenWithPerfLibs ALIAS eigen_with_perflibs)
print_target_properties(cmake_utils::EigenWithPerfLibs)

# set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_COMPILE_DEFINITIONS "EIGEN_NO_MALLOC")
