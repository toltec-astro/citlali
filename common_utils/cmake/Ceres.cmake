include_guard(GLOBAL)
option(USE_INSTALLED_CERES "Use installed Ceres-solver" OFF)
if (USE_INSTALLED_CERES)
    set(Ceres_DIR ${USE_INSTALLED_CERES}/lib/cmake/Ceres)
    if (EXISTS ${Ceres_DIR})
        message("Use ceres-solver from ${USE_INSTALLED_CERES}")
    else()
        message("Use ceres-solver from default location")
        unset(Ceres_DIR)
    endif()
    find_package(Ceres REQUIRED CONFIG)
else()
    include(FetchContentHelper)
    FetchContentHelper(ceres GIT "https://github.com/ceres-solver/ceres-solver.git" "master"
        ADD_SUBDIR CONFIG_SUBDIR
            EIGEN_PREFER_EXPORTED_EIGEN_CMAKE_CONFIGURATION=ON
            EIGENSPARSE=ON
            SUITESPARSE=ON
            ACCELERATESPARSE=OFF
            CXSPARSE=OFF
            GFLAGS=OFF
            MINIGLOG=ON
            CXX11=ON
            OPENMP=ON
            TBB=OFF
            CXX11_THREADS=OFF
            EXPORT_BUILD_DIR=ON
            BUILD_DOCUMENTATION=OFF
            BUILD_TESTING=OFF
            BUILD_EXAMPLES=OFF
            BUILD_BENCHMARKS=OFF
        PATCH_SUBDIR
            ${FCH_PATCH_DIR}/patch.sh "ceres*.patch"
        )
endif()
add_library(ceres_alias INTERFACE)
target_link_libraries(ceres_alias INTERFACE ceres)
add_library(ceres::ceres ALIAS ceres_alias)
