include_guard(GLOBAL)
include(FetchContentHelper)
FetchContentHelper(mxx GIT "https://github.com/patflick/mxx.git" master
    ADD_SUBDIR CONFIG_SUBDIR
        MXX_BUILD_TESTS=OFF
        MXX_BUILD_DOCS=OFF
    PATCH_SUBDIR
        ${FCH_PATCH_DIR}/patch.sh "mxx_fixcmake.patch"
        )
add_library(mxx::mxx ALIAS mxx)
