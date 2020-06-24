include_guard(GLOBAL)
include(FetchContentHelper)
FetchContentHelper(yaml GIT "https://github.com/jbeder/yaml-cpp.git" "master"
    ADD_SUBDIR CONFIG_SUBDIR
    YAML_CPP_BULID_CONTRIB=OFF
    YAML_CPP_BUILD_TOOLS=OFF
    YAML_CPP_BUILD_TESTS=OFF
    YAML_CPP_INSTALL=OFF
    YAML_BUILD_SHARED_LIBS=OFF
    YAML_MSVC_SHARED_RT=OFF
    )

