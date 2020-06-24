include_guard(GLOBAL)
include(FetchContentHelper)
FetchContentHelper(uriparser GIT https://github.com/uriparser/uriparser.git master
    ADD_SUBDIR CONFIG_SUBDIR
        BUILD_SHARED_LIBS=OFF
        URIPARSER_BUILD_DOCS=OFF
        URIPARSER_BUILD_TESTS=OFF
        URIPARSER_BUILD_TOOLS=OFF
    )
add_library(uriparser::uriparser ALIAS uriparser)
