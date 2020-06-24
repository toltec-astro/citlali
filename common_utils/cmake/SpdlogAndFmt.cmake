include_guard(GLOBAL)
include(FetchContentHelper)
FetchContentHelper(fmt GIT "https://github.com/fmtlib/fmt.git" "master"
    ADD_SUBDIR
    )
# set_target_properties(fmt PROPERTIES INTERFACE_COMPILE_DEFINITIONS "FMT_USE_CONSTEXPR")
FetchContentHelper(spdlog GIT "https://github.com/gabime/spdlog.git" v1.x
    ADD_SUBDIR CONFIG_SUBDIR
    SPDLOG_FMT_EXTERNAL=ON
    )
# translate log level
string(TOLOWER "${LOGLEVEL}" loglevel)
set(SPDLOG_ACTIVE_LEVEL "")
if (loglevel STREQUAL "trace")
    set(SPDLOG_ACTIVE_LEVEL "SPDLOG_LEVEL_TRACE")
elseif (loglevel STREQUAL "debug")
    set(SPDLOG_ACTIVE_LEVEL "SPDLOG_LEVEL_DEBUG")
elseif (loglevel STREQUAL "info")
    set(SPDLOG_ACTIVE_LEVEL "SPDLOG_LEVEL_INFO")
elseif (loglevel STREQUAL "warn")
    set(SPDLOG_ACTIVE_LEVEL "SPDLOG_LEVEL_WARN")
elseif (loglevel STREQUAL "error")
    set(SPDLOG_ACTIVE_LEVEL "SPDLOG_LEVEL_ERROR")
endif()
get_target_property(defs spdlog COMPILE_DEFINITIONS
   )
get_target_property(idefs spdlog INTERFACE_COMPILE_DEFINITIONS
   )
set_target_properties(spdlog PROPERTIES
   COMPILE_DEFINITIONS
       "SPDLOG_ACTIVE_LEVEL=${SPDLOG_ACTIVE_LEVEL};${defs}"
   INTERFACE_COMPILE_DEFINITIONS
       "SPDLOG_ACTIVE_LEVEL=${SPDLOG_ACTIVE_LEVEL};${idefs}"
       )
add_library(spdlog_and_fmt INTERFACE)
target_link_libraries(spdlog_and_fmt INTERFACE
    spdlog::spdlog
    fmt::fmt
    )
add_library(cmake_utils::spdlog_and_fmt ALIAS spdlog_and_fmt)
