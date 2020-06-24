include_guard(GLOBAL)
include(FetchContentHelper)
FetchContentHelper(re2 GIT "https://github.com/google/re2.git" "2019-08-01"
    ADD_SUBDIR CONFIG_SUBDIR
        RE2_BUILD_TESTING=OFF
        )
# without this flag CMake may resort to just -lpthread
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library(re2_wrapped INTERFACE)
target_include_directories(re2_wrapped INTERFACE
    ${re2_SOURCE_DIR})
target_link_libraries(re2_wrapped INTERFACE
    re2::re2
    Threads::Threads)
add_library(google::re2 ALIAS re2_wrapped)
