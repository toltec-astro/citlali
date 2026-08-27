cmake_minimum_required(VERSION 3.25)
project(kidscpp_real_reader LANGUAGES CXX)

find_package(kidscpp CONFIG REQUIRED)

add_executable(kidscpp_real_reader main.cpp)
target_link_libraries(kidscpp_real_reader PRIVATE kids::kids)
target_compile_features(kidscpp_real_reader PRIVATE cxx_std_23)

if(NOT KIDS_REAL_FIXTURE)
    message(FATAL_ERROR "KIDS_REAL_FIXTURE is required")
endif()

enable_testing()
add_test(
    NAME kidscpp_real_reader
    COMMAND kidscpp_real_reader "${KIDS_REAL_FIXTURE}"
)
