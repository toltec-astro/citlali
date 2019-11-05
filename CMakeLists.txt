cmake_minimum_required(VERSION 3.15)
project(citlali LANGUAGES C CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/common_utils/cmake/CommonSettings.cmake")

set(KIDS_BUILD_GUI OFF CACHE BOOL "" FORCE)
set(KIDS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(KIDS_BUILD_DOCS OFF CACHE BOOL "" FORCE)
include("./kidscpp_src/import.cmake")

include("./citlali_src/import.cmake")
