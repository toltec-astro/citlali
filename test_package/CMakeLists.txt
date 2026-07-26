cmake_minimum_required(VERSION 3.25)
project(citlali_package_test LANGUAGES CXX)

find_package(citlali CONFIG REQUIRED)

add_executable(citlali_package_test src/main.cpp)
target_link_libraries(citlali_package_test PRIVATE citlali::citlali)
target_compile_features(citlali_package_test PRIVATE cxx_std_23)
