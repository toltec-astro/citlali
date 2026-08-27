cmake_minimum_required(VERSION 3.25)

project(citlali_installed_consumer LANGUAGES C CXX)

find_package(citlali 4.0 CONFIG REQUIRED)

add_executable(citlali_installed_consumer main.cpp)
target_link_libraries(citlali_installed_consumer PRIVATE citlali::citlali)
target_compile_features(citlali_installed_consumer PRIVATE cxx_std_23)

enable_testing()
add_test(NAME citlali_installed_consumer COMMAND citlali_installed_consumer)
