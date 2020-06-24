include_guard(GLOBAL)
find_package(Python3 REQUIRED COMPONENTS Development Interpreter NumPy)
add_library(python3_and_numpy INTERFACE)
target_link_libraries(python3_and_numpy INTERFACE
    Python3::Python
    Python3::NumPy
    )
print_target_properties(Python3::Python)
print_target_properties(Python3::NumPy)
add_library(cmake_utils::python3_and_numpy ALIAS python3_and_numpy)
