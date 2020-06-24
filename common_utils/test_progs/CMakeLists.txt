add_custom_target(test_progs)
foreach(prog mpi)
    add_subdirectory(${prog})
endforeach()
