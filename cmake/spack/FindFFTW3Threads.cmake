include_guard(GLOBAL)

find_library(FFTW3Threads_LIBRARY NAMES fftw3_threads)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    FFTW3Threads
    REQUIRED_VARS FFTW3Threads_LIBRARY
)

if(FFTW3Threads_FOUND AND NOT TARGET FFTW3::Threads)
    add_library(FFTW3::Threads UNKNOWN IMPORTED)
    set_target_properties(
        FFTW3::Threads
        PROPERTIES IMPORTED_LOCATION "${FFTW3Threads_LIBRARY}"
    )
endif()

mark_as_advanced(FFTW3Threads_LIBRARY)
