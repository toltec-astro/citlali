#pragma once

#if defined(__has_include)
#if __has_include(<hdf5.h>)
#include <hdf5.h>
#define CITLALI_HAS_HDF5 1
#elif __has_include(<hdf5/serial/hdf5.h>)
#include <hdf5/serial/hdf5.h>
#define CITLALI_HAS_HDF5 1
#else
#define CITLALI_HAS_HDF5 0
#endif
#else
#define CITLALI_HAS_HDF5 0
#endif

namespace citlali::cli {

inline void suppress_optional_hdf5_diagnostics() {
#if CITLALI_HAS_HDF5
    // netCDF may probe optional HDF5 quantization attributes; suppress noisy
    // HDF5 diagnostics when those attributes are absent.
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
#endif
}

}  // namespace citlali::cli
