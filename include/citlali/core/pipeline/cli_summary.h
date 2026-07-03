#pragma once

namespace citlali::pipeline {

template <class MapBuffer>
double map_buffer_memory_gb(const MapBuffer &mb) {
    return 8 * mb.n_rows * mb.n_cols *
           (mb.signal.size() + mb.weight.size() + mb.kernel.size() +
            mb.coverage.size() + mb.grid_weight.size()) /
           1e9;
}

}  // namespace citlali::pipeline
