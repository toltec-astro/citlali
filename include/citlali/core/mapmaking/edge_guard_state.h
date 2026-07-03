#pragma once

#include <cstddef>
#include <limits>

namespace mapmaking {

inline double edge_guard_fill_double() {
    return std::numeric_limits<double>::quiet_NaN();
}

template <class MapBuffer>
void reset_edge_guard_storage(MapBuffer &mb, std::size_t n_maps) {
    mb.edge_guard_applied.assign(n_maps, 0);
    mb.edge_guard_support_radius_pix.assign(n_maps, 0);
    mb.edge_guard_science_npix.assign(n_maps, 0);
    mb.edge_guard_support_npix.assign(n_maps, 0);
    mb.edge_guard_guardband_npix.assign(n_maps, 0);
    mb.edge_guard_weight_threshold.assign(n_maps, edge_guard_fill_double());
    mb.edge_guard_hits_threshold.assign(n_maps, edge_guard_fill_double());
    mb.edge_guard_background_level.assign(n_maps, edge_guard_fill_double());
    mb.edge_guard_science_frac.assign(n_maps, edge_guard_fill_double());
    mb.edge_guard_support_frac.assign(n_maps, edge_guard_fill_double());
    mb.edge_guard_guardband_rms_pre.assign(n_maps, edge_guard_fill_double());
    mb.edge_guard_guardband_rms_post.assign(n_maps, edge_guard_fill_double());
    mb.edge_guard_exterior_rms_pre.assign(n_maps, edge_guard_fill_double());
    mb.edge_guard_exterior_rms_post.assign(n_maps, edge_guard_fill_double());
    mb.edge_guard_exterior_max_abs_pre.assign(n_maps, edge_guard_fill_double());
    mb.edge_guard_exterior_max_abs_post.assign(n_maps, edge_guard_fill_double());
    mb.edge_guard_window.resize(n_maps);
}

template <class MapBuffer>
void ensure_edge_guard_storage(MapBuffer &mb) {
    const auto n_maps = static_cast<std::size_t>(mb.signal.size());
    if (mb.edge_guard_applied.size() != n_maps) {
        reset_edge_guard_storage(mb, n_maps);
    }
}

template <class MapBuffer>
void reset_edge_guard_map(MapBuffer &mb, std::size_t map_index) {
    mb.edge_guard_applied[map_index] = 0;
    mb.edge_guard_support_radius_pix[map_index] = 0;
    mb.edge_guard_science_npix[map_index] = 0;
    mb.edge_guard_support_npix[map_index] = 0;
    mb.edge_guard_guardband_npix[map_index] = 0;
    mb.edge_guard_weight_threshold[map_index] = edge_guard_fill_double();
    mb.edge_guard_hits_threshold[map_index] = edge_guard_fill_double();
    mb.edge_guard_background_level[map_index] = edge_guard_fill_double();
    mb.edge_guard_science_frac[map_index] = edge_guard_fill_double();
    mb.edge_guard_support_frac[map_index] = edge_guard_fill_double();
    mb.edge_guard_guardband_rms_pre[map_index] = edge_guard_fill_double();
    mb.edge_guard_guardband_rms_post[map_index] = edge_guard_fill_double();
    mb.edge_guard_exterior_rms_pre[map_index] = edge_guard_fill_double();
    mb.edge_guard_exterior_rms_post[map_index] = edge_guard_fill_double();
    mb.edge_guard_exterior_max_abs_pre[map_index] = edge_guard_fill_double();
    mb.edge_guard_exterior_max_abs_post[map_index] = edge_guard_fill_double();
    if (map_index < mb.edge_guard_window.size()) {
        mb.edge_guard_window[map_index].resize(0, 0);
    }
}

} // namespace mapmaking
