#pragma once

namespace citlali::pipeline {

struct MapOutputDebugBreadcrumb {
    bool valid = false;
    const char *stage = "unset";
    const char *filepath = "";
    long long map_i = -1;
    long long map_index = -1;
    long long stokes_index = -1;
    long long array_index = -1;
    long long hdu_index = -1;
    long long hdu_count = -1;
    int flag_value = -999;
};

inline thread_local MapOutputDebugBreadcrumb map_output_debug_breadcrumb{};

inline void reset_map_output_debug_breadcrumb() {
    map_output_debug_breadcrumb = {};
}

inline void update_map_output_debug_breadcrumb(const char *stage,
                                               const char *filepath,
                                               long long map_i,
                                               long long map_index,
                                               long long stokes_index,
                                               long long array_index,
                                               long long hdu_index,
                                               long long hdu_count,
                                               int flag_value = -999) {
    map_output_debug_breadcrumb.valid = true;
    map_output_debug_breadcrumb.stage = stage;
    map_output_debug_breadcrumb.filepath = filepath;
    map_output_debug_breadcrumb.map_i = map_i;
    map_output_debug_breadcrumb.map_index = map_index;
    map_output_debug_breadcrumb.stokes_index = stokes_index;
    map_output_debug_breadcrumb.array_index = array_index;
    map_output_debug_breadcrumb.hdu_index = hdu_index;
    map_output_debug_breadcrumb.hdu_count = hdu_count;
    map_output_debug_breadcrumb.flag_value = flag_value;
}

inline const MapOutputDebugBreadcrumb &get_map_output_debug_breadcrumb() {
    return map_output_debug_breadcrumb;
}

}  // namespace citlali::pipeline
