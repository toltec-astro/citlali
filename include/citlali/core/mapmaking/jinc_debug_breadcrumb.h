#pragma once

namespace mapmaking {

struct JincDebugBreadcrumb {
    bool valid = false;
    const char *stage = "unset";
    long long det_col = -1;
    int det_uid = -1;
    long long sample = -1;
    long long map_index = -1;
    long long array_index = -1;
    int pixel_row = -1;
    int pixel_col = -1;
    int subpix_idx = -1;
    int lower_row = -1;
    int upper_row = -1;
    int lower_col = -1;
    int upper_col = -1;
    int jinc_lower_row = -1;
    int jinc_lower_col = -1;
    int size_rows = -1;
    int size_cols = -1;
};

inline thread_local JincDebugBreadcrumb jinc_debug_breadcrumb{};

inline void reset_jinc_debug_breadcrumb() {
    jinc_debug_breadcrumb = {};
}

inline void update_jinc_debug_breadcrumb(const char *stage,
                                         long long det_col,
                                         int det_uid,
                                         long long sample,
                                         long long map_index,
                                         long long array_index,
                                         int pixel_row,
                                         int pixel_col,
                                         int subpix_idx) {
    jinc_debug_breadcrumb.valid = true;
    jinc_debug_breadcrumb.stage = stage;
    jinc_debug_breadcrumb.det_col = det_col;
    jinc_debug_breadcrumb.det_uid = det_uid;
    jinc_debug_breadcrumb.sample = sample;
    jinc_debug_breadcrumb.map_index = map_index;
    jinc_debug_breadcrumb.array_index = array_index;
    jinc_debug_breadcrumb.pixel_row = pixel_row;
    jinc_debug_breadcrumb.pixel_col = pixel_col;
    jinc_debug_breadcrumb.subpix_idx = subpix_idx;
}

inline void update_jinc_debug_breadcrumb_block(const char *stage,
                                               int lower_row,
                                               int upper_row,
                                               int lower_col,
                                               int upper_col,
                                               int jinc_lower_row,
                                               int jinc_lower_col,
                                               int size_rows,
                                               int size_cols) {
    jinc_debug_breadcrumb.valid = true;
    jinc_debug_breadcrumb.stage = stage;
    jinc_debug_breadcrumb.lower_row = lower_row;
    jinc_debug_breadcrumb.upper_row = upper_row;
    jinc_debug_breadcrumb.lower_col = lower_col;
    jinc_debug_breadcrumb.upper_col = upper_col;
    jinc_debug_breadcrumb.jinc_lower_row = jinc_lower_row;
    jinc_debug_breadcrumb.jinc_lower_col = jinc_lower_col;
    jinc_debug_breadcrumb.size_rows = size_rows;
    jinc_debug_breadcrumb.size_cols = size_cols;
}

inline const JincDebugBreadcrumb &get_jinc_debug_breadcrumb() {
    return jinc_debug_breadcrumb;
}

}  // namespace mapmaking
