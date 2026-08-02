#pragma once

// Implementation detail included by kidsproc.h.

template <typename Derived>
auto KidsDataProc::load_rawobs(const RawObs &rawobs, const Eigen::Index scan,
                               Eigen::DenseBase<Derived> &scan_indices,
                               std::vector<Eigen::Index> &start_indices,
                               std::vector<Eigen::Index> &end_indices) {

    if (scan < 0 || scan >= scan_indices.cols() ||
        scan_indices.rows() < 4 || scan_indices(2, scan) < 0 ||
        scan_indices(3, scan) < scan_indices(2, scan) ||
        scan_indices(3, scan) ==
            std::numeric_limits<Eigen::Index>::max()) {
        throw std::runtime_error("invalid rawobs scan window");
    }
    const auto kids_data = rawobs.kidsdata();
    if (kids_data.size() != start_indices.size() ||
        kids_data.size() != end_indices.size()) {
        throw std::runtime_error(
            "rawobs KIDs and index cardinalities differ");
    }

    std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>> result;
    std::size_t stream_index = 0;
    for (const auto &data_item : kids_data) {
        const auto source_offset = start_indices[stream_index];
        const auto source_end = end_indices[stream_index];
        const auto context_start = scan_indices(2, scan);
        const auto context_stop_exclusive = scan_indices(3, scan) + 1;
        if (source_offset < 0 || source_end < source_offset ||
            source_end == std::numeric_limits<Eigen::Index>::max() ||
            context_start >
                std::numeric_limits<Eigen::Index>::max() - source_offset ||
            context_stop_exclusive >
                std::numeric_limits<Eigen::Index>::max() - source_offset) {
            throw std::overflow_error(
                "rawobs source slice exceeds the index range");
        }
        const auto source_start = context_start + source_offset;
        const auto source_stop = context_stop_exclusive + source_offset;
        if (source_stop > source_end + 1) {
            throw std::runtime_error("rawobs source slice exceeds support");
        }
        auto slice = tula::container_utils::Slice<int>{
            citlali::engine_detail::checked_kids_slice_index(
                source_start, "rawobs slice start"),
            citlali::engine_detail::checked_kids_slice_index(
                source_stop, "rawobs slice stop"),
            std::nullopt};
        result.push_back(load_data_item(data_item, slice));

        ++stream_index;
    }

    return result;
}

template <typename loaded_t>
auto KidsDataProc::populate_rtc(loaded_t &loaded,
                                const Eigen::Index n_pts,
                                const Eigen::Index n_det,
                                citlali::config::TodType data_type) {
    citlali::engine_detail::require_kids_matrix_dimensions(n_pts, n_det);
    Eigen::MatrixXd data(n_pts, n_det);

    Eigen::Index i = 0;
    // loop through raw timestream objects
    for (std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>>::
         iterator it = loaded.begin(); it != loaded.end(); ++it) {
        // run the solver
        auto result = this->solver()(*it, Solver::Config{});
        Eigen::Index n_cols = 0;
        citlali::pipeline::visit_kids_tod_channel(
            result, data_type, [&](const auto &channel) {
                const Eigen::Index n_rows = channel.rows();
                n_cols = channel.cols();
                if (n_rows != n_pts || n_cols < 0 || i > n_det ||
                    n_cols > n_det - i) {
                    throw std::runtime_error(
                        "populated RTC solver output shape exceeds its admitted matrix");
                }
                data.block(0, i, n_rows, n_cols) = channel;
            });
        // increment columns
        i += n_cols;
    }
    if (i != n_det) {
        throw std::runtime_error(
            "populated RTC solver output does not match configured detector cardinality");
    }

    citlali::pipeline::require_finite_kids_input(
        data, "populated RTC KIDs input");

    return data;
}
