#pragma once

// Implementation detail included by kidsproc.h.

template <typename Derived>
auto KidsDataProc::populate_rtc_from_rawobs(const RawObs &rawobs, const Eigen::Index scan,
                                            Eigen::DenseBase<Derived> &scan_indices,
                                            std::vector<Eigen::Index> &start_indices,
                                            std::vector<Eigen::Index> &end_indices,
                                            const Eigen::Index n_pts,
                                            const Eigen::Index n_det,
                                            citlali::config::TodType data_type) {
    citlali::engine_detail::require_kids_matrix_dimensions(n_pts, n_det);
    if (scan < 0 || scan >= scan_indices.cols() ||
        scan_indices.rows() < 4 ||
        scan_indices(2, scan) < 0 ||
        scan_indices(3, scan) < scan_indices(2, scan) ||
        scan_indices(3, scan) ==
            std::numeric_limits<Eigen::Index>::max() ||
        scan_indices(3, scan) - scan_indices(2, scan) + 1 != n_pts) {
        throw std::runtime_error("invalid direct RTC scan window");
    }
    const auto kids_data = rawobs.kidsdata();
    if (kids_data.size() != start_indices.size() ||
        kids_data.size() != end_indices.size()) {
        throw std::runtime_error(
            "direct RTC KIDs and index cardinalities differ");
    }

    Eigen::MatrixXd data(n_pts, n_det);

    Eigen::Index output_column = 0;
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
                "direct RTC source slice exceeds the index range");
        }
        const Eigen::Index source_start =
            context_start + source_offset;
        const Eigen::Index source_stop =
            context_stop_exclusive + source_offset;
        if (source_start < 0 || source_stop <= source_start ||
            source_stop > source_end + 1) {
            throw std::runtime_error("invalid direct RTC source slice");
        }
        auto slice = tula::container_utils::Slice<int>{
            citlali::engine_detail::checked_kids_slice_index(
                source_start, "direct RTC slice start"),
            citlali::engine_detail::checked_kids_slice_index(
                source_stop, "direct RTC slice stop"),
            std::nullopt};
        auto rts = load_data_item(data_item, slice);
        auto result = this->solver()(rts, Solver::Config{});

        Eigen::Index n_cols = 0;
        citlali::pipeline::visit_kids_tod_channel(
            result, data_type, [&](const auto &channel) {
                const Eigen::Index n_rows = channel.rows();
                n_cols = channel.cols();
                if (n_rows != n_pts || n_cols < 0 ||
                    output_column > n_det ||
                    n_cols > n_det - output_column) {
                    throw std::runtime_error(
                        "direct RTC solver output shape exceeds its admitted matrix");
                }
                data.block(0, output_column, n_rows, n_cols) = channel;
            });

        output_column += n_cols;
        ++stream_index;
    }
    if (output_column != n_det) {
        throw std::runtime_error(
            "direct RTC solver output does not match configured detector cardinality");
    }

    citlali::pipeline::require_finite_kids_input(
        data, "direct RTC KIDs input");

    return data;
}
