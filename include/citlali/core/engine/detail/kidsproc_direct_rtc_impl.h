#pragma once

// Implementation detail included by kidsproc.h.

template <typename Derived>
auto KidsDataProc::populate_rtc_from_rawobs(const RawObs &rawobs, const Eigen::Index scan,
                                            Eigen::DenseBase<Derived> &scan_indices,
                                            std::vector<Eigen::Index> &start_indices,
                                            std::vector<Eigen::Index> &end_indices,
                                            const int n_pts, const int n_det,
                                            citlali::config::TodType data_type) {
    // resize data
    Eigen::MatrixXd data(n_pts, n_det);

    Eigen::Index i = 0;
    for (const auto &data_item : rawobs.kidsdata()) {
        auto slice = tula::container_utils::Slice<int>{scan_indices(2,scan) + start_indices[i],
                                                       scan_indices(3,scan) + 1 + start_indices[i],
                                                       std::nullopt};
        auto rts = load_data_item(data_item, slice);
        auto result = this->solver()(rts, Solver::Config{});

        Eigen::Index n_cols = 0;
        citlali::pipeline::visit_kids_tod_channel(
            result, data_type, [&](const auto &channel) {
                Eigen::Index n_rows = channel.rows();
                n_cols = channel.cols();
                data.block(0, i, n_rows, n_cols) = channel;
            });

        // increment columns
        i += n_cols;
    }

    // check for nans
    if ((data.array().isNaN()).any()) {
        logger->error("nan found in data! Check that your KIDs data dir is correct.");
        std::exit(EXIT_FAILURE);
    }
    // check for infs
    if ((data.array().isInf()).any()) {
        logger->error("inf found in data! Check that your KIDs data dir is correct.");
        std::exit(EXIT_FAILURE);
    }

    return data;
}
