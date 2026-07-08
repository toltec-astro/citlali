#pragma once

// Implementation detail included by kidsproc.h.

template <typename Derived>
auto KidsDataProc::load_rawobs(const RawObs &rawobs, const Eigen::Index scan,
                               Eigen::DenseBase<Derived> &scan_indices,
                               std::vector<Eigen::Index> &start_indices,
                               std::vector<Eigen::Index> &end_indices) {

    std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>> result;
    Eigen::Index i = 0;
    for (const auto &data_item : rawobs.kidsdata()) {
        // get slice of data for current scan
        auto slice = tula::container_utils::Slice<int>{scan_indices(2,scan) + start_indices[i],
                                                       scan_indices(3,scan) + 1 + start_indices[i],
                                                       std::nullopt};
        result.push_back(load_data_item(data_item, slice));

        i++;
    }

    return result;
}

template <typename loaded_t>
auto KidsDataProc::populate_rtc(loaded_t &loaded,
                                const int n_pts, const int n_det,
                                citlali::config::TodType data_type) {
    // resize data
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
