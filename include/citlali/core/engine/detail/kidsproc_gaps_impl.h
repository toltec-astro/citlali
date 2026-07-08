#pragma once

// Implementation detail included by kidsproc.h.

template <typename DerivedA, typename DerivedB, typename DerivedC>
auto KidsDataProc::load_rawobs_gaps(const RawObs &rawobs, const Eigen::Index scan,
                                    Eigen::DenseBase<DerivedA>& scan_indices,
                                    std::vector<Eigen::Index>& start_indices,
                                    Eigen::DenseBase<DerivedB>& t_common,
                                    std::vector<DerivedC>& times,
                                    const double tol) {

    std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>> result;

    if (scan_indices(2, scan) < 0 || scan_indices(3, scan) >= t_common.size() ||
        scan_indices(3, scan) < scan_indices(2, scan)) {
        throw std::runtime_error(fmt::format(
            "invalid gap scan time window for scan {}: start={} end={} t_common.size={}",
            scan, scan_indices(2, scan), scan_indices(3, scan), t_common.size()));
    }
    double t0 = t_common(scan_indices(2, scan));
    double t1 = t_common(scan_indices(3, scan));

    auto find_sample_range = [&](const auto &time, Eigen::Index stream_index) {
        if (time.size() == 0) {
            throw std::runtime_error(fmt::format(
                "empty KIDs time vector for stream {} while loading scan {}", stream_index, scan));
        }

        Eigen::Index i_start = 0;
        while (i_start < time.size()) {
            double t = time(i_start);
            if (t >= t0 - tol) {
                if (t > t0 + tol) {
                    if (i_start == 0) {
                        throw std::runtime_error(fmt::format(
                            "no KIDs sample within start tolerance for stream {} scan {}", stream_index, scan));
                    }
                    i_start--;
                }
                break;
            }
            ++i_start;
        }
        if (i_start >= time.size()) {
            throw std::runtime_error(fmt::format(
                "failed to find KIDs scan start for stream {} scan {}", stream_index, scan));
        }

        Eigen::Index i_end = i_start;
        while (i_end < time.size()) {
            double t = time(i_end);
            if (t >= t1 - tol) {
                if (t > t1 + tol) {
                    if (i_end == 0) {
                        throw std::runtime_error(fmt::format(
                            "no KIDs sample within end tolerance for stream {} scan {}", stream_index, scan));
                    }
                    i_end--;
                }
                break;
            }
            ++i_end;
        }
        if (i_end >= time.size()) {
            throw std::runtime_error(fmt::format(
                "failed to find KIDs scan end for stream {} scan {}", stream_index, scan));
        }
        if (i_end < i_start) {
            throw std::runtime_error(fmt::format(
                "invalid KIDs sample range for stream {} scan {}: start={} end={}",
                stream_index, scan, i_start, i_end));
        }

        return std::pair<Eigen::Index, Eigen::Index>{i_start, i_end};
    };

    int i = 0;
    for (const auto &data_item : rawobs.kidsdata()) {
        if (i >= static_cast<int>(times.size())) {
            throw std::runtime_error("rawobs KIDs stream count exceeds time-vector count");
        }
        auto [i_start, i_end] = find_sample_range(times[i], i);

        // get slice of data for current scan
        auto slice = tula::container_utils::Slice<int>{i_start, i_end + 1,
                                                       std::nullopt};
        result.push_back(load_data_item(data_item, slice));

        ++i;
    }

    return result;
}

template <typename LoadedType, typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD>
auto KidsDataProc::populate_rtc_gaps(LoadedType &loaded, Eigen::DenseBase<DerivedA>& t_common,
                                     std::vector<DerivedB>& times,
                                     std::vector<DerivedC>& masks,
                                     const int scan,
                                     const double tol,
                                     Eigen::DenseBase<DerivedD>& scan_indices,
                                     const int n_pts, const int n_det,
                                     citlali::config::TodType data_type) {
    // resize data
    Eigen::MatrixXd data(n_pts, n_det);

    if (scan_indices(2, scan) < 0 || scan_indices(3, scan) >= t_common.size() ||
        scan_indices(3, scan) < scan_indices(2, scan)) {
        throw std::runtime_error(fmt::format(
            "invalid gap scan time window for scan {}: start={} end={} t_common.size={}",
            scan, scan_indices(2, scan), scan_indices(3, scan), t_common.size()));
    }
    double t0 = t_common(scan_indices(2, scan));
    double t1 = t_common(scan_indices(3, scan));

    auto find_sample_range = [&](const auto &time, Eigen::Index stream_index) {
        if (time.size() == 0) {
            throw std::runtime_error(fmt::format(
                "empty KIDs time vector for stream {} while populating scan {}", stream_index, scan));
        }

        Eigen::Index i_start = 0;
        while (i_start < time.size()) {
            double t = time(i_start);
            if (t >= t0 - tol) {
                if (t > t0 + tol) {
                    if (i_start == 0) {
                        throw std::runtime_error(fmt::format(
                            "no KIDs sample within start tolerance for stream {} scan {}", stream_index, scan));
                    }
                    i_start--;
                }
                break;
            }
            ++i_start;
        }
        if (i_start >= time.size()) {
            throw std::runtime_error(fmt::format(
                "failed to find KIDs scan start for stream {} scan {}", stream_index, scan));
        }

        Eigen::Index i_end = i_start;
        while (i_end < time.size()) {
            double t = time(i_end);
            if (t >= t1 - tol) {
                if (t > t1 + tol) {
                    if (i_end == 0) {
                        throw std::runtime_error(fmt::format(
                            "no KIDs sample within end tolerance for stream {} scan {}", stream_index, scan));
                    }
                    i_end--;
                }
                break;
            }
            ++i_end;
        }
        if (i_end >= time.size()) {
            throw std::runtime_error(fmt::format(
                "failed to find KIDs scan end for stream {} scan {}", stream_index, scan));
        }
        if (i_end < i_start) {
            throw std::runtime_error(fmt::format(
                "invalid KIDs sample range for stream {} scan {}: start={} end={}",
                stream_index, scan, i_start, i_end));
        }

        return std::pair<Eigen::Index, Eigen::Index>{i_start, i_end};
    };

    Eigen::Index i = 0, j = 0;
    // loop through raw timestream objects
    for (std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>>::
         iterator it = loaded.begin(); it != loaded.end(); ++it) {
        // run the solver
        auto result = this->solver()(*it, Solver::Config{});
        Eigen::Index n_cols = 0;
        Eigen::MatrixXd block;
        citlali::pipeline::visit_kids_tod_channel(
            result, data_type, [&](const auto &channel) {
                n_cols = channel.cols();
                block = channel;
            });

        if (j >= static_cast<Eigen::Index>(times.size()) || j >= static_cast<Eigen::Index>(masks.size())) {
            throw std::runtime_error("loaded KIDs stream count exceeds time or mask vector count");
        }
        auto [i_start, i_end] = find_sample_range(times[j], j);

        block = engine_utils::interp_data(t_common.segment(scan_indices(2,scan), n_pts),
                                          masks[j].segment(scan_indices(2,scan), n_pts),
                                          times[j].segment(i_start, i_end - i_start + 1),
                                          block);

        data.block(0, i, n_pts, n_cols) = block;
        // increment columns
        i += n_cols;
        j++;
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
