#pragma once

#include <kids/core/kidsdata.h>
#include <kids/sweep/fitter.h>
#include <kids/timestream/solver.h>
#include <kids/toltec/toltec.h>
#include <kidscpp_config/gitversion.h>

#include <tula/datatable.h>
#include <unordered_map>
#include <stdexcept>
#include <utility>

#include <citlali/core/engine/io.h>

/**
 * @brief The KIDs data solver struct
 * This wraps around the kids config
 */

bool extra_output = 0;
struct KidsDataProc : ConfigMapper<KidsDataProc> {
    using Base = ConfigMapper<KidsDataProc>;
    using Fitter = kids::SweepFitter;
    using Solver = kids::TimeStreamSolver;

    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    KidsDataProc(config_t config)
        : Base{std::move(config)},
          m_fitter{Fitter::Config{
              {"weight_window_type", this->config().get_str(std::tuple{
                                         "fitter", "weight_window", "type"})},
              {"weight_window_fwhm", this->config().get_typed<double>(
                   std::tuple{"fitter", "weight_window", "fwhm_Hz"})},
              {"modelspec", config.get_str(std::tuple{"fitter", "modelspec"})}}},
           m_solver{Solver::Config{
              {"fitreportdir", this->config().get_str(std::tuple{"solver", "fitreportdir"})},
              {"exmode", this->config().get_str(std::tuple{"solver", "parallel_policy"})},
              {"extra_output", extra_output},
          }} {}

    static auto check_config(const config_t &config)
        -> std::optional<std::string> {
        // get logger
        std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

        std::vector<std::string> missing_keys;
        logger->debug("check kids data proc config\n{}", config);
        if (!config.has("fitter")) {
            missing_keys.push_back("fitter");
        }
        if (!config.has("solver")) {
            missing_keys.push_back("solver");
        }
        if (missing_keys.empty()) {
            return std::nullopt;
        }
        return fmt::format("invalid or missing keys={}", missing_keys);
    }

    // get data item meta data
    auto get_data_item_meta(const RawObs::DataItem &);

    // get meta data from rawobs
    std::vector<kids::KidsData<>::meta_t> get_rawobs_meta(const RawObs &);

    // populate rtc meta data
    auto populate_rtc_meta(const RawObs &);

    // reduce data item
    auto reduce_data_item(const RawObs::DataItem &,
                          const tula::container_utils::Slice<int> &);
    // reduce rawobs
    auto reduce_rawobs(const RawObs &rawobs,
                       const tula::container_utils::Slice<int> &);
    // load data item
    auto load_data_item(const RawObs::DataItem &,
                        const tula::container_utils::Slice<int> &);
    // load kids fit report
    auto load_fit_report(const RawObs &);

    // load rawobs
    template <typename Derived>
    auto load_rawobs(const RawObs &, const Eigen::Index,
                     Eigen::DenseBase<Derived> &,
                     std::vector<Eigen::Index> &,
                     std::vector<Eigen::Index> &);

    // populate rtc
    template <typename loaded_t>
    auto populate_rtc(loaded_t &, const int, const int,
                      const std::string);

    // read+solve rawobs directly into rtc matrix (avoids intermediate loaded vector)
    template <typename Derived>
    auto populate_rtc_from_rawobs(const RawObs &, const Eigen::Index,
                                  Eigen::DenseBase<Derived> &,
                                  std::vector<Eigen::Index> &,
                                  std::vector<Eigen::Index> &,
                                  const int, const int, const std::string);

    // load rawobs with gaps
    template <typename DerivedA, typename DerivedB, typename DerivedC>
    auto load_rawobs_gaps(const RawObs &, const Eigen::Index,
                          Eigen::DenseBase<DerivedA>&,
                          std::vector<Eigen::Index>&,
                          Eigen::DenseBase<DerivedB>&,
                          std::vector<DerivedC>&,
                          const double);

    // populate rtc with gaps
    template <typename LoadedType, typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD>
    auto populate_rtc_gaps(LoadedType &, Eigen::DenseBase<DerivedA>&,
                          std::vector<DerivedB>&,
                          std::vector<DerivedC>&,
                          const int, const double,
                          Eigen::DenseBase<DerivedD>&,
                          const int, const int, const std::string);

    // TODO fix the const correctness
    Fitter &fitter() { return m_fitter; }
    Solver &solver() { return m_solver; }

    const Fitter &fitter() const { return m_fitter; }
    const Solver &solver() const { return m_solver; }

    template <typename OStream>
    friend OStream &operator<<(OStream &os, const KidsDataProc &kidsproc) {
        return os << fmt::format("KidsDataProc(fitter={}, solver={})",
                                 kidsproc.fitter().config.pformat(),
                                 kidsproc.solver().config.pformat());
    }

private:
    // fitter and solver
    Fitter m_fitter;
    Solver m_solver;
    // cache data kind lookup by filepath to avoid repeated metadata reads
    std::unordered_map<std::string, kids::KidsDataKind> m_data_item_kind_cache;
};

auto KidsDataProc::get_data_item_meta(const RawObs::DataItem &data_item) {
    namespace kidsdata = predefs::kidsdata;
    auto source = data_item.filepath();
    predefs::suppress_hdf5_diagnostics_for_this_thread();
    std::lock_guard<std::mutex> lock(predefs::netcdf_io_mutex());
    auto [kind, meta] = kidsdata::get_meta<>(source);
    return meta;
}

std::vector<kids::KidsData<>::meta_t> KidsDataProc::get_rawobs_meta(const RawObs &rawobs) {
    std::vector<kids::KidsData<>::meta_t> result;
    for (const auto &data_item : rawobs.kidsdata()) {
        result.push_back(get_data_item_meta(data_item));
    }
    return result;
}

auto KidsDataProc::populate_rtc_meta(const RawObs &rawobs) {
    std::vector<kids::KidsData<>::meta_t> result;
    for (const auto &data_item : rawobs.kidsdata()) {
        result.push_back(get_data_item_meta(data_item));
    }
    return result;
}

auto KidsDataProc::reduce_data_item(const RawObs::DataItem &data_item,
                                    const tula::container_utils::Slice<int> &slice) {
    logger->debug("kids reduce data_item {}", data_item);
    // read data
    namespace kidsdata = predefs::kidsdata;
    auto source = data_item.filepath();
    kids::KidsDataKind kind;
    kids::KidsData<>::meta_t meta;
    {
        predefs::suppress_hdf5_diagnostics_for_this_thread();
        std::lock_guard<std::mutex> lock(predefs::netcdf_io_mutex());
        auto km = kidsdata::get_meta<>(source);
        kind = km.first;
        meta = std::move(km.second);
    }
    if (!(kind & kids::KidsDataKind::TimeStream)) {
        throw std::runtime_error(
            fmt::format("wrong type of kids data {}", kind));
    }
    kids::KidsData<kids::KidsDataKind::RawTimeStream> rts;
    try {
        predefs::suppress_hdf5_diagnostics_for_this_thread();
        std::lock_guard<std::mutex> lock(predefs::netcdf_io_mutex());
        rts = kidsdata::read_data_slice<kids::KidsDataKind::RawTimeStream>(
            source, slice);
    } catch (const std::exception &e) {
        throw std::runtime_error(fmt::format(
            "failed to read raw timestream slice from {} slice {}: {}",
            source, slice, e.what()));
    }
    auto result = this->solver()(rts, Solver::Config{});
    return result;
}

auto KidsDataProc::reduce_rawobs(const RawObs &rawobs,
                                 const tula::container_utils::Slice<int> &slice) {
    logger->debug("kids reduce rawobs {}", rawobs);
    std::vector<kids::TimeStreamSolverResult> result;
    for (const auto &data_item : rawobs.kidsdata()) {
        result.push_back(reduce_data_item(data_item, slice));
    }
    return result;
}

auto KidsDataProc::load_data_item(const RawObs::DataItem &data_item,
                                  const tula::container_utils::Slice<int> &slice) {
    logger->debug("kids reduce data_item {}", data_item);
    // read data
    namespace kidsdata = predefs::kidsdata;
    auto source = data_item.filepath();
    kids::KidsDataKind kind;
    if (auto it = m_data_item_kind_cache.find(source); it != m_data_item_kind_cache.end()) {
        kind = it->second;
    }
    else {
        predefs::suppress_hdf5_diagnostics_for_this_thread();
        std::lock_guard<std::mutex> lock(predefs::netcdf_io_mutex());
        auto [kind_, meta] = kidsdata::get_meta<>(source);
        kind = kind_;
        m_data_item_kind_cache[source] = kind;
    }
    if (!(kind & kids::KidsDataKind::TimeStream)) {
        throw std::runtime_error(
            fmt::format("wrong type of kids data {}", kind));
    }
    kids::KidsData<kids::KidsDataKind::RawTimeStream> rts;
    try {
        predefs::suppress_hdf5_diagnostics_for_this_thread();
        std::lock_guard<std::mutex> lock(predefs::netcdf_io_mutex());
        rts = kidsdata::read_data_slice<kids::KidsDataKind::RawTimeStream>(
            source, slice);
    } catch (const std::exception &e) {
        throw std::runtime_error(fmt::format(
            "failed to read raw timestream slice from {} slice {}: {}",
            source, slice, e.what()));
    }
    return rts;
}

auto KidsDataProc::load_fit_report(const RawObs &rawobs) {
    std::vector<Eigen::MatrixXd> kids_models;
    std::vector<std::string> header;

    for (const auto &data_item : rawobs.kidsdata()) {
        auto meta = get_data_item_meta(data_item);
        //auto fitreport = this->solver().loadfitreport(this->config(),meta);

        namespace fs = std::filesystem;
        auto pattern = meta.get_str("cal_file");
        std::string filepath{};
        if (this->solver().config.has("fitreportfile")) {
            filepath = this->solver().config.get_str("fitreportfile");
        } else if (this->solver().config.has("fitreportdir")) {
            auto dir = this->solver().config.get_str("fitreportdir");
            logger->info("look for fitreport dir {} with pattern {}", dir, pattern);
            auto candidates = tula::filename_utils::find_regex(dir, pattern);
            if (!candidates.empty()) {
                filepath = candidates[0];
            } else {
                throw std::runtime_error(fmt::format(
                    "no fit report found in {} that matches {}", dir, pattern));
            }
        } else {
            throw std::runtime_error(
                fmt::format("no fit report location specified."));
        }
        logger->info("use fitreport file {}", filepath);
        //std::vector<std::string> header;
        header.clear();
        Eigen::MatrixXd table;
        using meta_t = kids::KidsData<>::meta_t;
        meta_t meta_cal{};

        try {
            YAML::Node meta_;
            table = datatable::read<double, datatable::Format::ecsv>(
                filepath, &header, &meta_);
            auto meta_map =
                tula::ecsv::meta_to_map<typename meta_t::storage_t::key_type,
                                        typename meta_t::storage_t::mapped_type>(
                    meta_, &meta_);
            meta_cal = meta_t{std::move(meta_map)};

            kids_models.push_back(std::move(table));
            if (!meta_.IsNull()) {
                logger->warn("un recongnized meta:\n{}", YAML::Dump(meta_));
            }
        } catch (datatable::ParseError &e) {
            logger->warn("unable to read fitreport file as ECSV {}: {}", filepath,
                        e.what());
            try {
                table = datatable::read<double, datatable::Format::ascii>(filepath,
                                                                          &header);
                kids_models.push_back(std::move(table));

            } catch (datatable::ParseError &e) {
                logger->warn("unable to read fitreport file as ASCII {}: {}",
                            filepath, e.what());
                throw e;
            }
        }
        logger->info("meta_cal: {}", meta_cal.pformat());
        logger->info("table {}",table);
        logger->info("header {}",header);

        //return std::tuple{
        //                  kids::ToneAxis(std::move(table).transpose(), std::move(header)),
        //                  std::move(meta_cal)};
    }

    return std::tuple{std::move(kids_models), std::move(header)};
}

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
                                const std::string data_type) {
    // resize data
    Eigen::MatrixXd data(n_pts, n_det);

    Eigen::Index i = 0;
    // loop through raw timestream objects
    for (std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>>::
         iterator it = loaded.begin(); it != loaded.end(); ++it) {
        // run the solver
        auto result = this->solver()(*it, Solver::Config{});
        // get number of rows
        Eigen::Index n_rows = result.data_out.xs.data.rows();
        // get number of cols
        Eigen::Index n_cols = result.data_out.xs.data.cols();

        // get xs
        if (data_type == "xs") {
            data.block(0, i, n_rows, n_cols) = result.data_out.xs.data;
        }
        // get rs
        else if (data_type == "rs") {
            data.block(0, i, n_rows, n_cols) = result.data_out.rs.data;
        }
        // get is
        else if (data_type == "is") {
            data.block(0, i, n_rows, n_cols) = result.data.is.data;
        }
        // get qs
        else if (data_type == "qs") {
            data.block(0, i, n_rows, n_cols) = result.data.qs.data;
        }
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

template <typename Derived>
auto KidsDataProc::populate_rtc_from_rawobs(const RawObs &rawobs, const Eigen::Index scan,
                                            Eigen::DenseBase<Derived> &scan_indices,
                                            std::vector<Eigen::Index> &start_indices,
                                            std::vector<Eigen::Index> &end_indices,
                                            const int n_pts, const int n_det, const std::string data_type) {
    // resize data
    Eigen::MatrixXd data(n_pts, n_det);

    Eigen::Index i = 0;
    for (const auto &data_item : rawobs.kidsdata()) {
        auto slice = tula::container_utils::Slice<int>{scan_indices(2,scan) + start_indices[i],
                                                       scan_indices(3,scan) + 1 + start_indices[i],
                                                       std::nullopt};
        auto rts = load_data_item(data_item, slice);
        auto result = this->solver()(rts, Solver::Config{});

        // get number of rows
        Eigen::Index n_rows = result.data_out.xs.data.rows();
        // get number of cols
        Eigen::Index n_cols = result.data_out.xs.data.cols();

        // copy requested channel
        if (data_type == "xs") {
            data.block(0, i, n_rows, n_cols) = result.data_out.xs.data;
        }
        else if (data_type == "rs") {
            data.block(0, i, n_rows, n_cols) = result.data_out.rs.data;
        }
        else if (data_type == "is") {
            data.block(0, i, n_rows, n_cols) = result.data.is.data;
        }
        else if (data_type == "qs") {
            data.block(0, i, n_rows, n_cols) = result.data.qs.data;
        }

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
                                     const int n_pts, const int n_det, const std::string data_type) {
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
        // get number of rows
        Eigen::Index n_rows = result.data_out.xs.data.rows();
        // get number of cols
        Eigen::Index n_cols = result.data_out.xs.data.cols();

        Eigen::MatrixXd block(n_rows, n_cols);

        // get xs
        if (data_type == "xs") {
            block = result.data_out.xs.data;
        } else if (data_type == "rs") {
            block = result.data_out.rs.data;
        } else if (data_type == "is") {
            block = result.data.is.data;
        } else if (data_type == "qs") {
            block = result.data.qs.data;
        }

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
