#pragma once

#include <kids/core/kidsdata.h>
#include <kids/sweep/fitter.h>
#include <kids/timestream/solver.h>
#include <kids/toltec/toltec.h>
#include <kidscpp_config/gitversion.h>

#include <tula/datatable.h>

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
    template <typename loaded_t, typename scanindices_t>
    auto populate_rtc(loaded_t &, scanindices_t &,
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
};

auto KidsDataProc::get_data_item_meta(const RawObs::DataItem &data_item) {
    namespace kidsdata = predefs::kidsdata;
    auto source = data_item.filepath();
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
    auto [kind, meta] = kidsdata::get_meta<>(source);
    if (!(kind & kids::KidsDataKind::TimeStream)) {
        throw std::runtime_error(
            fmt::format("wrong type of kids data {}", kind));
    }
    auto rts = kidsdata::read_data_slice<kids::KidsDataKind::RawTimeStream>(
        source, slice);
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
    auto [kind, meta] = kidsdata::get_meta<>(source);
    if (!(kind & kids::KidsDataKind::TimeStream)) {
        throw std::runtime_error(
            fmt::format("wrong type of kids data {}", kind));
    }
    auto rts = kidsdata::read_data_slice<kids::KidsDataKind::RawTimeStream>(
        source, slice);
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

    return std::move(result);
}

template <typename loaded_t, typename scanindices_t>
auto KidsDataProc::populate_rtc(loaded_t &loaded, scanindices_t &scanindex,
                                const int scanlength, const int n_det,
                                const std::string data_type) {
    // resize data
    Eigen::MatrixXd data(scanlength, n_det);

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
        logger->error("nan found in data!");
        std::exit(EXIT_FAILURE);
    }
    // check for infs
    if ((data.array().isInf()).any()) {
        logger->error("inf found in data!");
        std::exit(EXIT_FAILURE);
    }

    return data;
}
