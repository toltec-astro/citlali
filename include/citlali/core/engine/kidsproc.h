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

#include <citlali/core/config/timestream_config.h>
#include <citlali/core/engine/io.h>
#include <citlali/core/pipeline/kids_external_config.h>
#include <citlali/core/pipeline/kids_tod_channel.h>

/**
 * @brief The KIDs data solver struct
 * This wraps around the kids config
 */

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
              {"extra_output",
               citlali::pipeline::kids_solver_extra_output_effective},
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
                      citlali::config::TodType);

    // read+solve rawobs directly into rtc matrix (avoids intermediate loaded vector)
    template <typename Derived>
    auto populate_rtc_from_rawobs(const RawObs &, const Eigen::Index,
                                  Eigen::DenseBase<Derived> &,
                                  std::vector<Eigen::Index> &,
                                  std::vector<Eigen::Index> &,
                                  const int, const int,
                                  citlali::config::TodType);

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
                          const int, const int,
                          citlali::config::TodType);

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


#include <citlali/core/engine/detail/kidsproc_metadata_reduce_impl.h>
#include <citlali/core/engine/detail/kidsproc_load_rawobs_impl.h>
#include <citlali/core/engine/detail/kidsproc_direct_rtc_impl.h>
#include <citlali/core/engine/detail/kidsproc_gaps_impl.h>
