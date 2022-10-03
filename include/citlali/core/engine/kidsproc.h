#pragma once

/**
 * @brief The KIDs data solver struct
 * This wraps around the kids config
 */

bool extra_output = 0;
struct KidsDataProc : ConfigMapper<KidsDataProc> {
    using Base = ConfigMapper<KidsDataProc>;
    using Fitter = kids::SweepFitter;
    using Solver = kids::TimeStreamSolver;
    KidsDataProc(config_t config)
        : Base{std::move(config)},
          m_fitter{Fitter::Config{
              {"weight_window_type", this->config().get_str(std::tuple{
                                         "fitter", "weight_window", "type"})},
              {"weight_window_fwhm",
               this->config().get_typed<double>(
                   std::tuple{"fitter", "weight_window", "fwhm_Hz"})},
              {"modelspec",
               config.get_str(std::tuple{"fitter", "modelspec"})}}},
          m_solver{Solver::Config{
              {"fitreportdir", "/dev/null"},
              {"exmode", "seq"},
              {"extra_output", extra_output},
          }} {}

    static auto check_config(const config_t &config)
        -> std::optional<std::string> {
        std::vector<std::string> missing_keys;
        SPDLOG_TRACE("check kids data proc config\n{}", config);
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

    auto get_data_item_meta(const RawObs::DataItem &data_item) {
        namespace kidsdata = predefs::kidsdata;
        auto source = data_item.filepath();
        auto [kind, meta] = kidsdata::get_meta<>(source);
        return meta;
    }

    auto get_rawobs_meta(const RawObs &rawobs) {
        std::vector<kids::KidsData<>::meta_t> result;
        for (const auto &data_item : rawobs.kidsdata()) {
            result.push_back(get_data_item_meta(data_item));
        }
        return result;
    }

    auto populate_rtc_meta(const RawObs &rawobs) {
        std::vector<kids::KidsData<>::meta_t> result;
        for (const auto &data_item : rawobs.kidsdata()) {
            result.push_back(get_data_item_meta(data_item));
        }
        return result;
    }

    auto reduce_data_item(const RawObs::DataItem &data_item,
                          const tula::container_utils::Slice<int> &slice) {
        SPDLOG_TRACE("kids reduce data_item {}", data_item);
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

    auto reduce_rawobs(const RawObs &rawobs,
                       const tula::container_utils::Slice<int> &slice) {
        SPDLOG_TRACE("kids reduce rawobs {}", rawobs);
        std::vector<kids::TimeStreamSolverResult> result;
        for (const auto &data_item : rawobs.kidsdata()) {
            result.push_back(reduce_data_item(data_item, slice));
        }
        return result;
    }

    auto load_data_item(const RawObs::DataItem &data_item,
                        const tula::container_utils::Slice<int> &slice) {
        SPDLOG_TRACE("kids reduce data_item {}", data_item);
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

    template <typename Derived>
    auto load_rawobs(const RawObs &rawobs,
                     const Eigen::Index scan,
                     Eigen::DenseBase<Derived> &scan_indices,
                     std::vector<Eigen::Index> &start_indices,
                     std::vector<Eigen::Index> &end_indices) {

        std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>> result;
        Eigen::Index i = 0;
        for (const auto &data_item : rawobs.kidsdata()) {
            auto slice = tula::container_utils::Slice<int>{scan_indices(2,scan),
                                                           scan_indices(3,scan) + 1,
                                                           std::nullopt};
            result.push_back(load_data_item(data_item, slice));

            i++;
        }

        return std::move(result);
    }

    template <typename loaded_t, typename scanindices_t>
    auto populate_rtc(loaded_t &loaded, scanindices_t &scanindex,
                      const int scanlength, const int n_detectors, std::string data_type) {

        Eigen::MatrixXd data(scanlength, n_detectors);

        Eigen::Index i = 0;
        for (std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>>::
                 iterator it = loaded.begin();
            it != loaded.end(); ++it) {
            auto result = this->solver()(*it, Solver::Config{});
            Eigen::Index n_rows = result.data_out.xs.data.rows();
            Eigen::Index n_cols = result.data_out.xs.data.cols();

            if (data_type == "xs") {
                data.block(0, i, n_rows, n_cols) = result.data_out.xs.data;
            }
            else if (data_type == "rs") {
                data.block(0, i, n_rows, n_cols) = result.data_out.rs.data;
            }

            i += n_cols;
        }

        return std::move(data);
    }

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
