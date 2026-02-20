#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>

#include <tula/logging.h>
#include <tula/nc.h>
#include <tula/algorithm/ei_stats.h>

#include <citlali/core/utils/utils.h>
#include <citlali/core/utils/pointing.h>

#include <citlali/core/timestream/timestream.h>
#include <citlali/core/timestream/ptc/clean.h>

#include <citlali/core/utils/toltec_io.h>

namespace timestream {

using timestream::TCData;

class PTCProc: public TCProc {
public:
    // controls for timestream reduction
    bool run_clean;
    // median weight factor
    double med_weight_factor;
    // upper and lower weight limits for outliers
    double lower_weight_factor, upper_weight_factor;
    // weight type (full, approximate, const)
    std::string weighting_type;

    // ptc tod proc
    timestream::Cleaner cleaner;

    struct CorrNWDiagSummary {
        Eigen::Index nw = -1;
        Eigen::Index n_det_input = 0;
        Eigen::Index n_det_candidates = 0;
        Eigen::Index n_det_used = 0;
        Eigen::Index n_det_grouped = 0;
        Eigen::Index n_det_ungrouped = 0;
        Eigen::Index n_groups_raw = 0;
        Eigen::Index n_groups_final = 0;
        Eigen::Index sample_step = 1;
    };
    std::map<Eigen::Index, Eigen::VectorXi> corr_nw_group_ids_by_scan;
    std::map<Eigen::Index, std::vector<CorrNWDiagSummary>> corr_nw_summary_by_scan;

    // get config file
    template <typename config_t>
    void get_config(config_t &, std::vector<std::vector<std::string>> &,
                    std::vector<std::vector<std::string>> &);

    // subtract detector means
    void subtract_mean(TCData<TCDataKind::PTC, Eigen::MatrixXd> &,
                       const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> *flags_override = nullptr);

    // run main processing stage
    template <class calib_type>
    void run(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, TCData<TCDataKind::PTC, Eigen::MatrixXd> &,
             calib_type &, std::string, std::string);

    // calculate detector weights
    template <typename apt_type, class tel_type>
    void calc_weights(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, apt_type &, tel_type &);

    // reset outlier weights to the median
    template <typename calib_t>
    auto reset_weights(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, calib_t &, std::string);

    // append time chunk to tod netcdf file
    template <typename calib_t, typename pointing_offset_t>
    void append_to_netcdf(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, std::string, std::string, std::string &,
                          pointing_offset_t &, calib_t &, bool apply_det_offsets = false,
                          Eigen::Index scan_row_index = -1);
};

// get config file
template <typename config_t>
void PTCProc::get_config(config_t &config, std::vector<std::vector<std::string>> &missing_keys,
                         std::vector<std::vector<std::string>> &invalid_keys) {

    // weight type
    get_config_value(config, weighting_type, missing_keys, invalid_keys,
                     std::tuple{"timestream","processed_time_chunk","weighting","type"},{"full","approximate","const"});
    // median weight factor
    get_config_value(config, med_weight_factor, missing_keys, invalid_keys,
                     std::tuple{"timestream","processed_time_chunk","weighting","median_map_weight_factor"});
    // lower inv var factor
    get_config_value(config, lower_inv_var_factor, missing_keys, invalid_keys,
                     std::tuple{"timestream","processed_time_chunk","flagging","lower_tod_inv_var_factor"});
    // upper inv var factor
    get_config_value(config, upper_inv_var_factor, missing_keys, invalid_keys,
                     std::tuple{"timestream","processed_time_chunk","flagging","upper_tod_inv_var_factor"});

    // lower weight factor
    get_config_value(config, lower_weight_factor, missing_keys, invalid_keys,
                     std::tuple{"timestream","processed_time_chunk","weighting","lower_map_weight_factor"});
    // upper weight factor
    get_config_value(config, upper_weight_factor, missing_keys, invalid_keys,
                     std::tuple{"timestream","processed_time_chunk","weighting","upper_map_weight_factor"});

    // run fruit loops?
    get_config_value(config, run_fruit_loops, missing_keys, invalid_keys,
                     std::tuple{"timestream","fruit_loops","enabled"});

    if (run_fruit_loops) {
        // save all fruit loops iterations?
        get_config_value(config, save_all_iters, missing_keys, invalid_keys,
                         std::tuple{"timestream","fruit_loops","save_all_iters"});
        // fruit looops path
        get_config_value(config, fruit_loops_path, missing_keys, invalid_keys,
                         std::tuple{"timestream","fruit_loops","path"});
        // fruit looops type
        get_config_value(config, fruit_loops_type, missing_keys, invalid_keys,
                         std::tuple{"timestream","fruit_loops","type"});
	// fruit looops mode
        get_config_value(config, fruit_mode, missing_keys, invalid_keys,
                         std::tuple{"timestream","fruit_loops","mode"}, {"upper", "lower", "both"});
        // let user specify "coadd" or "coadded"
        if (fruit_loops_type == "coadded") {
            fruit_loops_type = "coadd";
        }
        // fruit loops signal-to-noise
        get_config_value(config, fruit_loops_sig2noise, missing_keys, invalid_keys,
                         std::tuple{"timestream","fruit_loops", "sig2noise_limit"});
        // fruit loops flux density limit
        auto fruit_loops_flux_vec = config.template get_typed<std::vector<double>>(std::tuple{"timestream","fruit_loops","array_flux_limit"});
        fruit_loops_flux = Eigen::Map<Eigen::VectorXd>(fruit_loops_flux_vec.data(), fruit_loops_flux_vec.size());

        // maximum fruit loops iterations
        get_config_value(config, fruit_loops_iters, missing_keys, invalid_keys,
                         std::tuple{"timestream","fruit_loops","max_iters"});
    }

    // run clean?
    get_config_value(config, run_clean, missing_keys, invalid_keys,
                     std::tuple{"timestream","processed_time_chunk","clean", "enabled"});

    if (run_clean) {
        // get cleaning grouping vector
        cleaner.grouping = config.template get_typed<std::vector<std::string>>(std::tuple{"timestream","processed_time_chunk","clean","grouping"});
        // get cleaning number of eigenvalues vector
        for (auto const& [arr_index, arr_name] : toltec_io.array_name_map) {
            auto n_eig_to_cut = config.template get_typed<std::vector<Eigen::Index>>(std::tuple{"timestream","processed_time_chunk","clean",
                                                                                                "n_eig_to_cut",arr_name});
            if (n_eig_to_cut.empty()) {
                logger->warn("clean.n_eig_to_cut.{} is empty; defaulting to 0 for all {} grouping pass(es)",
                             arr_name, cleaner.grouping.size());
                n_eig_to_cut.assign(cleaner.grouping.size(), 0);
            }
            else if (n_eig_to_cut.size() < cleaner.grouping.size()) {
                logger->warn("clean.n_eig_to_cut.{} has {} value(s) but clean.grouping has {} pass(es); padding with last value {}",
                             arr_name, n_eig_to_cut.size(), cleaner.grouping.size(), n_eig_to_cut.back());
                n_eig_to_cut.resize(cleaner.grouping.size(), n_eig_to_cut.back());
            }
            // add eigenvalues to cleaner class
            cleaner.n_eig_to_cut[arr_index] = (Eigen::Map<Eigen::VectorXI>(n_eig_to_cut.data(),n_eig_to_cut.size()));
        }

        // stddev limit
        get_config_value(config, cleaner.stddev_limit, missing_keys, invalid_keys,
                         std::tuple{"timestream","processed_time_chunk","clean","stddev_limit"});
        // optional: number of eigenvalues to calculate (0 => full spectrum)
        if (config.template has_typed<int>(std::tuple{"timestream","processed_time_chunk","clean","n_calc"})) {
            get_config_value(config, cleaner.n_calc, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","clean","n_calc"},{},{0});
        }
        // optional brute-force null-model mode selection
        if (config.template has_typed<bool>(std::tuple{"timestream","processed_time_chunk","clean","null_model","enabled"})) {
            get_config_value(config, cleaner.null_model.enabled, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","clean","null_model","enabled"});
        }
        // optional correlation-defined grouping inside each network
        if (config.template has_typed<bool>(std::tuple{"timestream","processed_time_chunk","clean","corr_grouping","enabled"})) {
            get_config_value(config, cleaner.corr_grouping.enabled, missing_keys, invalid_keys,
                             std::tuple{"timestream","processed_time_chunk","clean","corr_grouping","enabled"});
        }
        if (cleaner.corr_grouping.enabled) {
            if (config.template has_typed<std::string>(std::tuple{"timestream","processed_time_chunk","clean","corr_grouping","metric"})) {
                get_config_value(config, cleaner.corr_grouping.metric, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","corr_grouping","metric"},
                                 {"abs", "signed"});
            }
            if (config.template has_typed<double>(std::tuple{"timestream","processed_time_chunk","clean","corr_grouping","corr_min"})) {
                get_config_value(config, cleaner.corr_grouping.corr_min, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","corr_grouping","corr_min"},
                                 {}, {0.0}, {1.0});
            }
            if (config.template has_typed<int>(std::tuple{"timestream","processed_time_chunk","clean","corr_grouping","min_overlap"})) {
                get_config_value(config, cleaner.corr_grouping.min_overlap, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","corr_grouping","min_overlap"},
                                 {}, {1});
            }
            if (config.template has_typed<double>(std::tuple{"timestream","processed_time_chunk","clean","corr_grouping","min_good_frac"})) {
                get_config_value(config, cleaner.corr_grouping.min_good_frac, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","corr_grouping","min_good_frac"},
                                 {}, {0.0}, {1.0});
            }
            if (config.template has_typed<int>(std::tuple{"timestream","processed_time_chunk","clean","corr_grouping","min_group_size"})) {
                get_config_value(config, cleaner.corr_grouping.min_group_size, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","corr_grouping","min_group_size"},
                                 {}, {2});
            }
            if (config.template has_typed<int>(std::tuple{"timestream","processed_time_chunk","clean","corr_grouping","max_samples"})) {
                get_config_value(config, cleaner.corr_grouping.max_samples, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","corr_grouping","max_samples"},
                                 {}, {0});
            }
            if (config.template has_typed<bool>(std::tuple{"timestream","processed_time_chunk","clean","corr_grouping","clean_residual"})) {
                get_config_value(config, cleaner.corr_grouping.clean_residual, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","corr_grouping","clean_residual"});
            }
            logger->info("clean.corr_grouping enabled: metric={} corr_min={} min_overlap={} min_good_frac={} min_group_size={} max_samples={} clean_residual={}",
                         cleaner.corr_grouping.metric, cleaner.corr_grouping.corr_min, cleaner.corr_grouping.min_overlap,
                         cleaner.corr_grouping.min_good_frac, cleaner.corr_grouping.min_group_size,
                         cleaner.corr_grouping.max_samples, cleaner.corr_grouping.clean_residual);
        }
        if (cleaner.null_model.enabled) {
            if (config.template has_typed<int>(std::tuple{"timestream","processed_time_chunk","clean","null_model","n_surrogates"})) {
                get_config_value(config, cleaner.null_model.n_surrogates, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","null_model","n_surrogates"},{},{4});
            }
            if (config.template has_typed<double>(std::tuple{"timestream","processed_time_chunk","clean","null_model","quantile"})) {
                get_config_value(config, cleaner.null_model.quantile, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","null_model","quantile"},{},{0.5},{0.999999});
            }
            if (config.template has_typed<double>(std::tuple{"timestream","processed_time_chunk","clean","null_model","min_good_frac"})) {
                get_config_value(config, cleaner.null_model.min_good_frac, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","null_model","min_good_frac"},{},{0.0},{1.0});
            }
            if (config.template has_typed<int>(std::tuple{"timestream","processed_time_chunk","clean","null_model","max_modes"})) {
                get_config_value(config, cleaner.null_model.max_modes, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","null_model","max_modes"},{},{0});
            }
            if (config.template has_typed<int>(std::tuple{"timestream","processed_time_chunk","clean","null_model","max_samples"})) {
                get_config_value(config, cleaner.null_model.max_samples, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","null_model","max_samples"},{},{0});
            }
            int null_seed = static_cast<int>(cleaner.null_model.seed);
            if (config.template has_typed<int>(std::tuple{"timestream","processed_time_chunk","clean","null_model","seed"})) {
                get_config_value(config, null_seed, missing_keys, invalid_keys,
                                 std::tuple{"timestream","processed_time_chunk","clean","null_model","seed"},{},{0});
            }
            cleaner.null_model.seed = static_cast<std::uint32_t>(null_seed);
            // optional: restrict null-model mode selection to a subset of clean.grouping passes
            cleaner.null_model.grouping.clear();
            if (config.template has_typed<std::vector<std::string>>(
                    std::tuple{"timestream","processed_time_chunk","clean","null_model","grouping"})) {
                auto null_grouping = config.template get_typed<std::vector<std::string>>(
                    std::tuple{"timestream","processed_time_chunk","clean","null_model","grouping"});
                std::unordered_set<std::string> seen;
                for (const auto &g_raw : null_grouping) {
                    auto g = cleaner.normalize_group_name(g_raw);
                    if (g != "all" && g != "array" && g != "nw" && g != "detector" && g != "fg" && g != "corr_nw") {
                        logger->warn("clean.null_model.grouping contains unsupported entry '{}'; ignoring", g_raw);
                        continue;
                    }
                    if (seen.insert(g).second) {
                        cleaner.null_model.grouping.push_back(g);
                    }
                }
            }
            logger->info("clean.null_model enabled: n_surrogates={} quantile={} min_good_frac={} max_modes={} max_samples={} seed={}",
                         cleaner.null_model.n_surrogates, cleaner.null_model.quantile,
                         cleaner.null_model.min_good_frac, cleaner.null_model.max_modes,
                         cleaner.null_model.max_samples, cleaner.null_model.seed);
            if (!cleaner.null_model.grouping.empty()) {
                std::string groups_joined;
                for (std::size_t i = 0; i < cleaner.null_model.grouping.size(); ++i) {
                    if (i > 0) {
                        groups_joined += ",";
                    }
                    groups_joined += cleaner.null_model.grouping[i];
                }
                logger->info("clean.null_model active for grouping(s): {}", groups_joined);
            }
        }
        // mask radius in arcseconds
        get_config_value(config, mask_radius_arcsec, missing_keys, invalid_keys,
                         std::tuple{"timestream","processed_time_chunk","clean","mask_radius_arcsec"});

        // upper weight factor
        get_config_value(config, cleaner.tau, missing_keys, invalid_keys,
                         std::tuple{"timestream","processed_time_chunk","clean","tau"});
    }
}

void PTCProc::subtract_mean(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in,
                            const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> *flags_override) {
    const auto &flags_ref = flags_override ? *flags_override : in.flags.data;
    // cast flags to double and flip 1's and 0's so we can multiply by the data
    auto f = (flags_ref.derived().array().cast <double> ().array() - 1).abs();
    // mean of each detector
    Eigen::RowVectorXd col_mean = (in.scans.data.derived().array()*f).colwise().sum()/
                                   f.colwise().sum();

    // remove nans from completely flagged detectors
    Eigen::RowVectorXd dm = (col_mean).array().isNaN().select(0,col_mean);

    // subtract mean from data and copy into det matrix
    in.scans.data.noalias() = in.scans.data.derived().rowwise() - dm;

    // subtract kernel mean
    if (in.kernel.data.size()!=0) {
        Eigen::RowVectorXd col_mean = (in.kernel.data.derived().array()*f).colwise().sum()/
                                      f.colwise().sum();

        // remove nans from completely flagged detectors
        Eigen::RowVectorXd dm = (col_mean).array().isNaN().select(0,col_mean);

        // subtract mean from data and copy into det matrix
        in.kernel.data.noalias() = in.kernel.data.derived().rowwise() - dm;
    }
}

template <class calib_type>
void PTCProc::run(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, TCData<TCDataKind::PTC, Eigen::MatrixXd> &out,
                  calib_type &calib, std::string pixel_axes, std::string map_grouping) {

    Eigen::Index n_pts = in.scans.data.rows();
    Eigen::Index n_dets = in.scans.data.cols();

    // subtract mean from data and kernel, optionally masking the source region
    if (run_clean && mask_radius_arcsec > 0) {
        auto mean_flags = mask_region(in, calib, pixel_axes, map_grouping, n_pts, n_dets, 0);
        subtract_mean(in, &mean_flags);
    }
    else {
        subtract_mean(in);
    }

    if (run_clean) {
        logger->info("cleaning");
        // Use a local copy so per-pass state does not leak across concurrent run() calls.
        auto cleaner_local = cleaner;
        // number of samples
        Eigen::Index n_pts = in.scans.data.rows();
        // index for number of cleaning groups in vectors
        Eigen::Index indx = 0;
        const bool want_eigs = (run_tod_output || write_evals);
        const bool store_eigs = want_eigs && (cleaner_local.n_calc > 0);
        bool warned_eigs = false;

        // loop through config groupings
        const bool null_model_enabled_global = cleaner_local.null_model.enabled;
        for (const auto & group: cleaner_local.grouping) {
            std::string effective_group = group;
            if (group == "corr_nw" && !cleaner_local.corr_grouping.enabled) {
                logger->warn("cleaning group 'corr_nw' requested but clean.corr_grouping.enabled=false; falling back to nw");
                effective_group = "nw";
            }
            // optional per-group null-model gating
            const bool null_model_for_group =
                null_model_enabled_global && cleaner_local.null_model_enabled_for_group(effective_group);
            if (null_model_enabled_global && !null_model_for_group) {
                logger->debug("null_model disabled for {} grouping", effective_group);
            }

            logger->debug("cleaning with {} grouping", effective_group);

            if (store_eigs) {
                // add current group to eval/evec vectors
                out.evals.data.emplace_back();
                out.evecs.data.emplace_back();
            }
            else if (want_eigs && !warned_eigs) {
                logger->warn("n_calc=0; skipping eval/evec output");
                warned_eigs = true;
            }

            // map of tuples to hold detector limits
            std::map<Eigen::Index, std::tuple<Eigen::Index, Eigen::Index>> grp_limits;

            if (group == "corr_nw" && cleaner_local.corr_grouping.enabled) {
                    Eigen::VectorXi corr_group_ids_scan = Eigen::VectorXi::Constant(in.scans.data.cols(), -1);
                    std::vector<CorrNWDiagSummary> corr_summary_scan;
                    corr_summary_scan.reserve(static_cast<std::size_t>(calib.n_nws));
                    grp_limits = get_grouping("nw", calib, in.scans.data.cols());
                    for (auto const& [key, val] : grp_limits) {
                        const Eigen::Index nw_index = key;
                        const Eigen::Index arr_index = toltec_io.nw_to_array_map[key];
                        auto [start_index, n_dets] = std::make_tuple(std::get<0>(val), std::get<1>(val) - std::get<0>(val));

                        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> masked_flags;
                        if (mask_radius_arcsec > 0) {
                            masked_flags = mask_region(in, calib, pixel_axes, map_grouping, n_pts, n_dets, start_index);
                        }
                        else {
                            masked_flags = in.flags.data.block(0, start_index, n_pts, n_dets);
                        }

                        auto in_scans_block = in.scans.data.block(0, start_index, n_pts, n_dets);
                        auto out_scans_block = out.scans.data.block(0, start_index, n_pts, n_dets);
                        out_scans_block = in_scans_block;

                        auto apt_flags = calib.apt["flag"].segment(start_index, n_dets);

                        if (in.kernel.data.size()!=0) {
                            auto in_kernel_block = in.kernel.data.block(0, start_index, n_pts, n_dets);
                            auto out_kernel_block = out.kernel.data.block(0, start_index, n_pts, n_dets);
                            out_kernel_block = in_kernel_block;
                        }

                        auto corr_groups = cleaner_local.get_corr_groups(in_scans_block, masked_flags, apt_flags);
                        logger->info("cleaning corr_nw {} groups={} grouped={} ungrouped={} candidates={} used={} step={}",
                                     key, corr_groups.n_groups_final, corr_groups.n_det_grouped, corr_groups.n_det_ungrouped,
                                     corr_groups.n_det_candidates, corr_groups.n_det_used, corr_groups.sample_step);
                        corr_summary_scan.push_back(CorrNWDiagSummary{
                            .nw = nw_index,
                            .n_det_input = corr_groups.n_det_input,
                            .n_det_candidates = corr_groups.n_det_candidates,
                            .n_det_used = corr_groups.n_det_used,
                            .n_det_grouped = corr_groups.n_det_grouped,
                            .n_det_ungrouped = corr_groups.n_det_ungrouped,
                            .n_groups_raw = corr_groups.n_groups_raw,
                            .n_groups_final = corr_groups.n_groups_final,
                            .sample_step = corr_groups.sample_step,
                        });

                        auto extract_scans_cols = [&](const auto &m, const std::vector<Eigen::Index> &cols) {
                            Eigen::MatrixXd out_m(m.rows(), static_cast<Eigen::Index>(cols.size()));
                            for (Eigen::Index c = 0; c < static_cast<Eigen::Index>(cols.size()); ++c) {
                                out_m.col(c) = m.col(cols[static_cast<std::size_t>(c)]);
                            }
                            return out_m;
                        };
                        auto extract_flag_cols = [&](const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> &m,
                                                     const std::vector<Eigen::Index> &cols) {
                            Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> out_m(
                                m.rows(), static_cast<Eigen::Index>(cols.size()));
                            for (Eigen::Index c = 0; c < static_cast<Eigen::Index>(cols.size()); ++c) {
                                out_m.col(c) = m.col(cols[static_cast<std::size_t>(c)]);
                            }
                            return out_m;
                        };
                        auto extract_apt_cols = [&](const auto &v, const std::vector<Eigen::Index> &cols) {
                            Eigen::VectorXd out_v(static_cast<Eigen::Index>(cols.size()));
                            for (Eigen::Index c = 0; c < static_cast<Eigen::Index>(cols.size()); ++c) {
                                out_v(c) = v(cols[static_cast<std::size_t>(c)]);
                            }
                            return out_v;
                        };
                        auto scatter_cols = [&](auto &dst, const Eigen::MatrixXd &src, const std::vector<Eigen::Index> &cols) {
                            for (Eigen::Index c = 0; c < static_cast<Eigen::Index>(cols.size()); ++c) {
                                dst.col(cols[static_cast<std::size_t>(c)]) = src.col(c);
                            }
                        };

                        for (Eigen::Index gidx = 0; gidx < static_cast<Eigen::Index>(corr_groups.groups.size()); ++gidx) {
                            const auto &cols = corr_groups.groups[static_cast<std::size_t>(gidx)];
                            if (cols.size() < 2) {
                                continue;
                            }
                            for (const auto &local_col : cols) {
                                corr_group_ids_scan(start_index + local_col) = gidx;
                            }

                            auto in_scans_sub = extract_scans_cols(in_scans_block, cols);
                            auto out_scans_sub = in_scans_sub;
                            auto flags_sub = extract_flag_cols(masked_flags, cols);
                            auto apt_flags_sub = extract_apt_cols(apt_flags, cols);

                            if (!(apt_flags_sub.array() == 0).any()) {
                                continue;
                            }

                            auto [evals, evecs] = cleaner_local.calc_eig_values<timestream::Cleaner::SpectraBackend>(
                                in_scans_sub, flags_sub, apt_flags_sub, cleaner_local.n_eig_to_cut[arr_index](indx));
                            Eigen::Index forced_limit_index = -1;
                            if (null_model_for_group) {
                                forced_limit_index = cleaner_local.get_null_model_index(in_scans_sub, flags_sub, apt_flags_sub);
                            }

                            if (store_eigs) {
                                Eigen::Index n_keep = std::min<Eigen::Index>(cleaner_local.n_calc, evals.size());
                                if (n_keep > 0) {
                                    Eigen::VectorXd ev = evals.head(n_keep);
                                    Eigen::MatrixXd evc = evecs.leftCols(n_keep);
                                    out.evals.data[indx].push_back(std::move(ev));
                                    out.evecs.data[indx].push_back(std::move(evc));
                                }
                            }

                            cleaner_local.remove_eig_values<timestream::Cleaner::SpectraBackend>(
                                in_scans_sub, flags_sub, evals, evecs, out_scans_sub,
                                cleaner_local.n_eig_to_cut[arr_index](indx), forced_limit_index,
                                group, nw_index, arr_index);
                            scatter_cols(out_scans_block, out_scans_sub, cols);

                            if (in.kernel.data.size()!=0) {
                                auto in_kernel_block = in.kernel.data.block(0, start_index, n_pts, n_dets);
                                auto out_kernel_block = out.kernel.data.block(0, start_index, n_pts, n_dets);
                                auto in_kernel_sub = extract_scans_cols(in_kernel_block, cols);
                                auto out_kernel_sub = in_kernel_sub;
                                cleaner_local.remove_eig_values<timestream::Cleaner::SpectraBackend>(
                                    in_kernel_sub, flags_sub, evals, evecs, out_kernel_sub,
                                    cleaner_local.n_eig_to_cut[arr_index](indx), forced_limit_index,
                                    group, nw_index, arr_index);
                                scatter_cols(out_kernel_block, out_kernel_sub, cols);
                            }
                        }
                    }
                    corr_nw_group_ids_by_scan[in.index.data] = std::move(corr_group_ids_scan);
                    corr_nw_summary_by_scan[in.index.data] = std::move(corr_summary_scan);
                    indx++;
                    out.status.cleaned = true;
                    continue;
            }

            // use all detectors for cleaning
            if (effective_group == "all") {
                grp_limits[0] = std::make_tuple(0,in.scans.data.cols());
            }
            else {
                // get group limits
                grp_limits = get_grouping(effective_group, calib, in.scans.data.cols());
            }
            // loop through cleaning groups
            for (auto const& [key, val] : grp_limits) {
                Eigen::Index arr_index;
                Eigen::Index nw_index = -1;
                // use all detectors
                if (effective_group=="all") {
                    arr_index = calib.arrays(0);
                }
                // use network grouping
                else if (effective_group=="nw" || effective_group=="network") {
                    nw_index = key;
                    arr_index = toltec_io.nw_to_array_map[key];
                }
                // use array grouping
                else if (effective_group=="array") {
                    arr_index = key;
                }

                // start index and number of detectors
                auto [start_index, n_dets] = std::make_tuple(std::get<0>(val), std::get<1>(val) - std::get<0>(val));

                // matrix for flags so we don't overwrite the raw flags
                Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> masked_flags;

                // mask region if radius is >0
                if (mask_radius_arcsec > 0) {
                    // samples that were masked will be flagged
                    masked_flags = mask_region(in, calib, pixel_axes, map_grouping, n_pts, n_dets, start_index);
                }
                // otherwise just use input flags
                else {
                    masked_flags = in.flags.data.block(0, start_index, n_pts, n_dets);
                }

                auto in_scans_block = in.scans.data.block(0, start_index, n_pts, n_dets);
                auto out_scans_block = out.scans.data.block(0, start_index, n_pts, n_dets);

                // get the block of out scans that corresponds to the current array
                auto apt_flags = calib.apt["flag"].segment(start_index, n_dets);

                // check if any good flags
                if ((apt_flags.array()==0).any()) {
                    logger->info("cleaning {} {}", effective_group, key);
                    // calculate eigenvalues and eigenvalues
                    auto [evals, evecs] = cleaner_local.calc_eig_values<timestream::Cleaner::SpectraBackend>(
                        in_scans_block, masked_flags, apt_flags, cleaner_local.n_eig_to_cut[arr_index](indx));
                    Eigen::Index forced_limit_index = -1;
                    if (null_model_for_group) {
                        forced_limit_index = cleaner_local.get_null_model_index(in_scans_block, masked_flags, apt_flags);
                    }

                    if (store_eigs) {
                        // get first n_calc eigenvalues and eigenvectors
                        Eigen::Index n_keep = std::min<Eigen::Index>(cleaner_local.n_calc, evals.size());
                        if (n_keep > 0) {
                            Eigen::VectorXd ev = evals.head(n_keep);
                            Eigen::MatrixXd evc = evecs.leftCols(n_keep);

                            // avoid dumping full matrices in debug; can be huge and unstable
                            const Eigen::Index n_show = std::min<Eigen::Index>(n_keep, 8);
                            logger->debug("evals n={} head({})={}", n_keep, n_show, ev.head(n_show).transpose());
                            logger->debug("evecs shape={}x{} (values omitted)", evc.rows(), evc.cols());

                            // copy evals and evecs to ptcdata
                            out.evals.data[indx].push_back(std::move(ev));
                            out.evecs.data[indx].push_back(std::move(evc));
                        }
                    }

                    // remove eigenvalues from the data and reconstruct the tod
                    cleaner_local.remove_eig_values<timestream::Cleaner::SpectraBackend>(
                        in_scans_block, masked_flags, evals, evecs, out_scans_block,
                        cleaner_local.n_eig_to_cut[arr_index](indx), forced_limit_index,
                        effective_group, nw_index, arr_index);

                    if (in.kernel.data.size()!=0) {
                        // check if any good flags
                            logger->debug("cleaning kernel");
                            auto in_kernel_block = in.kernel.data.block(0, start_index, n_pts, n_dets);
                            auto out_kernel_block = in.kernel.data.block(0, start_index, n_pts, n_dets);

                            // remove eigenvalues from the kernel and reconstruct the tod
                            cleaner_local.remove_eig_values<timestream::Cleaner::SpectraBackend>(
                                in_kernel_block, masked_flags, evals, evecs, out_kernel_block,
                                cleaner_local.n_eig_to_cut[arr_index](indx), forced_limit_index,
                                effective_group, nw_index, arr_index);
                    }
                }
                // otherwise just copy the data
                else {
                    logger->debug("no good detectors found. skipping clean.");
                    // copy scans
                    out.scans.data.block(0, start_index, n_pts, n_dets) = in.scans.data.block(0, start_index, n_pts, n_dets);
                    // copy kernel
                    if (in.kernel.data.size()!=0) {
                        out.kernel.data.block(0, start_index, n_pts, n_dets) = in.kernel.data.block(0, start_index, n_pts, n_dets);
                    }
                }
            }
            indx++;
            // set as cleaned
            out.status.cleaned = true;
        }
    }
}

template <typename apt_type, class tel_type>
void PTCProc::calc_weights(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, apt_type &apt, tel_type &telescope) {
    // number of detectors
    Eigen::Index n_dets = in.scans.data.cols();
    const auto scan_index_1based = static_cast<long long>(in.index.data) + 1;

    // resize weights to number of detectors
    in.weights.data = Eigen::VectorXd::Zero(n_dets);

    // approximate weighting
    if (weighting_type == "approximate") {
        logger->debug("calculating weights using detector sensitivities");
        // unit conversion x flux calibration factor x 1/exp(-tau)
        double conversion_factor;

        // loop through detectors and calculate weights
        for (Eigen::Index i=0; i<n_dets; ++i) {
            // current detector index
            Eigen::Index det_index = i;
            if (apt["flag"](det_index)!=0) {
                in.weights.data(i) = 0;
                continue;
            }
            // if flux calibrated, get flux conversion factor
            if (in.status.calibrated) {
                conversion_factor = in.fcf.data(i);
            }
            // otherwise fcf is unity
            else {
                conversion_factor = 1;
            }
            // make sure flux conversion is not zero (otherwise weight=0)
            if (conversion_factor*apt["sens"](det_index)!=0) {
                // calculate weights while applying flux calibration
                in.weights.data(i) = pow(sqrt(telescope.d_fsmp)*apt["sens"](det_index)*conversion_factor,-2.0);
            }
            else {
                in.weights.data(i) = 0;
            }
        }
    }
    // use full weighting
    else if (weighting_type == "full"){
        logger->debug("calculating weights using timestream variance");

        // loop through detectors
        for (Eigen::Index i=0; i<n_dets; ++i) {
            // only calculate weights if detector is unflagged
            if (apt["flag"](i)==0) {
                // make Eigen::Maps for each detector's scan
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> scans(
                    in.scans.data.col(i).data(), in.scans.data.rows());
                Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> flags(
                    in.flags.data.col(i).data(), in.flags.data.rows());

                // unflagged detector stddev
                double det_std_dev = engine_utils::calc_std_dev(scans, flags);
                // if stddev is not zero
                if (det_std_dev !=0) {
                    // weight = 1/(stddev)^2
                    in.weights.data(i) = pow(det_std_dev,-2);
                }
                // otherwise weight = 0 (not included in maps)
                else {
                    in.weights.data(i) = 0;
                }
            }
            // otherwise weight = 0 (not included in maps)
            else {
                in.weights.data(i) = 0;
            }
        }
    }
    // constant weighting
    else if (weighting_type == "const") {
        for (Eigen::Index i=0; i<n_dets; ++i) {
            // only calculate weights if detector is unflagged
            if (apt["flag"](i)==0) {
                in.weights.data(i) = 1;
            }
            // otherwise set to zero
            else {
                in.weights.data(i) = 0;
            }
        }
    }

    Eigen::Index n_apt_unflagged = 0;
    Eigen::Index n_nonfinite = 0;
    Eigen::Index n_positive = 0;
    Eigen::Index n_zero = 0;
    Eigen::Index n_negative = 0;
    for (Eigen::Index i = 0; i < n_dets; ++i) {
        if (apt["flag"](i) == 0) {
            n_apt_unflagged++;
        }
        const auto w = in.weights.data(i);
        if (!std::isfinite(w)) {
            n_nonfinite++;
        } else if (w > 0) {
            n_positive++;
        } else if (w == 0) {
            n_zero++;
        } else {
            n_negative++;
        }
    }
    logger->info(
        "weight calc summary scan={} type={} n_dets={} apt_unflagged={} "
        "positive={} zero={} negative={} nonfinite={}",
        scan_index_1based, weighting_type, n_dets, n_apt_unflagged, n_positive,
        n_zero, n_negative, n_nonfinite);
}

template <typename calib_t>
auto PTCProc::reset_weights(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, calib_t &calib, std::string map_grouping) {

    // make a copy of the calib class for flagging
    calib_t calib_scan = calib;

    const auto scan_index_1based = static_cast<long long>(in.index.data) + 1;
    static std::atomic<long long> reset_weights_call_counter{0};
    const auto reset_call_id = ++reset_weights_call_counter;

    // only need to run if median weight factor >=1
    if (med_weight_factor >= 1 || lower_weight_factor > 0 || upper_weight_factor > 0) {
        // number of detectors
        Eigen::Index n_dets = in.scans.data.cols();

        // get group limits
        auto grp_limits = get_grouping("array", calib, n_dets);

        logger->info(
            "resetting weights call={} scan={} map_grouping={} n_dets={} "
            "med_weight_factor={} lower_weight_factor={} upper_weight_factor={}",
            reset_call_id, scan_index_1based, map_grouping, n_dets,
            med_weight_factor, lower_weight_factor, upper_weight_factor);

        // collect detectors that are un-flagged and have non-zero weights
        for (auto const& [key, val] : grp_limits) {
            // weights for current group
            auto grp_weights = in.weights.data(Eigen::seq(std::get<0>(grp_limits[key]),
                                                         std::get<1>(grp_limits[key])-1));
            const auto group_start = std::get<0>(grp_limits[key]);
            const auto group_end = std::get<1>(grp_limits[key]);
            const auto n_group_dets = group_end - group_start;
            // number of unflagged detectors, and unflagged with positive weights
            Eigen::Index n_unflagged = 0;
            Eigen::Index n_good_dets = 0;
            Eigen::Index n_nonfinite_weights = 0;
            Eigen::Index n_nonpositive_unflagged = 0;
            // start index of current group
            Eigen::Index j = group_start;

            // loop through detectors in current group
            for (Eigen::Index m=0; m<grp_weights.size(); ++m) {
                if (!std::isfinite(grp_weights(m))) {
                    n_nonfinite_weights++;
                }
                // count unflagged detectors
                if (calib.apt["flag"](j)==0) {
                    n_unflagged++;
                    if (grp_weights(m) > 0) {
                        n_good_dets++;
                    } else {
                        n_nonpositive_unflagged++;
                    }
                }
                j++;
            }

            // to hold good detectors
            Eigen::VectorXd good_wt;

            // if good detectors were found
            if (n_good_dets>0) {
                good_wt.resize(n_good_dets);

                // remove flagged dets
                j = std::get<0>(grp_limits[key]);
                Eigen::Index k = 0;
                for (Eigen::Index m=0; m<grp_weights.size(); ++m) {
                    if (calib.apt["flag"](j)==0 && grp_weights(m)>0) {
                        good_wt(k) = grp_weights(m);
                        k++;
                    }
                    j++;
                }
            }
            // otherwise just use all detectors
            else {
                good_wt = grp_weights;
            }

            // get median weight
            auto med_wt = tula::alg::median(good_wt);
            const auto lower_limit =
                lower_weight_factor != 0 ? lower_weight_factor * med_wt : 0.0;
            const auto upper_limit =
                upper_weight_factor != 0 ? upper_weight_factor * med_wt : 0.0;
            // store median weights
            in.median_weights.data.push_back(med_wt);

            int outliers = 0;
            int n_dets_low = 0;
            int n_dets_high = 0;

            // start index of current group
            j = group_start;
            // loop through detectors in current group
            for (Eigen::Index m=0; m<grp_weights.size(); ++m) {
                // if detector weight is med_weight_factor times larger than med_wt
                if (med_weight_factor >=1 && in.weights.data(j) > med_weight_factor*med_wt) {
                    // reset high weights to median
                    in.weights.data(j) = med_wt;
                    outliers++;
                }

                // only run if unflagged already
                if (calib.apt["flag"](j)==0) {
                    // flag those below limit
                    if ((in.weights.data(j) < (lower_weight_factor*med_wt)) && lower_weight_factor!=0) {
                        if (map_grouping!="detector") {
                            in.flags.data.col(j).setOnes();
                        }
                        else {
                            calib_scan.apt["flag"](j) = 1;
                        }
                        in.n_dets_low++;
                        n_dets_low++;
                    }

                    // flag those above limit
                    if ((in.weights.data(j) > (upper_weight_factor*med_wt)) && upper_weight_factor!=0) {
                        if (map_grouping!="detector") {
                            in.flags.data.col(j).setOnes();
                        }
                        else {
                            calib_scan.apt["flag"](j) = 1;
                        }
                        in.n_dets_high++;
                        n_dets_high++;
                    }
                }
                j++;
            }
            logger->info(
                "weight audit call={} scan={} array={} idx_range=[{}, {}) "
                "group_dets={} apt_unflagged={} apt_flagged={} "
                "positive_unflagged={} nonpositive_unflagged={} nonfinite_weights={} "
                "median_weight={} lower_limit={} upper_limit={}",
                reset_call_id, scan_index_1based, key, group_start, group_end,
                n_group_dets, n_unflagged, n_group_dets - n_unflagged, n_good_dets,
                n_nonpositive_unflagged, n_nonfinite_weights, med_wt, lower_limit,
                upper_limit);
            logger->info(
                "weight flags call={} scan={} array={} outlier_resets={} "
                "below_limit={}/{} above_limit={}/{}",
                reset_call_id, scan_index_1based, key, outliers, n_dets_low,
                n_unflagged, n_dets_high, n_unflagged);

            // sanity checks for impossible counter combinations
            if (n_unflagged < 0 || n_unflagged > n_group_dets ||
                n_good_dets < 0 || n_good_dets > n_unflagged ||
                n_dets_low < 0 || n_dets_low > n_unflagged ||
                n_dets_high < 0 || n_dets_high > n_unflagged) {
                logger->error(
                    "weight counter invariant failure call={} scan={} array={} "
                    "group_dets={} apt_unflagged={} positive_unflagged={} "
                    "below_count={} above_count={} outlier_count={}",
                    reset_call_id, scan_index_1based, key, n_group_dets,
                    n_unflagged, n_good_dets, n_dets_low, n_dets_high, outliers);
                const auto n_dump = std::min<Eigen::Index>(grp_weights.size(), 10);
                for (Eigen::Index m = 0; m < n_dump; ++m) {
                    const auto det_index = group_start + m;
                    logger->error(
                        "weight counter dump call={} scan={} array={} m={} det_index={} apt_flag={} weight={}",
                        reset_call_id, scan_index_1based, key, m, det_index,
                        calib.apt["flag"](det_index), in.weights.data(det_index));
                }
                std::exit(EXIT_FAILURE);
            }
        }

        // set up scan calib
        calib_scan.setup();
    }
    return std::move(calib_scan);
}

template <typename calib_t, typename pointing_offset_t>
void PTCProc::append_to_netcdf(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, std::string filepath, std::string map_grouping,
                              std::string &pixel_axes, pointing_offset_t &pointing_offsets_arcsec, calib_t &calib,
                              bool apply_det_offsets, Eigen::Index scan_row_index) {

    using netCDF::NcDim;
    using netCDF::NcFile;
    using netCDF::NcType;
    using netCDF::NcVar;
    using namespace netCDF::exceptions;

    try {
        // open netcdf file
        NcFile fo(filepath, netCDF::NcFile::write);

        // append common time chunk variables
        append_base_to_netcdf(fo, in, map_grouping, pixel_axes, pointing_offsets_arcsec, calib, apply_det_offsets,
                              scan_row_index);

        // get dimensions
        NcDim n_dets_dim = fo.getDim("n_dets");

        // number of detectors currently in file
        unsigned long n_dets_exists = n_dets_dim.getSize();

        // append weights
        const auto scan_row = static_cast<unsigned long>((scan_row_index >= 0) ? scan_row_index : in.index.data);
        std::vector<std::size_t> start_index_weights = {scan_row, 0};
        std::vector<std::size_t> size_weights = {1, n_dets_exists};

        // get weight variable
        NcVar weights_v = fo.getVar("weights");

        // add weights to tod output
        weights_v.putVar(start_index_weights, size_weights, in.weights.data.data());

        const auto corr_groups_it = corr_nw_group_ids_by_scan.find(in.index.data);
        const auto corr_summary_it = corr_nw_summary_by_scan.find(in.index.data);
        const int corr_fill_value = -2147483647;

        // optional corr_nw diagnostics: detector group IDs per scan x detector
        NcVar corr_group_id_v = fo.getVar("corr_nw_group_id");
        if (!corr_group_id_v.isNull()) {
            std::vector<int> group_ids(static_cast<std::size_t>(n_dets_exists), corr_fill_value);
            if (corr_groups_it != corr_nw_group_ids_by_scan.end()) {
                const auto &gid = corr_groups_it->second;
                const auto n_copy = std::min<unsigned long>(n_dets_exists, static_cast<unsigned long>(gid.size()));
                for (unsigned long i = 0; i < n_copy; ++i) {
                    group_ids[static_cast<std::size_t>(i)] = static_cast<int>(gid(static_cast<Eigen::Index>(i)));
                }
            }
            corr_group_id_v.putVar(start_index_weights, size_weights, group_ids.data());
        }

        // optional corr_nw diagnostics: per-network summaries per scan
        NcVar corr_n_groups_v = fo.getVar("corr_nw_n_groups");
        if (!corr_n_groups_v.isNull()) {
            NcDim n_nws_dim = fo.getDim("n_nws_corr");
            if (!n_nws_dim.isNull()) {
                const auto n_nws = n_nws_dim.getSize();
                std::vector<int> v_n_groups(n_nws, corr_fill_value);
                std::vector<int> v_n_groups_raw(n_nws, corr_fill_value);
                std::vector<int> v_n_det_input(n_nws, corr_fill_value);
                std::vector<int> v_n_det_candidates(n_nws, corr_fill_value);
                std::vector<int> v_n_det_used(n_nws, corr_fill_value);
                std::vector<int> v_n_det_grouped(n_nws, corr_fill_value);
                std::vector<int> v_n_det_ungrouped(n_nws, corr_fill_value);
                std::vector<int> v_sample_step(n_nws, corr_fill_value);

                std::unordered_map<Eigen::Index, std::size_t> nw_to_index;
                nw_to_index.reserve(static_cast<std::size_t>(calib.nws.size()));
                for (Eigen::Index i = 0; i < calib.nws.size(); ++i) {
                    nw_to_index[calib.nws(i)] = static_cast<std::size_t>(i);
                }

                if (corr_summary_it != corr_nw_summary_by_scan.end()) {
                    for (const auto &row : corr_summary_it->second) {
                        const auto it = nw_to_index.find(row.nw);
                        if (it == nw_to_index.end() || it->second >= n_nws) {
                            continue;
                        }
                        const auto j = it->second;
                        v_n_groups[j] = static_cast<int>(row.n_groups_final);
                        v_n_groups_raw[j] = static_cast<int>(row.n_groups_raw);
                        v_n_det_input[j] = static_cast<int>(row.n_det_input);
                        v_n_det_candidates[j] = static_cast<int>(row.n_det_candidates);
                        v_n_det_used[j] = static_cast<int>(row.n_det_used);
                        v_n_det_grouped[j] = static_cast<int>(row.n_det_grouped);
                        v_n_det_ungrouped[j] = static_cast<int>(row.n_det_ungrouped);
                        v_sample_step[j] = static_cast<int>(row.sample_step);
                    }
                }

                std::vector<std::size_t> start_scan_nw = {scan_row, 0};
                std::vector<std::size_t> size_scan_nw = {1, n_nws};

                corr_n_groups_v.putVar(start_scan_nw, size_scan_nw, v_n_groups.data());
                fo.getVar("corr_nw_n_groups_raw").putVar(start_scan_nw, size_scan_nw, v_n_groups_raw.data());
                fo.getVar("corr_nw_n_det_input").putVar(start_scan_nw, size_scan_nw, v_n_det_input.data());
                fo.getVar("corr_nw_n_det_candidates").putVar(start_scan_nw, size_scan_nw, v_n_det_candidates.data());
                fo.getVar("corr_nw_n_det_used").putVar(start_scan_nw, size_scan_nw, v_n_det_used.data());
                fo.getVar("corr_nw_n_det_grouped").putVar(start_scan_nw, size_scan_nw, v_n_det_grouped.data());
                fo.getVar("corr_nw_n_det_ungrouped").putVar(start_scan_nw, size_scan_nw, v_n_det_ungrouped.data());
                fo.getVar("corr_nw_sample_step").putVar(start_scan_nw, size_scan_nw, v_sample_step.data());
            }
        }

        // drop per-scan diagnostics once persisted to netCDF
        if (corr_groups_it != corr_nw_group_ids_by_scan.end()) {
            corr_nw_group_ids_by_scan.erase(corr_groups_it);
        }
        if (corr_summary_it != corr_nw_summary_by_scan.end()) {
            corr_nw_summary_by_scan.erase(corr_summary_it);
        }

        if (write_evals) {
            if (cleaner.n_calc <= 0 || in.evals.data.empty()) {
                logger->warn("n_calc=0 or evals empty; skipping eval/evec output");
                // sync file to make sure it gets updated
                fo.sync();
                // close file
                fo.close();
                logger->info("tod chunk written to {}", filepath);
                return;
            }
            // get number of eigenvalues to save
            NcDim n_eigs_dim = fo.getDim("n_eigs");
            netCDF::NcDim n_eig_grp_dim = fo.getDim("n_eig_grp");

            // if eigenvalue dimension is null, add it
            if (n_eig_grp_dim.isNull()) {
                n_eig_grp_dim = fo.addDim("n_eig_grp",in.evals.data[0].size());
            }

            // dimensions for eigenvalue data
            std::vector<netCDF::NcDim> eval_dims = {n_eig_grp_dim, n_eigs_dim};

            // loop through cleaner gropuing
            for (Eigen::Index i=0; i<in.evals.data.size(); ++i) {
                NcVar eval_v = fo.addVar("evals_" + cleaner.grouping[i] + "_" + std::to_string(i) +
                                             "_chunk_" + std::to_string(in.index.data), netCDF::ncDouble,eval_dims);
                std::vector<std::size_t> start_eig_index = {0, 0};
                std::vector<std::size_t> size = {1, TULA_SIZET(cleaner.n_calc)};

                // loop through eigenvalues in current group
                for (const auto &evals: in.evals.data[i]) {
                    eval_v.putVar(start_eig_index,size,evals.data());
                    start_eig_index[0] += 1;
                }
            }

            // number of dimensions for eigenvectors
            std::vector<netCDF::NcDim> eig_dims = {n_dets_dim, n_eigs_dim};

            // loop through cleaner gropuing
            for (Eigen::Index i=0; i<in.evecs.data.size(); ++i) {
                // start at first row and col
                std::vector<std::size_t> start_eig_index = {0, 0};

                NcVar evec_v = fo.addVar("evecs_" + cleaner.grouping[i] + "_" + std::to_string(i) + "_chunk_" +
                                             std::to_string(in.index.data),netCDF::ncDouble,eig_dims);

                // loop through eigenvectors in current group
                for (const auto &evecs: in.evecs.data[i]) {
                    std::vector<std::size_t> size = {TULA_SIZET(evecs.rows()), TULA_SIZET(cleaner.n_calc)};

                    // transpose eigenvectors
                    Eigen::MatrixXd ev = evecs.transpose();
                    evec_v.putVar(start_eig_index, size, ev.data());

                    // increment start
                    start_eig_index[0] += TULA_SIZET(evecs.rows());
                }
            }
        }

        // sync file to make sure it gets updated
        fo.sync();
        // close file
        fo.close();
        logger->info("tod chunk written to {}", filepath);

    } catch (NcException &e) {
        logger->error("{}", e.what());
    }
}

} // namespace timestream
