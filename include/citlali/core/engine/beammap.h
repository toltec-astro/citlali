#pragma once

#include <map>
#include <vector>
#include <string>
#include <cstdlib>
#include <limits>
#include <cmath>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <memory>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <condition_variable>
#include <queue>
#include <set>

#include <citlali/core/engine/engine.h>
#include <citlali/core/utils/ecsv_io.h>

using timestream::TCData;
using timestream::RTCProc;
using timestream::PTCProc;

// selects the type of TCData
using timestream::TCDataKind;

class Beammap: public Engine {
public:
    struct SoftPriorSlot {
        double x_arcsec = 0.0;
        double y_arcsec = 0.0;
        double sx_arcsec = 8.0;
        double sy_arcsec = 8.0;
        int slot_index = -1;
    };

    struct PriorArrayAlignment {
        bool valid = false;
        double cos_theta = 1.0;
        double sin_theta = 0.0;
        double theta_rad = 0.0;
        double dx_arcsec = 0.0;
        double dy_arcsec = 0.0;
        Eigen::Index n_matches = 0;
        double rms_arcsec = std::numeric_limits<double>::quiet_NaN();
    };

    bool beammap_soft_priors_loaded = false;
    bool beammap_soft_priors_are_centered = false;
    bool beammap_soft_priors_are_derotated = false;
    std::map<std::pair<int, int>, std::vector<SoftPriorSlot>> beammap_soft_prior_slots;
    std::map<int, double> beammap_prior_array_center_x_arcsec;
    std::map<int, double> beammap_prior_array_center_y_arcsec;
    std::map<int, PriorArrayAlignment> beammap_prior_array_alignment;

    // parallel policies for each section
    std::string  map_parallel_policy;

    // vector to store each scan's PTCData
    std::vector<TCData<TCDataKind::PTC,Eigen::MatrixXd>> ptcs0, ptcs;

    // copy of obs map buffer for map iteration
    mapmaking::MapBuffer omb_copy{"omb"};

    // vector to store each scan's calib class
    std::vector<engine::Calib> calib_scans0, calib_scans;

    // beammap iteration parameters
    Eigen::Index current_iter;

    // vector for convergence check
    Eigen::Matrix<bool, Eigen::Dynamic, 1> converged;

    // vector to record convergence iteration
    Eigen::Vector<int, Eigen::Dynamic> converge_iter;

    // previous iteration fit parameters
    Eigen::MatrixXd p0, perror0;

    // current iteration fit parameters
    Eigen::MatrixXd params, perrors;

    // reference detector
    Eigen::Index beammap_reference_det_found = -99;

    // bitwise flags
    enum AptFlags {
        Good         = 0,
        BadFit       = 1 << 0,
        AzFWHM       = 1 << 1,
        ElFWHM       = 1 << 2,
        Sig2Noise    = 1 << 3,
        Sens         = 1 << 4,
        Position     = 1 << 5,
        PriorDist    = 1 << 6,
        NetworkPos   = 1 << 7,
        };

    // holds bitwise flags
    Eigen::Matrix<uint16_t,Eigen::Dynamic,1> flag2;

    // good fits
    Eigen::Matrix<bool, Eigen::Dynamic, 1> good_fits;

    // per-map fit-bound diagnostics (for final QC outputs)
    Eigen::MatrixXd fit_diag_init_params;
    Eigen::MatrixXd fit_diag_lower_limits;
    Eigen::MatrixXd fit_diag_upper_limits;
    Eigen::MatrixXi fit_diag_hit_lower;
    Eigen::MatrixXi fit_diag_hit_upper;
    Eigen::VectorXi fit_diag_bound_code;
    Eigen::VectorXi fit_diag_bound_nhit;

    enum PriorDiagColumn {
        prior_init_mode_col = 0,
        prior_used_col,
        prior_fallback_blind_col,
        prior_no_candidate_reason_col,
        prior_slot_index_col,
        prior_match_d2_col,
        prior_match_score_col,
        prior_candidate_snr_col,
        prior_n_candidates_col,
        prior_n_candidates_keep_col,
        prior_n_candidates_gate_col,
        prior_candidate_x_raw_col,
        prior_candidate_y_raw_col,
        prior_candidate_x_prior_col,
        prior_candidate_y_prior_col,
        prior_center_x_col,
        prior_center_y_col,
        prior_derot_elev_col,
        prior_slot_x_col,
        prior_slot_y_col,
        prior_slot_sx_col,
        prior_slot_sy_col,
        n_prior_diag_cols
    };

    // per-map prior-init diagnostics (for final QC outputs)
    Eigen::MatrixXd prior_diag_values;

    struct RFIMaskScanSummary {
        Eigen::Index n_det_candidates = 0;
        Eigen::Index n_det_flagged = 0;
        Eigen::Index n_samples_flagged = 0;
        Eigen::Index n_det_rejected = 0;
    };

    struct ScanBandMaskSummary {
        Eigen::Index n_det_flagged = 0;
        Eigen::Index n_rows_flagged = 0;
        Eigen::Index n_samples_flagged = 0;
        Eigen::Index n_det_rejected = 0;
    };

    // diagnostics for sample-level beammap RFI masking
    Eigen::VectorXi rfi_mask_samples_flagged;
    Eigen::VectorXi rfi_mask_scans_flagged;
    Eigen::VectorXi scan_band_mask_samples_flagged;
    Eigen::VectorXi scan_band_mask_rows_flagged;
    Eigen::VectorXi scan_band_mask_edge_code;
    Eigen::VectorXi scan_band_mask_rejected;
    Eigen::VectorXd final_prior_d2_diag;
    Eigen::VectorXi final_prior_slot_index_diag;
    std::shared_ptr<std::mutex> rfi_mask_diag_mutex = std::make_shared<std::mutex>();

    // placeholder vectors for grppi maps
    std::vector<int> scan_in_vec, scan_out_vec;
    std::vector<int> det_in_vec, det_out_vec;

    // initial setup for each obs
    void setup();

    // timestream grppi pipeline
    template <class KidsProc, class RawObs>
    void timestream_pipeline(KidsProc &, RawObs &, bool write_outputs = true);

    // run the raw time chunk processing
    template <class KidsProc>
    auto run_timestream(KidsProc &, bool write_outputs = true);

    // run the loop pipeline
    template <class KidsProc, class RawObs>
    void loop_pipeline(KidsProc &, RawObs &);

    // run the iterative stage
    template <class KidsProc, class RawObs>
    void run_loop(KidsProc &, RawObs &);

    // robust sample-level masking for short RFI bursts in detector beammaps
    RFIMaskScanSummary apply_rfi_sample_mask(TCData<TCDataKind::PTC,Eigen::MatrixXd> &);

    // detector-map edge-band masking for coherent bad scan legs
    ScanBandMaskSummary apply_scan_band_mask(mapmaking::MapBuffer &);

    // optional prior-assisted peak initialization
    bool is_beammap_locator_iter(Eigen::Index iter) const;
    bool is_beammap_measurement_iter(Eigen::Index iter) const;
    bool is_beammap_first_measurement_iter(Eigen::Index iter) const;
    bool has_completed_beammap_measurement_iter(Eigen::Index iter) const;
    std::string beammap_iter_phase_name(Eigen::Index iter) const;
    std::filesystem::path resolve_soft_priors_filepath() const;
    bool load_soft_priors();
    bool find_map_weighted_peak(Eigen::Index map_index, Eigen::Index &best_row,
                                Eigen::Index &best_col, double &best_snr) const;
    double get_prior_derot_elev_rad() const;
    double effective_prior_max_d2() const;
    double effective_prior_score_lambda() const;
    bool observed_to_prior_frame(int array, double x_raw_arcsec, double y_raw_arcsec,
                                 double derot_elev_rad, double &x_prior_arcsec,
                                 double &y_prior_arcsec, double *center_x_arcsec = nullptr,
                                 double *center_y_arcsec = nullptr,
                                 bool apply_empirical_alignment = true) const;
    bool match_prior_slot(int array, int nw, double x_prior_arcsec, double y_prior_arcsec,
                          double &best_d2, int &best_slot, double *slot_x_arcsec = nullptr,
                          double *slot_y_arcsec = nullptr, double *slot_sx_arcsec = nullptr,
                          double *slot_sy_arcsec = nullptr) const;
    void update_prior_frame_estimates();
    bool choose_prior_guided_init(Eigen::Index map_index, double &init_row, double &init_col);
    void configure_detector_source_centers_from_previous_fit();
    double calc_map_support_stddev(Eigen::Index map_index, bool exclude_fit_core = false) const;
    double calc_beammap_convergence_delta(Eigen::Index map_index) const;
    void init_empirical_template_calibration_columns();
    void calc_empirical_template_calibration();

    // flag detectors
    void set_apt_flags();

    // derotate apt and subtract reference detector
    void process_apt();
    void apply_final_network_position_flags();
    void update_final_prior_match_diagnostics();
    void log_final_network_qc_summary();
    void write_detector_specific_ptc_tod(int output_iter);
    void write_detector_table_outputs();

    // main pipeline process
    template <class KidsProc, class RawObs>
    void pipeline(KidsProc &, RawObs &);

    // output files
    template <mapmaking::MapType map_type>
    void output();
    template <mapmaking::MapType map_type>
    void write_beammap_map_products(
        mapmaking::MapBuffer *mb,
        std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *f_io,
        std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *n_io,
        const std::string &dir_name);
};

#include <citlali/core/engine/detail/beammap_detector_tod_output_impl.h>
#include <citlali/core/engine/detail/beammap_detector_table_output_impl.h>
#include <citlali/core/engine/detail/beammap_map_product_output_impl.h>
#include <citlali/core/engine/detail/beammap_empirical_template_calibration_impl.h>
#include <citlali/core/engine/detail/beammap_masking_impl.h>
#include <citlali/core/engine/detail/beammap_pipeline_impl.h>
#include <citlali/core/engine/detail/beammap_prior_impl.h>

double Beammap::calc_map_support_stddev(Eigen::Index map_index, bool exclude_fit_core) const {
    if (map_index < 0 ||
        map_index >= static_cast<Eigen::Index>(omb.signal.size()) ||
        map_index >= static_cast<Eigen::Index>(omb.weight.size())) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const auto &sig = omb.signal[map_index];
    const auto &wt = omb.weight[map_index];
    if (sig.rows() <= 0 || sig.cols() <= 0 ||
        wt.rows() != sig.rows() || wt.cols() != sig.cols()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double core_row = std::numeric_limits<double>::quiet_NaN();
    double core_col = std::numeric_limits<double>::quiet_NaN();
    double core_radius_pix = 0.0;
    if (exclude_fit_core &&
        map_index < params.rows() &&
        params.cols() >= 5 &&
        std::isfinite(params(map_index, 1)) &&
        std::isfinite(params(map_index, 2)) &&
        std::isfinite(params(map_index, 3)) &&
        std::isfinite(params(map_index, 4)) &&
        params(map_index, 3) > 0.0 &&
        params(map_index, 4) > 0.0) {
        core_col = params(map_index, 1);
        core_row = params(map_index, 2);
        core_radius_pix = 2.0 * STD_TO_FWHM *
                          std::max(params(map_index, 3), params(map_index, 4));
    }

    auto collect = [&](bool exclude_core) {
        std::vector<double> values;
        values.reserve(static_cast<std::size_t>(sig.size()));
        const bool do_exclude =
            exclude_core &&
            std::isfinite(core_row) &&
            std::isfinite(core_col) &&
            core_radius_pix > 0.0;
        const double core_radius2 = core_radius_pix * core_radius_pix;
        for (Eigen::Index row = 0; row < sig.rows(); ++row) {
            for (Eigen::Index col = 0; col < sig.cols(); ++col) {
                const double s = sig(row, col);
                const double w = wt(row, col);
                if (!std::isfinite(s) || !std::isfinite(w) || w <= 0.0) {
                    continue;
                }
                if (do_exclude) {
                    const double dr = static_cast<double>(row) - core_row;
                    const double dc = static_cast<double>(col) - core_col;
                    if (dr * dr + dc * dc <= core_radius2) {
                        continue;
                    }
                }
                values.push_back(s);
            }
        }
        return values;
    };

    auto values = collect(exclude_fit_core);
    if (values.size() < static_cast<std::size_t>(std::max(16, map_fitter.n_params + 1))) {
        values = collect(false);
    }
    if (values.size() < 2) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    Eigen::Map<Eigen::VectorXd> vec(values.data(), static_cast<Eigen::Index>(values.size()));
    return engine_utils::calc_std_dev(vec);
}

double Beammap::calc_beammap_convergence_delta(Eigen::Index map_index) const {
    if (map_index < 0 ||
        map_index >= static_cast<Eigen::Index>(omb.signal.size()) ||
        map_index >= static_cast<Eigen::Index>(omb_copy.signal.size())) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (beammap_convergence_radius_arcsec <= 0.0 || omb.pixel_size_rad <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const auto &prev_sig = omb_copy.signal[map_index];
    const auto &cur_sig = omb.signal[map_index];
    if (prev_sig.rows() <= 0 || prev_sig.cols() <= 0 ||
        cur_sig.rows() != prev_sig.rows() || cur_sig.cols() != prev_sig.cols()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const bool have_prev_wt =
        map_index < static_cast<Eigen::Index>(omb_copy.weight.size()) &&
        omb_copy.weight[map_index].rows() == prev_sig.rows() &&
        omb_copy.weight[map_index].cols() == prev_sig.cols();
    const bool have_cur_wt =
        map_index < static_cast<Eigen::Index>(omb.weight.size()) &&
        omb.weight[map_index].rows() == cur_sig.rows() &&
        omb.weight[map_index].cols() == cur_sig.cols();

    auto choose_center = [&](const Eigen::MatrixXd &fit_params,
                             double &center_row, double &center_col) {
        if (map_index >= fit_params.rows() || fit_params.cols() < 3) {
            return false;
        }
        const double col = fit_params(map_index, 1);
        const double row = fit_params(map_index, 2);
        if (!std::isfinite(row) || !std::isfinite(col)) {
            return false;
        }
        center_row = row;
        center_col = col;
        return true;
    };

    double center_row = std::numeric_limits<double>::quiet_NaN();
    double center_col = std::numeric_limits<double>::quiet_NaN();
    if (!choose_center(params, center_row, center_col) &&
        !choose_center(p0, center_row, center_col)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double radius_pix = beammap_convergence_radius_arcsec * ASEC_TO_RAD / omb.pixel_size_rad;
    const double radius2 = radius_pix * radius_pix;
    const Eigen::Index row_min = std::max<Eigen::Index>(
        0, static_cast<Eigen::Index>(std::floor(center_row - radius_pix)));
    const Eigen::Index row_max = std::min<Eigen::Index>(
        prev_sig.rows() - 1, static_cast<Eigen::Index>(std::ceil(center_row + radius_pix)));
    const Eigen::Index col_min = std::max<Eigen::Index>(
        0, static_cast<Eigen::Index>(std::floor(center_col - radius_pix)));
    const Eigen::Index col_max = std::min<Eigen::Index>(
        prev_sig.cols() - 1, static_cast<Eigen::Index>(std::ceil(center_col + radius_pix)));

    if (row_min > row_max || col_min > col_max) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double diff2_sum = 0.0;
    double prev2_sum = 0.0;
    Eigen::Index n_pix = 0;
    for (Eigen::Index row = row_min; row <= row_max; ++row) {
        for (Eigen::Index col = col_min; col <= col_max; ++col) {
            const double dr = static_cast<double>(row) - center_row;
            const double dc = static_cast<double>(col) - center_col;
            if (dr * dr + dc * dc > radius2) {
                continue;
            }
            const double prev = prev_sig(row, col);
            const double cur = cur_sig(row, col);
            if (!std::isfinite(prev) || !std::isfinite(cur)) {
                continue;
            }
            if (have_prev_wt) {
                const double wt = omb_copy.weight[map_index](row, col);
                if (!std::isfinite(wt) || wt <= 0.0) {
                    continue;
                }
            }
            if (have_cur_wt) {
                const double wt = omb.weight[map_index](row, col);
                if (!std::isfinite(wt) || wt <= 0.0) {
                    continue;
                }
            }
            const double diff = cur - prev;
            diff2_sum += diff * diff;
            prev2_sum += prev * prev;
            ++n_pix;
        }
    }

    if (n_pix < std::max<Eigen::Index>(8, map_fitter.n_params + 1)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double eps = std::numeric_limits<double>::epsilon();
    if (prev2_sum <= eps) {
        if (diff2_sum <= eps) {
            return 0.0;
        }
        return std::numeric_limits<double>::infinity();
    }
    return std::sqrt(diff2_sum / prev2_sum);
}

template <class KidsProc, class RawObs>
void Beammap::run_loop(KidsProc &kidsproc, RawObs &rawobs) {
    // variable to control iteration
    bool keep_going = true;

    // declare random number generator
    boost::random::mt19937 eng;

    // boost random number generator (0,1)
    boost::random::uniform_int_distribution<> rands{0,1};

    if (beammap_rfi_mask_enabled && map_grouping == "detector") {
        logger->info("beammap rfi mask enabled: block_size={} min_good={} sigma_threshold={:.4g} sigma_floor={:.4g} dilate_blocks={} max_flagged_fraction={:.4f}",
                     beammap_rfi_mask_block_size_samples,
                     beammap_rfi_mask_min_good_samples,
                     beammap_rfi_mask_sigma_threshold,
                     beammap_rfi_mask_sigma_floor,
                     beammap_rfi_mask_dilate_blocks,
                     beammap_rfi_mask_max_flagged_fraction);
    }
    if (beammap_scan_band_mask_enabled && map_grouping == "detector") {
        logger->info(
            "beammap scan-band mask enabled: edge_rows={} min_row_pixels={} min_contiguous_rows={} row_median_sigma_threshold={:.4g} row_sigma_ratio_threshold={:.4g} max_flagged_fraction={:.4f}",
            beammap_scan_band_mask_edge_rows,
            beammap_scan_band_mask_min_row_pixels,
            beammap_scan_band_mask_min_contiguous_rows,
            beammap_scan_band_mask_row_median_sigma_threshold,
            beammap_scan_band_mask_row_sigma_ratio_threshold,
            beammap_scan_band_mask_max_flagged_fraction);
    }

    // iterative loop
    while (keep_going) {
        const bool locator_iter = is_beammap_locator_iter(current_iter);
        const bool measurement_iter = is_beammap_measurement_iter(current_iter);
        const bool first_measurement_iter = is_beammap_first_measurement_iter(current_iter);
        logger->info(
            "starting iter {} phase={} locator_iter={} measurement_start_iter={}",
            current_iter, beammap_iter_phase_name(current_iter),
            beammap_locator_iter, beammap_measurement_start_iter);

        configure_detector_source_centers_from_previous_fit();
        const bool detector_kernel_source_centers_active =
            map_grouping == "detector" &&
            rtcproc.run_kernel &&
            rtcproc.kernel.has_source_centers();
        const bool rerun_source_aware_rtc =
            first_measurement_iter && detector_kernel_source_centers_active;
        if (rerun_source_aware_rtc) {
            logger->info(
                "beammap iter {} rerunning RTC with previous-fit detector source centers; regular RTC TOD output disabled for this internal pass",
                current_iter);
            timestream_pipeline(kidsproc, rawobs, false);
        }

        // copy ptcs
        ptcs = ptcs0;
        // copy calibs
        calib_scans = calib_scans0;
        if (beammap_rfi_mask_enabled && map_grouping == "detector" &&
            rfi_mask_samples_flagged.size() == calib.n_dets &&
            rfi_mask_scans_flagged.size() == calib.n_dets) {
            rfi_mask_samples_flagged.setZero();
            rfi_mask_scans_flagged.setZero();
        }
        const bool skip_centered_kernel_map_feedback =
            rerun_source_aware_rtc;
        ptcproc.fruit_loops_kernel_feedback_enabled = !skip_centered_kernel_map_feedback;
        if (skip_centered_kernel_map_feedback) {
            logger->info(
                "beammap detector kernel map feedback disabled on iter {} while building the first source-aware kernel map",
                current_iter);
        }

        // copy previous-iteration maps for source-aperture convergence tests
        if (run_mapmaking && beammap_iter_tolerance > 0.0 && measurement_iter) {
            omb_copy.signal = omb.signal;
            omb_copy.weight = omb.weight;
        }

        if (ptcproc.run_fruit_loops) {
            // calc mean rms
            if (first_measurement_iter) {
                // use obs map buffer
                if (!omb.noise.empty()) {
                    omb.calc_median_rms();
                }
            }
            if (measurement_iter) {
                ptcproc.configure_fruit_loops_adaptive_gate(omb, calib, map_grouping, false);
            }
        }

        // progress bar
        tula::logging::progressbar pb(
            [&](const auto &msg) { logger->info("{}", msg); }, 100, "PTC progress ");


        auto ptc_line_audit_mutex = std::make_shared<std::mutex>();

        // cleaning (separate from mapmaking loop due to jinc mapmaking parallelization)
        grppi::map(tula::grppi_utils::dyn_ex(omb.parallel_policy), scan_in_vec, scan_out_vec, [&](auto i) {
            bool model_subtracted_for_ptc_line_audit = false;
            if (run_mapmaking) {
                if (measurement_iter) {
                    if (!ptcproc.run_fruit_loops) {
                        // if not running fruit loops use source fit
                        logger->info("subtracting gaussian from tod");
                        // subtract gaussian
                        ptcproc.add_gaussian<timestream::TCProc::SourceType::NegativeGaussian>(ptcs[i], params, telescope.pixel_axes, map_grouping,
                                                                                               calib.apt,omb.pixel_size_rad, omb.n_rows, omb.n_cols);
                    }
                    else {
                        logger->info("subtracting map from tod");
                        // subtract map
                        ptcproc.map_to_tod<timestream::TCProc::SourceType::NegativeMap>(omb, ptcs[i], calib, ptcs[i].map_indices.data, telescope.pixel_axes,
                                                                                        map_grouping);
                        model_subtracted_for_ptc_line_audit = true;
                    }
                }
            }

            {
                std::lock_guard<std::mutex> lock(*ptc_line_audit_mutex);
                apply_model_protected_ptc_line_audit(
                    ptcs[i], calib_scans[i], model_subtracted_for_ptc_line_audit);
            }

            // clean the maps
            logger->info("processed time chunk processing for scan {}", i + 1);
            ptcproc.run(ptcs[i], ptcs[i], calib_scans[i], telescope.pixel_axes, map_grouping);

            if (run_mapmaking) {
                if (measurement_iter) {
                    // if not running fruit loops use source fit
                    if (!ptcproc.run_fruit_loops) {
                        logger->info("adding gaussian to tod");
                        // add gaussian back
                        ptcproc.add_gaussian<timestream::TCProc::SourceType::Gaussian>(ptcs[i], params, telescope.pixel_axes, map_grouping, calib.apt,
                                                                                       omb.pixel_size_rad,omb.n_rows, omb.n_cols);
                    }
                    else {
                        logger->info("adding map to tod");
                        // add map back
                        ptcproc.map_to_tod<timestream::TCProc::SourceType::Map>(omb, ptcs[i], calib, ptcs[i].map_indices.data, telescope.pixel_axes,
                                                                                map_grouping);
                    }
                }
            }

            // For detector-grouped beammaps, keep the locator pass permissive so
            // bright-source scans are less likely to be rejected before we have
            // any source-location estimate to feed back into later iterations.
            if (map_grouping == "detector" && locator_iter) {
                logger->info("skipping remove_bad_dets on beammap locator iter {} for detector scan {}",
                             current_iter, ptcs[i].index.data + 1);
            }
            else {
                // remove outliers after clean
                calib_scans[i] = ptcproc.remove_bad_dets(ptcs[i], calib_scans[i], map_grouping);
            }

            if (map_grouping == "detector") {
                auto rfi_summary = apply_rfi_sample_mask(ptcs[i]);
                if (beammap_rfi_mask_enabled) {
                    if (rfi_summary.n_samples_flagged > 0 || rfi_summary.n_det_rejected > 0) {
                        logger->info("beammap rfi mask scan {}: masked {} samples across {}/{} detectors ({} rejected by max_flagged_fraction={:.4f})",
                                     ptcs[i].index.data + 1,
                                     rfi_summary.n_samples_flagged,
                                     rfi_summary.n_det_flagged,
                                     rfi_summary.n_det_candidates,
                                     rfi_summary.n_det_rejected,
                                     beammap_rfi_mask_max_flagged_fraction);
                    }
                    else {
                        logger->debug("beammap rfi mask scan {}: no samples masked", ptcs[i].index.data + 1);
                    }
                }
                const bool use_ptc_weights =
                    beammap_detector_weighting_mode == "ptc" ||
                    (beammap_detector_weighting_mode == "ptc_after_iter0" && measurement_iter);
                if (use_ptc_weights) {
                    logger->info("calculating detector-mode PTC weights for scan {} (mode={})",
                                 ptcs[i].index.data + 1, beammap_detector_weighting_mode);
                    ptcproc.calc_weights(ptcs[i], calib_scans[i].apt, telescope);
                    calib_scans[i] = ptcproc.reset_weights(ptcs[i], calib_scans[i], map_grouping);
                }
                else {
                    // Constant weights remain the safest default for bright beammaps.
                    ptcs[i].weights.data.resize(ptcs[i].scans.data.cols());
                    ptcs[i].weights.data.setOnes();
                }
            }
            else {
                // calculate weights
                logger->info("calculating weights for scan {}", ptcs[i].index.data + 1);
                ptcproc.calc_weights(ptcs[i], calib_scans[i].apt, telescope);

                // reset weights to median
                calib_scans[i] = ptcproc.reset_weights(ptcs[i], calib_scans[i], map_grouping);
            }

            // calc stats
            logger->debug("calculating stats");
            diagnostics.calc_stats(ptcs[i]);

            return 0;
        });

        auto clear_beammap_ptc_diagnostics = [&]() {
            for (Eigen::Index i=0; i<telescope.scan_indices.cols(); ++i) {
                ptcproc.clear_cached_diagnostics(ptcs[i].index.data);
            }
        };


        auto write_beammap_ptc_products = [&](int output_iter) {
            if (verbose_mode) {
                logger->debug("writing chunk summaries for beammap PTC iteration {}", output_iter);
                for (Eigen::Index i=0; i<telescope.scan_indices.cols(); ++i) {
                    write_chunk_summary(ptcs[i]);
                }
            }
            if (!ptcdiag_filename.empty()) {
                logger->info("writing ptc diagnostics sidecar chunks for beammap iteration {}", output_iter);
                for (Eigen::Index i=0; i<telescope.scan_indices.cols(); ++i) {
                    ptcproc.append_diag_to_netcdf(ptcs[i], ptcdiag_filename, calib_scans[i], ptcs[i].index.data);
                    if (!(run_tod_output && !tod_filename.empty() &&
                          (tod_output_type == "ptc" || tod_output_type == "both"))) {
                        ptcproc.clear_cached_diagnostics(ptcs[i].index.data);
                    }
                }
            }
            if (run_tod_output && !tod_filename.empty()) {
                if (tod_output_type == "ptc" || tod_output_type == "both") {
                    logger->info("writing processed time chunk for beammap iteration {}", output_iter);
                    auto ptc_filename_it = tod_filename.find("ptc");
                    if (ptc_filename_it != tod_filename.end() && !ptc_filename_it->second.empty()) {
                        try {
                            netCDF::NcFile ptc_tod_file(ptc_filename_it->second, netCDF::NcFile::write);
                            netCDF::NcVar fruit_iter_var = ptc_tod_file.getVar("FRUITLOOPS_ITER");
                            if (!fruit_iter_var.isNull()) {
                                fruit_iter_var.putVar(&output_iter);
                            }
                            else {
                                logger->warn("PTC TOD file {} has no FRUITLOOPS_ITER variable",
                                             ptc_filename_it->second);
                            }
                        } catch (const std::exception &e) {
                            logger->warn("failed to update PTC TOD FRUITLOOPS_ITER in {}: {}",
                                         ptc_filename_it->second, e.what());
                        }
                    }
                    for (Eigen::Index i=0; i<telescope.scan_indices.cols(); ++i) {
                        const auto ptc_scan_row = tod_output_scan_row(i, "ptc");
                        if (ptc_scan_row < 0) {
                            continue;
                        }
                        ptcproc.append_to_netcdf(ptcs[i], tod_filename["ptc"], map_grouping, telescope.pixel_axes,
                                                 ptcs[i].pointing_offsets_arcsec.data, calib_scans[i], true, ptc_scan_row);
                        ptcproc.clear_cached_diagnostics(ptcs[i].index.data);
                    }
                }
            }
            write_detector_specific_ptc_tod(output_iter);
        };

        logger->info("starting mapmaking");

        if (run_mapmaking) {
            auto run_mapmaking_pass = [&](bool update_progress) {
                Eigen::Matrix<bool, Eigen::Dynamic, 1> active_maps;
                const Eigen::Matrix<bool, Eigen::Dynamic, 1> *active_maps_ptr = nullptr;
                Eigen::Index n_active_maps = n_maps;
                if (map_grouping == "detector" && converged.size() == n_maps) {
                    const Eigen::Index n_converged = (converged.array() == true).count();
                    if (n_converged > 0 && n_converged < n_maps) {
                        active_maps.resize(n_maps);
                        n_active_maps = 0;
                        for (Eigen::Index i = 0; i < n_maps; ++i) {
                            active_maps(i) = !converged(i);
                            if (active_maps(i)) {
                                ++n_active_maps;
                            }
                        }
                        active_maps_ptr = &active_maps;
                        logger->info("beammap detector mapmaking: remaking {}/{} unconverged maps",
                                     n_active_maps, n_maps);
                    }
                }

                if (map_method == "jinc" &&
                    static_cast<Eigen::Index>(omb.grid_weight.size()) != n_maps) {
                    logger->info("allocating jinc grid_weight maps: current={} expected={}",
                                 omb.grid_weight.size(), n_maps);
                    omb.grid_weight.assign(
                        static_cast<size_t>(n_maps),
                        Eigen::MatrixXd::Zero(omb.n_rows, omb.n_cols));
                }

                // set maps to zero for each pass
                omb.clear_contribution_diag();
                for (Eigen::Index i = 0; i < n_maps; ++i) {
                    if (active_maps_ptr != nullptr && !(*active_maps_ptr)(i)) {
                        continue;
                    }
                    omb.signal[i].setZero();
                    omb.weight[i].setZero();
                    if (!omb.grid_weight.empty()) {
                        omb.grid_weight[i].setZero();
                    }

                    if (!omb.coverage.empty()) {
                        omb.coverage[i].setZero();
                    }
                    if (rtcproc.run_kernel) {
                        omb.kernel[i].setZero();
                    }
                    if (!omb.noise.empty()) {
                        omb.noise[i].setZero();
                    }

                    if (run_noise) {
                        for (auto &ptcdata : ptcs) {
                            if (omb.randomize_dets) {
                                ptcdata.noise.data = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>::Zero(
                                                         omb.n_noise, calib.n_dets)
                                                         .unaryExpr([&](int dummy) { return 2 * rands(eng) - 1; });
                            }
                            else {
                                ptcdata.noise.data = Eigen::Matrix<int, Eigen::Dynamic, 1>::Zero(omb.n_noise)
                                                         .unaryExpr([&](int dummy) { return 2 * rands(eng) - 1; });
                            }
                        }
                    }
                }

                logger->info("running mapmaking");

                if (map_grouping == "detector") {
                    bool run_omb = true;
                    for (std::size_t scan_vec_idx = 0; scan_vec_idx < ptcs.size(); ++scan_vec_idx) {
                        auto &ptc = ptcs[scan_vec_idx];
                        auto &scan_apt = calib_scans[scan_vec_idx].apt;
                        if (map_method == "naive") {
                            naive_mm.populate_maps_naive_parallel(ptc, omb, cmb, ptc.map_indices.data,
                                                                  telescope.pixel_axes, scan_apt,
                                                                  telescope.d_fsmp, run_omb, run_noise,
                                                                  active_maps_ptr);
                        }
                        else if (map_method == "jinc") {
                            std::array<Eigen::Index, 3> array_counts = {0, 0, 0};
                            for (Eigen::Index det = 0; det < ptc.scans.data.cols(); ++det) {
                                auto array_index = static_cast<int>(calib.apt["array"](det));
                                if (array_index >= 0 && array_index < static_cast<int>(array_counts.size())) {
                                    array_counts[static_cast<size_t>(array_index)]++;
                                }
                            }
                            Eigen::Index map_min = -1;
                            Eigen::Index map_max = -1;
                            if (ptc.map_indices.data.size() > 0) {
                                map_min = ptc.map_indices.data.minCoeff();
                                map_max = ptc.map_indices.data.maxCoeff();
                            }
                            std::ostringstream kernel_dims;
                            for (int array_index = 0; array_index < 3; ++array_index) {
                                auto it = jinc_mm.jinc_weights_mat.find(array_index);
                                if (it == jinc_mm.jinc_weights_mat.end()) {
                                    continue;
                                }
                                if (kernel_dims.tellp() > 0) {
                                    kernel_dims << ", ";
                                }
                                kernel_dims << "a" << array_index << "="
                                            << it->second.rows() << "x" << it->second.cols();
                            }
                            logger->info(
                                "beammap jinc preflight: n_dets={} n_pts={} n_maps={} map_index_range=[{}, {}] "
                                "subpixel_n={} kernel_dims=[{}] array_counts=[{},{},{}]",
                                ptc.scans.data.cols(),
                                ptc.scans.data.rows(),
                                omb.signal.size(),
                                map_min,
                                map_max,
                                jinc_mm.subpixel_n,
                                kernel_dims.str(),
                                array_counts[0],
                                array_counts[1],
                                array_counts[2]);
                            jinc_mm.populate_maps_jinc_parallel(ptc, omb, cmb, ptc.map_indices.data,
                                                                telescope.pixel_axes, scan_apt,
                                                                telescope.d_fsmp, run_omb, run_noise,
                                                                active_maps_ptr);
                        }
                        if (update_progress) {
                            pb.count(telescope.scan_indices.cols(), 1);
                        }
                    }
                }
                else {
                    grppi::map(tula::grppi_utils::dyn_ex(map_parallel_policy), scan_in_vec, scan_out_vec, [&](auto i) {
                        bool run_omb = true;
                        if (map_method == "naive") {
                            naive_mm.populate_maps_naive(ptcs[i], omb, cmb, ptcs[i].map_indices.data,
                                                        telescope.pixel_axes, calib_scans[i].apt, telescope.d_fsmp,
                                                        run_omb, run_noise);
                        }
                        else if (map_method == "jinc") {
                            jinc_mm.populate_maps_jinc(ptcs[i], omb, cmb, ptcs[i].map_indices.data,
                                                       telescope.pixel_axes, calib_scans[i].apt, telescope.d_fsmp,
                                                       run_omb, run_noise);
                        }
                        if (update_progress) {
                            pb.count(telescope.scan_indices.cols(), 1);
                        }
                        return 0;
                    });
                }

                logger->info("normalizing maps");
                if (rtcproc.run_kernel && !omb.grid_weight.empty()) {
                    timestream::log_kernel_map_diag(
                        logger,
                        "beammap iter " + std::to_string(current_iter) + " before normalize",
                        omb.kernel,
                        active_maps_ptr,
                        &omb.grid_weight);
                }
                omb.normalize_maps(active_maps_ptr);
                if (rtcproc.run_kernel) {
                    timestream::log_kernel_map_diag(
                        logger,
                        "beammap iter " + std::to_string(current_iter) + " after normalize",
                        omb.kernel,
                        active_maps_ptr);
                }
                if (!omb.normalize_support_diag.empty()) {
                    Eigen::Index n_diag_maps = 0;
                    Eigen::Index total_masked = 0;
                    Eigen::Index total_no_accum_weight = 0;
                    Eigen::Index total_bad_grid_weight = 0;
                    Eigen::Index total_support_threshold = 0;
                    Eigen::Index total_raw_signal_nonzero = 0;
                    Eigen::Index total_adjacent_support = 0;
                    std::vector<Eigen::Index> suspicious_maps;

                    for (Eigen::Index map_index = 0;
                         map_index < static_cast<Eigen::Index>(omb.normalize_support_diag.size());
                         ++map_index) {
                        const auto &diag = omb.normalize_support_diag[map_index];
                        if (diag.map_index < 0) {
                            continue;
                        }
                        n_diag_maps++;
                        total_masked += diag.n_masked;
                        total_no_accum_weight += diag.n_masked_no_accum_weight;
                        total_bad_grid_weight += diag.n_masked_bad_grid_weight_with_accum_weight;
                        total_support_threshold += diag.n_masked_by_support_threshold;
                        total_raw_signal_nonzero += diag.n_masked_raw_signal_nonzero;
                        total_adjacent_support += diag.n_masked_adjacent_support;
                        if (diag.n_masked_bad_grid_weight_with_accum_weight > 0 ||
                            diag.n_masked_by_support_threshold > 0 ||
                            diag.n_masked_adjacent_support > 0 ||
                            diag.n_masked_raw_signal_nonzero > 0) {
                            suspicious_maps.push_back(map_index);
                        }
                    }

                    logger->info(
                        "beammap normalize support summary iter={} maps={} masked={} no_accum_weight={} bad_grid_weight_with_accum_weight={} support_threshold={} raw_signal_nonzero={} adjacent_support_holes={}",
                        current_iter,
                        n_diag_maps,
                        total_masked,
                        total_no_accum_weight,
                        total_bad_grid_weight,
                        total_support_threshold,
                        total_raw_signal_nonzero,
                        total_adjacent_support);

                    auto support_diag_score = [&](Eigen::Index map_index) {
                        const auto &diag = omb.normalize_support_diag[map_index];
                        return diag.n_masked_adjacent_support +
                               diag.n_masked_bad_grid_weight_with_accum_weight +
                               diag.n_masked_by_support_threshold +
                               diag.n_masked_raw_signal_nonzero;
                    };
                    std::sort(suspicious_maps.begin(), suspicious_maps.end(),
                              [&](Eigen::Index lhs, Eigen::Index rhs) {
                                  const auto lhs_score = support_diag_score(lhs);
                                  const auto rhs_score = support_diag_score(rhs);
                                  if (lhs_score != rhs_score) {
                                      return lhs_score > rhs_score;
                                  }
                                  const double lhs_neighbor =
                                      omb.normalize_support_diag[lhs].max_masked_neighbor_weight;
                                  const double rhs_neighbor =
                                      omb.normalize_support_diag[rhs].max_masked_neighbor_weight;
                                  return std::isfinite(lhs_neighbor) && std::isfinite(rhs_neighbor)
                                             ? lhs_neighbor > rhs_neighbor
                                             : std::isfinite(lhs_neighbor);
                              });

                    auto cause_name = [](int cause) {
                        switch (cause) {
                        case 1:
                            return "no_accum_weight";
                        case 2:
                            return "bad_grid_weight";
                        case 3:
                            return "support_threshold";
                        default:
                            return "unknown";
                        }
                    };

                    const Eigen::Index n_log =
                        std::min<Eigen::Index>(10, static_cast<Eigen::Index>(suspicious_maps.size()));
                    for (Eigen::Index rank = 0; rank < n_log; ++rank) {
                        const Eigen::Index map_index = suspicious_maps[rank];
                        const auto &diag = omb.normalize_support_diag[map_index];
                        const int uid = (map_index < calib.apt["uid"].size())
                                            ? static_cast<int>(std::lround(calib.apt["uid"](map_index)))
                                            : -1;
                        const int array = (map_index < calib.apt["array"].size())
                                              ? static_cast<int>(std::lround(calib.apt["array"](map_index)))
                                              : -1;
                        const int nw = (map_index < calib.apt["nw"].size())
                                           ? static_cast<int>(std::lround(calib.apt["nw"](map_index)))
                                           : -1;
                        const double x_t = (map_index < calib.apt["x_t"].size())
                                               ? calib.apt["x_t"](map_index)
                                               : std::numeric_limits<double>::quiet_NaN();
                        const double y_t = (map_index < calib.apt["y_t"].size())
                                               ? calib.apt["y_t"](map_index)
                                               : std::numeric_limits<double>::quiet_NaN();
                        logger->info(
                            "beammap normalize support detail iter={} rank={} map={} uid={} array={} nw={} x_t={:.3f} y_t={:.3f} masked={} no_accum={} bad_grid_with_accum={} threshold={} raw_signal_nonzero={} adjacent_holes={} support_threshold={:.4g} max_raw_signal={:.4g} max_neighbor_weight={:.4g} max_neighbor_rc=({}, {}) max_neighbor_cause={}",
                            current_iter,
                            rank + 1,
                            map_index,
                            uid,
                            array,
                            nw,
                            x_t,
                            y_t,
                            diag.n_masked,
                            diag.n_masked_no_accum_weight,
                            diag.n_masked_bad_grid_weight_with_accum_weight,
                            diag.n_masked_by_support_threshold,
                            diag.n_masked_raw_signal_nonzero,
                            diag.n_masked_adjacent_support,
                            diag.support_weight_threshold,
                            diag.max_masked_abs_raw_signal,
                            diag.max_masked_neighbor_weight,
                            diag.max_neighbor_row,
                            diag.max_neighbor_col,
                            cause_name(diag.max_neighbor_cause));
                    }
                }
            };

            run_mapmaking_pass(true);

            if (beammap_scan_band_mask_enabled && map_grouping == "detector" && locator_iter) {
                auto scan_band_summary = apply_scan_band_mask(omb);
                if (scan_band_summary.n_samples_flagged > 0) {
                    logger->info(
                        "beammap scan-band mask summary: flagged {} samples in {} rows across {} detectors ({} rejected by max_flagged_fraction={:.4f}); rebuilding maps",
                        scan_band_summary.n_samples_flagged,
                        scan_band_summary.n_rows_flagged,
                        scan_band_summary.n_det_flagged,
                        scan_band_summary.n_det_rejected,
                        beammap_scan_band_mask_max_flagged_fraction);
                    run_mapmaking_pass(false);
                }
                else {
                    logger->info(
                        "beammap scan-band mask summary: no edge bands flagged ({} detectors rejected by max_flagged_fraction={:.4f})",
                        scan_band_summary.n_det_rejected,
                        beammap_scan_band_mask_max_flagged_fraction);
                }
            }

            Eigen::VectorXi iter_bound_low = Eigen::VectorXi::Zero(map_fitter.n_params);
            Eigen::VectorXi iter_bound_high = Eigen::VectorXi::Zero(map_fitter.n_params);
            Eigen::Index iter_bound_any = 0;
            Eigen::Index iter_init_prev = 0;
            Eigen::Index iter_init_prior = 0;
            Eigen::Index iter_init_blind = 0;
            Eigen::Index iter_init_skip = 0;
            Eigen::Index iter_attempt_prev = 0;
            Eigen::Index iter_attempt_prior = 0;
            Eigen::Index iter_attempt_blind = 0;
            Eigen::Index iter_fail_prev = 0;
            Eigen::Index iter_fail_prior = 0;
            Eigen::Index iter_fail_blind = 0;
            Eigen::Index iter_prev_rejected_by_peak = 0;
            Eigen::Index iter_init_amp_zero_prev = 0;
            Eigen::Index iter_init_amp_zero_prior = 0;
            Eigen::Index iter_init_amp_zero_blind = 0;
            Eigen::Index iter_amp_bounds_zero_prev = 0;
            Eigen::Index iter_amp_bounds_zero_prior = 0;
            Eigen::Index iter_amp_bounds_zero_blind = 0;

            logger->info("fitting maps");
            logger->info("beammap fit diagnostics enabled");
            if (beammap_priors_enabled && beammap_soft_priors_loaded && map_grouping == "detector") {
                update_prior_frame_estimates();
            }
            // Run beammap fits sequentially. This avoids allocator/covariance instability
            // observed with parallel Ceres fits on some systems.
            for (Eigen::Index i = 0; i < n_maps; ++i) {
                logger->debug("beammap fit checkpoint: map={} begin converged={}", i, converged(i));

                if (omb.signal[i].rows() != omb.n_rows || omb.signal[i].cols() != omb.n_cols ||
                    omb.weight[i].rows() != omb.n_rows || omb.weight[i].cols() != omb.n_cols) {
                    logger->error("beammap fit map={} geometry mismatch: signal={}x{} weight={}x{} expected={}x{}",
                                  i, omb.signal[i].rows(), omb.signal[i].cols(),
                                  omb.weight[i].rows(), omb.weight[i].cols(),
                                  omb.n_rows, omb.n_cols);
                    std::exit(EXIT_FAILURE);
                }

                const auto &sig = omb.signal[i];
                const auto &wt = omb.weight[i];
                const Eigen::Index n_pix = sig.size();
                const Eigen::Index sig_finite = sig.array().isFinite().count();
                const Eigen::Index wt_finite = wt.array().isFinite().count();
                const Eigen::Index wt_pos = (wt.array() > 0.0).count();
                logger->debug("beammap fit map={} stats: sig_finite={}/{} wt_finite={}/{} wt_pos={}/{} sig[min,max]=({:.6g}, {:.6g}) wt[min,max]=({:.6g}, {:.6g})",
                              i, sig_finite, n_pix, wt_finite, n_pix, wt_pos, n_pix,
                              sig.minCoeff(), sig.maxCoeff(), wt.minCoeff(), wt.maxCoeff());

                // only fit if not converged
                if (!converged(i)) {
                    if (prior_diag_values.rows() == n_maps && prior_diag_values.cols() == n_prior_diag_cols) {
                        prior_diag_values.row(i).setConstant(std::numeric_limits<double>::quiet_NaN());
                        prior_diag_values(i, prior_init_mode_col) = -1.0;
                        prior_diag_values(i, prior_used_col) = 0.0;
                        prior_diag_values(i, prior_fallback_blind_col) = 0.0;
                        prior_diag_values(i, prior_no_candidate_reason_col) = 0.0;
                        prior_diag_values(i, prior_slot_index_col) = -1.0;
                    }

                    const Eigen::Index n_weight_pos = (omb.weight[i].array() > 0.0).count();
                    if (n_weight_pos < map_fitter.n_params) {
                        logger->warn("beammap fit map={} skipped: insufficient weighted pixels ({})", i, n_weight_pos);
                        params.row(i).setZero();
                        perrors.row(i).setZero();
                        fit_diag_init_params.row(i).setZero();
                        fit_diag_lower_limits.row(i).setZero();
                        fit_diag_upper_limits.row(i).setZero();
                        fit_diag_hit_lower.row(i).setZero();
                        fit_diag_hit_upper.row(i).setZero();
                        fit_diag_bound_code(i) = 0;
                        fit_diag_bound_nhit(i) = 0;
                        good_fits(i) = false;
                        continue;
                    }

                    // get array number
                    auto array = maps_to_arrays(i);
                    // get initial guess fwhm from theoretical fwhms for the arrays
                    double init_fwhm = toltec_io.array_fwhm_arcsec[array]*ASEC_TO_RAD/omb.pixel_size_rad;
                    // choose fit initialization
                    double init_row = -99.0;
                    double init_col = -99.0;
                    bool init_from_prev = false;
                    bool init_from_prior = false;
                    enum class FitInitMode { Blind, Previous, Prior };
                    auto init_mode = FitInitMode::Blind;
                    const bool can_try_prior =
                        beammap_priors_enabled && beammap_soft_priors_loaded && map_grouping == "detector";
                    if (measurement_iter &&
                        good_fits(i) &&
                        p0.cols() > 2 &&
                        std::isfinite(p0(i,0)) && p0(i,0) > 0.0 &&
                        std::isfinite(p0(i,1)) && std::isfinite(p0(i,2))) {
                        const double prev_col = p0(i,1);
                        const double prev_row = p0(i,2);
                        Eigen::Index prev_row_i = static_cast<Eigen::Index>(std::llround(prev_row));
                        Eigen::Index prev_col_i = static_cast<Eigen::Index>(std::llround(prev_col));
                        bool prev_seed_valid = false;
                        if (prev_row_i >= 0 && prev_row_i < omb.signal[i].rows() &&
                            prev_col_i >= 0 && prev_col_i < omb.signal[i].cols()) {
                            const double seed_w = omb.weight[i](prev_row_i, prev_col_i);
                            const double seed_s = omb.signal[i](prev_row_i, prev_col_i);
                            prev_seed_valid = std::isfinite(seed_w) && seed_w > 0.0 &&
                                              std::isfinite(seed_s) && seed_s > 0.0;
                            if (prev_seed_valid) {
                                Eigen::Index peak_row = -1;
                                Eigen::Index peak_col = -1;
                                double peak_snr = -std::numeric_limits<double>::infinity();
                                if (find_map_weighted_peak(i, peak_row, peak_col, peak_snr) &&
                                    peak_row >= 0 && peak_col >= 0 && std::isfinite(peak_snr)) {
                                    const double prev_snr = seed_s * std::sqrt(seed_w);
                                    const double dr = static_cast<double>(peak_row) - prev_row;
                                    const double dc = static_cast<double>(peak_col) - prev_col;
                                    const double dist_pix = std::sqrt(dr * dr + dc * dc);
                                    const double min_switch_dist_pix = std::max(1.0, init_fwhm);
                                    constexpr double min_switch_snr_ratio = 1.25;
                                    bool prior_allows_switch = true;
                                    if (can_try_prior) {
                                        const int array_int = static_cast<int>(maps_to_arrays(i));
                                        const int nw_int = static_cast<int>(std::lround(calib.apt["nw"](i)));
                                        const double pix_to_arcsec = RAD_TO_ASEC * omb.pixel_size_rad;
                                        const double col0 = static_cast<double>(omb.n_cols - 1) / 2.0;
                                        const double row0 = static_cast<double>(omb.n_rows - 1) / 2.0;
                                        const double derot_elev_rad = get_prior_derot_elev_rad();
                                        const double prior_max_d2 = effective_prior_max_d2();

                                        auto prior_compatible = [&](double row, double col, double &d2_out) {
                                            const double x_raw = pix_to_arcsec * (col - col0);
                                            const double y_raw = pix_to_arcsec * (row - row0);
                                            double x_prior = std::numeric_limits<double>::quiet_NaN();
                                            double y_prior = std::numeric_limits<double>::quiet_NaN();
                                            d2_out = std::numeric_limits<double>::infinity();
                                            int slot_index = -1;
                                            if (!observed_to_prior_frame(array_int, x_raw, y_raw, derot_elev_rad,
                                                                         x_prior, y_prior, nullptr, nullptr, true)) {
                                                return false;
                                            }
                                            if (!match_prior_slot(array_int, nw_int, x_prior, y_prior,
                                                                  d2_out, slot_index)) {
                                                return false;
                                            }
                                            static_cast<void>(slot_index);
                                            return prior_max_d2 <= 0.0 || d2_out <= prior_max_d2;
                                        };

                                        double prev_prior_d2 = std::numeric_limits<double>::infinity();
                                        double peak_prior_d2 = std::numeric_limits<double>::infinity();
                                        const bool prev_prior_ok = prior_compatible(prev_row, prev_col, prev_prior_d2);
                                        const bool peak_prior_ok = prior_compatible(
                                            static_cast<double>(peak_row), static_cast<double>(peak_col), peak_prior_d2);
                                        prior_allows_switch = peak_prior_ok || !prev_prior_ok;
                                        if (!prior_allows_switch) {
                                            logger->debug(
                                                "beammap fit map={} kept previous init over stronger weighted peak because prior d2 prev={} peak={} max_d2={}",
                                                i, prev_prior_d2, peak_prior_d2, prior_max_d2);
                                        }
                                    }
                                    if (std::isfinite(prev_snr) &&
                                        peak_snr > min_switch_snr_ratio * prev_snr &&
                                        dist_pix > min_switch_dist_pix &&
                                        prior_allows_switch) {
                                        prev_seed_valid = false;
                                        iter_prev_rejected_by_peak++;
                                        logger->debug(
                                            "beammap fit map={} rejected previous init: current weighted peak row={} col={} snr={} is {} pix from previous row={} col={} snr={}",
                                            i, peak_row, peak_col, peak_snr, dist_pix,
                                            prev_row, prev_col, prev_snr);
                                    }
                                }
                            }
                        }
                        if (prev_seed_valid) {
                            init_col = prev_col;
                            init_row = prev_row;
                            init_from_prev = true;
                            init_mode = FitInitMode::Previous;
                            iter_init_prev++;
                        }
                        else {
                            logger->debug(
                                "beammap fit map={} rejected previous init at row={} col={} due to invalid/no-weight/non-positive seed pixel",
                                i, prev_row, prev_col);
                        }
                    }
                    if (!init_from_prev && can_try_prior) {
                        if (choose_prior_guided_init(i, init_row, init_col)) {
                            init_from_prior = true;
                            init_mode = FitInitMode::Prior;
                            iter_init_prior++;
                        }
                        else if (!beammap_priors_fallback_blind) {
                            if (prior_diag_values.rows() == n_maps && prior_diag_values.cols() == n_prior_diag_cols) {
                                prior_diag_values(i, prior_init_mode_col) = -1.0;
                            }
                            logger->warn("beammap fit map={} skipped: no prior-guided init candidate and fallback_blind=false", i);
                            params.row(i).setZero();
                            perrors.row(i).setZero();
                            fit_diag_init_params.row(i).setZero();
                            fit_diag_lower_limits.row(i).setZero();
                            fit_diag_upper_limits.row(i).setZero();
                            fit_diag_hit_lower.row(i).setZero();
                            fit_diag_hit_upper.row(i).setZero();
                            fit_diag_bound_code(i) = 0;
                            fit_diag_bound_nhit(i) = 0;
                            good_fits(i) = false;
                            iter_init_skip++;
                            continue;
                        }
                        else if (prior_diag_values.rows() == n_maps && prior_diag_values.cols() == n_prior_diag_cols) {
                            prior_diag_values(i, prior_fallback_blind_col) = 1.0;
                        }
                    }
                    if (!init_from_prev && !init_from_prior) {
                        iter_init_blind++;
                    }
                    if (prior_diag_values.rows() == n_maps && prior_diag_values.cols() == n_prior_diag_cols) {
                        if (init_from_prev) {
                            prior_diag_values(i, prior_init_mode_col) = 1.0;
                        }
                        else if (init_from_prior) {
                            prior_diag_values(i, prior_init_mode_col) = 2.0;
                        }
                        else {
                            prior_diag_values(i, prior_init_mode_col) = 0.0;
                        }
                    }
                    logger->debug("beammap fit map={} init mode={} row={:.3f} col={:.3f}",
                                  i, init_from_prev ? "previous" : (init_from_prior ? "prior" : "blind"),
                                  init_row, init_col);
                    // fit the maps
                    logger->debug("beammap fit checkpoint: map={} call fit_to_gaussian", i);
                    engine_utils::mapFitter::FitDiagnostics fit_diag;
                    auto [det_params, det_perror, good_fit] =
                        map_fitter.fit_to_gaussian<engine_utils::mapFitter::beammap>(omb.signal[i], omb.weight[i],
                                                                                     init_fwhm, init_row, init_col, &fit_diag);
                    logger->debug("beammap fit checkpoint: map={} fit_to_gaussian returned good_fit={}", i, good_fit);

                    if (!(det_params.array().isFinite().all() && det_perror.array().isFinite().all())) {
                        det_params.setZero();
                        det_perror.setZero();
                        good_fit = false;
                    }

                    params.row(i) = det_params;
                    perrors.row(i) = det_perror;
                    good_fits(i) = good_fit;

                    bool init_amp_zero = false;
                    bool amp_bounds_zero = false;
                    if (fit_diag.valid &&
                        fit_diag.init_params.size() > 0 &&
                        fit_diag.lower_limits.size() > 0 &&
                        fit_diag.upper_limits.size() > 0) {
                        const double init_amp = fit_diag.init_params(0);
                        const double amp_low = fit_diag.lower_limits(0);
                        const double amp_high = fit_diag.upper_limits(0);
                        init_amp_zero = std::isfinite(init_amp) && std::abs(init_amp) <= 1e-12;
                        amp_bounds_zero =
                            std::isfinite(amp_low) && std::isfinite(amp_high) &&
                            std::abs(amp_high - amp_low) <= 1e-12;
                    }
                    switch (init_mode) {
                        case FitInitMode::Previous:
                            iter_attempt_prev++;
                            if (!good_fit) {
                                iter_fail_prev++;
                            }
                            if (init_amp_zero) {
                                iter_init_amp_zero_prev++;
                            }
                            if (amp_bounds_zero) {
                                iter_amp_bounds_zero_prev++;
                            }
                            break;
                        case FitInitMode::Prior:
                            iter_attempt_prior++;
                            if (!good_fit) {
                                iter_fail_prior++;
                            }
                            if (init_amp_zero) {
                                iter_init_amp_zero_prior++;
                            }
                            if (amp_bounds_zero) {
                                iter_amp_bounds_zero_prior++;
                            }
                            break;
                        case FitInitMode::Blind:
                            iter_attempt_blind++;
                            if (!good_fit) {
                                iter_fail_blind++;
                            }
                            if (init_amp_zero) {
                                iter_init_amp_zero_blind++;
                            }
                            if (amp_bounds_zero) {
                                iter_amp_bounds_zero_blind++;
                            }
                            break;
                    }

                    if (fit_diag.valid &&
                        fit_diag.init_params.size() == map_fitter.n_params &&
                        fit_diag.lower_limits.size() == map_fitter.n_params &&
                        fit_diag.upper_limits.size() == map_fitter.n_params &&
                        fit_diag.hit_lower.size() == map_fitter.n_params &&
                        fit_diag.hit_upper.size() == map_fitter.n_params) {
                        fit_diag_init_params.row(i) = fit_diag.init_params.transpose();
                        fit_diag_lower_limits.row(i) = fit_diag.lower_limits.transpose();
                        fit_diag_upper_limits.row(i) = fit_diag.upper_limits.transpose();
                        fit_diag_hit_lower.row(i) = fit_diag.hit_lower.transpose();
                        fit_diag_hit_upper.row(i) = fit_diag.hit_upper.transpose();

                        int bound_code = 0;
                        int bound_nhit = 0;
                        for (int p = 0; p < map_fitter.n_params; ++p) {
                            const bool hit_low = fit_diag.hit_lower(p) != 0;
                            const bool hit_high = fit_diag.hit_upper(p) != 0;
                            if (hit_low) {
                                bound_code |= (1 << (2 * p));
                                iter_bound_low(p)++;
                                bound_nhit++;
                            }
                            if (hit_high) {
                                bound_code |= (1 << (2 * p + 1));
                                iter_bound_high(p)++;
                                bound_nhit++;
                            }
                        }
                        fit_diag_bound_code(i) = bound_code;
                        fit_diag_bound_nhit(i) = bound_nhit;
                        if (bound_nhit > 0) {
                            iter_bound_any++;
                        }
                    }
                    else {
                        fit_diag_init_params.row(i).setZero();
                        fit_diag_lower_limits.row(i).setZero();
                        fit_diag_upper_limits.row(i).setZero();
                        fit_diag_hit_lower.row(i).setZero();
                        fit_diag_hit_upper.row(i).setZero();
                        fit_diag_bound_code(i) = 0;
                        fit_diag_bound_nhit(i) = 0;
                    }
                }
                // otherwise keep value from previous iteration
                else {
                    params.row(i) = p0.row(i);
                    perrors.row(i) = perror0.row(i);
                }

                logger->debug("beammap fit checkpoint: map={} end good_fit={}", i, good_fits(i));
            }

            logger->info("beammap init summary (iter {}): previous={} prior={} blind={} skipped={} prev_rejected_by_peak={}",
                         current_iter, iter_init_prev, iter_init_prior, iter_init_blind, iter_init_skip,
                         iter_prev_rejected_by_peak);
            logger->info(
                "beammap fit diagnostics (iter {}): prev fail={}/{} init_amp_zero={}/{} amp_bounds_zero={}/{} | "
                "prior fail={}/{} init_amp_zero={}/{} amp_bounds_zero={}/{} | "
                "blind fail={}/{} init_amp_zero={}/{} amp_bounds_zero={}/{}",
                current_iter,
                iter_fail_prev, iter_attempt_prev, iter_init_amp_zero_prev, iter_attempt_prev,
                iter_amp_bounds_zero_prev, iter_attempt_prev,
                iter_fail_prior, iter_attempt_prior, iter_init_amp_zero_prior, iter_attempt_prior,
                iter_amp_bounds_zero_prior, iter_attempt_prior,
                iter_fail_blind, iter_attempt_blind, iter_init_amp_zero_blind, iter_attempt_blind,
                iter_amp_bounds_zero_blind, iter_attempt_blind);

            if (map_fitter.n_params >= 6) {
                logger->info(
                    "beammap fit bound summary (iter {}): any_hit={}/{} amp(lo/hi)={}/{} x(lo/hi)={}/{} y(lo/hi)={}/{} a(lo/hi)={}/{} b(lo/hi)={}/{} angle(lo/hi)={}/{}",
                    current_iter, iter_bound_any, n_maps,
                    iter_bound_low(0), iter_bound_high(0),
                    iter_bound_low(1), iter_bound_high(1),
                    iter_bound_low(2), iter_bound_high(2),
                    iter_bound_low(3), iter_bound_high(3),
                    iter_bound_low(4), iter_bound_high(4),
                    iter_bound_low(5), iter_bound_high(5));
            }
            else {
                logger->info("beammap fit bound summary (iter {}): any_hit={}/{}",
                             current_iter, iter_bound_any, n_maps);
            }
            logger->info("number of good fits {}/{}", static_cast<long long>(good_fits.cast<int>().sum()), n_maps);
        }

        const int completed_iter = current_iter;

        // increment loop iteration
        current_iter++;

        if (current_iter < beammap_iter_max) {
            // check if all detectors are converged
            if ((converged.array() == true).all()) {
                logger->info("all maps converged");
                keep_going = false;
            }
            else if (has_completed_beammap_measurement_iter(current_iter)) {
                // only do convergence test if tolerance is above zero, otherwise run all iterations
                if (run_mapmaking && beammap_iter_tolerance > 0) {
                    // loop through maps and check if it is converged
                    logger->info("checking convergence in fitted-source aperture radius={:.3f} arcsec",
                                 beammap_convergence_radius_arcsec);
                    Eigen::VectorXd convergence_delta =
                        Eigen::VectorXd::Constant(n_maps, std::numeric_limits<double>::quiet_NaN());
                    grppi::map(tula::grppi_utils::dyn_ex(omb.parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
                        if (!converged(i)) {
                            const double delta = calc_beammap_convergence_delta(i);
                            convergence_delta(i) = delta;
                            if (std::isfinite(delta) && delta <= beammap_iter_tolerance) {
                                // set as converged
                                converged(i) = true;
                                // set convergence iteration
                                converge_iter(i) = current_iter;
                            }
                        }
                        return 0;
                    });

                    Eigen::Index n_delta_finite = 0;
                    Eigen::Index n_delta_invalid = 0;
                    double max_delta = 0.0;
                    for (Eigen::Index i = 0; i < convergence_delta.size(); ++i) {
                        if (std::isfinite(convergence_delta(i))) {
                            n_delta_finite++;
                            max_delta = std::max(max_delta, convergence_delta(i));
                        }
                        else if (!converged(i)) {
                            n_delta_invalid++;
                        }
                    }

                    logger->info(
                        "{} maps converged on iter {} (finite_metrics={} invalid_metrics={} max_delta={})",
                        (converged.array() == true).count(), current_iter,
                        n_delta_finite, n_delta_invalid, max_delta);

                    // stop if all maps converged
                    if ((converged.array() == true).all()) {
                        logger->info("all maps converged");
                        keep_going = false;
                    }
                }
                else {
                    logger->info("bypassing convergence check");
                }
            }

            // set previous iteration fits to current iteration fits
            p0 = params;
            perror0 = perrors;
        }
        else {
            logger->info("max iteration reached");
            keep_going = false;
        }

        const bool beammap_iter_is_final = !keep_going;
        const bool write_beammap_ptc_this_iter =
            (beammap_tod_output_iter < 0 && beammap_iter_is_final) ||
            (beammap_tod_output_iter >= 0 && completed_iter == beammap_tod_output_iter);
        if (write_beammap_ptc_this_iter) {
            write_beammap_ptc_products(completed_iter);
        }
        else {
            clear_beammap_ptc_diagnostics();
        }
    }
}

void Beammap::set_apt_flags() {
    // setup bitwise flags
    flag2.resize(calib.n_dets);
    flag2.setConstant(AptFlags::Good);

    // track number of flagged detectors
    std::atomic<int> n_flagged_dets{0};

    logger->info("flagging detectors");
    // first flag based on fit values and signal-to-noise
    grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
        // get array of current detector
        auto array_index = calib.apt["array"](i);
        std::string array_name = toltec_io.array_name_map[array_index];

        // calculate map standard deviation
        double map_std_dev = calc_map_support_stddev(i, true);
        const bool valid_map_std = std::isfinite(map_std_dev) && map_std_dev > 0.0;

        // reject non-physical fit results before threshold checks
        const bool finite_params = params.row(i).array().isFinite().all();
        const bool finite_perrors = perrors.row(i).array().isFinite().all();
        const bool positive_amp = std::isfinite(params(i,0)) && params(i,0) > 0.0;
        const bool positive_fwhm =
            std::isfinite(calib.apt["a_fwhm"](i)) && std::isfinite(calib.apt["b_fwhm"](i)) &&
            calib.apt["a_fwhm"](i) > 0.0 && calib.apt["b_fwhm"](i) > 0.0;
        if (!(finite_params && finite_perrors && positive_amp && positive_fwhm && valid_map_std)) {
            good_fits(i) = false;
        }

        // set apt signal to noise
        if (std::isfinite(perrors(i,0)) && perrors(i,0) > 0) {
            calib.apt["sig2noise"](i) = params(i,0)/perrors(i,0);
        } else {
            calib.apt["sig2noise"](i) = 0;
        }

        // flag bad fits
        if (!good_fits(i)) {
            if (calib.apt["flag"](i)==0) {
                n_flagged_dets++;
                calib.apt["flag"](i) = 1;
            }
            flag2(i) |= AptFlags::BadFit;
        }
        // flag detectors with outlier a_fwhm values
        if (calib.apt["a_fwhm"](i) < lower_fwhm_arcsec[array_name] ||
            ((calib.apt["a_fwhm"](i) > upper_fwhm_arcsec[array_name]) && upper_fwhm_arcsec[array_name] > 0)) {
            if (calib.apt["flag"](i)==0) {
                n_flagged_dets++;
                calib.apt["flag"](i) = 1;
            }
            flag2(i) |= AptFlags::AzFWHM;
        }
        // flag detectors with outlier b_fwhm values
        if (calib.apt["b_fwhm"](i) < lower_fwhm_arcsec[array_name] ||
            ((calib.apt["b_fwhm"](i) > upper_fwhm_arcsec[array_name] && upper_fwhm_arcsec[array_name] > 0))) {
            if (calib.apt["flag"](i)==0) {
                n_flagged_dets++;
                calib.apt["flag"](i) = 1;
            }
            flag2(i) |= AptFlags::ElFWHM;
        }
        // flag detectors with outlier S/N values
        const double map_sig2noise = valid_map_std ? params(i,0)/map_std_dev : 0.0;
        if (!std::isfinite(map_sig2noise) ||
            (map_sig2noise < lower_sig2noise[array_name]) ||
            ((map_sig2noise > upper_sig2noise[array_name]) && (upper_sig2noise[array_name] > 0))) {
            if (calib.apt["flag"](i)==0) {
                n_flagged_dets++;
                calib.apt["flag"](i) = 1;
            }
            flag2(i) |= AptFlags::Sig2Noise;
        }
        return 0;
    });


    // median network sensitivity for flagging
    std::map<Eigen::Index, double> nw_median_sens;

    // calc median sens from unflagged detectors for each nw
    logger->debug("calculating mean sensitivities");
    for (Eigen::Index i=0; i<calib.n_nws; ++i) {
        Eigen::Index nw = calib.nws(i);

        // nw sensitivity
        auto nw_sens = calib.apt["sens"](Eigen::seq(std::get<0>(calib.nw_limits[nw]),
                                                    std::get<1>(calib.nw_limits[nw])-1));

        // number of good detectors
        Eigen::Index n_good_det = (calib.apt["flag"](Eigen::seq(std::get<0>(calib.nw_limits[nw]),
                                                               std::get<1>(calib.nw_limits[nw])-1)).array()==0).count();

        if (n_good_det>0) {
            // to hold good detectors
            Eigen::VectorXd sens(n_good_det);

            // remove flagged dets
            Eigen::Index j = std::get<0>(calib.nw_limits[nw]);
            Eigen::Index k = 0;
            for (Eigen::Index m=0; m<nw_sens.size(); m++) {
                if (calib.apt["flag"](j)==0) {
                    sens(k) = nw_sens(m);
                    k++;
                }
                j++;
            }
            // calculate median sens
            nw_median_sens[nw] = tula::alg::median(sens);
        }
        else {
            nw_median_sens[nw] = tula::alg::median(nw_sens);
        }
    }


    // flag too low/high sensitivies based on the median unflagged sensitivity of each nw
    logger->debug("flagging sensitivities");
    grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
        // get nw of current detector
        auto nw_index = calib.apt["nw"](i);

        // flag outlier sensitivities
        if (calib.apt["sens"](i) < lower_sens_factor*nw_median_sens[nw_index] ||
            (calib.apt["sens"](i) > upper_sens_factor*nw_median_sens[nw_index] && upper_sens_factor > 0)) {
            if (calib.apt["flag"](i)==0) {
                calib.apt["flag"](i) = 1;
                n_flagged_dets++;
            }
            flag2(i) |= AptFlags::Sens;
        }

        return 0;
    });

    // std maps to hold median unflagged x and y positions
    std::map<std::string, double> array_median_x_t, array_median_y_t;

    // calc median x_t and y_t values from unflagged detectors for each arrays
    logger->debug("calculating array median positions");
    for (Eigen::Index i=0; i<calib.n_arrays; ++i) {
        Eigen::Index array = calib.arrays(i);
        std::string array_name = toltec_io.array_name_map[array];

        // x_t
        auto array_x_t = calib.apt["x_t"](Eigen::seq(std::get<0>(calib.array_limits[array]),
                                                     std::get<1>(calib.array_limits[array])-1));
        // y_t
        auto array_y_t = calib.apt["y_t"](Eigen::seq(std::get<0>(calib.array_limits[array]),
                                                     std::get<1>(calib.array_limits[array])-1));
        // number of good detectors
        Eigen::Index n_good_det = (calib.apt["flag"](Eigen::seq(std::get<0>(calib.array_limits[array]),
                                                                std::get<1>(calib.array_limits[array])-1)).array()==0).count();

        // to hold good detectors
        Eigen::VectorXd x_t, y_t;

        if (n_good_det>0) {
            x_t.resize(n_good_det);
            y_t.resize(n_good_det);

            // remove flagged dets
            Eigen::Index j = std::get<0>(calib.array_limits[array]);
            Eigen::Index k = 0;
            for (Eigen::Index m=0; m<array_x_t.size(); m++) {
                if (calib.apt["flag"](j)==0) {
                    x_t(k) = array_x_t(m);
                    y_t(k) = array_y_t(m);
                    k++;
                }
                j++;
            }
            // calculate medians
            array_median_x_t[array_name] = tula::alg::median(x_t);
            array_median_y_t[array_name] = tula::alg::median(y_t);
        }
        else {
            // if no good dets, use all dets to calculate median
            array_median_x_t[array_name] = tula::alg::median(array_x_t);
            array_median_y_t[array_name] = tula::alg::median(array_y_t);
        }
    }

    // remove detectors above distance limits
    logger->debug("flagging detector positions");
    grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
        // get array of current detector
        auto array_index = calib.apt["array"](i);
        std::string array_name = toltec_io.array_name_map[array_index];

        // calculate distance of detector from mean position of all detectors
        double dist = sqrt(pow(calib.apt["x_t"](i) - array_median_x_t[array_name],2) +
                           pow(calib.apt["y_t"](i) - array_median_y_t[array_name],2));

        // flag detectors that are further than the mean value than the distance limit
        if (dist > max_dist_arcsec[array_name] && max_dist_arcsec[array_name] > 0) {
            if (calib.apt["flag"](i)==0) {
                n_flagged_dets++;
                calib.apt["flag"](i) = 1;
            }
            flag2(i) |= AptFlags::Position;
        }

        return 0;
    });

    const bool prior_dist_flag_enabled =
        beammap_flag_max_prior_d2 > 0.0 && beammap_soft_priors_loaded && !beammap_soft_prior_slots.empty();
    if (beammap_flag_max_prior_d2 > 0.0 && !prior_dist_flag_enabled) {
        logger->warn(
            "beammap.flagging.max_prior_d2={} requested but soft priors are unavailable; skipping prior-distance flagging",
            beammap_flag_max_prior_d2);
    }
    if (prior_dist_flag_enabled) {
        double prior_derot_elev_rad = telescope.tel_data["TelElAct"].mean();
        if (!std::isfinite(prior_derot_elev_rad)) {
            prior_derot_elev_rad = 0.0;
        }
        if (std::abs(prior_derot_elev_rad) > pi) {
            prior_derot_elev_rad *= DEG_TO_RAD;
        }
        const bool apply_derot = beammap_soft_priors_are_derotated && telescope.pixel_axes == "altaz";
        const double cos_rot = std::cos(-prior_derot_elev_rad);
        const double sin_rot = std::sin(-prior_derot_elev_rad);
        std::atomic<int> n_prior_dist_hits{0};

        logger->debug("flagging detector prior distances");
        grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
            const int array_index = static_cast<int>(std::lround(calib.apt["array"](i)));
            const int nw_index = static_cast<int>(std::lround(calib.apt["nw"](i)));
            std::string array_name = toltec_io.array_name_map[array_index];

            auto slots_it = beammap_soft_prior_slots.find({array_index, nw_index});
            if (slots_it == beammap_soft_prior_slots.end() || slots_it->second.empty()) {
                return 0;
            }

            double x_arcsec = calib.apt["x_t"](i);
            double y_arcsec = calib.apt["y_t"](i);
            if (!std::isfinite(x_arcsec) || !std::isfinite(y_arcsec)) {
                return 0;
            }

            if (beammap_soft_priors_are_centered) {
                x_arcsec -= array_median_x_t[array_name];
                y_arcsec -= array_median_y_t[array_name];
            }

            if (apply_derot) {
                const double rot_az_off = cos_rot * x_arcsec - sin_rot * y_arcsec;
                const double rot_alt_off = sin_rot * x_arcsec + cos_rot * y_arcsec;
                x_arcsec = -rot_az_off;
                y_arcsec = -rot_alt_off;
            }

            double min_d2 = std::numeric_limits<double>::infinity();
            for (const auto &slot : slots_it->second) {
                if (!std::isfinite(slot.x_arcsec) || !std::isfinite(slot.y_arcsec) ||
                    !std::isfinite(slot.sx_arcsec) || !std::isfinite(slot.sy_arcsec) ||
                    slot.sx_arcsec <= 0.0 || slot.sy_arcsec <= 0.0) {
                    continue;
                }
                const double dx = (x_arcsec - slot.x_arcsec) / slot.sx_arcsec;
                const double dy = (y_arcsec - slot.y_arcsec) / slot.sy_arcsec;
                const double d2 = dx * dx + dy * dy;
                if (std::isfinite(d2) && d2 < min_d2) {
                    min_d2 = d2;
                }
            }
            if (!std::isfinite(min_d2) || min_d2 <= beammap_flag_max_prior_d2) {
                return 0;
            }

            n_prior_dist_hits++;
            if (calib.apt["flag"](i)==0) {
                n_flagged_dets++;
                calib.apt["flag"](i) = 1;
            }
            flag2(i) |= AptFlags::PriorDist;
            return 0;
        });

        logger->info("beammap prior-distance flagging: {} detectors exceeded max_prior_d2={}",
                     n_prior_dist_hits.load(), beammap_flag_max_prior_d2);
    }

    // print number of flagged detectors
    logger->info("{} detectors were flagged", n_flagged_dets.load());

    // Derive the calibration amplitude from an empirical array template where
    // possible.  The Gaussian fit amplitude remains in amp for morphology/QC.
    calc_empirical_template_calibration();

    // calculate fcf
    logger->debug("calculating flux conversion factors");
    grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
        // get array of current detector
        auto array_index = calib.apt["array"](i);
        std::string array_name = toltec_io.array_name_map[array_index];

        const double template_cal_amp =
            (calib.apt.count("cal_amp") > 0 && calib.apt["cal_amp"].size() == calib.n_dets)
                ? calib.apt["cal_amp"](i)
                : std::numeric_limits<double>::quiet_NaN();
        const double amp =
            (std::isfinite(template_cal_amp) && template_cal_amp > 0.0)
                ? template_cal_amp
                : params(i,0);
        // calc flux scale (always in mJy/beam)
        if (calib.apt["flag"](i) == 0 && std::isfinite(amp) && amp > 0.0) {
            const double flxscale = beammap_fluxes_mJy_beam[array_name] / amp;
            if (std::isfinite(flxscale) && flxscale > 0.0) {
                calib.apt["flxscale"](i) = flxscale;
                calib.apt["sens"](i) = calib.apt["sens"](i) * flxscale;
            } else {
                calib.apt["flxscale"](i) = 0;
                calib.apt["sens"](i) = 0;
                calib.apt["flag"](i) = 1;
                flag2(i) |= AptFlags::Sens;
            }
        }
        // set fluxscale (fcf) to zero if flagged
        else {
            calib.apt["flxscale"](i) = 0;
            calib.apt["sens"](i) = 0;
        }
        return 0;
    });

    // re-run calib setup to get average fwhms and beam areas
    calib.setup();

    // calculate source flux in MJy/sr from average beamsizes
    for (Eigen::Index i=0; i<calib.n_arrays; ++i) {
        Eigen::Index array = calib.arrays(i);
        std::string array_name = toltec_io.array_name_map[array];

        // get source flux in MJy/Sr
        beammap_fluxes_MJy_Sr[array_name] = mJY_ASEC_to_MJY_SR*(beammap_fluxes_mJy_beam[array_name])/calib.array_beam_areas[array];
    }
}

void Beammap::process_apt() {
    // reference detector x and y
    double ref_det_x_t = 0;
    double ref_det_y_t = 0;

    // initial reference det
    beammap_reference_det_found = -99;

    // if particular reference detector is requested
    if (beammap_subtract_reference) {
        if (beammap_reference_det >= 0 && beammap_reference_det < calib.n_dets) {
            beammap_reference_det_found = beammap_reference_det;
            // set reference x_t and y_t
            ref_det_x_t = calib.apt["x_t"](beammap_reference_det_found);
            ref_det_y_t = calib.apt["y_t"](beammap_reference_det_found);
        }
        // else use detector closest to the median of selected networks
        else {
            if (beammap_reference_det >= 0) {
                logger->warn("configured beammap_reference_det={} is out of range [0, {}); using automatic reference selection",
                             beammap_reference_det, calib.n_dets);
            }
            logger->info("finding a reference detector");
            constexpr Eigen::Index min_reference_candidates = 25;
            auto nw_in_set = [](Eigen::Index nw, const std::vector<Eigen::Index> &set) {
                return std::find(set.begin(), set.end(), nw) != set.end();
            };

            using IndexVector = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>;
            auto gather_from_nws = [&](const std::vector<Eigen::Index> &ref_nws,
                                       Eigen::VectorXd &x_t, Eigen::VectorXd &y_t,
                                       IndexVector &det_indices) -> bool {
                Eigen::Index n_match = 0;
                for (Eigen::Index i = 0; i < calib.n_dets; ++i) {
                    if (calib.apt["flag"](i) == 0) {
                        auto nw = static_cast<Eigen::Index>(calib.apt["nw"](i));
                        const double x = calib.apt["x_t"](i);
                        const double y = calib.apt["y_t"](i);
                        if (nw_in_set(nw, ref_nws) && std::isfinite(x) && std::isfinite(y)) {
                            n_match++;
                        }
                    }
                }
                if (n_match < min_reference_candidates) {
                    return false;
                }

                x_t.resize(n_match);
                y_t.resize(n_match);
                det_indices.resize(n_match);
                Eigen::Index k = 0;
                for (Eigen::Index i = 0; i < calib.n_dets; ++i) {
                    if (calib.apt["flag"](i) == 0) {
                        auto nw = static_cast<Eigen::Index>(calib.apt["nw"](i));
                        const double x = calib.apt["x_t"](i);
                        const double y = calib.apt["y_t"](i);
                        if (nw_in_set(nw, ref_nws) && std::isfinite(x) && std::isfinite(y)) {
                            x_t(k) = x;
                            y_t(k) = y;
                            det_indices(k) = i;
                            k++;
                        }
                    }
                }
                return true;
            };

            Eigen::VectorXd x_t, y_t, dist;
            IndexVector det_indices;
            double med_x_t = 0.0;
            double med_y_t = 0.0;

            const std::vector<Eigen::Index> primary_nws = {3};
            const std::vector<Eigen::Index> fallback_nws = {2, 3, 4};

            bool have_ref = false;
            if (gather_from_nws(primary_nws, x_t, y_t, det_indices)) {
                logger->info("using median of nw=3 for reference");
                have_ref = true;
            }
            else if (gather_from_nws(fallback_nws, x_t, y_t, det_indices)) {
                logger->info("using median of nw=2,3,4 for reference");
                have_ref = true;
            }

            if (!have_ref) {
                logger->warn("no robust reference from nw=3 or nw=2,3,4; using all unflagged detectors");
                Eigen::Index n_unflagged = 0;
                for (Eigen::Index i = 0; i < calib.n_dets; ++i) {
                    if (calib.apt["flag"](i) == 0 &&
                        std::isfinite(calib.apt["x_t"](i)) &&
                        std::isfinite(calib.apt["y_t"](i))) {
                        n_unflagged++;
                    }
                }
                if (n_unflagged > 0) {
                    x_t.resize(n_unflagged);
                    y_t.resize(n_unflagged);
                    det_indices.resize(n_unflagged);
                    Eigen::Index k = 0;
                    for (Eigen::Index i = 0; i < calib.n_dets; ++i) {
                        const double x = calib.apt["x_t"](i);
                        const double y = calib.apt["y_t"](i);
                        if (calib.apt["flag"](i) == 0 && std::isfinite(x) && std::isfinite(y)) {
                            x_t(k) = x;
                            y_t(k) = y;
                            det_indices(k) = i;
                            k++;
                        }
                    }
                    have_ref = true;
                }
            }

            if (!have_ref) {
                logger->warn("all detectors are flagged; disabling reference subtraction");
            } else {
                logger->info("beammap reference candidate count: {}", x_t.size());
                med_x_t = tula::alg::median(x_t);
                med_y_t = tula::alg::median(y_t);

                if (!std::isfinite(med_x_t) || !std::isfinite(med_y_t)) {
                    logger->warn("beammap reference median is non-finite ({},{}); disabling reference subtraction",
                                 med_x_t, med_y_t);
                    beammap_reference_det_found = -99;
                } else {
                    dist = (x_t.array() - med_x_t).square().matrix() +
                           (y_t.array() - med_y_t).square().matrix();
                    Eigen::Index nearest_candidate = -1;
                    dist.minCoeff(&nearest_candidate);
                    if (nearest_candidate >= 0 && nearest_candidate < det_indices.size()) {
                        beammap_reference_det_found = det_indices(nearest_candidate);

                        // set reference x_t and y_t to the median location
                        ref_det_x_t = med_x_t;
                        ref_det_y_t = med_y_t;
                    } else {
                        logger->warn("beammap reference nearest candidate index {} is invalid; disabling reference subtraction",
                                     nearest_candidate);
                        beammap_reference_det_found = -99;
                    }
                }
            }
        }
        if (beammap_reference_det_found >= 0 && beammap_reference_det_found < calib.n_dets) {
            double ref_det_actual_x_t = calib.apt["x_t"](beammap_reference_det_found);
            double ref_det_actual_y_t = calib.apt["y_t"](beammap_reference_det_found);
            logger->info("using reference median ({:.3f},{:.3f}) arcsec; nearest detector {} at ({:.3f},{:.3f}) arcsec",
                         ref_det_x_t, ref_det_y_t,
                         beammap_reference_det_found,
                         ref_det_actual_x_t, ref_det_actual_y_t);
            // record resolved reference detector for metadata; keep config value unchanged
            calib.apt_meta["reference_det"] = beammap_reference_det_found;
        } else {
            logger->warn("reference detector is invalid; leaving reference offsets at ({:.3f},{:.3f}) arcsec",
                         ref_det_x_t, ref_det_y_t);
        }
    }
    else {
        logger->info("no reference detector selected");
    }

    // add reference detector to APT meta data
    calib.apt_meta["reference_x_t"] = ref_det_x_t;
    calib.apt_meta["reference_y_t"] = ref_det_y_t;

    // raw (not derotated or reference detector subtracted) detector x and y values
    calib.apt["x_t_raw"] = calib.apt["x_t"];
    calib.apt["y_t_raw"] = calib.apt["y_t"];

    // per-detector derotation elevation for altaz beammaps
    calib.apt["derot_elev"].setConstant(telescope.tel_data["TelElAct"].mean());
    if (telescope.pixel_axes == "altaz" && map_grouping == "detector" && !ptcs.empty()) {
        Eigen::MatrixXd elev_best(omb.n_rows, omb.n_cols);
        Eigen::MatrixXd dist2_best(omb.n_rows, omb.n_cols);
        elev_best.setConstant(std::numeric_limits<double>::quiet_NaN());
        dist2_best.setConstant(std::numeric_limits<double>::infinity());

        for (const auto &ptc : ptcs) {
            const auto &alt = ptc.tel_data.data.at("alt_phys");
            const auto &az = ptc.tel_data.data.at("az_phys");
            const auto &el = ptc.tel_data.data.at("TelElAct");
            for (Eigen::Index k = 0; k < alt.size(); ++k) {
                double row = alt(k) / omb.pixel_size_rad + (omb.n_rows - 1) / 2.0;
                double col = az(k) / omb.pixel_size_rad + (omb.n_cols - 1) / 2.0;
                Eigen::Index ir = static_cast<Eigen::Index>(std::llround(row));
                Eigen::Index ic = static_cast<Eigen::Index>(std::llround(col));
                if ((ir >= 0) && (ir < omb.n_rows) && (ic >= 0) && (ic < omb.n_cols)) {
                    double lat_center = (static_cast<double>(ir) - (omb.n_rows - 1) / 2.0) * omb.pixel_size_rad;
                    double lon_center = (static_cast<double>(ic) - (omb.n_cols - 1) / 2.0) * omb.pixel_size_rad;
                    double dlat = alt(k) - lat_center;
                    double dlon = az(k) - lon_center;
                    double dist2 = dlat * dlat + dlon * dlon;
                    if (dist2 < dist2_best(ir, ic)) {
                        dist2_best(ir, ic) = dist2;
                        elev_best(ir, ic) = el(k);
                    }
                }
            }
        }

        for (Eigen::Index i = 0; i < calib.n_dets; ++i) {
            double row = (calib.apt["y_t_raw"](i) * ASEC_TO_RAD) / omb.pixel_size_rad + (omb.n_rows - 1) / 2.0;
            double col = (calib.apt["x_t_raw"](i) * ASEC_TO_RAD) / omb.pixel_size_rad + (omb.n_cols - 1) / 2.0;
            Eigen::Index ir = static_cast<Eigen::Index>(std::llround(row));
            Eigen::Index ic = static_cast<Eigen::Index>(std::llround(col));
            if ((ir >= 0) && (ir < omb.n_rows) && (ic >= 0) && (ic < omb.n_cols)) {
                double elev = elev_best(ir, ic);
                if (std::isfinite(elev)) {
                    calib.apt["derot_elev"](i) = elev;
                }
            }
        }
    }

    // align to reference detector if specified and subtract its position from x and y
    calib.apt["x_t"] =  calib.apt["x_t"].array() - ref_det_x_t;
    calib.apt["y_t"] =  calib.apt["y_t"].array() - ref_det_y_t;

    // derotated detector x and y values
    calib.apt["x_t_derot"] = calib.apt["x_t"];
    calib.apt["y_t_derot"] = calib.apt["y_t"];

    // tolerate telescope streams that provide elevation in degrees.
    Eigen::VectorXd derot_elev_rad = calib.apt["derot_elev"];
    const double max_abs_elev = derot_elev_rad.array().abs().maxCoeff();
    if (std::isfinite(max_abs_elev) && max_abs_elev > 2.0 * pi + 0.1) {
        logger->warn("derot_elev appears to be in degrees (max |elev|={:.4g}); converting to radians", max_abs_elev);
        derot_elev_rad *= DEG_TO_RAD;
    }

    // calculate derotated positions
    Eigen::VectorXd rot_az_off = cos(-derot_elev_rad.array())*calib.apt["x_t_derot"].array() -
                                 sin(-derot_elev_rad.array())*calib.apt["y_t_derot"].array();
    Eigen::VectorXd rot_alt_off = sin(-derot_elev_rad.array())*calib.apt["x_t_derot"].array() +
                                  cos(-derot_elev_rad.array())*calib.apt["y_t_derot"].array();

    // overwrite x_t and y_t
    calib.apt["x_t_derot"] = -rot_az_off;
    calib.apt["y_t_derot"] = -rot_alt_off;

    if (beammap_derotate) {
        logger->info("derotating apt");
        // if derotation requested set default positions to derotated positions
        calib.apt["x_t"] = calib.apt["x_t_derot"];
        calib.apt["y_t"] = calib.apt["y_t_derot"];
    }
}

void Beammap::apply_final_network_position_flags() {
    if (map_grouping != "detector") {
        return;
    }

    bool enabled = false;
    for (const auto &[arr_index, arr_name] : toltec_io.array_name_map) {
        auto it = network_robust_z.find(arr_name);
        if (it != network_robust_z.end() && it->second > 0.0) {
            enabled = true;
            break;
        }
    }
    if (!enabled) {
        return;
    }

    struct NetworkStats {
        bool valid = false;
        double median_x = 0.0;
        double median_y = 0.0;
        double sigma_x = 0.0;
        double sigma_y = 0.0;
        double threshold = 0.0;
    };

    std::map<std::pair<int, int>, NetworkStats> stats_by_network;
    constexpr Eigen::Index min_network_samples = 16;

    logger->debug("flagging final detector network positions");
    for (Eigen::Index i = 0; i < calib.n_arrays; ++i) {
        Eigen::Index array = calib.arrays(i);
        std::string array_name = toltec_io.array_name_map[array];
        const double threshold = network_robust_z[array_name];
        if (!(threshold > 0.0)) {
            continue;
        }

        for (Eigen::Index j = 0; j < calib.n_nws; ++j) {
            Eigen::Index nw = calib.nws(j);
            if (std::get<0>(calib.nw_limits[nw]) < 0 ||
                std::get<1>(calib.nw_limits[nw]) <= std::get<0>(calib.nw_limits[nw])) {
                continue;
            }
            if (static_cast<Eigen::Index>(calib.apt["array"](std::get<0>(calib.nw_limits[nw]))) != array) {
                continue;
            }

            std::vector<double> x_vals;
            std::vector<double> y_vals;
            x_vals.reserve(static_cast<std::size_t>(std::get<1>(calib.nw_limits[nw]) - std::get<0>(calib.nw_limits[nw])));
            y_vals.reserve(x_vals.capacity());

            for (Eigen::Index k = std::get<0>(calib.nw_limits[nw]); k < std::get<1>(calib.nw_limits[nw]); ++k) {
                if (calib.apt["flag"](k) != 0) {
                    continue;
                }
                const double x = calib.apt["x_t"](k);
                const double y = calib.apt["y_t"](k);
                if (!std::isfinite(x) || !std::isfinite(y)) {
                    continue;
                }
                x_vals.push_back(x);
                y_vals.push_back(y);
            }
            if (static_cast<Eigen::Index>(x_vals.size()) < min_network_samples) {
                continue;
            }

            Eigen::Map<Eigen::VectorXd> x_vec(x_vals.data(), static_cast<Eigen::Index>(x_vals.size()));
            Eigen::Map<Eigen::VectorXd> y_vec(y_vals.data(), static_cast<Eigen::Index>(y_vals.size()));
            const double median_x = tula::alg::median(x_vec);
            const double median_y = tula::alg::median(y_vec);
            Eigen::VectorXd x_abs_dev = (x_vec.array() - median_x).abs().matrix();
            Eigen::VectorXd y_abs_dev = (y_vec.array() - median_y).abs().matrix();
            double sigma_x = 1.4826 * tula::alg::median(x_abs_dev);
            double sigma_y = 1.4826 * tula::alg::median(y_abs_dev);
            if (!std::isfinite(sigma_x) || sigma_x <= std::numeric_limits<double>::epsilon()) {
                sigma_x = engine_utils::calc_std_dev(x_vec);
            }
            if (!std::isfinite(sigma_y) || sigma_y <= std::numeric_limits<double>::epsilon()) {
                sigma_y = engine_utils::calc_std_dev(y_vec);
            }
            if (!std::isfinite(sigma_x) || !std::isfinite(sigma_y) ||
                sigma_x <= std::numeric_limits<double>::epsilon() ||
                sigma_y <= std::numeric_limits<double>::epsilon()) {
                continue;
            }

            stats_by_network[{static_cast<int>(array), static_cast<int>(nw)}] =
                {true, median_x, median_y, sigma_x, sigma_y, threshold};
        }
    }

    std::atomic<int> n_flagged{0};
    grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
        if (calib.apt["flag"](i) != 0) {
            return 0;
        }

        const int array_index = static_cast<int>(std::lround(calib.apt["array"](i)));
        const int nw_index = static_cast<int>(std::lround(calib.apt["nw"](i)));
        auto it = stats_by_network.find({array_index, nw_index});
        if (it == stats_by_network.end() || !it->second.valid) {
            return 0;
        }

        const double x = calib.apt["x_t"](i);
        const double y = calib.apt["y_t"](i);
        if (!std::isfinite(x) || !std::isfinite(y)) {
            return 0;
        }

        const double zx = (x - it->second.median_x) / it->second.sigma_x;
        const double zy = (y - it->second.median_y) / it->second.sigma_y;
        const double z = std::sqrt(zx * zx + zy * zy);
        if (!std::isfinite(z) || z <= it->second.threshold) {
            return 0;
        }

        calib.apt["flag"](i) = 1;
        calib.apt["flxscale"](i) = 0.0;
        calib.apt["sens"](i) = 0.0;
        flag2(i) |= AptFlags::NetworkPos;
        n_flagged++;
        return 0;
    });

    if (n_flagged.load() > 0) {
        std::string by_array;
        for (Eigen::Index i = 0; i < calib.n_arrays; ++i) {
            Eigen::Index array = calib.arrays(i);
            std::string array_name = toltec_io.array_name_map[array];
            Eigen::Index n_array_flagged = 0;
            if (calib.array_limits.count(array) > 0) {
                for (Eigen::Index k = std::get<0>(calib.array_limits[array]);
                     k < std::get<1>(calib.array_limits[array]); ++k) {
                    if ((flag2(k) & AptFlags::NetworkPos) != 0) {
                        n_array_flagged++;
                    }
                }
            }
            if (!by_array.empty()) {
                by_array += ", ";
            }
            by_array += array_name + "=" + std::to_string(n_array_flagged);
        }
        logger->info(
            "beammap final network-position flagging: {} detectors exceeded per-array robust-z limits ({})",
            n_flagged.load(), by_array);
    }
}

void Beammap::update_final_prior_match_diagnostics() {
    final_prior_d2_diag = Eigen::VectorXd::Constant(
        calib.n_dets, std::numeric_limits<double>::quiet_NaN());
    final_prior_slot_index_diag = Eigen::VectorXi::Constant(calib.n_dets, -1);

    if (map_grouping != "detector" || !beammap_soft_priors_loaded || beammap_soft_prior_slots.empty()) {
        return;
    }

    struct ArrayCenter {
        bool valid = false;
        double x = 0.0;
        double y = 0.0;
    };

    std::map<int, ArrayCenter> centers;
    auto median_from = [](std::vector<double> &values, double &median) -> bool {
        if (values.empty()) {
            median = std::numeric_limits<double>::quiet_NaN();
            return false;
        }
        Eigen::Map<Eigen::VectorXd> vec(values.data(), static_cast<Eigen::Index>(values.size()));
        median = tula::alg::median(vec);
        return std::isfinite(median);
    };

    for (Eigen::Index i = 0; i < calib.n_arrays; ++i) {
        const Eigen::Index array = calib.arrays(i);
        std::vector<double> x_vals;
        std::vector<double> y_vals;

        auto gather = [&](bool unflagged_only) {
            x_vals.clear();
            y_vals.clear();
            for (Eigen::Index k = 0; k < calib.n_dets; ++k) {
                if (static_cast<Eigen::Index>(std::lround(calib.apt["array"](k))) != array) {
                    continue;
                }
                if (unflagged_only && calib.apt["flag"](k) != 0) {
                    continue;
                }
                const double x = calib.apt["x_t_raw"](k);
                const double y = calib.apt["y_t_raw"](k);
                if (!std::isfinite(x) || !std::isfinite(y)) {
                    continue;
                }
                x_vals.push_back(x);
                y_vals.push_back(y);
            }
        };

        gather(true);
        if (x_vals.size() < 8) {
            gather(false);
        }
        if (x_vals.empty()) {
            continue;
        }

        double median_x = std::numeric_limits<double>::quiet_NaN();
        double median_y = std::numeric_limits<double>::quiet_NaN();
        if (!median_from(x_vals, median_x) || !median_from(y_vals, median_y)) {
            continue;
        }
        centers[static_cast<int>(array)] = {true, median_x, median_y};
    }

    for (Eigen::Index i = 0; i < calib.n_dets; ++i) {
        const int array = static_cast<int>(std::lround(calib.apt["array"](i)));
        const int nw = static_cast<int>(std::lround(calib.apt["nw"](i)));
        auto slots_it = beammap_soft_prior_slots.find({array, nw});
        if (slots_it == beammap_soft_prior_slots.end() || slots_it->second.empty()) {
            continue;
        }

        double x_arcsec = calib.apt["x_t_raw"](i);
        double y_arcsec = calib.apt["y_t_raw"](i);
        if (!std::isfinite(x_arcsec) || !std::isfinite(y_arcsec)) {
            continue;
        }

        if (beammap_soft_priors_are_centered) {
            auto center_it = centers.find(array);
            if (center_it != centers.end() && center_it->second.valid) {
                x_arcsec -= center_it->second.x;
                y_arcsec -= center_it->second.y;
            }
        }

        if (beammap_soft_priors_are_derotated && telescope.pixel_axes == "altaz") {
            double derot_elev_rad = calib.apt["derot_elev"](i);
            if (!std::isfinite(derot_elev_rad)) {
                derot_elev_rad = telescope.tel_data["TelElAct"].mean();
            }
            if (!std::isfinite(derot_elev_rad)) {
                derot_elev_rad = 0.0;
            }
            if (std::abs(derot_elev_rad) > pi) {
                derot_elev_rad *= DEG_TO_RAD;
            }
            const double rot_az_off = std::cos(-derot_elev_rad) * x_arcsec -
                                      std::sin(-derot_elev_rad) * y_arcsec;
            const double rot_alt_off = std::sin(-derot_elev_rad) * x_arcsec +
                                       std::cos(-derot_elev_rad) * y_arcsec;
            x_arcsec = -rot_az_off;
            y_arcsec = -rot_alt_off;
        }

        double best_d2 = std::numeric_limits<double>::infinity();
        int best_slot = -1;
        for (const auto &slot : slots_it->second) {
            const double sx = std::max(slot.sx_arcsec, std::numeric_limits<double>::epsilon());
            const double sy = std::max(slot.sy_arcsec, std::numeric_limits<double>::epsilon());
            const double dx = (x_arcsec - slot.x_arcsec) / sx;
            const double dy = (y_arcsec - slot.y_arcsec) / sy;
            const double d2 = dx * dx + dy * dy;
            if (std::isfinite(d2) && d2 < best_d2) {
                best_d2 = d2;
                best_slot = slot.slot_index;
            }
        }
        if (std::isfinite(best_d2)) {
            final_prior_d2_diag(i) = best_d2;
            final_prior_slot_index_diag(i) = best_slot;
        }
    }
}

void Beammap::log_final_network_qc_summary() {
    if (map_grouping != "detector") {
        return;
    }

    auto median_or_nan = [](std::vector<double> &values) {
        if (values.empty()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        Eigen::Map<Eigen::VectorXd> vec(values.data(), static_cast<Eigen::Index>(values.size()));
        return tula::alg::median(vec);
    };

    logger->info("beammap final per-network qc summary follows");
    for (Eigen::Index i = 0; i < calib.n_arrays; ++i) {
        const Eigen::Index array = calib.arrays(i);
        const std::string array_name = toltec_io.array_name_map[array];

        for (Eigen::Index j = 0; j < calib.n_nws; ++j) {
            const Eigen::Index nw = calib.nws(j);
            if (calib.nw_limits.count(nw) == 0) {
                continue;
            }
            const auto [k0, k1] = calib.nw_limits[nw];
            if (k0 < 0 || k1 <= k0) {
                continue;
            }
            if (static_cast<Eigen::Index>(std::lround(calib.apt["array"](k0))) != array) {
                continue;
            }

            std::vector<double> a_vals;
            std::vector<double> b_vals;
            std::vector<double> snr_vals;
            std::vector<double> prior_d2_vals;
            Eigen::Index n_total = 0;
            Eigen::Index n_good = 0;
            for (Eigen::Index k = k0; k < k1; ++k) {
                n_total++;
                if (calib.apt["flag"](k) != 0) {
                    continue;
                }
                n_good++;
                if (std::isfinite(calib.apt["a_fwhm"](k))) {
                    a_vals.push_back(calib.apt["a_fwhm"](k));
                }
                if (std::isfinite(calib.apt["b_fwhm"](k))) {
                    b_vals.push_back(calib.apt["b_fwhm"](k));
                }
                if (std::isfinite(calib.apt["sig2noise"](k))) {
                    snr_vals.push_back(calib.apt["sig2noise"](k));
                }
                if (final_prior_d2_diag.size() == calib.n_dets &&
                    std::isfinite(final_prior_d2_diag(k))) {
                    prior_d2_vals.push_back(final_prior_d2_diag(k));
                }
            }

            const double good_frac =
                static_cast<double>(n_good) / static_cast<double>(std::max<Eigen::Index>(1, n_total));
            logger->info(
                "beammap network qc: array={} nw={} good={}/{} ({:.3f}) med_a_fwhm={} med_b_fwhm={} med_sig2noise={} med_final_prior_d2={}",
                array_name,
                static_cast<int>(nw),
                n_good,
                n_total,
                good_frac,
                median_or_nan(a_vals),
                median_or_nan(b_vals),
                median_or_nan(snr_vals),
                median_or_nan(prior_d2_vals));
        }
    }
}

template <mapmaking::MapType map_type>
void Beammap::output() {
    // pointer to map buffer
    mapmaking::MapBuffer* mb = nullptr;
    // pointer to data file fits vector
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>* f_io = nullptr;
    // pointer to noise file fits vector
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>* n_io = nullptr;

    // directory name
    std::string dir_name;

    // raw obs maps
    if constexpr (map_type == mapmaking::RawObs) {
        mb = &omb;
        f_io = &fits_io_vec;
        n_io = &noise_fits_io_vec;
        dir_name = obsnum_dir_name + "raw/";

        // write stats file
        write_stats();

        // add header informqtion to tod
        if (run_tod_output && !tod_filename.empty()) {
            add_tod_header(mb);
        }

        write_detector_table_outputs();
    }

    // filtered obs maps
    else if constexpr (map_type == mapmaking::FilteredObs) {
        mb = &omb;
        f_io = &filtered_fits_io_vec;
        n_io = &filtered_noise_fits_io_vec;
        dir_name = obsnum_dir_name + "filtered/";
    }

    // raw coadded maps
    else if constexpr (map_type == mapmaking::RawCoadd) {
        mb = &cmb;
        f_io = &coadd_fits_io_vec;
        n_io = &coadd_noise_fits_io_vec;
        dir_name = coadd_dir_name + "raw/";
    }

    // filtered coadded maps
    else if constexpr (map_type == mapmaking::FilteredCoadd) {
        mb = &cmb;
        f_io = &filtered_coadd_fits_io_vec;
        n_io = &filtered_coadd_noise_fits_io_vec;
        dir_name = coadd_dir_name + "filtered/";
    }

    write_beammap_map_products<map_type>(mb, f_io, n_io, dir_name);
}
