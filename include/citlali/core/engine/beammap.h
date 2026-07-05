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
#include <citlali/core/engine/detail/beammap_run_loop_impl.h>
#include <citlali/core/engine/detail/beammap_finalization_impl.h>

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
