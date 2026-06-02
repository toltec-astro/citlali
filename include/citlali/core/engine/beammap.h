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

    // main pipeline process
    template <class KidsProc, class RawObs>
    void pipeline(KidsProc &, RawObs &);

    // output files
    template <mapmaking::MapType map_type>
    void output();
};

void Beammap::setup() {
    // assign parallel policies
    map_parallel_policy = parallel_policy;

    // run obsnum setup
    obsnum_setup();

    // create kids tone apt row
    calib.apt["kids_tone"].resize(calib.n_dets);

    Eigen::Index j = 0;
    // set kids tone (det number on network)
    calib.apt["kids_tone"](0) = 0;
    for (Eigen::Index i=1; i<calib.n_dets; ++i) {
        if (calib.apt["nw"](i) > calib.apt["nw"](i-1)) {
            j = 0;
        }
        else {
            j++;
        }

        calib.apt["kids_tone"](i) = j;
    }

    // add kids tone to apt header
    calib.apt_header_keys.push_back("kids_tone");
    calib.apt_header_units["kids_tone"] = "N/A";

    // resize the PTCData vector to number of scans
    ptcs0.resize(telescope.scan_indices.cols());

    // resize the calib vector to number of scans
    calib_scans0.resize(telescope.scan_indices.cols());

    // resize the initial fit matrix
    p0.setZero(n_maps, map_fitter.n_params);
    // resize the initial fit error matrix
    perror0.setZero(n_maps, map_fitter.n_params);
    // resize the current fit matrix
    params.setZero(n_maps, map_fitter.n_params);
    perrors.setZero(n_maps, map_fitter.n_params);
    fit_diag_init_params.setZero(n_maps, map_fitter.n_params);
    fit_diag_lower_limits.setZero(n_maps, map_fitter.n_params);
    fit_diag_upper_limits.setZero(n_maps, map_fitter.n_params);
    fit_diag_hit_lower.setZero(n_maps, map_fitter.n_params);
    fit_diag_hit_upper.setZero(n_maps, map_fitter.n_params);
    fit_diag_bound_code.setZero(n_maps);
    fit_diag_bound_nhit.setZero(n_maps);
    prior_diag_values.resize(n_maps, n_prior_diag_cols);
    prior_diag_values.setConstant(std::numeric_limits<double>::quiet_NaN());

    // resize good fits
    good_fits.setZero(n_maps);
    rfi_mask_samples_flagged = Eigen::VectorXi::Zero(calib.n_dets);
    rfi_mask_scans_flagged = Eigen::VectorXi::Zero(calib.n_dets);
    scan_band_mask_samples_flagged = Eigen::VectorXi::Zero(calib.n_dets);
    scan_band_mask_rows_flagged = Eigen::VectorXi::Zero(calib.n_dets);
    scan_band_mask_edge_code = Eigen::VectorXi::Zero(calib.n_dets);
    scan_band_mask_rejected = Eigen::VectorXi::Zero(calib.n_dets);
    final_prior_d2_diag = Eigen::VectorXd::Constant(calib.n_dets, std::numeric_limits<double>::quiet_NaN());
    final_prior_slot_index_diag = Eigen::VectorXi::Constant(calib.n_dets, -1);

    // initially all detectors are unconverged
    converged.setZero(n_maps);
    // convergence iteration
    converge_iter.resize(n_maps);
    converge_iter.setConstant(1);
    // set the initial iteration
    current_iter = 0;

    /* update apt table meta data */
    calib.apt_meta.reset();

    // add obsnum to meta data
    calib.apt_meta["obsnum"] = obsnum;

    // add source name
    calib.apt_meta["source"] = telescope.source_name;

    // add project id to meta data
    calib.apt_meta["project_id"] = telescope.project_id;

    // add input source flux
    for (const auto &beammap_flux: beammap_fluxes_mJy_beam) {
        auto key = beammap_flux.first + "_flux";
        calib.apt_meta[key].push_back(beammap_flux.second);
        calib.apt_meta[key].push_back("units: mJy/beam");
        calib.apt_meta[key].push_back(beammap_flux.first + " flux density");
    }

    // add date of file creation
    calib.apt_meta["creation_date"] = engine_utils::current_date_time();

    // add observation date
    calib.apt_meta["date"] = date_obs.back();

    // mean Modified Julian Date
    calib.apt_meta["mjd"] = engine_utils::unix_to_modified_julian_date(telescope.tel_data["TelTime"].mean());

    // reference frame
    calib.apt_meta["Radesys"] = telescope.pixel_axes;

    // add mean tau to apt meta
    if (rtcproc.run_extinction) {
        Eigen::VectorXd tau_el(1);
        tau_el << telescope.tel_data["TelElAct"].mean();
        auto tau_freq = rtcproc.calibration.calc_tau(tau_el, telescope.tau_225_GHz);

        Eigen::Index i = 0;
        for (auto const& [key, val] : tau_freq) {
            calib.apt_meta[toltec_io.array_name_map[calib.arrays(i)]+"_tau"] = val[0];
            i++;
        }
    }
    else {
        for (Eigen::Index i=0; i<calib.arrays.size(); ++i) {
            calib.apt_meta[toltec_io.array_name_map[calib.arrays(i)]+"_tau"] = 0.;
        }
    }

    // add apt header keys
    for (const auto &[param,unit]: calib.apt_header_units) {
        calib.apt_meta[param].push_back("units: " + unit);
    }
    // add apt header descriptions
    for (const auto &[param,description]: calib.apt_header_description) {
        calib.apt_meta[param].push_back(description);
    }

    // kids tone
    calib.apt_meta["kids_tone"].push_back("units: N/A");
    calib.apt_meta["kids_tone"].push_back("index of tone in network");

    // diagnostics for robust sample masking of beammap RFI
    calib.apt["rfi_masked_samples"] = Eigen::VectorXd::Zero(calib.n_dets);
    calib.apt_header_units["rfi_masked_samples"] = "samples";
    calib.apt_header_keys.push_back("rfi_masked_samples");
    calib.apt_meta["rfi_masked_samples"].push_back("units: samples");
    calib.apt_meta["rfi_masked_samples"].push_back("number of timestream samples masked by beammap rfi_mask");

    calib.apt["rfi_masked_scans"] = Eigen::VectorXd::Zero(calib.n_dets);
    calib.apt_header_units["rfi_masked_scans"] = "scans";
    calib.apt_header_keys.push_back("rfi_masked_scans");
    calib.apt_meta["rfi_masked_scans"].push_back("units: scans");
    calib.apt_meta["rfi_masked_scans"].push_back("number of scans with at least one sample masked by beammap rfi_mask");

    calib.apt["scan_band_masked_samples"] = Eigen::VectorXd::Zero(calib.n_dets);
    calib.apt_header_units["scan_band_masked_samples"] = "samples";
    calib.apt_header_keys.push_back("scan_band_masked_samples");
    calib.apt_meta["scan_band_masked_samples"].push_back("units: samples");
    calib.apt_meta["scan_band_masked_samples"].push_back(
        "number of timestream samples masked by beammap scan_band_mask");

    calib.apt["scan_band_masked_rows"] = Eigen::VectorXd::Zero(calib.n_dets);
    calib.apt_header_units["scan_band_masked_rows"] = "rows";
    calib.apt_header_keys.push_back("scan_band_masked_rows");
    calib.apt_meta["scan_band_masked_rows"].push_back("units: rows");
    calib.apt_meta["scan_band_masked_rows"].push_back(
        "number of detector-map edge rows flagged by beammap scan_band_mask");

    calib.apt["scan_band_masked_edge"] = Eigen::VectorXd::Zero(calib.n_dets);
    calib.apt_header_units["scan_band_masked_edge"] = "N/A";
    calib.apt_header_keys.push_back("scan_band_masked_edge");
    calib.apt_meta["scan_band_masked_edge"].push_back("units: N/A");
    calib.apt_meta["scan_band_masked_edge"].push_back(
        "scan-band edge code (0 none, 1 top, 2 bottom, 3 both)");
    calib.apt_meta["scan_band_masked_edge"].push_back("0=none");
    calib.apt_meta["scan_band_masked_edge"].push_back("1=top");
    calib.apt_meta["scan_band_masked_edge"].push_back("2=bottom");
    calib.apt_meta["scan_band_masked_edge"].push_back("3=both");

    calib.apt["scan_band_mask_rejected"] = Eigen::VectorXd::Zero(calib.n_dets);
    calib.apt_header_units["scan_band_mask_rejected"] = "N/A";
    calib.apt_header_keys.push_back("scan_band_mask_rejected");
    calib.apt_meta["scan_band_mask_rejected"].push_back("units: N/A");
    calib.apt_meta["scan_band_mask_rejected"].push_back(
        "1 if scan_band_mask proposed a mask but rejected it due to max_flagged_fraction");

    calib.apt["final_prior_slot_index"] =
        Eigen::VectorXd::Constant(calib.n_dets, -1.0);
    calib.apt_header_units["final_prior_slot_index"] = "N/A";
    calib.apt_header_keys.push_back("final_prior_slot_index");
    calib.apt_meta["final_prior_slot_index"].push_back("units: N/A");
    calib.apt_meta["final_prior_slot_index"].push_back(
        "nearest prior slot index for final detector position in prior frame (-1 if unavailable)");

    calib.apt["final_prior_d2"] =
        Eigen::VectorXd::Constant(calib.n_dets, std::numeric_limits<double>::quiet_NaN());
    calib.apt_header_units["final_prior_d2"] = "N/A";
    calib.apt_header_keys.push_back("final_prior_d2");
    calib.apt_meta["final_prior_d2"].push_back("units: N/A");
    calib.apt_meta["final_prior_d2"].push_back(
        "nearest-slot Mahalanobis d^2 for final detector position in the soft-prior frame");

    init_empirical_template_calibration_columns();

    // bitwise flag
    calib.apt_meta["flag2"].push_back("units: N/A");
    calib.apt_meta["flag2"].push_back("bitwise flag");
    calib.apt_meta["flag2"].push_back("Good=0");
    calib.apt_meta["flag2"].push_back("BadFit=1");
    calib.apt_meta["flag2"].push_back("AzFWHM=2");
    calib.apt_meta["flag2"].push_back("ElFWHM=4");
    calib.apt_meta["flag2"].push_back("Sig2Noise=8");
    calib.apt_meta["flag2"].push_back("Sens=16");
    calib.apt_meta["flag2"].push_back("Position=32");
    calib.apt_meta["flag2"].push_back("PriorDist=64");
    calib.apt_meta["flag2"].push_back("NetworkPos=128");

    // add array mapping
    for (const auto &[arr_index,arr_name]: toltec_io.array_name_map) {
        calib.apt_meta["array_order"].push_back(std::to_string(arr_index) + ": " + arr_name);
    }

    calib.apt_header_units["flag2"] = "N/A";
    calib.apt_header_keys.push_back("flag2");

    // is the detector rotated?
    calib.apt_meta["is_derotated"] = beammap_derotate;
    // was a reference detector subtracted?
    calib.apt_meta["reference_detector_subtracted"] = beammap_subtract_reference;
    // reference detector
    calib.apt_meta["reference_det"] = beammap_reference_det_found;
    calib.apt_meta["rfi_mask_enabled"] = beammap_rfi_mask_enabled;
    calib.apt_meta["rfi_mask_block_size_samples"] = beammap_rfi_mask_block_size_samples;
    calib.apt_meta["rfi_mask_min_good_samples"] = beammap_rfi_mask_min_good_samples;
    calib.apt_meta["rfi_mask_dilate_blocks"] = beammap_rfi_mask_dilate_blocks;
    calib.apt_meta["rfi_mask_sigma_threshold"] = beammap_rfi_mask_sigma_threshold;
    calib.apt_meta["rfi_mask_sigma_floor"] = beammap_rfi_mask_sigma_floor;
    calib.apt_meta["rfi_mask_max_flagged_fraction"] = beammap_rfi_mask_max_flagged_fraction;
    calib.apt_meta["detector_weighting_mode"] = beammap_detector_weighting_mode;
    calib.apt_meta["beammap_fit_radius_fwhm"] = beammap_fit_radius_fwhm;
    beammap_soft_prior_slots.clear();
    beammap_soft_priors_loaded = false;
    beammap_soft_priors_are_centered = false;
    beammap_soft_priors_are_derotated = false;
    beammap_prior_array_center_x_arcsec.clear();
    beammap_prior_array_center_y_arcsec.clear();
    beammap_prior_array_alignment.clear();
    if (beammap_priors_enabled) {
        if (map_grouping != "detector") {
            logger->warn("beammap priors requested but map_grouping={} (requires detector); disabling priors",
                         map_grouping);
            beammap_priors_enabled = false;
        }
        else if (!load_soft_priors()) {
            logger->warn("beammap priors failed to load; disabling prior-guided initialization");
            beammap_priors_enabled = false;
        }
    }
    calib.apt_meta["beammap_priors_enabled"] = beammap_priors_enabled;
    calib.apt_meta["beammap_priors_filepath"] = beammap_priors_filepath;
    calib.apt_meta["beammap_priors_candidate_top_n"] = beammap_priors_candidate_top_n;
    calib.apt_meta["beammap_priors_min_snr"] = beammap_priors_min_snr;
    calib.apt_meta["beammap_priors_max_d2"] = beammap_priors_max_d2;
    calib.apt_meta["beammap_priors_max_d2_iter0"] = beammap_priors_max_d2_iter0;
    calib.apt_meta["beammap_priors_max_d2_after_iter0"] = beammap_priors_max_d2_after_iter0;
    calib.apt_meta["beammap_priors_score_lambda"] = beammap_priors_score_lambda;
    calib.apt_meta["beammap_priors_score_lambda_iter0"] = beammap_priors_score_lambda_iter0;
    calib.apt_meta["beammap_priors_score_lambda_after_iter0"] = beammap_priors_score_lambda_after_iter0;
    calib.apt_meta["beammap_priors_fallback_blind"] = beammap_priors_fallback_blind;
    calib.apt_meta["beammap_priors_align_after_iter0"] = beammap_priors_align_after_iter0;
    calib.apt_meta["beammap_priors_alignment_scope"] = beammap_priors_alignment_scope;
    calib.apt_meta["beammap_priors_alignment_common_support"] = beammap_priors_alignment_common_support;
    calib.apt_meta["beammap_priors_alignment_common_support_quantile"] =
        beammap_priors_alignment_common_support_quantile;
    calib.apt_meta["beammap_priors_alignment_min_matches"] = beammap_priors_alignment_min_matches;
    calib.apt_meta["beammap_priors_alignment_max_d2"] = beammap_priors_alignment_max_d2;
    calib.apt_meta["beammap_priors_alignment_fit_rotation"] = beammap_priors_alignment_fit_rotation;
    calib.apt_meta["beammap_priors_alignment_max_rotation_deg"] = beammap_priors_alignment_max_rotation_deg;
}

template <class KidsProc, class RawObs>
void Beammap::pipeline(KidsProc &kidsproc, RawObs &rawobs) {
    // only get kids params if not simulation
    if (!telescope.sim_obs) {
        // add kids models to apt
        auto [kids_models, kids_model_header] = kidsproc.load_fit_report(rawobs);

        Eigen::Index i = 0;
        // loop through kids header
        for (const auto &h: kids_model_header) {
            std::string name = h;
            if (name=="flag") {
                name = "kids_flag";
            }
            calib.apt[name].resize(calib.n_dets);
            Eigen::Index j = 0;
            for (const auto &v: kids_models) {
                calib.apt[name].segment(j,v.rows()) = v.col(i);
                j = j + v.rows();
            }

            // search for key
            bool found = false;
            for (const auto &key: calib.apt_header_keys){
                if (key==name) {
                    found = true;
                }
            }
            // if not found, push back placeholder
            if (!found) {
                calib.apt_header_keys.push_back(name);
                calib.apt_header_units[name] = "N/A";
            }

            // detector orientation
            calib.apt_meta[name].push_back("units: N/A");
            calib.apt_meta[name].push_back(name);
            i++;
        }
    }

    // run timestream pipeline
    rtcproc.kernel.clear_source_centers();
    timestream_pipeline(kidsproc, rawobs);

    // placeholder vectors of size nscans for grppi maps
    scan_in_vec.resize(ptcs0.size());
    std::iota(scan_in_vec.begin(), scan_in_vec.end(), 0);
    scan_out_vec.resize(ptcs0.size());

    // placeholder vectors of size ndet for grppi maps
    det_in_vec.resize(n_maps);
    std::iota(det_in_vec.begin(), det_in_vec.end(), 0);
    det_out_vec.resize(n_maps);

    // run iterative pipeline
    loop_pipeline(kidsproc, rawobs);
}

template <class KidsProc, class RawObs>
void Beammap::timestream_pipeline(KidsProc &kidsproc, RawObs &rawobs, bool write_outputs) {
    using input_t = TCData<TCDataKind::RTC, Eigen::MatrixXd>;
    // initialize number of completed scans
    n_scans_done = 0;

    // progress bar
    tula::logging::progressbar pb(
        [&](const auto &msg) { logger->info("{}", msg); }, 100, "RTC progress ");

    // grppi generator function. gets time chunk data from files sequentially and passes them to grppi::farm
    grppi::pipeline(tula::grppi_utils::dyn_ex(parallel_policy),
        [&]() -> std::optional<input_t> {

            // variable to hold current scan
            static int scan = 0;
            // loop through scans
            while (scan < telescope.scan_indices.cols()) {
                // update progress bar
                pb.count(telescope.scan_indices.cols(), 1);

                // create rtcdata
                TCData<TCDataKind::RTC, Eigen::MatrixXd> rtcdata;
                // get scan indices
                rtcdata.scan_indices.data = telescope.scan_indices.col(scan);
                // current scan
                rtcdata.index.data = scan;

                // current length of outer scans
                Eigen::Index sl = rtcdata.scan_indices.data(3) - rtcdata.scan_indices.data(2) + 1;

                // get raw tod from files
                if (!interp_over_gaps) {
                    rtcdata.scans.data = kidsproc.populate_rtc_from_rawobs(rawobs, scan, telescope.scan_indices,
                                                                           start_indices, end_indices,
                                                                           sl, calib.n_dets, tod_type);
                }
                else {
                    auto scan_rawobs = kidsproc.load_rawobs_gaps(rawobs, scan, telescope.scan_indices, start_indices,
                                                                 t_common, nw_times, 1 / (2 * telescope.fsmp));
                    rtcdata.scans.data = kidsproc.populate_rtc_gaps(scan_rawobs, t_common, nw_times, masks, scan, 1 / (2 * telescope.fsmp),
                                                                telescope.scan_indices, sl, calib.n_dets, tod_type);
                    std::vector<kids::KidsData<kids::KidsDataKind::RawTimeStream>>().swap(scan_rawobs);
                }

                // increment scan
                scan++;
                // return rtcdata
                return rtcdata;
            }
            // reset scan to zero for each obs
            scan = 0;
            return {};
        },
        // run the raw time chunk processing
        run_timestream(kidsproc, write_outputs));
}

template <class KidsProc>
auto Beammap::run_timestream(KidsProc &kidsproc, bool write_outputs) {
    auto scans_done_mutex = std::make_shared<std::mutex>();

    struct OrderedWriter {
        std::mutex mutex;
        std::condition_variable cv;
        Eigen::Index next = 0;
        void wait_turn(Eigen::Index idx) {
            std::unique_lock<std::mutex> lk(mutex);
            cv.wait(lk, [&] { return idx == next; });
        }
        void advance() {
            std::lock_guard<std::mutex> lk(mutex);
            ++next;
            cv.notify_all();
        }
    };

    const bool write_rtc = write_outputs && run_tod_output && !tod_filename.empty() &&
        (tod_output_type == "rtc" || tod_output_type == "both");
    const bool write_rtcdiag = write_outputs && !rtcdiag_filename.empty();
    auto rtc_writer = write_rtc ? std::make_shared<OrderedWriter>() : nullptr;
    auto rtcdiag_writer = write_rtcdiag ? std::make_shared<OrderedWriter>() : nullptr;

    auto farm = grppi::farm(n_threads,[&, scans_done_mutex, rtc_writer, rtcdiag_writer,
                                       write_rtc, write_rtcdiag](auto &rtcdata) -> TCData<TCDataKind::PTC,Eigen::MatrixXd> {

        // allocate up bitwise timestream flags
        rtcdata.flags2.data.setConstant(timestream::TimestreamFlags::Good);

        // starting index for scan (outer scan)
        Eigen::Index si = rtcdata.scan_indices.data(2);

        // current length of outer scans
        Eigen::Index sl = rtcdata.scan_indices.data(3) - rtcdata.scan_indices.data(2) + 1;

        // copy scan's telescope vectors
        for (const auto& x: telescope.tel_data) {
            rtcdata.tel_data.data[x.first] = telescope.tel_data[x.first].segment(si,sl);
        }

        // copy pointing offsets
        for (const auto& [axis,offset]: pointing_offsets_arcsec) {
            rtcdata.pointing_offsets_arcsec.data[axis] = offset.segment(si,sl);
        }

        // get hwpr
        if (rtcproc.run_polarization) {
            rtcdata.hwpr_angle.data = calib.hwpr_angle.segment(si + hwpr_start_indices, sl);
        }

        // set up flags
        rtcdata.flags.data.resize(rtcdata.scans.data.rows(), rtcdata.scans.data.cols());
        rtcdata.flags.data.setConstant(0);

        if (interp_over_gaps) {
            for (auto const& [key, val] : calib.nw_limits) {
                auto mask_it = nw_masks.find(key);
                if (mask_it == nw_masks.end()) {
                    logger->error("missing gap mask for nw {}; cannot apply gap flagging", key);
                    std::exit(EXIT_FAILURE);
                }
                auto& mask = mask_it->second;

                Eigen::Index start = std::get<0>(calib.nw_limits[key]);
                Eigen::Index end = std::get<1>(calib.nw_limits[key]) - 1;

                for (int j = 0; j < rtcdata.flags.data.rows(); ++j) {
                    int start_index = j;
                    int size = 1;
                    if (rtcproc.filter_edge_guard.context_samples > 0) {
                        const int context = static_cast<int>(rtcproc.filter_edge_guard.context_samples);
                        start_index = std::max(0, j - context);
                        int end_index = std::min(j + context, static_cast<int>(rtcdata.flags.data.rows() - 1));
                        size = end_index - start_index + 1;
                    }
                    if (mask(j + si) == 0) {
                        rtcdata.flags.data.block(start_index, start, size, end - start + 1).setOnes();
                    }
                }
                logger->debug("{}/{} gaps flagged", rtcdata.flags.data.col(start).template cast<int>().sum(), rtcdata.flags.data.rows());
            }
        }

        // create PTCData
        TCData<TCDataKind::PTC,Eigen::MatrixXd> ptcdata;
        TCData<TCDataKind::RTC,Eigen::MatrixXd> rtc_outer_output;
        const auto rtc_scan_row = tod_output_scan_row(rtcdata.index.data, "rtc");
        const bool write_this_rtc = write_rtc && rtc_scan_row >= 0;
        auto *rtc_outer_output_ptr =
            (write_this_rtc && rtcproc.tod_output_outer) ? &rtc_outer_output : nullptr;

        {
            std::lock_guard<std::mutex> lk(*scans_done_mutex);
            logger->info("starting scan {}. {}/{} scans completed", rtcdata.index.data + 1, n_scans_done,
                         telescope.scan_indices.cols());
        }

        // run rtcproc
        logger->info("raw time chunk processing for scan {}", rtcdata.index.data + 1);
        auto map_indices = rtcproc.run(rtcdata, ptcdata, calib, telescope, omb.pixel_size_rad, map_grouping,
                                       rtc_outer_output_ptr);

        if (map_grouping!="detector") {
            // remove flagged detectors
            rtcproc.remove_flagged_dets(ptcdata, calib.apt);
        }

        // remove outliers before cleaning
        auto calib_scan = rtcproc.remove_bad_dets(ptcdata, calib, map_grouping);

        // remove duplicate tones
        if (!telescope.sim_obs) {
            calib_scan = rtcproc.remove_nearby_tones(ptcdata, calib_scan, map_grouping);
        }

        if (write_rtcdiag) {
            rtcdiag_writer->wait_turn(ptcdata.index.data);
            logger->info("writing rtc diagnostics sidecar chunk");
            rtcproc.append_diag_to_netcdf(ptcdata, rtcdiag_filename, calib_scan, ptcdata.index.data);
            rtcdiag_writer->advance();
        }

        // write rtc timestreams
        if (write_this_rtc) {
            rtc_writer->wait_turn(rtc_scan_row);
            if (rtcproc.tod_output_outer) {
                logger->info("writing outer raw time chunk");
                rtcproc.append_to_netcdf(rtc_outer_output, tod_filename["rtc"], map_grouping, telescope.pixel_axes,
                                         rtc_outer_output.pointing_offsets_arcsec.data, calib, true, rtc_scan_row);
            }
            else {
                logger->info("writing raw time chunk");
                rtcproc.append_to_netcdf(ptcdata, tod_filename["rtc"], map_grouping, telescope.pixel_axes,
                                         ptcdata.pointing_offsets_arcsec.data, calib_scan, true, rtc_scan_row);
            }
            rtc_writer->advance();
        }
        rtcproc.clear_cached_diagnostics(ptcdata.index.data);

        // store indices for each ptcdata
        ptcdata.map_indices.data = std::move(map_indices);

        // move out ptcdata the PTCData vector at corresponding index
        ptcs0.at(ptcdata.index.data) = std::move(ptcdata);
        calib_scans0.at(ptcdata.index.data) = std::move(calib_scan);

        // increment number of completed scans
        {
            std::lock_guard<std::mutex> lk(*scans_done_mutex);
            n_scans_done++;
            logger->info("done with scan {}. {}/{} scans completed", ptcdata.index.data + 1, n_scans_done,
                         telescope.scan_indices.cols());
        }

        return ptcdata;
    });

    return farm;
}

template <class KidsProc, class RawObs>
void Beammap::loop_pipeline(KidsProc &kidsproc, RawObs &rawobs) {
    // run iterative stage
    run_loop(kidsproc, rawobs);
    ptcproc.fruit_loops_kernel_feedback_enabled = true;

    // write map summary
    if (verbose_mode) {
        write_map_summary(omb);
    }

    // empty initial ptcdata vector to save memory
    ptcs0.clear();

    // set to input parallel policy
    parallel_policy = omb.parallel_policy;

    if (map_grouping=="detector") {
        logger->info("calculating sensitivity");
        // parallelize on detectors
        grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto i) {
            Eigen::MatrixXd det_sens, noise_flux;
            // calc sensitivity within psd freq range
            calc_sensitivity(ptcs, det_sens, noise_flux, telescope.d_fsmp, i, {sens_psd_limits_Hz(0), sens_psd_limits_Hz(1)});
            // copy into apt table
            calib.apt["sens"](i) = tula::alg::median(det_sens);

            return 0;
        });
    }

    // apt and sensitivity only relevant if beammapping
    if (map_grouping=="detector") {
        // rescale fit params from pixel to on-sky units
        calib.apt["amp"] = params.col(0);
        calib.apt["x_t"] = RAD_TO_ASEC*omb.pixel_size_rad*(params.col(1).array() - (omb.n_cols - 1)/2.0);
        calib.apt["y_t"] = RAD_TO_ASEC*omb.pixel_size_rad*(params.col(2).array() - (omb.n_rows - 1)/2.0);
        calib.apt["a_fwhm"] = RAD_TO_ASEC*STD_TO_FWHM*omb.pixel_size_rad*(params.col(3));
        calib.apt["b_fwhm"] = RAD_TO_ASEC*STD_TO_FWHM*omb.pixel_size_rad*(params.col(4));
        calib.apt["angle"] = params.col(5);

        // rescale fit errors from pixel to on-sky units
        calib.apt["amp_err"] = perrors.col(0);
        calib.apt["x_t_err"] = RAD_TO_ASEC*omb.pixel_size_rad*(perrors.col(1));
        calib.apt["y_t_err"] = RAD_TO_ASEC*omb.pixel_size_rad*(perrors.col(2));
        calib.apt["a_fwhm_err"] = RAD_TO_ASEC*STD_TO_FWHM*omb.pixel_size_rad*(perrors.col(3));
        calib.apt["b_fwhm_err"] = RAD_TO_ASEC*STD_TO_FWHM*omb.pixel_size_rad*(perrors.col(4));
        calib.apt["angle_err"] = perrors.col(5);

        // add convergence iteration to apt table
        calib.apt["converge_iter"] = converge_iter.cast<double> ();

        if (rfi_mask_samples_flagged.size() == calib.n_dets) {
            calib.apt["rfi_masked_samples"] = rfi_mask_samples_flagged.cast<double>();
        }
        if (rfi_mask_scans_flagged.size() == calib.n_dets) {
            calib.apt["rfi_masked_scans"] = rfi_mask_scans_flagged.cast<double>();
        }
        if (scan_band_mask_samples_flagged.size() == calib.n_dets) {
            calib.apt["scan_band_masked_samples"] = scan_band_mask_samples_flagged.cast<double>();
        }
        if (scan_band_mask_rows_flagged.size() == calib.n_dets) {
            calib.apt["scan_band_masked_rows"] = scan_band_mask_rows_flagged.cast<double>();
        }
        if (scan_band_mask_edge_code.size() == calib.n_dets) {
            calib.apt["scan_band_masked_edge"] = scan_band_mask_edge_code.cast<double>();
        }
        if (scan_band_mask_rejected.size() == calib.n_dets) {
            calib.apt["scan_band_mask_rejected"] = scan_band_mask_rejected.cast<double>();
        }
        if (beammap_rfi_mask_enabled &&
            rfi_mask_samples_flagged.size() == calib.n_dets &&
            rfi_mask_scans_flagged.size() == calib.n_dets) {
            const Eigen::Index n_det_masked = (rfi_mask_scans_flagged.array() > 0).count();
            logger->info("beammap rfi mask summary: {} detectors affected, {} total samples masked",
                         n_det_masked, static_cast<long long>(rfi_mask_samples_flagged.cast<double>().sum()));
        }

        if (fit_diag_bound_nhit.size() == n_maps &&
            fit_diag_hit_lower.rows() == n_maps && fit_diag_hit_upper.rows() == n_maps &&
            fit_diag_hit_lower.cols() >= 6 && fit_diag_hit_upper.cols() >= 6) {
            const Eigen::Index n_bound_any = (fit_diag_bound_nhit.array() > 0).count();
            Eigen::VectorXi low_hits = fit_diag_hit_lower.colwise().sum().transpose();
            Eigen::VectorXi high_hits = fit_diag_hit_upper.colwise().sum().transpose();
            logger->info(
                "beammap final bound-hit summary: any_hit={}/{} amp(lo/hi)={}/{} x(lo/hi)={}/{} y(lo/hi)={}/{} a(lo/hi)={}/{} b(lo/hi)={}/{} angle(lo/hi)={}/{}",
                n_bound_any, n_maps,
                low_hits(0), high_hits(0),
                low_hits(1), high_hits(1),
                low_hits(2), high_hits(2),
                low_hits(3), high_hits(3),
                low_hits(4), high_hits(4),
                low_hits(5), high_hits(5));
        }

        // flag detectors in apt based on config limits
        set_apt_flags();

        // subtract reference detector position and derotate
        process_apt();
        apply_final_network_position_flags();
        update_final_prior_match_diagnostics();
        if (final_prior_slot_index_diag.size() == calib.n_dets) {
            calib.apt["final_prior_slot_index"] = final_prior_slot_index_diag.cast<double>();
        }
        if (final_prior_d2_diag.size() == calib.n_dets) {
            calib.apt["final_prior_d2"] = final_prior_d2_diag;
        }
        calib.setup();
        for (Eigen::Index i = 0; i < calib.n_arrays; ++i) {
            Eigen::Index array = calib.arrays(i);
            std::string array_name = toltec_io.array_name_map[array];
            beammap_fluxes_MJy_Sr[array_name] =
                mJY_ASEC_to_MJY_SR * (beammap_fluxes_mJy_beam[array_name]) / calib.array_beam_areas[array];
        }
        log_final_network_qc_summary();

        // add final apt table to timestream files
        if (run_tod_output && !tod_filename.empty()) {
            // vectors to hold tangent plane pointing for all ptcs (n_chunks x [n_pts x n_dets])
            std::vector<Eigen::MatrixXd> lat, lon;

            // recalculate tangent plane pointing for tod output
            for (Eigen::Index i=0; i<ptcs.size(); ++i) {
                // tangent plane pointing for each detector
                Eigen::MatrixXd ptc_lat(ptcs[i].scans.data.rows(), ptcs[i].scans.data.cols());
                Eigen::MatrixXd ptc_lon(ptcs[i].scans.data.rows(), ptcs[i].scans.data.cols());
                // loop through detectors
                grppi::map(tula::grppi_utils::dyn_ex(parallel_policy), det_in_vec, det_out_vec, [&](auto j) {
                    // det indices
                    auto det_index = j;
                    double az_off = calib.apt["x_t"](det_index);
                    double el_off = calib.apt["y_t"](det_index);

                    // get tangent pointing
                    auto [det_lat, det_lon] = engine_utils::calc_det_pointing(ptcs[i].tel_data.data, az_off,
                                                                              el_off, telescope.pixel_axes,
                                                                              ptcs[i].pointing_offsets_arcsec.data,
                                                                              map_grouping, true);
                    ptc_lat.col(j) = std::move(det_lat);
                    ptc_lon.col(j) = std::move(det_lon);

                    return 0;
                });
                lat.push_back(std::move(ptc_lat));
                lon.push_back(std::move(ptc_lon));
            }

            logger->info("adding final apt and detector pointing to tod files");
            // loop through tod files
            for (const auto & [key, val]: tod_filename) {
                netCDF::NcFile fo(val, netCDF::NcFile::write);
                // overwrite apt table
                for (auto const& x: calib.apt) {
                    if (x.first!="flag2") {
                        // start index for apt table
                        std::vector<std::size_t> start_index_apt = {0};
                        // size for apt
                        std::vector<std::size_t> size_apt = {1};
                        netCDF::NcVar apt_v = fo.getVar("apt_" + x.first);
                        if (!apt_v.isNull()) {
                            for (std::size_t i=0; i< TULA_SIZET(calib.n_dets); ++i) {
                                start_index_apt[0] = i;
                                apt_v.putVar(start_index_apt, size_apt, &calib.apt[x.first](i));
                            }
                        }
                    }
                }

                // detector tangent plane pointing
                netCDF::NcVar det_lat_v = fo.getVar("det_lat");
                netCDF::NcVar det_lon_v = fo.getVar("det_lon");

                // detector absolute pointing
                netCDF::NcVar det_ra_v = fo.getVar("det_ra");
                netCDF::NcVar det_dec_v = fo.getVar("det_dec");
                const bool write_tangent_pointing = !det_lat_v.isNull() && !det_lon_v.isNull();
                const bool write_abs_pointing = telescope.pixel_axes == "radec" &&
                                                !det_ra_v.isNull() && !det_dec_v.isNull();
                if (!write_tangent_pointing && !write_abs_pointing) {
                    logger->debug("tod file {} has no detector pointing variables; skipping final detector pointing update", val);
                    continue;
                }

                // start indices for data
                std::vector<std::size_t> start_index = {0, 0};
                // size for data
                std::vector<std::size_t> size = {1, TULA_SIZET(calib.n_dets)};
                std::size_t k = 0;
                // loop through ptcs
                for (Eigen::Index i=0; i<lat.size(); ++i) {
                    // loop through n_pts
                    for (std::size_t j=0; j < TULA_SIZET(lat[i].rows()); ++j) {
                        start_index[0] = k;
                        k++;
                        // append detector latitudes
                        Eigen::VectorXd lat_row = lat[i].row(j);

                        // append detector longitudes
                        Eigen::VectorXd lon_row = lon[i].row(j);
                        if (write_tangent_pointing) {
                            det_lat_v.putVar(start_index, size, lat_row.data());
                            det_lon_v.putVar(start_index, size, lon_row.data());
                        }

                        if (write_abs_pointing) {
                            // get absolute pointing
                            auto [dec, ra] = engine_utils::tangent_to_abs(lat_row, lon_row, telescope.tel_header["Header.Source.Ra"](0),
                                                                          telescope.tel_header["Header.Source.Dec"](0));
                            // append detector ra
                            det_ra_v.putVar(start_index, size, ra.data());

                            // append detector dec
                            det_dec_v.putVar(start_index, size, dec.data());
                        }
                    }
                }
            }

            // empty ptcdata vector to save memory
            ptcs.clear();
        }
    }

    else {
        // calculate map psds
        logger->info("calculating map psd");
        omb.calc_map_psd();
        // calculate map histograms
        logger->info("calculating map histogram");
        omb.calc_map_hist();
    }
}

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

void Beammap::init_empirical_template_calibration_columns() {
    auto add_column = [&](const std::string &name,
                          const std::string &unit,
                          const std::string &description,
                          double fill_value) {
        calib.apt[name] = Eigen::VectorXd::Constant(calib.n_dets, fill_value);
        calib.apt_header_units[name] = unit;
        if (std::find(calib.apt_header_keys.begin(), calib.apt_header_keys.end(), name) ==
            calib.apt_header_keys.end()) {
            calib.apt_header_keys.push_back(name);
        }
        calib.apt_header_description[name] = description;
        calib.apt_meta[name].push_back("units: " + unit);
        calib.apt_meta[name].push_back(description);
    };

    add_column("cal_amp", "xs",
               "amplitude used for beammap flux calibration; empirical template when valid, Gaussian fallback otherwise",
               std::numeric_limits<double>::quiet_NaN());
    add_column("cal_amp_method", "N/A",
               "calibration amplitude method code (0 Gaussian fallback, 1 empirical template)",
               0.0);
    add_column("template_amp", "xs",
               "empirical array-template matched amplitude with fitted local offset",
               std::numeric_limits<double>::quiet_NaN());
    add_column("template_offset", "xs",
               "local offset fitted with empirical array-template amplitude",
               std::numeric_limits<double>::quiet_NaN());
    add_column("template_resid_rms", "xs",
               "weighted residual RMS for empirical-template amplitude fit",
               std::numeric_limits<double>::quiet_NaN());
    add_column("template_npix", "pix",
               "number of pixels used for empirical-template amplitude fit",
               0.0);
    add_column("template_amp_over_fit_amp", "N/A",
               "ratio of empirical-template matched amplitude to Gaussian fit amplitude",
               std::numeric_limits<double>::quiet_NaN());
    add_column("cal_amp_over_fit_amp", "N/A",
               "ratio of calibration amplitude to Gaussian fit amplitude",
               std::numeric_limits<double>::quiet_NaN());
    add_column("map_peak_amp", "xs",
               "baseline-subtracted local map peak within 8 arcsec of Gaussian fit center",
               std::numeric_limits<double>::quiet_NaN());
    add_column("map_peak_amp_over_fit_amp", "N/A",
               "ratio of local map peak amplitude to Gaussian fit amplitude",
               std::numeric_limits<double>::quiet_NaN());
    calib.apt_meta["cal_amp_method"].push_back("0: Gaussian fit amplitude fallback");
    calib.apt_meta["cal_amp_method"].push_back("1: empirical array-template matched amplitude");
}

void Beammap::calc_empirical_template_calibration() {
    auto ensure_column = [&](const std::string &name, double fill_value) {
        if (calib.apt.find(name) == calib.apt.end() ||
            calib.apt[name].size() != calib.n_dets) {
            calib.apt[name] = Eigen::VectorXd::Constant(calib.n_dets, fill_value);
        }
        else {
            calib.apt[name].setConstant(fill_value);
        }
    };

    ensure_column("cal_amp", std::numeric_limits<double>::quiet_NaN());
    ensure_column("cal_amp_method", 0.0);
    ensure_column("template_amp", std::numeric_limits<double>::quiet_NaN());
    ensure_column("template_offset", std::numeric_limits<double>::quiet_NaN());
    ensure_column("template_resid_rms", std::numeric_limits<double>::quiet_NaN());
    ensure_column("template_npix", 0.0);
    ensure_column("template_amp_over_fit_amp", std::numeric_limits<double>::quiet_NaN());
    ensure_column("cal_amp_over_fit_amp", std::numeric_limits<double>::quiet_NaN());
    ensure_column("map_peak_amp", std::numeric_limits<double>::quiet_NaN());
    ensure_column("map_peak_amp_over_fit_amp", std::numeric_limits<double>::quiet_NaN());

    const Eigen::Index n_fallback = std::min<Eigen::Index>(n_maps, std::min(calib.n_dets, params.rows()));
    if (params.cols() > 0) {
        for (Eigen::Index i = 0; i < n_fallback; ++i) {
            const double fit_amp = params(i, 0);
            calib.apt["cal_amp"](i) = fit_amp;
            calib.apt["cal_amp_method"](i) = 0.0;
            if (std::isfinite(fit_amp) && fit_amp > 0.0) {
                calib.apt["cal_amp_over_fit_amp"](i) = 1.0;
            }
        }
    }

    if (map_grouping != "detector" ||
        static_cast<Eigen::Index>(omb.signal.size()) != n_maps ||
        static_cast<Eigen::Index>(omb.weight.size()) != n_maps ||
        omb.pixel_size_rad <= 0.0) {
        return;
    }

    const double pix_to_arcsec = RAD_TO_ASEC * omb.pixel_size_rad;
    if (!std::isfinite(pix_to_arcsec) || pix_to_arcsec <= 0.0) {
        return;
    }

    const Eigen::Index template_radius_pix =
        std::max<Eigen::Index>(8, static_cast<Eigen::Index>(std::ceil(40.0 / pix_to_arcsec)));
    const Eigen::Index match_radius_pix =
        std::max<Eigen::Index>(4, static_cast<Eigen::Index>(std::ceil(35.0 / pix_to_arcsec)));
    const Eigen::Index peak_radius_pix =
        std::max<Eigen::Index>(2, static_cast<Eigen::Index>(std::ceil(8.0 / pix_to_arcsec)));
    const Eigen::Index template_peak_radius_pix =
        std::max<Eigen::Index>(1, static_cast<Eigen::Index>(std::ceil(4.0 / pix_to_arcsec)));
    const Eigen::Index side = 2 * template_radius_pix + 1;
    const Eigen::Index center = template_radius_pix;
    constexpr Eigen::Index min_template_detectors = 25;
    constexpr Eigen::Index max_template_detectors = 500;
    constexpr double min_template_snr = 20.0;
    constexpr double min_template_value = 0.015;

    auto median_vector = [](std::vector<double> values) -> double {
        values.erase(std::remove_if(values.begin(), values.end(),
                                    [](double v) { return !std::isfinite(v); }),
                     values.end());
        if (values.empty()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        Eigen::Map<Eigen::VectorXd> vec(values.data(), static_cast<Eigen::Index>(values.size()));
        return tula::alg::median(vec);
    };

    auto bilinear_sample = [](const Eigen::MatrixXd &map, double row, double col) -> double {
        if (!std::isfinite(row) || !std::isfinite(col)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const Eigen::Index r0 = static_cast<Eigen::Index>(std::floor(row));
        const Eigen::Index c0 = static_cast<Eigen::Index>(std::floor(col));
        const Eigen::Index r1 = r0 + 1;
        const Eigen::Index c1 = c0 + 1;
        if (r0 < 0 || c0 < 0 || r1 >= map.rows() || c1 >= map.cols()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double v00 = map(r0, c0);
        const double v01 = map(r0, c1);
        const double v10 = map(r1, c0);
        const double v11 = map(r1, c1);
        if (!std::isfinite(v00) || !std::isfinite(v01) ||
            !std::isfinite(v10) || !std::isfinite(v11)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double fr = row - static_cast<double>(r0);
        const double fc = col - static_cast<double>(c0);
        const double v0 = (1.0 - fc) * v00 + fc * v01;
        const double v1 = (1.0 - fc) * v10 + fc * v11;
        return (1.0 - fr) * v0 + fr * v1;
    };

    auto edge_baseline = [&](const Eigen::MatrixXd &map, double row0, double col0) -> double {
        std::vector<double> edge;
        edge.reserve(static_cast<std::size_t>(4 * side));
        for (Eigen::Index k = -template_radius_pix; k <= template_radius_pix; ++k) {
            edge.push_back(bilinear_sample(map, row0 - template_radius_pix, col0 + k));
            edge.push_back(bilinear_sample(map, row0 + template_radius_pix, col0 + k));
            edge.push_back(bilinear_sample(map, row0 + k, col0 - template_radius_pix));
            edge.push_back(bilinear_sample(map, row0 + k, col0 + template_radius_pix));
        }
        return median_vector(std::move(edge));
    };

    auto local_peak = [&](const Eigen::MatrixXd &map, double row0, double col0, double baseline) -> double {
        double peak = -std::numeric_limits<double>::infinity();
        for (Eigen::Index dr = -peak_radius_pix; dr <= peak_radius_pix; ++dr) {
            for (Eigen::Index dc = -peak_radius_pix; dc <= peak_radius_pix; ++dc) {
                if (dr * dr + dc * dc > peak_radius_pix * peak_radius_pix) {
                    continue;
                }
                const double value = bilinear_sample(map, row0 + dr, col0 + dc);
                if (std::isfinite(value)) {
                    peak = std::max(peak, value - baseline);
                }
            }
        }
        return std::isfinite(peak) ? peak : std::numeric_limits<double>::quiet_NaN();
    };

    auto extract_normalized_cut = [&](Eigen::Index map_index,
                                      Eigen::MatrixXd &cut,
                                      double &peak_amp) -> bool {
        if (map_index < 0 || map_index >= n_maps ||
            map_index >= params.rows() || params.cols() < 3) {
            return false;
        }
        const double amp = params(map_index, 0);
        const double col0 = params(map_index, 1);
        const double row0 = params(map_index, 2);
        if (!std::isfinite(amp) || amp <= 0.0 ||
            !std::isfinite(row0) || !std::isfinite(col0)) {
            return false;
        }
        const double baseline = edge_baseline(omb.signal[map_index], row0, col0);
        if (!std::isfinite(baseline)) {
            return false;
        }
        peak_amp = local_peak(omb.signal[map_index], row0, col0, baseline);
        if (!std::isfinite(peak_amp) || peak_amp <= 0.0) {
            return false;
        }
        cut.resize(side, side);
        cut.setConstant(std::numeric_limits<double>::quiet_NaN());
        for (Eigen::Index rr = 0; rr < side; ++rr) {
            const Eigen::Index dr = rr - center;
            for (Eigen::Index cc = 0; cc < side; ++cc) {
                const Eigen::Index dc = cc - center;
                const double value = bilinear_sample(omb.signal[map_index], row0 + dr, col0 + dc);
                if (std::isfinite(value)) {
                    cut(rr, cc) = (value - baseline) / peak_amp;
                }
            }
        }
        return true;
    };

    struct TemplateCandidate {
        Eigen::Index map_index = -1;
        double shape_score = std::numeric_limits<double>::infinity();
        double snr = 0.0;
    };

    struct ArrayTemplate {
        bool valid = false;
        Eigen::MatrixXd shape;
        Eigen::Index n_detectors = 0;
    };

    std::map<int, ArrayTemplate> templates;

    for (Eigen::Index arr_i = 0; arr_i < calib.n_arrays; ++arr_i) {
        const int array = static_cast<int>(calib.arrays(arr_i));
        std::vector<double> a_values;
        std::vector<double> b_values;
        for (Eigen::Index i = 0; i < n_maps; ++i) {
            if (i >= calib.n_dets || calib.apt["flag"](i) != 0 || !good_fits(i)) {
                continue;
            }
            if (static_cast<int>(std::lround(calib.apt["array"](i))) != array) {
                continue;
            }
            if (!std::isfinite(calib.apt["a_fwhm"](i)) || !std::isfinite(calib.apt["b_fwhm"](i)) ||
                calib.apt["a_fwhm"](i) <= 0.0 || calib.apt["b_fwhm"](i) <= 0.0) {
                continue;
            }
            a_values.push_back(calib.apt["a_fwhm"](i));
            b_values.push_back(calib.apt["b_fwhm"](i));
        }
        const double med_a = median_vector(a_values);
        const double med_b = median_vector(b_values);
        if (!std::isfinite(med_a) || !std::isfinite(med_b) || med_a <= 0.0 || med_b <= 0.0) {
            continue;
        }

        std::vector<TemplateCandidate> candidates;
        for (Eigen::Index i = 0; i < n_maps; ++i) {
            if (i >= calib.n_dets || calib.apt["flag"](i) != 0 || !good_fits(i)) {
                continue;
            }
            if (static_cast<int>(std::lround(calib.apt["array"](i))) != array) {
                continue;
            }
            if (fit_diag_bound_nhit.size() == n_maps && fit_diag_bound_nhit(i) > 0) {
                continue;
            }
            const double rms = calc_map_support_stddev(i, true);
            const double snr = (std::isfinite(rms) && rms > 0.0 && std::isfinite(params(i, 0)))
                                   ? params(i, 0) / rms
                                   : 0.0;
            if (!std::isfinite(snr) || snr < min_template_snr) {
                continue;
            }
            const double a = calib.apt["a_fwhm"](i);
            const double b = calib.apt["b_fwhm"](i);
            if (!std::isfinite(a) || !std::isfinite(b) || a <= 0.0 || b <= 0.0) {
                continue;
            }
            const double shape_score = std::abs(a - med_a) / med_a + std::abs(b - med_b) / med_b;
            candidates.push_back({i, shape_score, snr});
        }

        std::sort(candidates.begin(), candidates.end(),
                  [](const TemplateCandidate &lhs, const TemplateCandidate &rhs) {
                      if (lhs.shape_score == rhs.shape_score) {
                          return lhs.snr > rhs.snr;
                      }
                      return lhs.shape_score < rhs.shape_score;
                  });
        if (static_cast<Eigen::Index>(candidates.size()) > max_template_detectors) {
            candidates.resize(static_cast<std::size_t>(max_template_detectors));
        }

        std::vector<Eigen::MatrixXd> cuts;
        cuts.reserve(candidates.size());
        for (const auto &candidate : candidates) {
            Eigen::MatrixXd cut;
            double peak_amp = std::numeric_limits<double>::quiet_NaN();
            if (extract_normalized_cut(candidate.map_index, cut, peak_amp)) {
                cuts.push_back(std::move(cut));
            }
        }

        if (static_cast<Eigen::Index>(cuts.size()) < min_template_detectors) {
            logger->warn(
                "beammap empirical template skipped for array={} candidates={} usable={} min_required={}",
                toltec_io.array_name_map[array], candidates.size(), cuts.size(), min_template_detectors);
            continue;
        }

        Eigen::MatrixXd templ(side, side);
        templ.setConstant(std::numeric_limits<double>::quiet_NaN());
        std::vector<double> values;
        values.reserve(cuts.size());
        for (Eigen::Index rr = 0; rr < side; ++rr) {
            for (Eigen::Index cc = 0; cc < side; ++cc) {
                values.clear();
                for (const auto &cut : cuts) {
                    const double value = cut(rr, cc);
                    if (std::isfinite(value)) {
                        values.push_back(value);
                    }
                }
                if (!values.empty()) {
                    templ(rr, cc) = median_vector(values);
                }
            }
        }

        double template_peak = -std::numeric_limits<double>::infinity();
        for (Eigen::Index rr = 0; rr < side; ++rr) {
            const Eigen::Index dr = rr - center;
            for (Eigen::Index cc = 0; cc < side; ++cc) {
                const Eigen::Index dc = cc - center;
                if (dr * dr + dc * dc > template_peak_radius_pix * template_peak_radius_pix) {
                    continue;
                }
                const double value = templ(rr, cc);
                if (std::isfinite(value)) {
                    template_peak = std::max(template_peak, value);
                }
            }
        }
        if (!std::isfinite(template_peak) || template_peak <= 0.0) {
            logger->warn("beammap empirical template skipped for array={} due to invalid template peak={:.4g}",
                         toltec_io.array_name_map[array], template_peak);
            continue;
        }
        templ.array() /= template_peak;
        templates[array] = {true, std::move(templ), static_cast<Eigen::Index>(cuts.size())};
        logger->info("beammap empirical template built for array={} using {} detectors",
                     toltec_io.array_name_map[array], cuts.size());
    }

    auto solve_template = [&](Eigen::Index map_index,
                              const Eigen::MatrixXd &templ,
                              double &amp,
                              double &offset,
                              double &resid_rms,
                              Eigen::Index &npix) -> bool {
        amp = std::numeric_limits<double>::quiet_NaN();
        offset = std::numeric_limits<double>::quiet_NaN();
        resid_rms = std::numeric_limits<double>::quiet_NaN();
        npix = 0;
        if (map_index < 0 || map_index >= n_maps ||
            map_index >= params.rows() || params.cols() < 3) {
            return false;
        }
        const double col0 = params(map_index, 1);
        const double row0 = params(map_index, 2);
        if (!std::isfinite(row0) || !std::isfinite(col0)) {
            return false;
        }
        const double baseline = edge_baseline(omb.signal[map_index], row0, col0);
        if (!std::isfinite(baseline)) {
            return false;
        }
        const double peak_amp = local_peak(omb.signal[map_index], row0, col0, baseline);
        if (std::isfinite(peak_amp)) {
            calib.apt["map_peak_amp"](map_index) = peak_amp;
            if (std::isfinite(params(map_index, 0)) && params(map_index, 0) > 0.0) {
                calib.apt["map_peak_amp_over_fit_amp"](map_index) = peak_amp / params(map_index, 0);
            }
        }

        std::vector<double> y;
        std::vector<double> t;
        std::vector<double> w;
        y.reserve(static_cast<std::size_t>(side * side));
        t.reserve(static_cast<std::size_t>(side * side));
        w.reserve(static_cast<std::size_t>(side * side));

        for (Eigen::Index rr = 0; rr < side; ++rr) {
            const Eigen::Index dr = rr - center;
            for (Eigen::Index cc = 0; cc < side; ++cc) {
                const Eigen::Index dc = cc - center;
                if (dr * dr + dc * dc > match_radius_pix * match_radius_pix) {
                    continue;
                }
                const double template_value = templ(rr, cc);
                if (!std::isfinite(template_value) || std::abs(template_value) <= min_template_value) {
                    continue;
                }
                const double row = row0 + static_cast<double>(dr);
                const double col = col0 + static_cast<double>(dc);
                const double signal_value = bilinear_sample(omb.signal[map_index], row, col);
                const double weight_value = bilinear_sample(omb.weight[map_index], row, col);
                if (!std::isfinite(signal_value) || !std::isfinite(weight_value) || weight_value <= 0.0) {
                    continue;
                }
                y.push_back(signal_value - baseline);
                t.push_back(template_value);
                w.push_back(weight_value);
            }
        }

        npix = static_cast<Eigen::Index>(y.size());
        if (npix < 30) {
            return false;
        }

        std::vector<double> w_sorted = w;
        std::sort(w_sorted.begin(), w_sorted.end());
        const std::size_t q95_index =
            std::min<std::size_t>(w_sorted.size() - 1,
                                  static_cast<std::size_t>(std::floor(0.95 * (w_sorted.size() - 1))));
        const double w_cap = w_sorted[q95_index];

        double sw = 0.0;
        double st = 0.0;
        double sy = 0.0;
        double stt = 0.0;
        double sty = 0.0;
        for (std::size_t k = 0; k < y.size(); ++k) {
            const double wk = (std::isfinite(w_cap) && w_cap > 0.0) ? std::min(w[k], w_cap) : w[k];
            sw += wk;
            st += wk * t[k];
            sy += wk * y[k];
            stt += wk * t[k] * t[k];
            sty += wk * t[k] * y[k];
        }
        const double det = stt * sw - st * st;
        if (!std::isfinite(det) || std::abs(det) <= std::numeric_limits<double>::epsilon()) {
            return false;
        }
        amp = (sty * sw - sy * st) / det;
        offset = (stt * sy - st * sty) / det;
        if (!std::isfinite(amp) || !std::isfinite(offset) || amp <= 0.0) {
            return false;
        }

        double resid2_sum = 0.0;
        double weight_sum = 0.0;
        for (std::size_t k = 0; k < y.size(); ++k) {
            const double wk = (std::isfinite(w_cap) && w_cap > 0.0) ? std::min(w[k], w_cap) : w[k];
            const double resid = y[k] - (amp * t[k] + offset);
            resid2_sum += wk * resid * resid;
            weight_sum += wk;
        }
        if (weight_sum > 0.0) {
            resid_rms = std::sqrt(resid2_sum / weight_sum);
        }
        return std::isfinite(resid_rms);
    };

    Eigen::Index n_template_amp = 0;
    Eigen::Index n_template_fallback = 0;
    for (Eigen::Index i = 0; i < n_maps; ++i) {
        if (i >= calib.n_dets) {
            continue;
        }
        const int array = static_cast<int>(std::lround(calib.apt["array"](i)));
        const double fit_amp = params(i, 0);
        calib.apt["cal_amp"](i) = fit_amp;
        calib.apt["cal_amp_method"](i) = 0.0;
        if (std::isfinite(fit_amp) && fit_amp > 0.0) {
            calib.apt["cal_amp_over_fit_amp"](i) = 1.0;
        }

        auto templ_it = templates.find(array);
        if (templ_it == templates.end() || !templ_it->second.valid) {
            n_template_fallback++;
            continue;
        }
        double template_amp = std::numeric_limits<double>::quiet_NaN();
        double template_offset = std::numeric_limits<double>::quiet_NaN();
        double template_resid_rms = std::numeric_limits<double>::quiet_NaN();
        Eigen::Index template_npix = 0;
        if (!solve_template(i, templ_it->second.shape, template_amp, template_offset,
                            template_resid_rms, template_npix)) {
            n_template_fallback++;
            continue;
        }

        calib.apt["template_amp"](i) = template_amp;
        calib.apt["template_offset"](i) = template_offset;
        calib.apt["template_resid_rms"](i) = template_resid_rms;
        calib.apt["template_npix"](i) = static_cast<double>(template_npix);
        if (std::isfinite(fit_amp) && fit_amp > 0.0) {
            calib.apt["template_amp_over_fit_amp"](i) = template_amp / fit_amp;
        }

        calib.apt["cal_amp"](i) = template_amp;
        calib.apt["cal_amp_method"](i) = 1.0;
        if (std::isfinite(fit_amp) && fit_amp > 0.0) {
            calib.apt["cal_amp_over_fit_amp"](i) = template_amp / fit_amp;
        }
        n_template_amp++;
    }

    logger->info("beammap empirical-template calibration amplitudes: template={} fallback={}",
                 n_template_amp, n_template_fallback);
}

Beammap::RFIMaskScanSummary Beammap::apply_rfi_sample_mask(TCData<TCDataKind::PTC,Eigen::MatrixXd> &ptc) {
    RFIMaskScanSummary summary;
    if (!beammap_rfi_mask_enabled) {
        return summary;
    }

    const Eigen::Index n_samples = ptc.scans.data.rows();
    const Eigen::Index n_dets = ptc.scans.data.cols();
    if (n_samples < 4 || n_dets <= 0 || ptc.flags.data.rows() != n_samples || ptc.flags.data.cols() != n_dets) {
        return summary;
    }

    const Eigen::Index block_size = std::max<Eigen::Index>(8, beammap_rfi_mask_block_size_samples);
    const Eigen::Index min_good = std::max<Eigen::Index>(4, std::min<Eigen::Index>(beammap_rfi_mask_min_good_samples, block_size));
    const int dilate_blocks = std::max(0, beammap_rfi_mask_dilate_blocks);
    const double sigma_threshold = std::max(1.0, beammap_rfi_mask_sigma_threshold);
    const double sigma_floor = std::max(0.0, beammap_rfi_mask_sigma_floor);
    const double max_flagged_fraction = std::clamp(beammap_rfi_mask_max_flagged_fraction, 0.0, 1.0);
    const double eps = std::numeric_limits<double>::epsilon();

    const Eigen::Index n_blocks = (n_samples + block_size - 1) / block_size;
    std::vector<unsigned char> bad_blocks(static_cast<std::size_t>(n_blocks), 0);
    std::vector<unsigned char> dilated_blocks(static_cast<std::size_t>(n_blocks), 0);
    std::vector<double> diffs;
    std::vector<Eigen::Index> to_flag;

    Eigen::VectorXi local_samples = Eigen::VectorXi::Zero(n_dets);
    Eigen::VectorXi local_scans = Eigen::VectorXi::Zero(n_dets);

    for (Eigen::Index det = 0; det < n_dets; ++det) {
        Eigen::Index n_good_samples = 0;
        for (Eigen::Index t = 0; t < n_samples; ++t) {
            const double s = ptc.scans.data(t, det);
            if (!ptc.flags.data(t, det) && std::isfinite(s)) {
                n_good_samples++;
            }
        }
        if (n_good_samples < min_good + 1) {
            continue;
        }
        summary.n_det_candidates++;

        diffs.clear();
        diffs.reserve(static_cast<std::size_t>(n_good_samples));
        for (Eigen::Index t = 1; t < n_samples; ++t) {
            if (ptc.flags.data(t, det) || ptc.flags.data(t - 1, det)) {
                continue;
            }
            const double s0 = ptc.scans.data(t - 1, det);
            const double s1 = ptc.scans.data(t, det);
            if (!std::isfinite(s0) || !std::isfinite(s1)) {
                continue;
            }
            diffs.push_back(s1 - s0);
        }
        if (static_cast<Eigen::Index>(diffs.size()) < min_good - 1) {
            continue;
        }

        Eigen::Map<Eigen::VectorXd> diff_global(diffs.data(), static_cast<Eigen::Index>(diffs.size()));
        const double diff_med = tula::alg::median(diff_global);
        Eigen::VectorXd diff_abs_dev = (diff_global.array() - diff_med).abs().matrix();
        const double diff_mad = tula::alg::median(diff_abs_dev);
        const double global_sigma = 1.4826 * diff_mad;
        if (!std::isfinite(global_sigma) || global_sigma <= eps) {
            continue;
        }

        std::fill(bad_blocks.begin(), bad_blocks.end(), 0);
        bool any_bad = false;
        for (Eigen::Index b = 0; b < n_blocks; ++b) {
            const Eigen::Index b_start = b * block_size;
            const Eigen::Index b_end = std::min(b_start + block_size, n_samples);
            diffs.clear();
            diffs.reserve(static_cast<std::size_t>(b_end - b_start));
            for (Eigen::Index t = std::max<Eigen::Index>(b_start + 1, 1); t < b_end; ++t) {
                if (ptc.flags.data(t, det) || ptc.flags.data(t - 1, det)) {
                    continue;
                }
                const double s0 = ptc.scans.data(t - 1, det);
                const double s1 = ptc.scans.data(t, det);
                if (!std::isfinite(s0) || !std::isfinite(s1)) {
                    continue;
                }
                diffs.push_back(s1 - s0);
            }
            if (static_cast<Eigen::Index>(diffs.size()) < min_good - 1) {
                continue;
            }

            Eigen::Map<Eigen::VectorXd> diff_block(diffs.data(), static_cast<Eigen::Index>(diffs.size()));
            const double block_med = tula::alg::median(diff_block);
            Eigen::VectorXd block_abs_dev = (diff_block.array() - block_med).abs().matrix();
            const double block_mad = tula::alg::median(block_abs_dev);
            const double block_sigma = 1.4826 * block_mad;
            if (!std::isfinite(block_sigma) || block_sigma <= eps) {
                continue;
            }
            if (block_sigma >= sigma_floor && block_sigma > sigma_threshold * global_sigma) {
                bad_blocks[static_cast<std::size_t>(b)] = 1;
                any_bad = true;
            }
        }

        if (!any_bad) {
            continue;
        }

        if (dilate_blocks > 0) {
            std::fill(dilated_blocks.begin(), dilated_blocks.end(), 0);
            for (Eigen::Index b = 0; b < n_blocks; ++b) {
                if (!bad_blocks[static_cast<std::size_t>(b)]) {
                    continue;
                }
                const Eigen::Index b0 = std::max<Eigen::Index>(0, b - dilate_blocks);
                const Eigen::Index b1 = std::min<Eigen::Index>(n_blocks - 1, b + dilate_blocks);
                for (Eigen::Index bb = b0; bb <= b1; ++bb) {
                    dilated_blocks[static_cast<std::size_t>(bb)] = 1;
                }
            }
            bad_blocks.swap(dilated_blocks);
        }

        to_flag.clear();
        for (Eigen::Index b = 0; b < n_blocks; ++b) {
            if (!bad_blocks[static_cast<std::size_t>(b)]) {
                continue;
            }
            const Eigen::Index b_start = b * block_size;
            const Eigen::Index b_end = std::min(b_start + block_size, n_samples);
            for (Eigen::Index t = b_start; t < b_end; ++t) {
                const double s = ptc.scans.data(t, det);
                if (!ptc.flags.data(t, det) && std::isfinite(s)) {
                    to_flag.push_back(t);
                }
            }
        }

        if (to_flag.empty()) {
            continue;
        }
        const double flagged_fraction =
            static_cast<double>(to_flag.size()) / static_cast<double>(std::max<Eigen::Index>(1, n_good_samples));
        if (max_flagged_fraction > 0.0 && flagged_fraction > max_flagged_fraction) {
            summary.n_det_rejected++;
            continue;
        }

        for (const auto t: to_flag) {
            ptc.flags.data(t, det) = true;
        }

        summary.n_det_flagged++;
        summary.n_samples_flagged += static_cast<Eigen::Index>(to_flag.size());
        local_samples(det) += static_cast<int>(to_flag.size());
        local_scans(det) = 1;
    }

    if (summary.n_samples_flagged > 0 &&
        rfi_mask_samples_flagged.size() == n_dets &&
        rfi_mask_scans_flagged.size() == n_dets) {
        if (!rfi_mask_diag_mutex) {
            rfi_mask_diag_mutex = std::make_shared<std::mutex>();
        }
        std::lock_guard<std::mutex> lock(*rfi_mask_diag_mutex);
        rfi_mask_samples_flagged += local_samples;
        rfi_mask_scans_flagged += local_scans;
    }

    return summary;
}

Beammap::ScanBandMaskSummary Beammap::apply_scan_band_mask(mapmaking::MapBuffer &map_buffer) {
    ScanBandMaskSummary summary;

    if (!beammap_scan_band_mask_enabled || map_grouping != "detector") {
        return summary;
    }

    const Eigen::Index n_det_maps = std::min<Eigen::Index>(
        static_cast<Eigen::Index>(map_buffer.signal.size()), calib.n_dets);
    if (n_det_maps <= 0 || map_buffer.n_rows <= 0 || map_buffer.n_cols <= 0) {
        return summary;
    }

    const Eigen::Index search_rows = std::min<Eigen::Index>(
        std::max<Eigen::Index>(1, beammap_scan_band_mask_edge_rows), map_buffer.n_rows / 2);
    if (search_rows <= 0) {
        return summary;
    }

    const Eigen::Index min_row_pixels = std::max<Eigen::Index>(1, beammap_scan_band_mask_min_row_pixels);
    const Eigen::Index min_contiguous_rows = std::max<Eigen::Index>(1, beammap_scan_band_mask_min_contiguous_rows);
    const double median_sigma_threshold = std::max(0.0, beammap_scan_band_mask_row_median_sigma_threshold);
    const double sigma_ratio_threshold = std::max(0.0, beammap_scan_band_mask_row_sigma_ratio_threshold);
    const double max_flagged_fraction = std::clamp(beammap_scan_band_mask_max_flagged_fraction, 0.0, 1.0);
    const double eps = std::numeric_limits<double>::epsilon();
    const double row0 = static_cast<double>(map_buffer.n_rows - 1) / 2.0;

    auto robust_stats = [&](const std::vector<double> &values, double &median, double &sigma) -> bool {
        if (values.empty()) {
            median = std::numeric_limits<double>::quiet_NaN();
            sigma = std::numeric_limits<double>::quiet_NaN();
            return false;
        }
        Eigen::Map<const Eigen::VectorXd> vec(values.data(), static_cast<Eigen::Index>(values.size()));
        median = tula::alg::median(vec);
        Eigen::VectorXd abs_dev = (vec.array() - median).abs().matrix();
        sigma = 1.4826 * tula::alg::median(abs_dev);
        if (!std::isfinite(sigma)) {
            sigma = std::numeric_limits<double>::quiet_NaN();
        }
        return std::isfinite(median);
    };

    auto row_is_bad = [&](double row_median, double row_sigma, double interior_median,
                          double interior_median_sigma, double interior_row_sigma_median) {
        bool bad = false;
        if (median_sigma_threshold > 0.0 &&
            std::isfinite(row_median) &&
            std::isfinite(interior_median) &&
            std::isfinite(interior_median_sigma) &&
            interior_median_sigma > eps) {
            bad = std::abs(row_median - interior_median) > median_sigma_threshold * interior_median_sigma;
        }
        if (!bad &&
            sigma_ratio_threshold > 0.0 &&
            std::isfinite(row_sigma) &&
            std::isfinite(interior_row_sigma_median) &&
            interior_row_sigma_median > eps) {
            bad = row_sigma > sigma_ratio_threshold * interior_row_sigma_median;
        }
        return bad;
    };

    for (Eigen::Index det = 0; det < n_det_maps; ++det) {
        const auto &sig = map_buffer.signal[det];
        const auto &wt = map_buffer.weight[det];
        if (sig.rows() != map_buffer.n_rows || sig.cols() != map_buffer.n_cols ||
            wt.rows() != sig.rows() || wt.cols() != sig.cols()) {
            continue;
        }

        std::vector<double> row_medians(static_cast<std::size_t>(map_buffer.n_rows),
                                        std::numeric_limits<double>::quiet_NaN());
        std::vector<double> row_sigmas(static_cast<std::size_t>(map_buffer.n_rows),
                                       std::numeric_limits<double>::quiet_NaN());
        std::vector<Eigen::Index> row_counts(static_cast<std::size_t>(map_buffer.n_rows), 0);
        std::vector<double> row_values;
        std::vector<double> interior_row_medians;
        std::vector<double> interior_row_sigmas;

        for (Eigen::Index row = 0; row < map_buffer.n_rows; ++row) {
            row_values.clear();
            row_values.reserve(static_cast<std::size_t>(map_buffer.n_cols));
            for (Eigen::Index col = 0; col < map_buffer.n_cols; ++col) {
                const double w = wt(row, col);
                const double s = sig(row, col);
                if (!std::isfinite(w) || w <= 0.0 || !std::isfinite(s)) {
                    continue;
                }
                row_values.push_back(s);
            }
            row_counts[static_cast<std::size_t>(row)] = static_cast<Eigen::Index>(row_values.size());
            if (row_counts[static_cast<std::size_t>(row)] < min_row_pixels) {
                continue;
            }
            double row_median = std::numeric_limits<double>::quiet_NaN();
            double row_sigma = std::numeric_limits<double>::quiet_NaN();
            if (!robust_stats(row_values, row_median, row_sigma)) {
                continue;
            }
            row_medians[static_cast<std::size_t>(row)] = row_median;
            row_sigmas[static_cast<std::size_t>(row)] = row_sigma;
            if (row >= search_rows && row < map_buffer.n_rows - search_rows) {
                interior_row_medians.push_back(row_median);
                if (std::isfinite(row_sigma)) {
                    interior_row_sigmas.push_back(row_sigma);
                }
            }
        }

        if (interior_row_medians.size() < static_cast<std::size_t>(min_contiguous_rows)) {
            continue;
        }

        double interior_median = std::numeric_limits<double>::quiet_NaN();
        double interior_median_sigma = std::numeric_limits<double>::quiet_NaN();
        if (!robust_stats(interior_row_medians, interior_median, interior_median_sigma)) {
            continue;
        }

        double interior_row_sigma_median = std::numeric_limits<double>::quiet_NaN();
        double dummy_sigma = std::numeric_limits<double>::quiet_NaN();
        if (!interior_row_sigmas.empty()) {
            robust_stats(interior_row_sigmas, interior_row_sigma_median, dummy_sigma);
        }

        auto collect_edge_rows = [&](bool from_top) {
            std::vector<Eigen::Index> flagged_rows;
            bool saw_eligible_row = false;
            for (Eigen::Index edge_idx = 0; edge_idx < search_rows; ++edge_idx) {
                const Eigen::Index row = from_top ? edge_idx : (map_buffer.n_rows - 1 - edge_idx);
                if (row < 0 || row >= map_buffer.n_rows) {
                    continue;
                }
                if (row_counts[static_cast<std::size_t>(row)] < min_row_pixels ||
                    !std::isfinite(row_medians[static_cast<std::size_t>(row)])) {
                    if (saw_eligible_row) {
                        break;
                    }
                    continue;
                }
                saw_eligible_row = true;
                const bool bad = row_is_bad(
                    row_medians[static_cast<std::size_t>(row)],
                    row_sigmas[static_cast<std::size_t>(row)],
                    interior_median,
                    interior_median_sigma,
                    interior_row_sigma_median);
                if (!bad) {
                    break;
                }
                flagged_rows.push_back(row);
            }
            if (flagged_rows.size() < static_cast<std::size_t>(min_contiguous_rows)) {
                flagged_rows.clear();
            }
            return flagged_rows;
        };

        auto top_rows = collect_edge_rows(true);
        auto bottom_rows = collect_edge_rows(false);
        if (top_rows.empty() && bottom_rows.empty()) {
            continue;
        }

        std::vector<unsigned char> bad_row_mask(static_cast<std::size_t>(map_buffer.n_rows), 0);
        Eigen::Index n_bad_rows = 0;
        for (const auto row : top_rows) {
            if (!bad_row_mask[static_cast<std::size_t>(row)]) {
                bad_row_mask[static_cast<std::size_t>(row)] = 1;
                n_bad_rows++;
            }
        }
        for (const auto row : bottom_rows) {
            if (!bad_row_mask[static_cast<std::size_t>(row)]) {
                bad_row_mask[static_cast<std::size_t>(row)] = 1;
                n_bad_rows++;
            }
        }
        if (n_bad_rows <= 0) {
            continue;
        }

        std::vector<std::pair<Eigen::Index, Eigen::Index>> proposed_flags;
        Eigen::Index n_good_samples = 0;
        for (Eigen::Index chunk_idx = 0; chunk_idx < static_cast<Eigen::Index>(ptcs.size()); ++chunk_idx) {
            auto &ptc = ptcs[chunk_idx];
            if (det >= ptc.scans.data.cols() || det >= ptc.flags.data.cols()) {
                continue;
            }
            Eigen::VectorXd lat;
            auto lat_it = ptc.pointing.data.find("lat");
            if (lat_it != ptc.pointing.data.end() &&
                lat_it->second.rows() == ptc.scans.data.rows() &&
                det < lat_it->second.cols()) {
                lat = lat_it->second.col(det);
            }
            else {
                auto latlon = engine_utils::calc_det_pointing(
                    ptc.tel_data.data,
                    calib.apt["x_t"](det),
                    calib.apt["y_t"](det),
                    telescope.pixel_axes,
                    ptc.pointing_offsets_arcsec.data,
                    map_grouping);
                lat = std::get<0>(latlon);
            }
            if (lat.size() != ptc.scans.data.rows()) {
                continue;
            }
            for (Eigen::Index t = 0; t < ptc.scans.data.rows(); ++t) {
                const double s = ptc.scans.data(t, det);
                if (ptc.flags.data(t, det) || !std::isfinite(s)) {
                    continue;
                }
                n_good_samples++;
                const double lat_v = lat(t);
                if (!std::isfinite(lat_v)) {
                    continue;
                }
                const Eigen::Index row = static_cast<Eigen::Index>(std::llround(lat_v / map_buffer.pixel_size_rad + row0));
                if (row < 0 || row >= map_buffer.n_rows) {
                    continue;
                }
                if (bad_row_mask[static_cast<std::size_t>(row)]) {
                    proposed_flags.emplace_back(chunk_idx, t);
                }
            }
        }

        if (proposed_flags.empty()) {
            continue;
        }

        const double flagged_fraction =
            static_cast<double>(proposed_flags.size()) /
            static_cast<double>(std::max<Eigen::Index>(1, n_good_samples));
        if (max_flagged_fraction > 0.0 && flagged_fraction > max_flagged_fraction) {
            if (scan_band_mask_rejected.size() == calib.n_dets) {
                scan_band_mask_rejected(det) = 1;
            }
            summary.n_det_rejected++;
            logger->debug(
                "beammap scan-band mask det={} rejected: proposed rows={} samples={} flagged_fraction={:.4f} exceeds limit={:.4f}",
                det, n_bad_rows, proposed_flags.size(), flagged_fraction, max_flagged_fraction);
            continue;
        }

        for (const auto &[chunk_idx, sample_idx] : proposed_flags) {
            ptcs[chunk_idx].flags.data(sample_idx, det) = true;
            if (chunk_idx < static_cast<Eigen::Index>(ptcs0.size()) &&
                sample_idx < ptcs0[chunk_idx].flags.data.rows() &&
                det < ptcs0[chunk_idx].flags.data.cols()) {
                ptcs0[chunk_idx].flags.data(sample_idx, det) = true;
            }
        }

        summary.n_det_flagged++;
        summary.n_rows_flagged += n_bad_rows;
        summary.n_samples_flagged += static_cast<Eigen::Index>(proposed_flags.size());
        if (scan_band_mask_samples_flagged.size() == calib.n_dets) {
            scan_band_mask_samples_flagged(det) += static_cast<int>(proposed_flags.size());
        }
        if (scan_band_mask_rows_flagged.size() == calib.n_dets) {
            scan_band_mask_rows_flagged(det) += static_cast<int>(n_bad_rows);
        }
        if (scan_band_mask_edge_code.size() == calib.n_dets) {
            const int edge_code = (!top_rows.empty() ? 1 : 0) + (!bottom_rows.empty() ? 2 : 0);
            scan_band_mask_edge_code(det) = edge_code;
        }

        logger->info(
            "beammap scan-band mask det={} array={} nw={} rows={} samples={} flagged_fraction={:.4f} top_rows={} bottom_rows={}",
            det,
            static_cast<int>(calib.apt["array"](det)),
            static_cast<int>(calib.apt["nw"](det)),
            n_bad_rows,
            proposed_flags.size(),
            flagged_fraction,
            static_cast<int>(top_rows.size()),
            static_cast<int>(bottom_rows.size()));
    }

    return summary;
}

std::filesystem::path Beammap::resolve_soft_priors_filepath() const {
    namespace fs = std::filesystem;

    if (beammap_priors_filepath.empty() || beammap_priors_filepath == "null") {
        return {};
    }

    fs::path requested(beammap_priors_filepath);
    std::vector<fs::path> candidates;

    if (requested.is_absolute()) {
        candidates.push_back(requested);
    }
    else {
        candidates.push_back(requested);

        fs::path source_path(__FILE__);
        if (source_path.is_relative()) {
            source_path = fs::current_path() / source_path;
        }
        source_path = source_path.lexically_normal();
        fs::path repo_root = source_path;
        for (int i = 0; i < 5 && !repo_root.empty(); ++i) {
            repo_root = repo_root.parent_path();
        }
        if (!repo_root.empty()) {
            candidates.push_back(repo_root / requested);
        }
    }

    for (const auto &candidate : candidates) {
        try {
            if (fs::exists(candidate)) {
                return fs::absolute(candidate).lexically_normal();
            }
        }
        catch (const std::exception &) {
        }
    }

    return {};
}

bool Beammap::load_soft_priors() {
    beammap_soft_prior_slots.clear();
    beammap_soft_priors_loaded = false;
    beammap_soft_priors_are_centered = false;
    beammap_soft_priors_are_derotated = false;

    if (!beammap_priors_enabled) {
        return false;
    }

    if (beammap_priors_filepath.empty() || beammap_priors_filepath == "null") {
        logger->warn("beammap priors filepath is empty/null");
        return false;
    }
    const auto resolved_priors_filepath = resolve_soft_priors_filepath();
    if (resolved_priors_filepath.empty()) {
        logger->warn("beammap priors file does not exist: {}", beammap_priors_filepath);
        return false;
    }
    if (resolved_priors_filepath.string() != beammap_priors_filepath) {
        logger->info("beammap priors resolved {} -> {}", beammap_priors_filepath, resolved_priors_filepath.string());
        beammap_priors_filepath = resolved_priors_filepath.string();
    }

    auto [priors_table, priors_header, priors_meta] =
        to_map_from_ecsv_mixted_type(beammap_priors_filepath);
    static_cast<void>(priors_header);

    auto prior_frame_it = priors_meta.find("prior_frame");
    if (prior_frame_it != priors_meta.end()) {
        std::string prior_frame = prior_frame_it->second;
        std::transform(prior_frame.begin(), prior_frame.end(), prior_frame.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        beammap_soft_priors_are_centered = (prior_frame.find("center") != std::string::npos);
        beammap_soft_priors_are_derotated = (prior_frame.find("derot") != std::string::npos);
    }

    const std::vector<std::string> required_columns = {
        "array",
        "nw",
        "slot_index",
        "x_rel_med_arcsec",
        "y_rel_med_arcsec",
        "x_rel_sigma_soft_arcsec",
        "y_rel_sigma_soft_arcsec"
    };

    for (const auto &col : required_columns) {
        if (priors_table.find(col) == priors_table.end()) {
            logger->warn("beammap priors missing required column '{}': {}", col, beammap_priors_filepath);
            return false;
        }
    }

    const Eigen::Index n_rows = priors_table.at("array").size();
    for (const auto &col : required_columns) {
        if (priors_table.at(col).size() != n_rows) {
            logger->warn("beammap priors column '{}' has wrong size {} (expected {})",
                         col, priors_table.at(col).size(), n_rows);
            return false;
        }
    }
    if (n_rows <= 0) {
        logger->warn("beammap priors table has no rows: {}", beammap_priors_filepath);
        return false;
    }

    constexpr double sigma_floor_arcsec = 1e-3;
    Eigen::Index n_valid_rows = 0;
    Eigen::Index n_dropped_rows = 0;
    for (Eigen::Index i = 0; i < n_rows; ++i) {
        const double array_d = priors_table.at("array")(i);
        const double nw_d = priors_table.at("nw")(i);
        const double slot_d = priors_table.at("slot_index")(i);
        const double x_d = priors_table.at("x_rel_med_arcsec")(i);
        const double y_d = priors_table.at("y_rel_med_arcsec")(i);
        const double sx_d = priors_table.at("x_rel_sigma_soft_arcsec")(i);
        const double sy_d = priors_table.at("y_rel_sigma_soft_arcsec")(i);

        if (!(std::isfinite(array_d) && std::isfinite(nw_d) && std::isfinite(slot_d) &&
              std::isfinite(x_d) && std::isfinite(y_d) && std::isfinite(sx_d) && std::isfinite(sy_d))) {
            n_dropped_rows++;
            continue;
        }

        const int array = static_cast<int>(std::lround(array_d));
        const int nw = static_cast<int>(std::lround(nw_d));

        SoftPriorSlot slot;
        slot.slot_index = static_cast<int>(std::lround(slot_d));
        slot.x_arcsec = x_d;
        slot.y_arcsec = y_d;
        slot.sx_arcsec = std::max(sigma_floor_arcsec, std::abs(sx_d));
        slot.sy_arcsec = std::max(sigma_floor_arcsec, std::abs(sy_d));

        beammap_soft_prior_slots[{array, nw}].push_back(slot);
        n_valid_rows++;
    }

    for (auto &entry : beammap_soft_prior_slots) {
        auto &slots = entry.second;
        std::sort(slots.begin(), slots.end(),
                  [](const SoftPriorSlot &a, const SoftPriorSlot &b) {
                      if (a.slot_index == b.slot_index) {
                          return a.y_arcsec < b.y_arcsec;
                      }
                      return a.slot_index < b.slot_index;
                  });
    }

    if (beammap_soft_prior_slots.empty()) {
        logger->warn("beammap priors produced no valid slots: {}", beammap_priors_filepath);
        return false;
    }

    Eigen::Index n_slots = 0;
    for (const auto &entry : beammap_soft_prior_slots) {
        n_slots += static_cast<Eigen::Index>(entry.second.size());
    }
    beammap_soft_priors_loaded = true;
    logger->info("loaded beammap soft priors: {} slot rows across {} (array,nw) groups from {}",
                 n_slots, beammap_soft_prior_slots.size(), beammap_priors_filepath);
    if (n_dropped_rows > 0) {
        logger->warn("dropped {} non-finite prior rows (kept {})", n_dropped_rows, n_valid_rows);
    }

    return true;
}

bool Beammap::find_map_weighted_peak(Eigen::Index map_index, Eigen::Index &best_row,
                                     Eigen::Index &best_col, double &best_snr) const {
    best_row = -1;
    best_col = -1;
    best_snr = -std::numeric_limits<double>::infinity();

    if (map_index < 0 || map_index >= n_maps) {
        return false;
    }

    const auto &sig = omb.signal[map_index];
    const auto &wt = omb.weight[map_index];
    if (sig.rows() <= 0 || sig.cols() <= 0 || wt.rows() != sig.rows() || wt.cols() != sig.cols()) {
        return false;
    }

    const double center_row = static_cast<double>(sig.rows() - 1) / 2.0;
    const double center_col = static_cast<double>(sig.cols() - 1) / 2.0;
    const double radius_pix = map_fitter.fitting_region_pix;
    const double radius2 = radius_pix * radius_pix;

    auto scan = [&](bool apply_radius) {
        bool found = false;
        for (Eigen::Index row = 0; row < sig.rows(); ++row) {
            for (Eigen::Index col = 0; col < sig.cols(); ++col) {
                const double s = sig(row, col);
                const double w = wt(row, col);
                if (!std::isfinite(s) || !std::isfinite(w) || w <= 0.0) {
                    continue;
                }
                if (apply_radius) {
                    const double dr = static_cast<double>(row) - center_row;
                    const double dc = static_cast<double>(col) - center_col;
                    if (dr * dr + dc * dc >= radius2) {
                        continue;
                    }
                }
                const double snr = s * std::sqrt(w);
                if (!std::isfinite(snr)) {
                    continue;
                }
                if (!found || snr > best_snr) {
                    best_row = row;
                    best_col = col;
                    best_snr = snr;
                    found = true;
                }
            }
        }
        return found;
    };

    if (radius_pix > 0.0 && scan(true)) {
        return true;
    }
    return scan(false);
}

void Beammap::configure_detector_source_centers_from_previous_fit() {
    if (map_grouping != "detector") {
        ptcproc.fruit_loops_source_lat.resize(0);
        ptcproc.fruit_loops_source_lon.resize(0);
        ptcproc.fruit_loops_source_valid.resize(0);
        rtcproc.kernel.clear_source_centers();
        return;
    }

    if (current_iter <= 0) {
        ptcproc.fruit_loops_source_lat.resize(0);
        ptcproc.fruit_loops_source_lon.resize(0);
        ptcproc.fruit_loops_source_valid.resize(0);
        rtcproc.kernel.clear_source_centers();
        logger->info(
            "beammap detector source centers unavailable on iter {}: no previous fits yet "
            "(ptc_mask_radius={:.3f} arcsec)",
            current_iter, ptcproc.mask_radius_arcsec);
        return;
    }

    if (p0.rows() != n_maps || p0.cols() < 3 || good_fits.size() != n_maps) {
        ptcproc.fruit_loops_source_lat.resize(0);
        ptcproc.fruit_loops_source_lon.resize(0);
        ptcproc.fruit_loops_source_valid.resize(0);
        rtcproc.kernel.clear_source_centers();
        logger->warn(
            "beammap detector source centers unavailable on iter {}: previous-fit state is incomplete "
            "(p0={}x{}, good_fits={})",
            current_iter, p0.rows(), p0.cols(), good_fits.size());
        return;
    }

    ptcproc.fruit_loops_source_lat = Eigen::VectorXd::Zero(n_maps);
    ptcproc.fruit_loops_source_lon = Eigen::VectorXd::Zero(n_maps);
    ptcproc.fruit_loops_source_valid = Eigen::VectorXi::Zero(n_maps);

    Eigen::Index n_valid = 0;
    for (Eigen::Index i = 0; i < n_maps; ++i) {
        if (!good_fits(i) ||
            !std::isfinite(p0(i, 0)) || p0(i, 0) <= 0.0 ||
            !std::isfinite(p0(i, 1)) || !std::isfinite(p0(i, 2))) {
            continue;
        }
        ptcproc.fruit_loops_source_lat(i) =
            (p0(i, 2) - (omb.n_rows - 1) / 2.0) * omb.pixel_size_rad;
        ptcproc.fruit_loops_source_lon(i) =
            (p0(i, 1) - (omb.n_cols - 1) / 2.0) * omb.pixel_size_rad;
        ptcproc.fruit_loops_source_valid(i) = 1;
        n_valid++;
    }

    logger->info(
        "beammap detector source centers using previous-fit centers for {}/{} detector maps "
        "on iter {} (ptc_mask_radius={:.3f} arcsec)",
        n_valid, n_maps, current_iter, ptcproc.mask_radius_arcsec);

    if (rtcproc.run_kernel) {
        rtcproc.kernel.set_source_centers(ptcproc.fruit_loops_source_lat,
                                          ptcproc.fruit_loops_source_lon,
                                          ptcproc.fruit_loops_source_valid);
        logger->info(
            "beammap detector kernel placement using previous-fit centers for {}/{} detector maps on iter {}",
            n_valid, n_maps, current_iter);
    }
}

double Beammap::get_prior_derot_elev_rad() const {
    double derot_elev_rad = 0.0;
    auto tel_el_it = telescope.tel_data.find("TelElAct");
    if (tel_el_it != telescope.tel_data.end() && tel_el_it->second.size() > 0) {
        derot_elev_rad = tel_el_it->second.mean();
    }
    if (!std::isfinite(derot_elev_rad)) {
        derot_elev_rad = 0.0;
    }
    if (std::abs(derot_elev_rad) > pi) {
        derot_elev_rad *= DEG_TO_RAD;
    }
    return derot_elev_rad;
}

double Beammap::effective_prior_max_d2() const {
    return (current_iter <= 0) ? beammap_priors_max_d2_iter0 : beammap_priors_max_d2_after_iter0;
}

double Beammap::effective_prior_score_lambda() const {
    return (current_iter <= 0) ? beammap_priors_score_lambda_iter0 : beammap_priors_score_lambda_after_iter0;
}

bool Beammap::observed_to_prior_frame(int array, double x_raw_arcsec, double y_raw_arcsec,
                                      double derot_elev_rad, double &x_prior_arcsec,
                                      double &y_prior_arcsec, double *center_x_arcsec,
                                      double *center_y_arcsec,
                                      bool apply_empirical_alignment) const {
    if (!std::isfinite(x_raw_arcsec) || !std::isfinite(y_raw_arcsec)) {
        return false;
    }

    double x = x_raw_arcsec;
    double y = y_raw_arcsec;
    double center_x = std::numeric_limits<double>::quiet_NaN();
    double center_y = std::numeric_limits<double>::quiet_NaN();

    if (beammap_soft_priors_are_centered) {
        auto x_it = beammap_prior_array_center_x_arcsec.find(array);
        auto y_it = beammap_prior_array_center_y_arcsec.find(array);
        if (x_it == beammap_prior_array_center_x_arcsec.end() ||
            y_it == beammap_prior_array_center_y_arcsec.end() ||
            !std::isfinite(x_it->second) || !std::isfinite(y_it->second)) {
            return false;
        }
        center_x = x_it->second;
        center_y = y_it->second;
        x -= center_x;
        y -= center_y;
    }

    if (center_x_arcsec != nullptr) {
        *center_x_arcsec = center_x;
    }
    if (center_y_arcsec != nullptr) {
        *center_y_arcsec = center_y;
    }

    if (beammap_soft_priors_are_derotated && telescope.pixel_axes == "altaz") {
        if (!std::isfinite(derot_elev_rad)) {
            derot_elev_rad = 0.0;
        }
        if (std::abs(derot_elev_rad) > pi) {
            derot_elev_rad *= DEG_TO_RAD;
        }
        const double cos_rot = std::cos(-derot_elev_rad);
        const double sin_rot = std::sin(-derot_elev_rad);
        const double rot_az_off = cos_rot * x - sin_rot * y;
        const double rot_alt_off = sin_rot * x + cos_rot * y;
        x = -rot_az_off;
        y = -rot_alt_off;
    }

    if (apply_empirical_alignment) {
        auto align_it = beammap_prior_array_alignment.find(array);
        if (align_it != beammap_prior_array_alignment.end() && align_it->second.valid) {
            const auto &align = align_it->second;
            const double x_rot = align.cos_theta * x - align.sin_theta * y;
            const double y_rot = align.sin_theta * x + align.cos_theta * y;
            x = x_rot + align.dx_arcsec;
            y = y_rot + align.dy_arcsec;
        }
    }

    x_prior_arcsec = x;
    y_prior_arcsec = y;
    return std::isfinite(x_prior_arcsec) && std::isfinite(y_prior_arcsec);
}

bool Beammap::match_prior_slot(int array, int nw, double x_prior_arcsec, double y_prior_arcsec,
                               double &best_d2, int &best_slot, double *slot_x_arcsec,
                               double *slot_y_arcsec, double *slot_sx_arcsec,
                               double *slot_sy_arcsec) const {
    best_d2 = std::numeric_limits<double>::infinity();
    best_slot = -1;
    auto slots_it = beammap_soft_prior_slots.find({array, nw});
    if (slots_it == beammap_soft_prior_slots.end() || slots_it->second.empty() ||
        !std::isfinite(x_prior_arcsec) || !std::isfinite(y_prior_arcsec)) {
        return false;
    }

    for (const auto &slot : slots_it->second) {
        if (!std::isfinite(slot.x_arcsec) || !std::isfinite(slot.y_arcsec) ||
            !std::isfinite(slot.sx_arcsec) || !std::isfinite(slot.sy_arcsec) ||
            slot.sx_arcsec <= 0.0 || slot.sy_arcsec <= 0.0) {
            continue;
        }
        const double dx = (x_prior_arcsec - slot.x_arcsec) / slot.sx_arcsec;
        const double dy = (y_prior_arcsec - slot.y_arcsec) / slot.sy_arcsec;
        const double d2 = dx * dx + dy * dy;
        if (std::isfinite(d2) && d2 < best_d2) {
            best_d2 = d2;
            best_slot = slot.slot_index;
            if (slot_x_arcsec != nullptr) {
                *slot_x_arcsec = slot.x_arcsec;
            }
            if (slot_y_arcsec != nullptr) {
                *slot_y_arcsec = slot.y_arcsec;
            }
            if (slot_sx_arcsec != nullptr) {
                *slot_sx_arcsec = slot.sx_arcsec;
            }
            if (slot_sy_arcsec != nullptr) {
                *slot_sy_arcsec = slot.sy_arcsec;
            }
        }
    }
    return std::isfinite(best_d2);
}

void Beammap::update_prior_frame_estimates() {
    beammap_prior_array_center_x_arcsec.clear();
    beammap_prior_array_center_y_arcsec.clear();
    beammap_prior_array_alignment.clear();

    std::map<int, std::vector<double>> x_by_array;
    std::map<int, std::vector<double>> y_by_array;
    std::set<int> arrays_missing;
    for (Eigen::Index i = 0; i < n_maps; ++i) {
        arrays_missing.insert(static_cast<int>(maps_to_arrays(i)));
    }

    Eigen::Index n_prev = 0;
    if (current_iter > 0 && p0.rows() == n_maps && p0.cols() > 2) {
        for (Eigen::Index i = 0; i < n_maps; ++i) {
            if (i < good_fits.size() && !good_fits(i)) {
                continue;
            }
            if (fit_diag_bound_nhit.size() == n_maps && fit_diag_bound_nhit(i) > 0) {
                continue;
            }
            if (!(std::isfinite(p0(i, 0)) && p0(i, 0) > 0.0 &&
                  std::isfinite(p0(i, 1)) && std::isfinite(p0(i, 2)))) {
                continue;
            }
            const int array = static_cast<int>(maps_to_arrays(i));
            const double x_arcsec =
                RAD_TO_ASEC * omb.pixel_size_rad * (p0(i, 1) - (omb.n_cols - 1) / 2.0);
            const double y_arcsec =
                RAD_TO_ASEC * omb.pixel_size_rad * (p0(i, 2) - (omb.n_rows - 1) / 2.0);
            x_by_array[array].push_back(x_arcsec);
            y_by_array[array].push_back(y_arcsec);
            arrays_missing.erase(array);
            n_prev++;
        }
    }

    Eigen::Index n_blind = 0;
    if (!arrays_missing.empty()) {
        for (Eigen::Index i = 0; i < n_maps; ++i) {
            const int array = static_cast<int>(maps_to_arrays(i));
            if (!arrays_missing.count(array)) {
                continue;
            }

            Eigen::Index peak_row = -1;
            Eigen::Index peak_col = -1;
            double peak_snr = -std::numeric_limits<double>::infinity();
            if (!find_map_weighted_peak(i, peak_row, peak_col, peak_snr)) {
                continue;
            }

            const double x_arcsec =
                RAD_TO_ASEC * omb.pixel_size_rad * (static_cast<double>(peak_col) - (omb.n_cols - 1) / 2.0);
            const double y_arcsec =
                RAD_TO_ASEC * omb.pixel_size_rad * (static_cast<double>(peak_row) - (omb.n_rows - 1) / 2.0);
            x_by_array[array].push_back(x_arcsec);
            y_by_array[array].push_back(y_arcsec);
            n_blind++;
        }
    }

    for (const auto &[array, xs] : x_by_array) {
        if (xs.empty()) {
            continue;
        }
        Eigen::Map<const Eigen::VectorXd> x_vec(xs.data(), static_cast<Eigen::Index>(xs.size()));
        auto y_it = y_by_array.find(array);
        if (y_it == y_by_array.end() || y_it->second.size() != xs.size()) {
            continue;
        }
        Eigen::Map<const Eigen::VectorXd> y_vec(y_it->second.data(), static_cast<Eigen::Index>(y_it->second.size()));
        beammap_prior_array_center_x_arcsec[array] = tula::alg::median(x_vec);
        beammap_prior_array_center_y_arcsec[array] = tula::alg::median(y_vec);
    }

    Eigen::Index n_alignment_matches = 0;
    if (beammap_priors_align_after_iter0 && current_iter > 0 && p0.rows() == n_maps && p0.cols() > 2) {
        struct PriorPair {
            double obs_x = 0.0;
            double obs_y = 0.0;
            double slot_x = 0.0;
            double slot_y = 0.0;
        };
        std::map<int, std::vector<PriorPair>> pairs_by_array;
        std::vector<PriorPair> all_pairs;
        std::set<int> arrays_with_alignment_pairs;
        const double derot_elev_rad = get_prior_derot_elev_rad();

        for (Eigen::Index i = 0; i < n_maps; ++i) {
            if (i >= good_fits.size() || !good_fits(i)) {
                continue;
            }
            if (fit_diag_bound_nhit.size() == n_maps && fit_diag_bound_nhit(i) > 0) {
                continue;
            }
            if (!(std::isfinite(p0(i, 0)) && p0(i, 0) > 0.0 &&
                  std::isfinite(p0(i, 1)) && std::isfinite(p0(i, 2)))) {
                continue;
            }
            const int array = static_cast<int>(maps_to_arrays(i));
            const int nw = static_cast<int>(std::lround(calib.apt["nw"](i)));
            const double x_raw =
                RAD_TO_ASEC * omb.pixel_size_rad * (p0(i, 1) - (omb.n_cols - 1) / 2.0);
            const double y_raw =
                RAD_TO_ASEC * omb.pixel_size_rad * (p0(i, 2) - (omb.n_rows - 1) / 2.0);
            double x_prior = std::numeric_limits<double>::quiet_NaN();
            double y_prior = std::numeric_limits<double>::quiet_NaN();
            if (!observed_to_prior_frame(array, x_raw, y_raw, derot_elev_rad,
                                         x_prior, y_prior, nullptr, nullptr, false)) {
                continue;
            }
            double d2 = std::numeric_limits<double>::infinity();
            int slot_index = -1;
            double slot_x = std::numeric_limits<double>::quiet_NaN();
            double slot_y = std::numeric_limits<double>::quiet_NaN();
            if (!match_prior_slot(array, nw, x_prior, y_prior, d2, slot_index, &slot_x, &slot_y)) {
                continue;
            }
            static_cast<void>(slot_index);
            if (beammap_priors_alignment_max_d2 > 0.0 && d2 > beammap_priors_alignment_max_d2) {
                continue;
            }
            PriorPair pair{x_prior, y_prior, slot_x, slot_y};
            pairs_by_array[array].push_back(pair);
            all_pairs.push_back(pair);
            arrays_with_alignment_pairs.insert(array);
            n_alignment_matches++;
        }

        auto fit_prior_alignment = [&](const std::vector<PriorPair> &pairs,
                                       const std::string &label,
                                       PriorArrayAlignment &alignment) {
            if (pairs.size() < static_cast<std::size_t>(beammap_priors_alignment_min_matches)) {
                logger->debug("beammap prior alignment skipped {} matches={} min_matches={}",
                              label, pairs.size(), beammap_priors_alignment_min_matches);
                return false;
            }

            std::vector<double> dx_vals;
            std::vector<double> dy_vals;
            dx_vals.reserve(pairs.size());
            dy_vals.reserve(pairs.size());
            for (const auto &pair : pairs) {
                dx_vals.push_back(pair.slot_x - pair.obs_x);
                dy_vals.push_back(pair.slot_y - pair.obs_y);
            }
            Eigen::Map<Eigen::VectorXd> dx_vec(dx_vals.data(), static_cast<Eigen::Index>(dx_vals.size()));
            Eigen::Map<Eigen::VectorXd> dy_vec(dy_vals.data(), static_cast<Eigen::Index>(dy_vals.size()));
            double tx = tula::alg::median(dx_vec);
            double ty = tula::alg::median(dy_vec);

            double theta = 0.0;
            if (beammap_priors_alignment_fit_rotation) {
                double obs_mean_x = 0.0;
                double obs_mean_y = 0.0;
                double slot_mean_x = 0.0;
                double slot_mean_y = 0.0;
                for (const auto &pair : pairs) {
                    obs_mean_x += pair.obs_x + tx;
                    obs_mean_y += pair.obs_y + ty;
                    slot_mean_x += pair.slot_x;
                    slot_mean_y += pair.slot_y;
                }
                const double inv_n = 1.0 / static_cast<double>(pairs.size());
                obs_mean_x *= inv_n;
                obs_mean_y *= inv_n;
                slot_mean_x *= inv_n;
                slot_mean_y *= inv_n;

                double a = 0.0;
                double b = 0.0;
                for (const auto &pair : pairs) {
                    const double ox = pair.obs_x + tx - obs_mean_x;
                    const double oy = pair.obs_y + ty - obs_mean_y;
                    const double sx = pair.slot_x - slot_mean_x;
                    const double sy = pair.slot_y - slot_mean_y;
                    a += ox * sx + oy * sy;
                    b += ox * sy - oy * sx;
                }
                if (std::isfinite(a) && std::isfinite(b) &&
                    (std::abs(a) > 0.0 || std::abs(b) > 0.0)) {
                    theta = std::atan2(b, a);
                }
                const double max_theta = beammap_priors_alignment_max_rotation_deg * DEG_TO_RAD;
                if (!std::isfinite(theta) || std::abs(theta) > max_theta) {
                    logger->debug(
                        "beammap prior alignment {} rejected residual rotation {} deg (limit={} deg)",
                        label, theta * RAD_TO_DEG, beammap_priors_alignment_max_rotation_deg);
                    theta = 0.0;
                }
            }

            const double cos_theta = std::cos(theta);
            const double sin_theta = std::sin(theta);
            dx_vals.clear();
            dy_vals.clear();
            for (const auto &pair : pairs) {
                const double x_rot = cos_theta * pair.obs_x - sin_theta * pair.obs_y;
                const double y_rot = sin_theta * pair.obs_x + cos_theta * pair.obs_y;
                dx_vals.push_back(pair.slot_x - x_rot);
                dy_vals.push_back(pair.slot_y - y_rot);
            }
            Eigen::Map<Eigen::VectorXd> dx_vec_final(dx_vals.data(), static_cast<Eigen::Index>(dx_vals.size()));
            Eigen::Map<Eigen::VectorXd> dy_vec_final(dy_vals.data(), static_cast<Eigen::Index>(dy_vals.size()));
            tx = tula::alg::median(dx_vec_final);
            ty = tula::alg::median(dy_vec_final);

            double rss = 0.0;
            for (const auto &pair : pairs) {
                const double x_fit = cos_theta * pair.obs_x - sin_theta * pair.obs_y + tx;
                const double y_fit = sin_theta * pair.obs_x + cos_theta * pair.obs_y + ty;
                const double rx = x_fit - pair.slot_x;
                const double ry = y_fit - pair.slot_y;
                rss += rx * rx + ry * ry;
            }
            const double rms = std::sqrt(rss / static_cast<double>(pairs.size()));
            if (!(std::isfinite(tx) && std::isfinite(ty) && std::isfinite(rms))) {
                return false;
            }

            alignment.valid = true;
            alignment.cos_theta = cos_theta;
            alignment.sin_theta = sin_theta;
            alignment.theta_rad = theta;
            alignment.dx_arcsec = tx;
            alignment.dy_arcsec = ty;
            alignment.n_matches = static_cast<Eigen::Index>(pairs.size());
            alignment.rms_arcsec = rms;
            return true;
        };

        if (beammap_priors_alignment_scope == "common") {
            auto common_pairs = all_pairs;
            if (beammap_priors_alignment_common_support == "overlap_box" &&
                pairs_by_array.size() >= 2) {
                auto quantile = [](std::vector<double> values, double q) {
                    if (values.empty()) {
                        return std::numeric_limits<double>::quiet_NaN();
                    }
                    q = std::clamp(q, 0.0, 1.0);
                    std::sort(values.begin(), values.end());
                    const double pos = q * static_cast<double>(values.size() - 1);
                    const auto lo = static_cast<std::size_t>(std::floor(pos));
                    const auto hi = static_cast<std::size_t>(std::ceil(pos));
                    if (lo == hi) {
                        return values[lo];
                    }
                    const double frac = pos - static_cast<double>(lo);
                    return values[lo] * (1.0 - frac) + values[hi] * frac;
                };

                const double q_low = beammap_priors_alignment_common_support_quantile;
                const double q_high = 1.0 - beammap_priors_alignment_common_support_quantile;
                double overlap_x_low = -std::numeric_limits<double>::infinity();
                double overlap_x_high = std::numeric_limits<double>::infinity();
                double overlap_y_low = -std::numeric_limits<double>::infinity();
                double overlap_y_high = std::numeric_limits<double>::infinity();
                bool overlap_valid = true;

                for (const auto &[array, pairs] : pairs_by_array) {
                    static_cast<void>(array);
                    std::vector<double> xs;
                    std::vector<double> ys;
                    xs.reserve(pairs.size());
                    ys.reserve(pairs.size());
                    for (const auto &pair : pairs) {
                        if (std::isfinite(pair.slot_x) && std::isfinite(pair.slot_y)) {
                            xs.push_back(pair.slot_x);
                            ys.push_back(pair.slot_y);
                        }
                    }
                    const double x_low = quantile(xs, q_low);
                    const double x_high = quantile(xs, q_high);
                    const double y_low = quantile(ys, q_low);
                    const double y_high = quantile(ys, q_high);
                    if (!(std::isfinite(x_low) && std::isfinite(x_high) &&
                          std::isfinite(y_low) && std::isfinite(y_high))) {
                        overlap_valid = false;
                        break;
                    }
                    overlap_x_low = std::max(overlap_x_low, x_low);
                    overlap_x_high = std::min(overlap_x_high, x_high);
                    overlap_y_low = std::max(overlap_y_low, y_low);
                    overlap_y_high = std::min(overlap_y_high, y_high);
                }

                if (overlap_valid && overlap_x_low < overlap_x_high &&
                    overlap_y_low < overlap_y_high) {
                    std::vector<PriorPair> filtered_pairs;
                    filtered_pairs.reserve(all_pairs.size());
                    for (const auto &pair : all_pairs) {
                        if (pair.slot_x >= overlap_x_low && pair.slot_x <= overlap_x_high &&
                            pair.slot_y >= overlap_y_low && pair.slot_y <= overlap_y_high) {
                            filtered_pairs.push_back(pair);
                        }
                    }
                    if (filtered_pairs.size() >= static_cast<std::size_t>(beammap_priors_alignment_min_matches)) {
                        common_pairs.swap(filtered_pairs);
                    }
                    logger->info(
                        "beammap prior common alignment overlap_box (iter {}): q={} x=[{}, {}] y=[{}, {}] kept={}/{}",
                        current_iter, beammap_priors_alignment_common_support_quantile,
                        overlap_x_low, overlap_x_high, overlap_y_low, overlap_y_high,
                        common_pairs.size(), all_pairs.size());
                }
                else {
                    logger->debug(
                        "beammap prior common alignment overlap_box skipped: invalid overlap x=[{}, {}] y=[{}, {}]",
                        overlap_x_low, overlap_x_high, overlap_y_low, overlap_y_high);
                }
            }

            PriorArrayAlignment alignment;
            if (fit_prior_alignment(common_pairs, "scope=common", alignment)) {
                for (int array : arrays_with_alignment_pairs) {
                    beammap_prior_array_alignment[array] = alignment;
                }
                logger->info(
                    "beammap prior empirical alignment (iter {} scope=common): arrays={} matches={} dx={} dy={} rot_deg={} rms={}",
                    current_iter, arrays_with_alignment_pairs.size(), alignment.n_matches,
                    alignment.dx_arcsec, alignment.dy_arcsec,
                    alignment.theta_rad * RAD_TO_DEG, alignment.rms_arcsec);
            }
        }
        else {
            for (auto &[array, pairs] : pairs_by_array) {
                PriorArrayAlignment alignment;
                if (!fit_prior_alignment(pairs, fmt::format("array={}", array), alignment)) {
                    continue;
                }
                beammap_prior_array_alignment[array] = alignment;

                logger->info(
                    "beammap prior empirical alignment (iter {} array={}): matches={} dx={} dy={} rot_deg={} rms={}",
                    current_iter, array, alignment.n_matches, alignment.dx_arcsec,
                    alignment.dy_arcsec, alignment.theta_rad * RAD_TO_DEG, alignment.rms_arcsec);
            }
        }
    }

    logger->info(
        "beammap priors frame estimate (iter {}): previous={} blind={} arrays={} alignment_matches={} aligned_arrays={}",
        current_iter, n_prev, n_blind, beammap_prior_array_center_x_arcsec.size(),
        n_alignment_matches, beammap_prior_array_alignment.size());
}

bool Beammap::choose_prior_guided_init(Eigen::Index map_index, double &init_row, double &init_col) {
    init_row = -99.0;
    init_col = -99.0;

    auto set_prior_diag = [&](PriorDiagColumn col, double value) {
        if (map_index >= 0 && map_index < prior_diag_values.rows() &&
            col >= 0 && col < prior_diag_values.cols()) {
            prior_diag_values(map_index, col) = value;
        }
    };

    constexpr int prior_reason_none = 0;
    constexpr int prior_reason_no_slot_group = 1;
    constexpr int prior_reason_no_valid_weighted_pixels = 2;
    constexpr int prior_reason_invalid_sigma = 3;
    constexpr int prior_reason_below_min_snr = 4;
    constexpr int prior_reason_gate_rejected = 5;

    if (!beammap_soft_priors_loaded || map_grouping != "detector") {
        return false;
    }
    if (map_index < 0 || map_index >= n_maps || map_index >= calib.n_dets) {
        return false;
    }
    if (map_index >= maps_to_arrays.size() || map_index >= calib.apt["nw"].size()) {
        return false;
    }

    const int array = static_cast<int>(maps_to_arrays(map_index));
    const int nw = static_cast<int>(std::lround(calib.apt["nw"](map_index)));
    auto slots_it = beammap_soft_prior_slots.find({array, nw});
    if (slots_it == beammap_soft_prior_slots.end() || slots_it->second.empty()) {
        set_prior_diag(prior_no_candidate_reason_col, prior_reason_no_slot_group);
        return false;
    }

    const auto &sig = omb.signal[map_index];
    const auto &wt = omb.weight[map_index];
    if (sig.rows() <= 0 || sig.cols() <= 0 || wt.rows() != sig.rows() || wt.cols() != sig.cols()) {
        set_prior_diag(prior_no_candidate_reason_col, prior_reason_no_valid_weighted_pixels);
        return false;
    }

    struct Candidate {
        double snr = 0.0;
        Eigen::Index row = 0;
        Eigen::Index col = 0;
    };

    std::vector<double> valid_signal;
    std::vector<double> valid_weight;
    valid_signal.reserve(static_cast<std::size_t>(sig.size()));
    valid_weight.reserve(static_cast<std::size_t>(sig.size()));
    for (Eigen::Index row = 0; row < sig.rows(); ++row) {
        for (Eigen::Index col = 0; col < sig.cols(); ++col) {
            const double s = sig(row, col);
            const double w = wt(row, col);
            if (!std::isfinite(s) || !std::isfinite(w) || w <= 0.0) {
                continue;
            }
            valid_signal.push_back(s);
            valid_weight.push_back(w);
        }
    }
    if (valid_signal.empty()) {
        set_prior_diag(prior_no_candidate_reason_col, prior_reason_no_valid_weighted_pixels);
        return false;
    }

    Eigen::Map<Eigen::VectorXd> sig_vec(valid_signal.data(), static_cast<Eigen::Index>(valid_signal.size()));
    const double sig_med = tula::alg::median(sig_vec);
    Eigen::VectorXd sig_abs_dev = (sig_vec.array() - sig_med).abs().matrix();
    double sig_sigma = 1.4826 * tula::alg::median(sig_abs_dev);
    if (!std::isfinite(sig_sigma) || sig_sigma <= std::numeric_limits<double>::epsilon()) {
        sig_sigma = engine_utils::calc_std_dev(sig_vec);
    }
    if (!std::isfinite(sig_sigma) || sig_sigma <= std::numeric_limits<double>::epsilon()) {
        set_prior_diag(prior_no_candidate_reason_col, prior_reason_invalid_sigma);
        return false;
    }

    Eigen::Map<Eigen::VectorXd> wt_vec(valid_weight.data(), static_cast<Eigen::Index>(valid_weight.size()));
    double wt_med = tula::alg::median(wt_vec);
    if (!std::isfinite(wt_med) || wt_med <= std::numeric_limits<double>::epsilon()) {
        wt_med = 1.0;
    }

    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<std::size_t>(sig.size()));
    for (Eigen::Index row = 0; row < sig.rows(); ++row) {
        for (Eigen::Index col = 0; col < sig.cols(); ++col) {
            const double s = sig(row, col);
            const double w = wt(row, col);
            if (!std::isfinite(s) || !std::isfinite(w) || w <= 0.0) {
                continue;
            }
            const double snr = ((s - sig_med) / sig_sigma) * std::sqrt(w / wt_med);
            if (!std::isfinite(snr) || snr < beammap_priors_min_snr) {
                continue;
            }
            candidates.push_back({snr, row, col});
        }
    }
    if (candidates.empty()) {
        logger->debug("beammap priors init map={} no candidates above min_snr={:.4g} (med={:.4g} sigma={:.4g} wt_med={:.4g})",
                      map_index, beammap_priors_min_snr, sig_med, sig_sigma, wt_med);
        set_prior_diag(prior_n_candidates_col, 0.0);
        set_prior_diag(prior_n_candidates_keep_col, 0.0);
        set_prior_diag(prior_n_candidates_gate_col, 0.0);
        set_prior_diag(prior_no_candidate_reason_col, prior_reason_below_min_snr);
        return false;
    }

    set_prior_diag(prior_n_candidates_col, static_cast<double>(candidates.size()));

    const std::size_t n_keep = std::min<std::size_t>(
        candidates.size(), static_cast<std::size_t>(std::max(1, beammap_priors_candidate_top_n)));
    set_prior_diag(prior_n_candidates_keep_col, static_cast<double>(n_keep));
    std::partial_sort(candidates.begin(), candidates.begin() + n_keep, candidates.end(),
                      [](const Candidate &a, const Candidate &b) { return a.snr > b.snr; });

    const double col0 = static_cast<double>(omb.n_cols - 1) / 2.0;
    const double row0 = static_cast<double>(omb.n_rows - 1) / 2.0;
    const double pix_to_arcsec = RAD_TO_ASEC * omb.pixel_size_rad;
    double derot_elev_rad = get_prior_derot_elev_rad();
    set_prior_diag(prior_derot_elev_col, derot_elev_rad);
    const double prior_max_d2 = effective_prior_max_d2();
    const double prior_score_lambda = effective_prior_score_lambda();

    bool found = false;
    double best_score = -std::numeric_limits<double>::infinity();
    double best_snr = -std::numeric_limits<double>::infinity();
    double best_d2 = std::numeric_limits<double>::infinity();
    Eigen::Index best_row = -1;
    Eigen::Index best_col = -1;
    int best_slot = -1;
    double best_x_raw = std::numeric_limits<double>::quiet_NaN();
    double best_y_raw = std::numeric_limits<double>::quiet_NaN();
    double best_x_prior = std::numeric_limits<double>::quiet_NaN();
    double best_y_prior = std::numeric_limits<double>::quiet_NaN();
    double best_slot_x = std::numeric_limits<double>::quiet_NaN();
    double best_slot_y = std::numeric_limits<double>::quiet_NaN();
    double best_slot_sx = std::numeric_limits<double>::quiet_NaN();
    double best_slot_sy = std::numeric_limits<double>::quiet_NaN();
    Eigen::Index n_gate = 0;

    for (std::size_t i = 0; i < n_keep; ++i) {
        const auto &cand = candidates[i];
        double x_arcsec_raw = pix_to_arcsec * (static_cast<double>(cand.col) - col0);
        double y_arcsec_raw = pix_to_arcsec * (static_cast<double>(cand.row) - row0);
        double center_x = std::numeric_limits<double>::quiet_NaN();
        double center_y = std::numeric_limits<double>::quiet_NaN();
        double x_arcsec = std::numeric_limits<double>::quiet_NaN();
        double y_arcsec = std::numeric_limits<double>::quiet_NaN();
        if (!observed_to_prior_frame(array, x_arcsec_raw, y_arcsec_raw, derot_elev_rad,
                                     x_arcsec, y_arcsec, &center_x, &center_y, true)) {
            continue;
        }

        double min_d2 = std::numeric_limits<double>::infinity();
        int min_slot = -1;
        double slot_x = std::numeric_limits<double>::quiet_NaN();
        double slot_y = std::numeric_limits<double>::quiet_NaN();
        double slot_sx = std::numeric_limits<double>::quiet_NaN();
        double slot_sy = std::numeric_limits<double>::quiet_NaN();
        if (!match_prior_slot(array, nw, x_arcsec, y_arcsec, min_d2, min_slot,
                              &slot_x, &slot_y, &slot_sx, &slot_sy)) {
            continue;
        }
        if (prior_max_d2 > 0.0 && min_d2 > prior_max_d2) {
            continue;
        }
        n_gate++;

        const double score = cand.snr - prior_score_lambda * min_d2;
        if (!found || score > best_score || (score == best_score && cand.snr > best_snr)) {
            found = true;
            best_score = score;
            best_snr = cand.snr;
            best_d2 = min_d2;
            best_row = cand.row;
            best_col = cand.col;
            best_slot = min_slot;
            best_x_raw = x_arcsec_raw;
            best_y_raw = y_arcsec_raw;
            best_x_prior = x_arcsec;
            best_y_prior = y_arcsec;
            best_slot_x = slot_x;
            best_slot_y = slot_y;
            best_slot_sx = slot_sx;
            best_slot_sy = slot_sy;
            if (std::isfinite(center_x) && std::isfinite(center_y)) {
                set_prior_diag(prior_center_x_col, center_x);
                set_prior_diag(prior_center_y_col, center_y);
            }
        }
    }

    set_prior_diag(prior_n_candidates_gate_col, static_cast<double>(n_gate));

    if (!found) {
        set_prior_diag(prior_no_candidate_reason_col, prior_reason_gate_rejected);
        return false;
    }

    init_row = static_cast<double>(best_row);
    init_col = static_cast<double>(best_col);
    set_prior_diag(prior_used_col, 1.0);
    set_prior_diag(prior_no_candidate_reason_col, prior_reason_none);
    set_prior_diag(prior_slot_index_col, static_cast<double>(best_slot));
    set_prior_diag(prior_match_d2_col, best_d2);
    set_prior_diag(prior_match_score_col, best_score);
    set_prior_diag(prior_candidate_snr_col, best_snr);
    set_prior_diag(prior_candidate_x_raw_col, best_x_raw);
    set_prior_diag(prior_candidate_y_raw_col, best_y_raw);
    set_prior_diag(prior_candidate_x_prior_col, best_x_prior);
    set_prior_diag(prior_candidate_y_prior_col, best_y_prior);
    set_prior_diag(prior_slot_x_col, best_slot_x);
    set_prior_diag(prior_slot_y_col, best_slot_y);
    set_prior_diag(prior_slot_sx_col, best_slot_sx);
    set_prior_diag(prior_slot_sy_col, best_slot_sy);
    logger->debug(
        "beammap priors init map={} det={} array={} nw={} row={} col={} snr={} d2={} slot={} lambda={} max_d2={}",
        map_index, map_index, array, nw, init_row, init_col, best_snr, best_d2,
        best_slot, prior_score_lambda, prior_max_d2);
    return true;
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
        logger->info("starting iter {}", current_iter);

        configure_detector_source_centers_from_previous_fit();
        const bool detector_kernel_source_centers_active =
            map_grouping == "detector" &&
            rtcproc.run_kernel &&
            rtcproc.kernel.has_source_centers();
        const bool rerun_source_aware_rtc =
            current_iter == 1 && detector_kernel_source_centers_active;
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
        if (run_mapmaking && beammap_iter_tolerance > 0.0 && current_iter > 0) {
            omb_copy.signal = omb.signal;
            omb_copy.weight = omb.weight;
        }

        if (ptcproc.run_fruit_loops) {
            // calc mean rms
            if (current_iter == 1) {
                // use obs map buffer
                if (!omb.noise.empty()) {
                    omb.calc_median_rms();
                }
            }
            if (current_iter > 0) {
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
                if (current_iter > 0) {
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
                if (current_iter > 0) {
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

            // For detector-grouped beammaps, keep the first pass permissive so
            // bright-source scans are less likely to be rejected before we have
            // any source-location estimate to feed back into later iterations.
            if (map_grouping == "detector" && current_iter == 0) {
                logger->info("skipping remove_bad_dets on iter 0 for beammap detector scan {}", ptcs[i].index.data + 1);
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
                    (beammap_detector_weighting_mode == "ptc_after_iter0" && current_iter > 0);
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

        auto write_detector_specific_ptc_tod = [&](int output_iter) {
            if (!beammap_detector_tod_output_enabled) {
                return;
            }
            const Eigen::Index n_scans = telescope.scan_indices.cols();
            if (n_scans <= 0) {
                logger->error("cannot write detector-specific PTC TOD: no scans");
                std::exit(EXIT_FAILURE);
            }
            if (map_grouping != "detector") {
                logger->warn(
                    "beammap.detector_tod_output requires detector map grouping; skipping detector-specific PTC TOD");
                return;
            }
            const int n_uniform = std::max(0, beammap_detector_tod_output_n_uniform);
            const int n_dense = std::max(0, beammap_detector_tod_output_n_source_dense);
            const Eigen::Index n_slots = static_cast<Eigen::Index>(n_uniform + n_dense);
            if (n_slots <= 0) {
                logger->warn("beammap.detector_tod_output requested with no output slots; skipping");
                return;
            }
            Eigen::Index n_samples_max = 0;
            for (const auto &ptc : ptcs) {
                n_samples_max = std::max<Eigen::Index>(n_samples_max, ptc.scans.data.rows());
            }
            if (n_samples_max <= 0) {
                logger->warn("beammap.detector_tod_output has no PTC samples to write; skipping");
                return;
            }

            std::vector<Eigen::Index> uniform_scans;
            uniform_scans.reserve(static_cast<std::size_t>(n_uniform));
            for (int i = 0; i < n_uniform; ++i) {
                Eigen::Index scan_index = (n_scans - 1) / 2;
                if (n_uniform > 1) {
                    const double frac = static_cast<double>(i) /
                                        static_cast<double>(n_uniform - 1);
                    scan_index = static_cast<Eigen::Index>(std::llround(frac * (n_scans - 1)));
                }
                uniform_scans.push_back(std::clamp<Eigen::Index>(scan_index, 0, n_scans - 1));
            }

            auto dense_window = [&](Eigen::Index center_scan) {
                std::vector<Eigen::Index> scans;
                scans.reserve(static_cast<std::size_t>(n_dense));
                if (n_dense <= 0) {
                    return scans;
                }
                if (n_dense > n_scans) {
                    for (int i = 0; i < n_dense; ++i) {
                        Eigen::Index scan_index = 0;
                        if (n_dense > 1) {
                            const double frac = static_cast<double>(i) /
                                                static_cast<double>(n_dense - 1);
                            scan_index = static_cast<Eigen::Index>(std::llround(frac * (n_scans - 1)));
                        }
                        scans.push_back(std::clamp<Eigen::Index>(scan_index, 0, n_scans - 1));
                    }
                    return scans;
                }
                Eigen::Index first_dense =
                    center_scan - static_cast<Eigen::Index>((n_dense - 1) / 2);
                first_dense = std::clamp<Eigen::Index>(
                    first_dense, 0, std::max<Eigen::Index>(0, n_scans - static_cast<Eigen::Index>(n_dense)));
                for (int i = 0; i < n_dense; ++i) {
                    scans.push_back(first_dense + static_cast<Eigen::Index>(i));
                }
                return scans;
            };

            std::vector<Eigen::Index> sampled_indices;
            std::vector<Eigen::Index> sampled_scan;
            for (Eigen::Index scan_index = 0; scan_index < n_scans; ++scan_index) {
                const Eigen::Index start = std::max<Eigen::Index>(0, telescope.scan_indices(0, scan_index));
                const Eigen::Index end = std::min<Eigen::Index>(
                    telescope.scan_indices(1, scan_index),
                    telescope.tel_data.empty() ? telescope.scan_indices(1, scan_index)
                                               : telescope.tel_data.begin()->second.size() - 1);
                if (end < start) {
                    continue;
                }
                const Eigen::Index n_scan_pts = end - start + 1;
                const Eigen::Index stride = std::max<Eigen::Index>(1, n_scan_pts / 96);
                Eigen::Index last_sample = -1;
                for (Eigen::Index sample = start; sample <= end; sample += stride) {
                    sampled_indices.push_back(sample);
                    sampled_scan.push_back(scan_index);
                    last_sample = sample;
                }
                if (last_sample != end) {
                    sampled_indices.push_back(end);
                    sampled_scan.push_back(scan_index);
                }
            }
            const Eigen::Index n_sampled = static_cast<Eigen::Index>(sampled_indices.size());
            if (n_sampled <= 0) {
                logger->warn("beammap.detector_tod_output cannot sample telescope pointing; skipping");
                return;
            }

            std::map<std::string, Eigen::VectorXd> sampled_tel_data;
            for (const auto &[key, values] : telescope.tel_data) {
                Eigen::VectorXd sampled(n_sampled);
                for (Eigen::Index i = 0; i < n_sampled; ++i) {
                    const Eigen::Index sample = sampled_indices[static_cast<std::size_t>(i)];
                    sampled(i) = (sample >= 0 && sample < values.size())
                                     ? values(sample)
                                     : std::numeric_limits<double>::quiet_NaN();
                }
                sampled_tel_data[key] = std::move(sampled);
            }
            std::map<std::string, Eigen::VectorXd> pointing_offsets;
            auto make_sampled_offset = [&](const std::string &axis) {
                Eigen::VectorXd sampled = Eigen::VectorXd::Zero(n_sampled);
                auto it = pointing_offsets_arcsec.find(axis);
                if (it == pointing_offsets_arcsec.end()) {
                    return sampled;
                }
                for (Eigen::Index i = 0; i < n_sampled; ++i) {
                    const Eigen::Index sample = sampled_indices[static_cast<std::size_t>(i)];
                    if (sample >= 0 && sample < it->second.size()) {
                        sampled(i) = it->second(sample);
                    }
                }
                return sampled;
            };
            pointing_offsets["az"] = make_sampled_offset("az");
            pointing_offsets["alt"] = make_sampled_offset("alt");

            auto scan_distances_for_detector = [&](Eigen::Index det, double source_x_arcsec,
                                                   double source_y_arcsec,
                                                   std::vector<double> &distances_arcsec) {
                Eigen::Index best_scan = (n_scans - 1) / 2;
                distances_arcsec.assign(static_cast<std::size_t>(n_scans),
                                        std::numeric_limits<double>::quiet_NaN());
                if (!std::isfinite(source_x_arcsec) || !std::isfinite(source_y_arcsec) ||
                    det < 0 || det >= calib.n_dets ||
                    det >= calib.apt["x_t"].size() || det >= calib.apt["y_t"].size() ||
                    !std::isfinite(calib.apt["x_t"](det)) ||
                    !std::isfinite(calib.apt["y_t"](det))) {
                    return best_scan;
                }
                double best_d2 = std::numeric_limits<double>::infinity();
                const double source_x_rad = source_x_arcsec * ASEC_TO_RAD;
                const double source_y_rad = source_y_arcsec * ASEC_TO_RAD;
                // Use the detector pointing that built the map, then find where
                // that pointing passes closest to the fitted source location.
                auto [lat, lon] = engine_utils::calc_det_pointing(
                    sampled_tel_data, calib.apt["x_t"](det), calib.apt["y_t"](det),
                    telescope.pixel_axes,
                    pointing_offsets, map_grouping, true);

                std::vector<double> best_d2_by_scan(static_cast<std::size_t>(n_scans),
                                                    std::numeric_limits<double>::infinity());
                for (Eigen::Index sample_i = 0; sample_i < n_sampled; ++sample_i) {
                    if (sample_i >= lat.size() || sample_i >= lon.size()) {
                        continue;
                    }
                    const double y = lat(sample_i) - source_y_rad;
                    const double x = lon(sample_i) - source_x_rad;
                    if (!std::isfinite(x) || !std::isfinite(y)) {
                        continue;
                    }
                    const auto scan_index = sampled_scan[static_cast<std::size_t>(sample_i)];
                    if (scan_index < 0 || scan_index >= n_scans) {
                        continue;
                    }
                    const double d2 = x * x + y * y;
                    auto &scan_best = best_d2_by_scan[static_cast<std::size_t>(scan_index)];
                    if (d2 < scan_best) {
                        scan_best = d2;
                    }
                    if (d2 < best_d2) {
                        best_d2 = d2;
                        best_scan = scan_index;
                    }
                }
                for (Eigen::Index scan_index = 0; scan_index < n_scans; ++scan_index) {
                    const double d2 = best_d2_by_scan[static_cast<std::size_t>(scan_index)];
                    if (std::isfinite(d2)) {
                        distances_arcsec[static_cast<std::size_t>(scan_index)] = std::sqrt(d2) * RAD_TO_ASEC;
                    }
                }
                return best_scan;
            };

            const int fill_int = -2147483647;
            const double fill_double = std::numeric_limits<double>::quiet_NaN();
            const float fill_float = std::numeric_limits<float>::quiet_NaN();
            const signed char fill_flag = static_cast<signed char>(-1);
            const auto total_det_slots =
                static_cast<std::size_t>(calib.n_dets) * static_cast<std::size_t>(n_slots);
            std::vector<int> slot_scan_index(total_det_slots, fill_int);
            std::vector<int> slot_kind(total_det_slots, fill_int);
            std::vector<int> slot_n_samples(total_det_slots, fill_int);
            std::vector<int> slot_inner_start(total_det_slots, fill_int);
            std::vector<int> slot_inner_end(total_det_slots, fill_int);
            std::vector<int> slot_outer_start(total_det_slots, fill_int);
            std::vector<int> slot_outer_end(total_det_slots, fill_int);
            std::vector<double> slot_source_distance_arcsec(total_det_slots, fill_double);
            std::vector<int> det_center_scan_index(static_cast<std::size_t>(calib.n_dets), fill_int);
            std::vector<double> det_center_distance_arcsec(static_cast<std::size_t>(calib.n_dets), fill_double);
            std::vector<double> det_fit_x_arcsec(static_cast<std::size_t>(calib.n_dets), fill_double);
            std::vector<double> det_fit_y_arcsec(static_cast<std::size_t>(calib.n_dets), fill_double);
            std::vector<int> det_fit_good(static_cast<std::size_t>(calib.n_dets), 0);
            Eigen::Index n_det_fit_positions = 0;
            Eigen::Index n_det_fallback_positions = 0;
            std::map<Eigen::Index, Eigen::Index> center_scan_counts;
            std::vector<double> center_distances;
            center_distances.reserve(static_cast<std::size_t>(calib.n_dets));

            auto flat = [&](Eigen::Index det, Eigen::Index slot) {
                return static_cast<std::size_t>(det) * static_cast<std::size_t>(n_slots) +
                       static_cast<std::size_t>(slot);
            };
            auto get_detector_source_position = [&](Eigen::Index det, bool &used_fit) {
                used_fit = false;
                double x_arcsec = std::numeric_limits<double>::quiet_NaN();
                double y_arcsec = std::numeric_limits<double>::quiet_NaN();
                const bool fit_ok = det < good_fits.size() && good_fits(det) &&
                                    det < params.rows() && params.cols() > 2 &&
                                    std::isfinite(params(det, 1)) &&
                                    std::isfinite(params(det, 2));
                if (fit_ok) {
                    x_arcsec = RAD_TO_ASEC * omb.pixel_size_rad *
                               (params(det, 1) - (omb.n_cols - 1) / 2.0);
                    y_arcsec = RAD_TO_ASEC * omb.pixel_size_rad *
                               (params(det, 2) - (omb.n_rows - 1) / 2.0);
                    used_fit = true;
                }
                if ((!std::isfinite(x_arcsec) || !std::isfinite(y_arcsec)) &&
                    det < calib.apt["x_t"].size() && det < calib.apt["y_t"].size()) {
                    x_arcsec = calib.apt["x_t"](det);
                    y_arcsec = calib.apt["y_t"](det);
                }
                return std::pair<double, double>{x_arcsec, y_arcsec};
            };

            for (Eigen::Index det = 0; det < calib.n_dets; ++det) {
                bool used_fit = false;
                auto [x_arcsec, y_arcsec] = get_detector_source_position(det, used_fit);
                det_fit_x_arcsec[static_cast<std::size_t>(det)] = x_arcsec;
                det_fit_y_arcsec[static_cast<std::size_t>(det)] = y_arcsec;
                det_fit_good[static_cast<std::size_t>(det)] = used_fit ? 1 : 0;
                if (used_fit) {
                    n_det_fit_positions++;
                }
                else if (std::isfinite(x_arcsec) && std::isfinite(y_arcsec)) {
                    n_det_fallback_positions++;
                }
                std::vector<double> distances_arcsec;
                const Eigen::Index center_scan = scan_distances_for_detector(
                    det, x_arcsec, y_arcsec, distances_arcsec);
                det_center_scan_index[static_cast<std::size_t>(det)] = static_cast<int>(center_scan + 1);
                center_scan_counts[center_scan]++;
                if (center_scan >= 0 && center_scan < n_scans &&
                    std::isfinite(distances_arcsec[static_cast<std::size_t>(center_scan)])) {
                    det_center_distance_arcsec[static_cast<std::size_t>(det)] =
                        distances_arcsec[static_cast<std::size_t>(center_scan)];
                    center_distances.push_back(distances_arcsec[static_cast<std::size_t>(center_scan)]);
                }

                Eigen::Index slot = 0;
                for (const auto scan_index : uniform_scans) {
                    const auto idx = flat(det, slot);
                    slot_scan_index[idx] = static_cast<int>(scan_index + 1);
                    slot_kind[idx] = 1;
                    if (scan_index >= 0 && scan_index < n_scans) {
                        slot_inner_start[idx] = static_cast<int>(telescope.scan_indices(0, scan_index));
                        slot_inner_end[idx] = static_cast<int>(telescope.scan_indices(1, scan_index));
                        slot_outer_start[idx] = static_cast<int>(telescope.scan_indices(2, scan_index));
                        slot_outer_end[idx] = static_cast<int>(telescope.scan_indices(3, scan_index));
                        if (scan_index < static_cast<Eigen::Index>(ptcs.size())) {
                            slot_n_samples[idx] = static_cast<int>(ptcs[scan_index].scans.data.rows());
                        }
                        if (scan_index < static_cast<Eigen::Index>(distances_arcsec.size())) {
                            slot_source_distance_arcsec[idx] =
                                distances_arcsec[static_cast<std::size_t>(scan_index)];
                        }
                    }
                    slot++;
                }
                for (const auto scan_index : dense_window(center_scan)) {
                    const auto idx = flat(det, slot);
                    slot_scan_index[idx] = static_cast<int>(scan_index + 1);
                    slot_kind[idx] = 2;
                    if (scan_index >= 0 && scan_index < n_scans) {
                        slot_inner_start[idx] = static_cast<int>(telescope.scan_indices(0, scan_index));
                        slot_inner_end[idx] = static_cast<int>(telescope.scan_indices(1, scan_index));
                        slot_outer_start[idx] = static_cast<int>(telescope.scan_indices(2, scan_index));
                        slot_outer_end[idx] = static_cast<int>(telescope.scan_indices(3, scan_index));
                        if (scan_index < static_cast<Eigen::Index>(ptcs.size())) {
                            slot_n_samples[idx] = static_cast<int>(ptcs[scan_index].scans.data.rows());
                        }
                        if (scan_index < static_cast<Eigen::Index>(distances_arcsec.size())) {
                            slot_source_distance_arcsec[idx] =
                                distances_arcsec[static_cast<std::size_t>(scan_index)];
                        }
                    }
                    slot++;
                }
            }

            std::vector<std::pair<Eigen::Index, Eigen::Index>> center_hist(center_scan_counts.begin(),
                                                                           center_scan_counts.end());
            std::sort(center_hist.begin(), center_hist.end(),
                      [](const auto &lhs, const auto &rhs) {
                          if (lhs.second != rhs.second) {
                              return lhs.second > rhs.second;
                          }
                          return lhs.first < rhs.first;
                      });
            std::ostringstream center_os;
            center_os << "[";
            for (std::size_t i = 0; i < std::min<std::size_t>(8, center_hist.size()); ++i) {
                if (i != 0) {
                    center_os << ", ";
                }
                center_os << center_hist[i].first + 1 << ":" << center_hist[i].second;
            }
            center_os << "]";

            double median_center_distance_arcsec = std::numeric_limits<double>::quiet_NaN();
            if (!center_distances.empty()) {
                Eigen::Map<Eigen::VectorXd> dist_vec(
                    center_distances.data(),
                    static_cast<Eigen::Index>(center_distances.size()));
                median_center_distance_arcsec = tula::alg::median(dist_vec);
            }

            namespace fs = std::filesystem;
            std::string dir_name = obsnum_dir_name + "raw/";
            if (beammap_detector_tod_output_subdir_name != "null") {
                dir_name += beammap_detector_tod_output_subdir_name + "/";
            }
            fs::create_directories(dir_name);
            std::string filename = dir_name + "toltec";
            filename += telescope.sim_obs ? "_simu" : "_commissioning";
            filename += "_" + redu_type + "_" + obsnum + "_ptc_detector_tod.nc";

            logger->info(
                "writing detector-specific PTC TOD iter={} file={} n_dets={} n_slots={} n_uniform={} n_source_dense={} fit_positions={} fallback_positions={} median_center_distance_arcsec={} top_center_scans={}",
                output_iter,
                filename,
                calib.n_dets,
                n_slots,
                n_uniform,
                n_dense,
                n_det_fit_positions,
                n_det_fallback_positions,
                median_center_distance_arcsec,
                center_os.str());

            write_netcdf_atomic(filename, [&](netCDF::NcFile &fo) {
                netCDF::NcDim n_tod_output_type_dim = fo.addDim("n_tod_output_type", 1);
                netCDF::NcVar tod_output_type_var =
                    fo.addVar("tod_output_type", netCDF::ncString, n_tod_output_type_dim);
                const std::vector<size_t> tod_output_type_index = {0};
                std::string tod_output_type_name = "ptc_detector_tod";
                tod_output_type_var.putVar(tod_output_type_index, tod_output_type_name);

                netCDF::NcVar obsnum_v = fo.addVar("obsnum", netCDF::ncInt);
                obsnum_v.putAtt("units", "N/A");
                int obsnum_int = std::stoi(obsnum);
                obsnum_v.putVar(&obsnum_int);
                add_netcdf_var<std::string>(fo, "SOURCE", telescope.source_name);
                add_netcdf_var<std::string>(fo, "PROJID", telescope.project_id);
                add_netcdf_var<std::string>(fo, "GOAL", redu_type);
                add_netcdf_var<std::string>(fo, "OBSGOAL", telescope.obs_goal);
                add_netcdf_var<std::string>(fo, "TYPE", tod_type);
                add_netcdf_var<std::string>(fo, "PIPELINE", "CITLALI");
                add_netcdf_var<std::string>(fo, "VERSION", CITLALI_GIT_VERSION);
                add_netcdf_var<std::string>(fo, "KIDS", KIDSCPP_GIT_VERSION);
                add_netcdf_var<std::string>(fo, "TULA", TULA_GIT_VERSION);
                add_netcdf_var(fo, "SourceRa", telescope.tel_header["Header.Source.Ra"](0));
                add_netcdf_var(fo, "SourceDec", telescope.tel_header["Header.Source.Dec"](0));
                add_netcdf_var(fo, "PTC_SAMPRATE", processed_time_chunk_fs_hz());
                add_netcdf_var(fo, "FRUITLOOPS_ITER", output_iter);
                add_netcdf_var(fo, "CONFIG.BEAMMAP.DETECTOR_TOD.N_UNIFORM", n_uniform);
                add_netcdf_var(fo, "CONFIG.BEAMMAP.DETECTOR_TOD.N_SOURCE_DENSE", n_dense);

                netCDF::NcDim n_dets_dim = fo.addDim("n_dets", calib.n_dets);
                netCDF::NcDim n_slots_dim = fo.addDim("n_detector_tod_slots", n_slots);
                netCDF::NcDim n_samples_dim = fo.addDim("n_samples_max", n_samples_max);
                std::vector<netCDF::NcDim> det_dims = {n_dets_dim};
                std::vector<netCDF::NcDim> det_slot_dims = {n_dets_dim, n_slots_dim};
                std::vector<netCDF::NcDim> data_dims = {n_dets_dim, n_slots_dim, n_samples_dim};
                std::vector<std::size_t> det_slot_chunks = {1, static_cast<std::size_t>(n_slots)};
                std::vector<std::size_t> data_chunks = {
                    1, 1, static_cast<std::size_t>(n_samples_max)};

                auto put_det_int = [&](const std::string &name, const std::string &comment,
                                       const std::vector<int> &values) {
                    netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, det_dims);
                    v.putAtt("units", "N/A");
                    v.putAtt("comment", comment);
                    v.putVar(values.data());
                };
                auto put_det_double = [&](const std::string &name, const std::string &units,
                                          const std::string &comment,
                                          const std::vector<double> &values) {
                    netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, det_dims);
                    v.putAtt("units", units);
                    v.putAtt("comment", comment);
                    v.putVar(values.data());
                };
                auto put_slot_int = [&](const std::string &name, const std::string &comment,
                                        const std::vector<int> &values) {
                    netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, det_slot_dims);
                    v.putAtt("units", "N/A");
                    v.putAtt("comment", comment);
                    set_netcdf_chunking_and_compression(v, det_slot_chunks, 1);
                    v.putVar(values.data());
                };
                auto put_slot_double = [&](const std::string &name, const std::string &units,
                                           const std::string &comment,
                                           const std::vector<double> &values) {
                    netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, det_slot_dims);
                    v.putAtt("units", units);
                    v.putAtt("comment", comment);
                    set_netcdf_chunking_and_compression(v, det_slot_chunks, 1);
                    v.putVar(values.data());
                };
                auto apt_int_values = [&](const std::string &key) {
                    std::vector<int> values(static_cast<std::size_t>(calib.n_dets), fill_int);
                    auto it = calib.apt.find(key);
                    if (it != calib.apt.end() && it->second.size() == calib.n_dets) {
                        for (Eigen::Index det = 0; det < calib.n_dets; ++det) {
                            values[static_cast<std::size_t>(det)] =
                                static_cast<int>(std::llround(it->second(det)));
                        }
                    }
                    return values;
                };

                put_det_int("detector_tod_uid", "detector UID along n_dets", apt_int_values("uid"));
                put_det_int("detector_tod_array", "array index along n_dets", apt_int_values("array"));
                put_det_int("detector_tod_network", "network index along n_dets", apt_int_values("nw"));
                put_det_int("detector_tod_fit_good", "1 when the source-crossing scan was centered on a good fit, else 0",
                            det_fit_good);
                put_det_int("detector_tod_source_center_scan_index",
                            "1-based full-observation scan with closest approach to the fitted detector source position",
                            det_center_scan_index);
                put_det_double("detector_tod_source_center_distance_arcsec", "arcsec",
                               "closest sampled distance from source in detector source-center scan",
                               det_center_distance_arcsec);
                put_det_double("detector_tod_fit_x_t_arcsec", "arcsec",
                               "fitted detector source x_t used for dense scan selection",
                               det_fit_x_arcsec);
                put_det_double("detector_tod_fit_y_t_arcsec", "arcsec",
                               "fitted detector source y_t used for dense scan selection",
                               det_fit_y_arcsec);

                put_slot_int("detector_tod_slot_kind",
                             "slot kind: 1=uniform over full raster, 2=dense around detector source crossing",
                             slot_kind);
                put_slot_int("detector_tod_scan_index",
                             "1-based full-observation scan index selected for this detector/slot",
                             slot_scan_index);
                put_slot_int("detector_tod_n_samples",
                             "number of PTC samples populated for this detector/slot",
                             slot_n_samples);
                put_slot_int("detector_tod_scan_inner_start_sample",
                             "raw inner-scan start sample from telescope scan definition",
                             slot_inner_start);
                put_slot_int("detector_tod_scan_inner_end_sample",
                             "raw inner-scan end sample from telescope scan definition",
                             slot_inner_end);
                put_slot_int("detector_tod_scan_outer_start_sample",
                             "raw outer-scan start sample from telescope scan definition",
                             slot_outer_start);
                put_slot_int("detector_tod_scan_outer_end_sample",
                             "raw outer-scan end sample from telescope scan definition",
                             slot_outer_end);
                put_slot_double("detector_tod_source_distance_arcsec", "arcsec",
                                "closest sampled source distance for this detector on this selected scan",
                                slot_source_distance_arcsec);

                netCDF::NcVar signal_v = fo.addVar("signal", netCDF::ncFloat, data_dims);
                signal_v.putAtt("units", omb.sig_unit);
                signal_v.putAtt("comment", "PTC signal for each detector-specific selected scan; unused samples are NaN");
                set_netcdf_chunking_and_compression(signal_v, data_chunks, 1);

                netCDF::NcVar flags_v = fo.addVar("flags", netCDF::ncByte, data_dims);
                flags_v.putAtt("units", "N/A");
                flags_v.putAtt("comment", "0=good, 1=flagged, -1=unused sample");
                set_netcdf_chunking_and_compression(flags_v, data_chunks, 1);

                std::vector<float> signal_block(
                    static_cast<std::size_t>(n_slots) * static_cast<std::size_t>(n_samples_max),
                    fill_float);
                std::vector<signed char> flags_block(
                    static_cast<std::size_t>(n_slots) * static_cast<std::size_t>(n_samples_max),
                    fill_flag);
                for (Eigen::Index det = 0; det < calib.n_dets; ++det) {
                    std::fill(signal_block.begin(), signal_block.end(), fill_float);
                    std::fill(flags_block.begin(), flags_block.end(), fill_flag);
                    for (Eigen::Index slot = 0; slot < n_slots; ++slot) {
                        const auto meta_idx = flat(det, slot);
                        const int scan_1based = slot_scan_index[meta_idx];
                        if (scan_1based <= 0) {
                            continue;
                        }
                        const Eigen::Index scan_index = static_cast<Eigen::Index>(scan_1based - 1);
                        if (scan_index < 0 || scan_index >= static_cast<Eigen::Index>(ptcs.size())) {
                            continue;
                        }
                        const auto &ptc = ptcs[scan_index];
                        if (det >= ptc.scans.data.cols() || det >= ptc.flags.data.cols()) {
                            continue;
                        }
                        const Eigen::Index n_copy = std::min<Eigen::Index>(
                            n_samples_max, ptc.scans.data.rows());
                        for (Eigen::Index sample = 0; sample < n_copy; ++sample) {
                            const auto data_idx =
                                static_cast<std::size_t>(slot) * static_cast<std::size_t>(n_samples_max) +
                                static_cast<std::size_t>(sample);
                            signal_block[data_idx] = static_cast<float>(ptc.scans.data(sample, det));
                            flags_block[data_idx] =
                                ptc.flags.data(sample, det) ? static_cast<signed char>(1)
                                                            : static_cast<signed char>(0);
                        }
                    }
                    std::vector<std::size_t> start = {static_cast<std::size_t>(det), 0, 0};
                    std::vector<std::size_t> size = {
                        1, static_cast<std::size_t>(n_slots), static_cast<std::size_t>(n_samples_max)};
                    signal_v.putVar(start, size, signal_block.data());
                    flags_v.putVar(start, size, flags_block.data());
                }
            });
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

            if (beammap_scan_band_mask_enabled && map_grouping == "detector" && current_iter == 0) {
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
                    if (current_iter > 0 &&
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
            else if (current_iter > 1) {
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

        // only write apt table if beammapping
        if (map_grouping=="detector") {
            logger->info("writing apt table");
            auto apt_filename = toltec_io.create_filename<engine_utils::toltecIO::apt, engine_utils::toltecIO::map,
                                                          engine_utils::toltecIO::raw>
                                (obsnum_dir_name + "raw/", redu_type, "", obsnum, telescope.sim_obs);

            Eigen::MatrixXd apt_table(calib.n_dets, calib.apt_header_keys.size());

            // copy to table
            Eigen::Index i = 0;
            for (auto const& x: calib.apt_header_keys) {
                if (x != "flag2") {
                    apt_table.col(i) = calib.apt[x];
                }
                else {
                    apt_table.col(i) = flag2.cast<double> ();
                }
                i++;
            }

            // write to ecsv
            to_ecsv_from_matrix(apt_filename, apt_table, calib.apt_header_keys, calib.apt_meta);

            logger->info("done writing apt table {}.ecsv",apt_filename);

            logger->info("writing beammap fit qc table");
            std::string fit_qc_filename = apt_filename + "_fit_qc";

            std::vector<std::string> fit_qc_header = {
                "uid",
                "array",
                "nw",
                "kids_tone",
                "good_fit",
                "converged",
                "converge_iter",
                "flag",
                "flag2",
                "amp",
                "amp_err",
                "cal_amp",
                "cal_amp_method",
                "template_amp",
                "template_offset",
                "template_resid_rms",
                "template_npix",
                "template_amp_over_fit_amp",
                "cal_amp_over_fit_amp",
                "map_peak_amp",
                "map_peak_amp_over_fit_amp",
                "fit_sig2noise",
                "map_rms",
                "map_sig2noise",
                "n_weight_pos",
                "fruitloops_source_x_t",
                "fruitloops_source_y_t",
                "fruitloops_local_sigma",
                "fruitloops_local_sigma_npix",
                "fruitloops_amp_ref",
                "fruitloops_peak_threshold",
                "fruitloops_snr_threshold",
                "fruitloops_adaptive_threshold",
                "fruitloops_support_radius_arcsec",
                "fruitloops_support_npix",
                "fruitloops_support_signal_sum",
                "fruitloops_support_x_span_arcsec",
                "fruitloops_support_y_span_arcsec",
                "rfi_masked_samples",
                "rfi_masked_scans",
                "scan_band_masked_samples",
                "scan_band_masked_rows",
                "scan_band_masked_edge",
                "scan_band_mask_rejected",
                "fit_bound_nhit",
                "fit_bound_code",
                "fit_bound_amp",
                "fit_bound_x",
                "fit_bound_y",
                "fit_bound_a",
                "fit_bound_b",
                "fit_bound_angle",
                "fit_init_amp",
                "fit_init_x_t",
                "fit_init_y_t",
                "fit_init_a_fwhm",
                "fit_init_b_fwhm",
                "fit_low_a_fwhm",
                "fit_high_a_fwhm",
                "fit_low_b_fwhm",
                "fit_high_b_fwhm",
                "prior_init_mode",
                "prior_used",
                "prior_fallback_blind",
                "prior_no_candidate_reason",
                "prior_slot_index",
                "prior_match_d2",
                "prior_match_score",
                "prior_candidate_snr",
                "prior_n_candidates",
                "prior_n_candidates_keep",
                "prior_n_candidates_gate",
                "prior_candidate_x_t_raw",
                "prior_candidate_y_t_raw",
                "prior_candidate_x_t_prior",
                "prior_candidate_y_t_prior",
                "prior_center_x_t",
                "prior_center_y_t",
                "prior_derot_elev",
                "prior_slot_x_t",
                "prior_slot_y_t",
                "prior_slot_sx",
                "prior_slot_sy",
                "final_prior_slot_index",
                "final_prior_d2",
                "x_t_raw",
                "y_t_raw",
                "x_t",
                "y_t",
                "x_t_derot",
                "y_t_derot",
                "a_fwhm",
                "a_fwhm_err",
                "b_fwhm",
                "b_fwhm_err",
                "angle",
                "angle_err",
                "flxscale",
                "sens"
            };

            auto apt_or_zero = [&](const std::string &key) -> Eigen::VectorXd {
                auto it = calib.apt.find(key);
                if (it != calib.apt.end() && it->second.size() == calib.n_dets) {
                    return it->second;
                }
                return Eigen::VectorXd::Zero(calib.n_dets);
            };
            auto get_unit = [&](const std::string &key, const std::string &fallback) {
                auto it = calib.apt_header_units.find(key);
                if (it != calib.apt_header_units.end()) {
                    return it->second;
                }
                return fallback;
            };
            auto get_description = [&](const std::string &key, const std::string &fallback) {
                auto it = calib.apt_header_description.find(key);
                if (it != calib.apt_header_description.end()) {
                    return it->second;
                }
                return fallback;
            };
            auto prior_diag_or = [&](PriorDiagColumn diag_col, double fallback_value) -> Eigen::VectorXd {
                Eigen::VectorXd out(calib.n_dets);
                if (prior_diag_values.rows() == calib.n_dets && prior_diag_values.cols() == n_prior_diag_cols) {
                    out = prior_diag_values.col(diag_col);
                }
                else {
                    out.setConstant(fallback_value);
                }
                return out;
            };

            Eigen::VectorXd map_rms(calib.n_dets);
            Eigen::VectorXd fit_sig2noise(calib.n_dets);
            Eigen::VectorXd map_sig2noise(calib.n_dets);
            Eigen::VectorXd n_weight_pos(calib.n_dets);
            map_rms.setZero();
            fit_sig2noise.setZero();
            map_sig2noise.setZero();
            n_weight_pos.setZero();
            for (Eigen::Index i = 0; i < calib.n_dets; ++i) {
                const double amp = params(i, 0);
                const double amp_err = perrors(i, 0);
                const double rms = calc_map_support_stddev(i, true);
                const double npos = static_cast<double>((omb.weight[i].array() > 0.0).count());
                n_weight_pos(i) = npos;
                if (std::isfinite(rms) && rms > 0.0) {
                    map_rms(i) = rms;
                    if (std::isfinite(amp)) {
                        map_sig2noise(i) = amp / rms;
                    }
                }
                if (std::isfinite(amp) && std::isfinite(amp_err) && amp_err > 0.0) {
                    fit_sig2noise(i) = amp / amp_err;
                }
            }

            const double pix_to_arcsec = RAD_TO_ASEC * omb.pixel_size_rad;
            const double sigma_to_fwhm_arcsec = pix_to_arcsec * STD_TO_FWHM;

            Eigen::VectorXd fit_bound_nhit = fit_diag_bound_nhit.cast<double>();
            Eigen::VectorXd fit_bound_code = fit_diag_bound_code.cast<double>();
            auto bound_state = [&](Eigen::Index p) -> Eigen::VectorXd {
                return fit_diag_hit_upper.col(p).cast<double>() - fit_diag_hit_lower.col(p).cast<double>();
            };
            Eigen::VectorXd fit_bound_amp = bound_state(0);
            Eigen::VectorXd fit_bound_x = bound_state(1);
            Eigen::VectorXd fit_bound_y = bound_state(2);
            Eigen::VectorXd fit_bound_a = bound_state(3);
            Eigen::VectorXd fit_bound_b = bound_state(4);
            Eigen::VectorXd fit_bound_angle = bound_state(5);

            Eigen::VectorXd fit_init_amp = fit_diag_init_params.col(0);
            Eigen::VectorXd fit_init_x_t =
                (pix_to_arcsec * (fit_diag_init_params.col(1).array() - (omb.n_cols - 1) / 2.0)).matrix();
            Eigen::VectorXd fit_init_y_t =
                (pix_to_arcsec * (fit_diag_init_params.col(2).array() - (omb.n_rows - 1) / 2.0)).matrix();
            Eigen::VectorXd fit_init_a_fwhm = (sigma_to_fwhm_arcsec * fit_diag_init_params.col(3).array()).matrix();
            Eigen::VectorXd fit_init_b_fwhm = (sigma_to_fwhm_arcsec * fit_diag_init_params.col(4).array()).matrix();
            Eigen::VectorXd fit_low_a_fwhm = (sigma_to_fwhm_arcsec * fit_diag_lower_limits.col(3).array()).matrix();
            Eigen::VectorXd fit_high_a_fwhm = (sigma_to_fwhm_arcsec * fit_diag_upper_limits.col(3).array()).matrix();
            Eigen::VectorXd fit_low_b_fwhm = (sigma_to_fwhm_arcsec * fit_diag_lower_limits.col(4).array()).matrix();
            Eigen::VectorXd fit_high_b_fwhm = (sigma_to_fwhm_arcsec * fit_diag_upper_limits.col(4).array()).matrix();

            const double fill_double = std::numeric_limits<double>::quiet_NaN();
            auto fruitloops_or_nan = [&](const Eigen::VectorXd &values,
                                         double scale = 1.0) -> Eigen::VectorXd {
                Eigen::VectorXd out = Eigen::VectorXd::Constant(calib.n_dets, fill_double);
                if (values.size() == calib.n_dets) {
                    out = (scale * values.array()).matrix();
                }
                return out;
            };
            auto fruitloops_int_or_nan = [&](const Eigen::VectorXi &values) -> Eigen::VectorXd {
                Eigen::VectorXd out = Eigen::VectorXd::Constant(calib.n_dets, fill_double);
                if (values.size() == calib.n_dets) {
                    out = values.cast<double>();
                }
                return out;
            };

            Eigen::VectorXd fruitloops_source_x_t =
                fruitloops_or_nan(ptcproc.fruit_loops_source_lon, RAD_TO_ASEC);
            Eigen::VectorXd fruitloops_source_y_t =
                fruitloops_or_nan(ptcproc.fruit_loops_source_lat, RAD_TO_ASEC);
            Eigen::VectorXd fruitloops_local_sigma =
                fruitloops_or_nan(ptcproc.fruit_loops_local_sigma_map);
            Eigen::VectorXd fruitloops_local_sigma_npix =
                fruitloops_int_or_nan(ptcproc.fruit_loops_local_sigma_npix);
            Eigen::VectorXd fruitloops_amp_ref =
                fruitloops_or_nan(ptcproc.fruit_loops_amp_ref);
            Eigen::VectorXd fruitloops_adaptive_threshold =
                fruitloops_or_nan(ptcproc.fruit_loops_adaptive_threshold);
            Eigen::VectorXd fruitloops_support_radius_arcsec =
                fruitloops_or_nan(ptcproc.fruit_loops_adaptive_support_radius_rad, RAD_TO_ASEC);
            Eigen::VectorXd fruitloops_peak_threshold =
                Eigen::VectorXd::Constant(calib.n_dets, fill_double);
            Eigen::VectorXd fruitloops_snr_threshold =
                Eigen::VectorXd::Constant(calib.n_dets, fill_double);
            Eigen::VectorXd fruitloops_support_npix =
                Eigen::VectorXd::Constant(calib.n_dets, fill_double);
            Eigen::VectorXd fruitloops_support_signal_sum =
                Eigen::VectorXd::Constant(calib.n_dets, fill_double);
            Eigen::VectorXd fruitloops_support_x_span_arcsec =
                Eigen::VectorXd::Constant(calib.n_dets, fill_double);
            Eigen::VectorXd fruitloops_support_y_span_arcsec =
                Eigen::VectorXd::Constant(calib.n_dets, fill_double);

            for (Eigen::Index i = 0; i < calib.n_dets; ++i) {
                const double amp_ref = fruitloops_amp_ref(i);
                if (ptcproc.fruit_loops_peak_fraction_limit > 0.0 &&
                    std::isfinite(amp_ref) && amp_ref > 0.0) {
                    fruitloops_peak_threshold(i) =
                        ptcproc.fruit_loops_peak_fraction_limit * amp_ref;
                }
                const double sigma = fruitloops_local_sigma(i);
                if (ptcproc.fruit_loops_local_snr_floor > 0.0 &&
                    std::isfinite(sigma) && sigma > 0.0) {
                    fruitloops_snr_threshold(i) =
                        ptcproc.fruit_loops_local_snr_floor * sigma;
                }

                if (i >= static_cast<Eigen::Index>(omb.signal.size()) ||
                    i >= static_cast<Eigen::Index>(omb.weight.size()) ||
                    omb.signal[i].rows() != omb.n_rows ||
                    omb.signal[i].cols() != omb.n_cols ||
                    omb.weight[i].rows() != omb.n_rows ||
                    omb.weight[i].cols() != omb.n_cols) {
                    continue;
                }
                const double threshold = fruitloops_adaptive_threshold(i);
                if (!std::isfinite(threshold) || threshold <= 0.0) {
                    continue;
                }
                if (ptcproc.fruit_loops_source_valid.size() != calib.n_dets ||
                    ptcproc.fruit_loops_source_valid(i) == 0 ||
                    !std::isfinite(ptcproc.fruit_loops_source_lat(i)) ||
                    !std::isfinite(ptcproc.fruit_loops_source_lon(i))) {
                    continue;
                }

                const double center_row =
                    ptcproc.fruit_loops_source_lat(i) / omb.pixel_size_rad +
                    (omb.n_rows - 1) / 2.0;
                const double center_col =
                    ptcproc.fruit_loops_source_lon(i) / omb.pixel_size_rad +
                    (omb.n_cols - 1) / 2.0;
                if (!std::isfinite(center_row) || !std::isfinite(center_col)) {
                    continue;
                }

                double support_radius_pix = std::numeric_limits<double>::infinity();
                const double support_radius_rad =
                    (ptcproc.fruit_loops_adaptive_support_radius_rad.size() == calib.n_dets)
                        ? ptcproc.fruit_loops_adaptive_support_radius_rad(i)
                        : fill_double;
                if (std::isfinite(support_radius_rad) && support_radius_rad > 0.0) {
                    support_radius_pix = support_radius_rad / omb.pixel_size_rad;
                }

                Eigen::Index npix = 0;
                double signal_sum = 0.0;
                double min_x = std::numeric_limits<double>::infinity();
                double max_x = -std::numeric_limits<double>::infinity();
                double min_y = std::numeric_limits<double>::infinity();
                double max_y = -std::numeric_limits<double>::infinity();
                for (Eigen::Index row = 0; row < omb.n_rows; ++row) {
                    const double drow_pix = static_cast<double>(row) - center_row;
                    for (Eigen::Index col = 0; col < omb.n_cols; ++col) {
                        const double weight = omb.weight[i](row, col);
                        const double signal = omb.signal[i](row, col);
                        if (!std::isfinite(weight) || weight <= 0.0 ||
                            !std::isfinite(signal)) {
                            continue;
                        }
                        const double dcol_pix = static_cast<double>(col) - center_col;
                        if (std::sqrt(drow_pix * drow_pix + dcol_pix * dcol_pix) >
                            support_radius_pix) {
                            continue;
                        }
                        bool include_pixel = false;
                        if (ptcproc.fruit_mode == "upper") {
                            include_pixel = signal >= threshold;
                        }
                        else if (ptcproc.fruit_mode == "lower") {
                            include_pixel = signal <= -std::abs(threshold);
                        }
                        else {
                            include_pixel = std::abs(signal) >= threshold;
                        }
                        if (!include_pixel) {
                            continue;
                        }
                        const double x_arcsec = dcol_pix * pix_to_arcsec;
                        const double y_arcsec = drow_pix * pix_to_arcsec;
                        min_x = std::min(min_x, x_arcsec);
                        max_x = std::max(max_x, x_arcsec);
                        min_y = std::min(min_y, y_arcsec);
                        max_y = std::max(max_y, y_arcsec);
                        signal_sum += signal;
                        ++npix;
                    }
                }
                fruitloops_support_npix(i) = static_cast<double>(npix);
                fruitloops_support_signal_sum(i) = signal_sum;
                if (npix > 0) {
                    fruitloops_support_x_span_arcsec(i) = max_x - min_x;
                    fruitloops_support_y_span_arcsec(i) = max_y - min_y;
                }
            }

            Eigen::MatrixXd fit_qc_table(calib.n_dets, fit_qc_header.size());
            Eigen::Index col = 0;
            fit_qc_table.col(col++) = apt_or_zero("uid");
            fit_qc_table.col(col++) = apt_or_zero("array");
            fit_qc_table.col(col++) = apt_or_zero("nw");
            fit_qc_table.col(col++) = apt_or_zero("kids_tone");
            fit_qc_table.col(col++) = good_fits.cast<double>();
            fit_qc_table.col(col++) = converged.cast<double>();
            fit_qc_table.col(col++) = converge_iter.cast<double>();
            fit_qc_table.col(col++) = apt_or_zero("flag");
            fit_qc_table.col(col++) = flag2.cast<double>();
            fit_qc_table.col(col++) = apt_or_zero("amp");
            fit_qc_table.col(col++) = apt_or_zero("amp_err");
            fit_qc_table.col(col++) = apt_or_zero("cal_amp");
            fit_qc_table.col(col++) = apt_or_zero("cal_amp_method");
            fit_qc_table.col(col++) = apt_or_zero("template_amp");
            fit_qc_table.col(col++) = apt_or_zero("template_offset");
            fit_qc_table.col(col++) = apt_or_zero("template_resid_rms");
            fit_qc_table.col(col++) = apt_or_zero("template_npix");
            fit_qc_table.col(col++) = apt_or_zero("template_amp_over_fit_amp");
            fit_qc_table.col(col++) = apt_or_zero("cal_amp_over_fit_amp");
            fit_qc_table.col(col++) = apt_or_zero("map_peak_amp");
            fit_qc_table.col(col++) = apt_or_zero("map_peak_amp_over_fit_amp");
            fit_qc_table.col(col++) = fit_sig2noise;
            fit_qc_table.col(col++) = map_rms;
            fit_qc_table.col(col++) = map_sig2noise;
            fit_qc_table.col(col++) = n_weight_pos;
            fit_qc_table.col(col++) = fruitloops_source_x_t;
            fit_qc_table.col(col++) = fruitloops_source_y_t;
            fit_qc_table.col(col++) = fruitloops_local_sigma;
            fit_qc_table.col(col++) = fruitloops_local_sigma_npix;
            fit_qc_table.col(col++) = fruitloops_amp_ref;
            fit_qc_table.col(col++) = fruitloops_peak_threshold;
            fit_qc_table.col(col++) = fruitloops_snr_threshold;
            fit_qc_table.col(col++) = fruitloops_adaptive_threshold;
            fit_qc_table.col(col++) = fruitloops_support_radius_arcsec;
            fit_qc_table.col(col++) = fruitloops_support_npix;
            fit_qc_table.col(col++) = fruitloops_support_signal_sum;
            fit_qc_table.col(col++) = fruitloops_support_x_span_arcsec;
            fit_qc_table.col(col++) = fruitloops_support_y_span_arcsec;
            fit_qc_table.col(col++) = apt_or_zero("rfi_masked_samples");
            fit_qc_table.col(col++) = apt_or_zero("rfi_masked_scans");
            fit_qc_table.col(col++) = apt_or_zero("scan_band_masked_samples");
            fit_qc_table.col(col++) = apt_or_zero("scan_band_masked_rows");
            fit_qc_table.col(col++) = apt_or_zero("scan_band_masked_edge");
            fit_qc_table.col(col++) = apt_or_zero("scan_band_mask_rejected");
            fit_qc_table.col(col++) = fit_bound_nhit;
            fit_qc_table.col(col++) = fit_bound_code;
            fit_qc_table.col(col++) = fit_bound_amp;
            fit_qc_table.col(col++) = fit_bound_x;
            fit_qc_table.col(col++) = fit_bound_y;
            fit_qc_table.col(col++) = fit_bound_a;
            fit_qc_table.col(col++) = fit_bound_b;
            fit_qc_table.col(col++) = fit_bound_angle;
            fit_qc_table.col(col++) = fit_init_amp;
            fit_qc_table.col(col++) = fit_init_x_t;
            fit_qc_table.col(col++) = fit_init_y_t;
            fit_qc_table.col(col++) = fit_init_a_fwhm;
            fit_qc_table.col(col++) = fit_init_b_fwhm;
            fit_qc_table.col(col++) = fit_low_a_fwhm;
            fit_qc_table.col(col++) = fit_high_a_fwhm;
            fit_qc_table.col(col++) = fit_low_b_fwhm;
            fit_qc_table.col(col++) = fit_high_b_fwhm;
            fit_qc_table.col(col++) = prior_diag_or(prior_init_mode_col, -1.0);
            fit_qc_table.col(col++) = prior_diag_or(prior_used_col, 0.0);
            fit_qc_table.col(col++) = prior_diag_or(prior_fallback_blind_col, 0.0);
            fit_qc_table.col(col++) = prior_diag_or(prior_no_candidate_reason_col, 0.0);
            fit_qc_table.col(col++) = prior_diag_or(prior_slot_index_col, -1.0);
            fit_qc_table.col(col++) = prior_diag_or(prior_match_d2_col, std::numeric_limits<double>::quiet_NaN());
            fit_qc_table.col(col++) = prior_diag_or(prior_match_score_col, std::numeric_limits<double>::quiet_NaN());
            fit_qc_table.col(col++) = prior_diag_or(prior_candidate_snr_col, std::numeric_limits<double>::quiet_NaN());
            fit_qc_table.col(col++) = prior_diag_or(prior_n_candidates_col, 0.0);
            fit_qc_table.col(col++) = prior_diag_or(prior_n_candidates_keep_col, 0.0);
            fit_qc_table.col(col++) = prior_diag_or(prior_n_candidates_gate_col, 0.0);
            fit_qc_table.col(col++) = prior_diag_or(prior_candidate_x_raw_col, std::numeric_limits<double>::quiet_NaN());
            fit_qc_table.col(col++) = prior_diag_or(prior_candidate_y_raw_col, std::numeric_limits<double>::quiet_NaN());
            fit_qc_table.col(col++) = prior_diag_or(prior_candidate_x_prior_col, std::numeric_limits<double>::quiet_NaN());
            fit_qc_table.col(col++) = prior_diag_or(prior_candidate_y_prior_col, std::numeric_limits<double>::quiet_NaN());
            fit_qc_table.col(col++) = prior_diag_or(prior_center_x_col, std::numeric_limits<double>::quiet_NaN());
            fit_qc_table.col(col++) = prior_diag_or(prior_center_y_col, std::numeric_limits<double>::quiet_NaN());
            fit_qc_table.col(col++) = prior_diag_or(prior_derot_elev_col, std::numeric_limits<double>::quiet_NaN());
            fit_qc_table.col(col++) = prior_diag_or(prior_slot_x_col, std::numeric_limits<double>::quiet_NaN());
            fit_qc_table.col(col++) = prior_diag_or(prior_slot_y_col, std::numeric_limits<double>::quiet_NaN());
            fit_qc_table.col(col++) = prior_diag_or(prior_slot_sx_col, std::numeric_limits<double>::quiet_NaN());
            fit_qc_table.col(col++) = prior_diag_or(prior_slot_sy_col, std::numeric_limits<double>::quiet_NaN());
            fit_qc_table.col(col++) = apt_or_zero("final_prior_slot_index");
            fit_qc_table.col(col++) = apt_or_zero("final_prior_d2");
            fit_qc_table.col(col++) = apt_or_zero("x_t_raw");
            fit_qc_table.col(col++) = apt_or_zero("y_t_raw");
            fit_qc_table.col(col++) = apt_or_zero("x_t");
            fit_qc_table.col(col++) = apt_or_zero("y_t");
            fit_qc_table.col(col++) = apt_or_zero("x_t_derot");
            fit_qc_table.col(col++) = apt_or_zero("y_t_derot");
            fit_qc_table.col(col++) = apt_or_zero("a_fwhm");
            fit_qc_table.col(col++) = apt_or_zero("a_fwhm_err");
            fit_qc_table.col(col++) = apt_or_zero("b_fwhm");
            fit_qc_table.col(col++) = apt_or_zero("b_fwhm_err");
            fit_qc_table.col(col++) = apt_or_zero("angle");
            fit_qc_table.col(col++) = apt_or_zero("angle_err");
            fit_qc_table.col(col++) = apt_or_zero("flxscale");
            fit_qc_table.col(col++) = apt_or_zero("sens");

            YAML::Node fit_qc_meta;
            fit_qc_meta["obsnum"] = obsnum;
            fit_qc_meta["source"] = telescope.source_name;
            fit_qc_meta["creation_date"] = engine_utils::current_date_time();
            fit_qc_meta["date"] = date_obs.back();
            fit_qc_meta["map_grouping"] = map_grouping;
            fit_qc_meta["beammap_iter_max"] = beammap_iter_max;
            fit_qc_meta["beammap_iter_tolerance"] = beammap_iter_tolerance;
            fit_qc_meta["beammap_convergence_radius_arcsec"] = beammap_convergence_radius_arcsec;
            fit_qc_meta["reference_detector_subtracted"] = beammap_subtract_reference;
            fit_qc_meta["reference_det"] = beammap_reference_det_found;
            fit_qc_meta["rfi_mask_enabled"] = beammap_rfi_mask_enabled;
            fit_qc_meta["rfi_mask_block_size_samples"] = beammap_rfi_mask_block_size_samples;
            fit_qc_meta["rfi_mask_min_good_samples"] = beammap_rfi_mask_min_good_samples;
            fit_qc_meta["rfi_mask_dilate_blocks"] = beammap_rfi_mask_dilate_blocks;
            fit_qc_meta["rfi_mask_sigma_threshold"] = beammap_rfi_mask_sigma_threshold;
            fit_qc_meta["rfi_mask_sigma_floor"] = beammap_rfi_mask_sigma_floor;
            fit_qc_meta["rfi_mask_max_flagged_fraction"] = beammap_rfi_mask_max_flagged_fraction;
            fit_qc_meta["detector_weighting_mode"] = beammap_detector_weighting_mode;
            fit_qc_meta["beammap_fit_radius_fwhm"] = beammap_fit_radius_fwhm;
            fit_qc_meta["rfi_mask_detectors_affected"] =
                static_cast<int>((apt_or_zero("rfi_masked_scans").array() > 0.0).count());
            fit_qc_meta["scan_band_mask_enabled"] = beammap_scan_band_mask_enabled;
            fit_qc_meta["scan_band_mask_edge_rows"] = beammap_scan_band_mask_edge_rows;
            fit_qc_meta["scan_band_mask_min_row_pixels"] = beammap_scan_band_mask_min_row_pixels;
            fit_qc_meta["scan_band_mask_min_contiguous_rows"] = beammap_scan_band_mask_min_contiguous_rows;
            fit_qc_meta["scan_band_mask_row_median_sigma_threshold"] =
                beammap_scan_band_mask_row_median_sigma_threshold;
            fit_qc_meta["scan_band_mask_row_sigma_ratio_threshold"] =
                beammap_scan_band_mask_row_sigma_ratio_threshold;
            fit_qc_meta["scan_band_mask_max_flagged_fraction"] =
                beammap_scan_band_mask_max_flagged_fraction;
            fit_qc_meta["scan_band_mask_detectors_affected"] =
                static_cast<int>((apt_or_zero("scan_band_masked_rows").array() > 0.0).count());
            fit_qc_meta["scan_band_mask_detectors_rejected"] =
                static_cast<int>((apt_or_zero("scan_band_mask_rejected").array() > 0.0).count());
            fit_qc_meta["fit_bound_any"] = static_cast<int>((fit_diag_bound_nhit.array() > 0).count());
            fit_qc_meta["beammap_priors_enabled"] = beammap_priors_enabled;
            fit_qc_meta["beammap_priors_filepath"] = beammap_priors_filepath;
            fit_qc_meta["beammap_priors_centered"] = beammap_soft_priors_are_centered;
            fit_qc_meta["beammap_priors_derotated"] = beammap_soft_priors_are_derotated;
            fit_qc_meta["beammap_priors_max_d2_iter0"] = beammap_priors_max_d2_iter0;
            fit_qc_meta["beammap_priors_max_d2_after_iter0"] = beammap_priors_max_d2_after_iter0;
            fit_qc_meta["beammap_priors_score_lambda_iter0"] = beammap_priors_score_lambda_iter0;
            fit_qc_meta["beammap_priors_score_lambda_after_iter0"] = beammap_priors_score_lambda_after_iter0;
            fit_qc_meta["beammap_priors_align_after_iter0"] = beammap_priors_align_after_iter0;
            fit_qc_meta["beammap_priors_alignment_scope"] = beammap_priors_alignment_scope;
            fit_qc_meta["beammap_priors_alignment_common_support"] =
                beammap_priors_alignment_common_support;
            fit_qc_meta["beammap_priors_alignment_common_support_quantile"] =
                beammap_priors_alignment_common_support_quantile;
            fit_qc_meta["beammap_priors_alignment_min_matches"] = beammap_priors_alignment_min_matches;
            fit_qc_meta["beammap_priors_alignment_max_d2"] = beammap_priors_alignment_max_d2;
            fit_qc_meta["beammap_priors_alignment_fit_rotation"] = beammap_priors_alignment_fit_rotation;
            fit_qc_meta["beammap_priors_alignment_max_rotation_deg"] = beammap_priors_alignment_max_rotation_deg;
            fit_qc_meta["beammap_priors_aligned_arrays"] = static_cast<int>(beammap_prior_array_alignment.size());

            std::map<std::string, std::string> fit_qc_units = {
                {"uid", "N/A"},
                {"array", "N/A"},
                {"nw", "N/A"},
                {"kids_tone", "N/A"},
                {"good_fit", "N/A"},
                {"converged", "N/A"},
                {"converge_iter", "N/A"},
                {"flag", "N/A"},
                {"flag2", "N/A"},
                {"amp", get_unit("amp", omb.sig_unit)},
                {"amp_err", get_unit("amp_err", omb.sig_unit)},
                {"cal_amp", get_unit("cal_amp", omb.sig_unit)},
                {"cal_amp_method", "N/A"},
                {"template_amp", get_unit("template_amp", omb.sig_unit)},
                {"template_offset", get_unit("template_offset", omb.sig_unit)},
                {"template_resid_rms", get_unit("template_resid_rms", omb.sig_unit)},
                {"template_npix", "pix"},
                {"template_amp_over_fit_amp", "N/A"},
                {"cal_amp_over_fit_amp", "N/A"},
                {"map_peak_amp", get_unit("map_peak_amp", omb.sig_unit)},
                {"map_peak_amp_over_fit_amp", "N/A"},
                {"fit_sig2noise", "N/A"},
                {"map_rms", omb.sig_unit},
                {"map_sig2noise", "N/A"},
                {"n_weight_pos", "pix"},
                {"fruitloops_source_x_t", "arcsec"},
                {"fruitloops_source_y_t", "arcsec"},
                {"fruitloops_local_sigma", omb.sig_unit},
                {"fruitloops_local_sigma_npix", "pix"},
                {"fruitloops_amp_ref", omb.sig_unit},
                {"fruitloops_peak_threshold", omb.sig_unit},
                {"fruitloops_snr_threshold", omb.sig_unit},
                {"fruitloops_adaptive_threshold", omb.sig_unit},
                {"fruitloops_support_radius_arcsec", "arcsec"},
                {"fruitloops_support_npix", "pix"},
                {"fruitloops_support_signal_sum", omb.sig_unit},
                {"fruitloops_support_x_span_arcsec", "arcsec"},
                {"fruitloops_support_y_span_arcsec", "arcsec"},
                {"rfi_masked_samples", "samples"},
                {"rfi_masked_scans", "scans"},
                {"scan_band_masked_samples", "samples"},
                {"scan_band_masked_rows", "rows"},
                {"scan_band_masked_edge", "N/A"},
                {"scan_band_mask_rejected", "N/A"},
                {"fit_bound_nhit", "N/A"},
                {"fit_bound_code", "N/A"},
                {"fit_bound_amp", "N/A"},
                {"fit_bound_x", "N/A"},
                {"fit_bound_y", "N/A"},
                {"fit_bound_a", "N/A"},
                {"fit_bound_b", "N/A"},
                {"fit_bound_angle", "N/A"},
                {"fit_init_amp", get_unit("amp", omb.sig_unit)},
                {"fit_init_x_t", "arcsec"},
                {"fit_init_y_t", "arcsec"},
                {"fit_init_a_fwhm", "arcsec"},
                {"fit_init_b_fwhm", "arcsec"},
                {"fit_low_a_fwhm", "arcsec"},
                {"fit_high_a_fwhm", "arcsec"},
                {"fit_low_b_fwhm", "arcsec"},
                {"fit_high_b_fwhm", "arcsec"},
                {"prior_init_mode", "N/A"},
                {"prior_used", "N/A"},
                {"prior_fallback_blind", "N/A"},
                {"prior_no_candidate_reason", "N/A"},
                {"prior_slot_index", "N/A"},
                {"prior_match_d2", "N/A"},
                {"prior_match_score", "N/A"},
                {"prior_candidate_snr", "N/A"},
                {"prior_n_candidates", "pix"},
                {"prior_n_candidates_keep", "pix"},
                {"prior_n_candidates_gate", "pix"},
                {"prior_candidate_x_t_raw", "arcsec"},
                {"prior_candidate_y_t_raw", "arcsec"},
                {"prior_candidate_x_t_prior", "arcsec"},
                {"prior_candidate_y_t_prior", "arcsec"},
                {"prior_center_x_t", "arcsec"},
                {"prior_center_y_t", "arcsec"},
                {"prior_derot_elev", "rad"},
                {"prior_slot_x_t", "arcsec"},
                {"prior_slot_y_t", "arcsec"},
                {"prior_slot_sx", "arcsec"},
                {"prior_slot_sy", "arcsec"},
                {"final_prior_slot_index", "N/A"},
                {"final_prior_d2", "N/A"},
                {"x_t_raw", get_unit("x_t", "arcsec")},
                {"y_t_raw", get_unit("y_t", "arcsec")},
                {"x_t", get_unit("x_t", "arcsec")},
                {"y_t", get_unit("y_t", "arcsec")},
                {"x_t_derot", get_unit("x_t", "arcsec")},
                {"y_t_derot", get_unit("y_t", "arcsec")},
                {"a_fwhm", get_unit("a_fwhm", "arcsec")},
                {"a_fwhm_err", get_unit("a_fwhm_err", "arcsec")},
                {"b_fwhm", get_unit("b_fwhm", "arcsec")},
                {"b_fwhm_err", get_unit("b_fwhm_err", "arcsec")},
                {"angle", get_unit("angle", "rad")},
                {"angle_err", get_unit("angle_err", "rad")},
                {"flxscale", get_unit("flxscale", "N/A")},
                {"sens", get_unit("sens", "N/A")}
            };
            std::map<std::string, std::string> fit_qc_desc = {
                {"uid", get_description("uid", "detector uid")},
                {"array", get_description("array", "array index")},
                {"nw", get_description("nw", "network index")},
                {"kids_tone", get_description("kids_tone", "index of tone in network")},
                {"good_fit", "fit returned a usable solution"},
                {"converged", "beammap iterative convergence flag"},
                {"converge_iter", get_description("converge_iter", "beammap convergence iteration")},
                {"flag", get_description("flag", "detector quality flag")},
                {"flag2", "bitwise detector quality flag"},
                {"amp", get_description("amp", "fitted beam amplitude")},
                {"amp_err", get_description("amp_err", "fitted beam amplitude uncertainty")},
                {"cal_amp", get_description("cal_amp", "amplitude used for beammap flux calibration")},
                {"cal_amp_method", get_description("cal_amp_method", "calibration amplitude method code")},
                {"template_amp", get_description("template_amp", "empirical-template matched amplitude")},
                {"template_offset", get_description("template_offset", "empirical-template fitted local offset")},
                {"template_resid_rms", get_description("template_resid_rms", "empirical-template residual RMS")},
                {"template_npix", get_description("template_npix", "number of pixels used by empirical-template amplitude fit")},
                {"template_amp_over_fit_amp", get_description("template_amp_over_fit_amp", "empirical-template amplitude divided by Gaussian fit amplitude")},
                {"cal_amp_over_fit_amp", get_description("cal_amp_over_fit_amp", "calibration amplitude divided by Gaussian fit amplitude")},
                {"map_peak_amp", get_description("map_peak_amp", "local map peak near Gaussian fit center")},
                {"map_peak_amp_over_fit_amp", get_description("map_peak_amp_over_fit_amp", "local map peak divided by Gaussian fit amplitude")},
                {"fit_sig2noise", "fitted amplitude divided by fitted amplitude uncertainty"},
                {"map_rms", "standard deviation of positive-weight detector map pixels, excluding fitted source core when possible"},
                {"map_sig2noise", "fitted amplitude divided by support-only detector map rms"},
                {"n_weight_pos", "number of detector-map pixels with positive weight"},
                {"fruitloops_source_x_t", "source-support x center used by adaptive fruit loops feedback"},
                {"fruitloops_source_y_t", "source-support y center used by adaptive fruit loops feedback"},
                {"fruitloops_local_sigma", "local annulus robust sigma used by adaptive fruit loops feedback"},
                {"fruitloops_local_sigma_npix", "number of local-annulus pixels used for fruit loops local sigma"},
                {"fruitloops_amp_ref", "amplitude reference used by adaptive fruit loops feedback"},
                {"fruitloops_peak_threshold", "peak-fraction threshold component for adaptive fruit loops feedback"},
                {"fruitloops_snr_threshold", "local-S/N threshold component for adaptive fruit loops feedback"},
                {"fruitloops_adaptive_threshold", "final adaptive threshold used for fruit loops map feedback"},
                {"fruitloops_support_radius_arcsec", "radial source-support limit used by adaptive fruit loops feedback"},
                {"fruitloops_support_npix", "number of final-map positive-weight pixels passing adaptive fruit loops threshold and support cuts"},
                {"fruitloops_support_signal_sum", "sum of final-map signal over pixels passing adaptive fruit loops threshold and support cuts"},
                {"fruitloops_support_x_span_arcsec", "x span of final-map pixels passing adaptive fruit loops threshold and support cuts"},
                {"fruitloops_support_y_span_arcsec", "y span of final-map pixels passing adaptive fruit loops threshold and support cuts"},
                {"rfi_masked_samples", "number of timestream samples masked by beammap rfi_mask"},
                {"rfi_masked_scans", "number of scans with at least one sample masked by beammap rfi_mask"},
                {"scan_band_masked_samples", "number of timestream samples masked by beammap scan_band_mask"},
                {"scan_band_masked_rows", "number of detector-map edge rows flagged by beammap scan_band_mask"},
                {"scan_band_masked_edge", "scan-band edge code (0 none, 1 top, 2 bottom, 3 both)"},
                {"scan_band_mask_rejected", "1 if scan_band_mask proposed a mask but rejected it due to max_flagged_fraction"},
                {"fit_bound_nhit", "number of fitted parameters at lower/upper bounds"},
                {"fit_bound_code", "bitmask of bound-hit parameters (see metadata legend)"},
                {"fit_bound_amp", "bound state for amplitude (-1 lower, 0 none, +1 upper)"},
                {"fit_bound_x", "bound state for fitted x center (-1 lower, 0 none, +1 upper)"},
                {"fit_bound_y", "bound state for fitted y center (-1 lower, 0 none, +1 upper)"},
                {"fit_bound_a", "bound state for fitted a sigma/FWHM (-1 lower, 0 none, +1 upper)"},
                {"fit_bound_b", "bound state for fitted b sigma/FWHM (-1 lower, 0 none, +1 upper)"},
                {"fit_bound_angle", "bound state for fitted angle (-1 lower, 0 none, +1 upper)"},
                {"fit_init_amp", "initial amplitude used by Gaussian fitter"},
                {"fit_init_x_t", "initial x position converted to arcsec offset"},
                {"fit_init_y_t", "initial y position converted to arcsec offset"},
                {"fit_init_a_fwhm", "initial a FWHM implied by fitter initialization"},
                {"fit_init_b_fwhm", "initial b FWHM implied by fitter initialization"},
                {"fit_low_a_fwhm", "active lower bound for a FWHM"},
                {"fit_high_a_fwhm", "active upper bound for a FWHM"},
                {"fit_low_b_fwhm", "active lower bound for b FWHM"},
                {"fit_high_b_fwhm", "active upper bound for b FWHM"},
                {"prior_init_mode", "prior-init mode code (0 blind, 1 previous, 2 prior, -1 skipped/not fit)"},
                {"prior_used", "1 if the final initialization seed came from priors, else 0"},
                {"prior_fallback_blind", "1 if priors were attempted but blind fallback was used, else 0"},
                {"prior_no_candidate_reason", "reason code when priors produced no accepted candidate (see metadata legend)"},
                {"prior_slot_index", "matched prior slot index for the chosen prior-guided seed"},
                {"prior_match_d2", "Mahalanobis d^2 of chosen prior-guided seed in prior frame"},
                {"prior_match_score", "prior ranking score of chosen prior-guided seed"},
                {"prior_candidate_snr", "S/N metric of chosen prior-guided seed candidate"},
                {"prior_n_candidates", "number of weighted pixels above prior min_snr before top-N truncation"},
                {"prior_n_candidates_keep", "number of top-ranked candidates retained for prior scoring"},
                {"prior_n_candidates_gate", "number of retained candidates that passed the prior d^2 gate"},
                {"prior_candidate_x_t_raw", "chosen prior-guided candidate x offset before prior-frame transforms"},
                {"prior_candidate_y_t_raw", "chosen prior-guided candidate y offset before prior-frame transforms"},
                {"prior_candidate_x_t_prior", "chosen prior-guided candidate x offset in the prior frame"},
                {"prior_candidate_y_t_prior", "chosen prior-guided candidate y offset in the prior frame"},
                {"prior_center_x_t", "array-center x offset subtracted before prior matching"},
                {"prior_center_y_t", "array-center y offset subtracted before prior matching"},
                {"prior_derot_elev", "derotation elevation used for prior-frame matching"},
                {"prior_slot_x_t", "matched prior slot x center in the prior frame"},
                {"prior_slot_y_t", "matched prior slot y center in the prior frame"},
                {"prior_slot_sx", "matched prior slot soft x sigma"},
                {"prior_slot_sy", "matched prior slot soft y sigma"},
                {"final_prior_slot_index", "nearest prior slot index for final detector position in the prior frame"},
                {"final_prior_d2", "nearest-slot Mahalanobis d^2 for final detector position in the soft-prior frame"},
                {"x_t_raw", "raw x position before reference subtraction/derotation"},
                {"y_t_raw", "raw y position before reference subtraction/derotation"},
                {"x_t", get_description("x_t", "detector x position")},
                {"y_t", get_description("y_t", "detector y position")},
                {"x_t_derot", "detector x position after derotation transform"},
                {"y_t_derot", "detector y position after derotation transform"},
                {"a_fwhm", get_description("a_fwhm", "fitted major-axis FWHM")},
                {"a_fwhm_err", get_description("a_fwhm_err", "fitted major-axis FWHM uncertainty")},
                {"b_fwhm", get_description("b_fwhm", "fitted minor-axis FWHM")},
                {"b_fwhm_err", get_description("b_fwhm_err", "fitted minor-axis FWHM uncertainty")},
                {"angle", get_description("angle", "fitted beam angle")},
                {"angle_err", get_description("angle_err", "fitted beam angle uncertainty")},
                {"flxscale", get_description("flxscale", "flux conversion factor")},
                {"sens", get_description("sens", "detector sensitivity")}
            };

            for (const auto &key: fit_qc_header) {
                fit_qc_meta[key].push_back("units: " + fit_qc_units[key]);
                fit_qc_meta[key].push_back(fit_qc_desc[key]);
            }
            fit_qc_meta["flag2"].push_back("Good=0");
            fit_qc_meta["flag2"].push_back("BadFit=1");
            fit_qc_meta["flag2"].push_back("AzFWHM=2");
            fit_qc_meta["flag2"].push_back("ElFWHM=4");
            fit_qc_meta["flag2"].push_back("Sig2Noise=8");
            fit_qc_meta["flag2"].push_back("Sens=16");
            fit_qc_meta["flag2"].push_back("Position=32");
            fit_qc_meta["flag2"].push_back("PriorDist=64");
            fit_qc_meta["flag2"].push_back("NetworkPos=128");
            fit_qc_meta["cal_amp_method"].push_back("0: Gaussian fit amplitude fallback");
            fit_qc_meta["cal_amp_method"].push_back("1: empirical array-template matched amplitude");
            fit_qc_meta["fit_bound_code"].push_back("bit 0: amp lower");
            fit_qc_meta["fit_bound_code"].push_back("bit 1: amp upper");
            fit_qc_meta["fit_bound_code"].push_back("bit 2: x lower");
            fit_qc_meta["fit_bound_code"].push_back("bit 3: x upper");
            fit_qc_meta["fit_bound_code"].push_back("bit 4: y lower");
            fit_qc_meta["fit_bound_code"].push_back("bit 5: y upper");
            fit_qc_meta["fit_bound_code"].push_back("bit 6: a lower");
            fit_qc_meta["fit_bound_code"].push_back("bit 7: a upper");
            fit_qc_meta["fit_bound_code"].push_back("bit 8: b lower");
            fit_qc_meta["fit_bound_code"].push_back("bit 9: b upper");
            fit_qc_meta["fit_bound_code"].push_back("bit 10: angle lower");
            fit_qc_meta["fit_bound_code"].push_back("bit 11: angle upper");
            fit_qc_meta["prior_init_mode"].push_back("-1: skipped before fitting on last attempted iteration");
            fit_qc_meta["prior_init_mode"].push_back("0: blind seed");
            fit_qc_meta["prior_init_mode"].push_back("1: previous-iteration seed");
            fit_qc_meta["prior_init_mode"].push_back("2: prior-guided seed");
            fit_qc_meta["prior_no_candidate_reason"].push_back("0: none");
            fit_qc_meta["prior_no_candidate_reason"].push_back("1: no slot group for (array,nw)");
            fit_qc_meta["prior_no_candidate_reason"].push_back("2: no valid weighted pixels");
            fit_qc_meta["prior_no_candidate_reason"].push_back("3: invalid robust sigma estimate");
            fit_qc_meta["prior_no_candidate_reason"].push_back("4: no candidates above min_snr");
            fit_qc_meta["prior_no_candidate_reason"].push_back("5: all retained candidates failed max_d2 gate");
            fit_qc_meta["scan_band_masked_edge"].push_back("0: none");
            fit_qc_meta["scan_band_masked_edge"].push_back("1: top");
            fit_qc_meta["scan_band_masked_edge"].push_back("2: bottom");
            fit_qc_meta["scan_band_masked_edge"].push_back("3: both");

            to_ecsv_from_matrix(fit_qc_filename, fit_qc_table, fit_qc_header, fit_qc_meta);
            logger->info("done writing beammap fit qc table {}.ecsv", fit_qc_filename);
        }
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

    if (run_mapmaking) {
        bool split_by_flag_mode = false;
        if constexpr (map_type == mapmaking::RawObs) {
            split_by_flag_mode = (map_grouping == "detector") && beammap_split_fits_by_flag;
            if (split_by_flag_mode && beammap_split_flag_values.empty()) {
                logger->warn("beammap.split_fits_by_flag enabled but no flag_values specified; using standard map output");
                split_by_flag_mode = false;
            }
        }

        // wiener filtered maps write before this and are deleted from the vector.
        if (!f_io->empty()) {
            auto write_standard_maps = [&]() {
                // progress bar
                tula::logging::progressbar pb(
                    [&](const auto &msg) { logger->info("{}", msg); }, 100, "output progress ");

                for (Eigen::Index i=0; i<f_io->size(); ++i) {
                    // get the array for the given map
                    // add primary hdu
                    logger->debug("adding primary header to file {}",i);
                    add_phdu(f_io, mb, i);

                    if (!mb->noise.empty() && !n_io->empty()) {
                        logger->debug("adding primary header to noise file {}",i);
                        add_phdu(n_io, mb, i);
                    }
                }

                logger->debug("done adding primary headers");

                // write the maps
                Eigen::Index k = 0;
                Eigen::Index step = 2;

                if (!mb->kernel.empty()) {
                    step++;
                    timestream::log_kernel_map_diag(
                        logger,
                        "map output " + dir_name + " before write",
                        mb->kernel);
                }
                if (!mb->coverage.empty()) {
                    step++;
                }

                // write the maps
                for (Eigen::Index i=0; i<n_maps; ++i) {
                    // update progress bar
                    pb.count(n_maps, 1);
                    logger->debug("adding map");
                    write_maps(f_io,n_io,mb,i);

                    if (map_grouping=="detector") {
                        if constexpr (map_type == mapmaking::RawObs) {
                            // get the array for the given map
                            Eigen::Index map_index = arrays_to_maps(i);

                            // check if we move from one file to the next
                            // if so go back to first hdu layer
                            if (i>0) {
                                if (map_index > arrays_to_maps(i-1)) {
                                    k = 0;
                                }
                            }

                            // add apt table
                            logger->debug("adding beammap header keys");
                            for (auto const& key: calib.apt_header_keys) {
                                if (key!="flag2") {
                                    try {
                                        f_io->at(map_index).hdus.at(k)->addKey("BEAMMAP." + key, calib.apt[key](i), key
                                                                              + " (" + calib.apt_header_units[key] + ")");
                                    } catch(...) {
                                        f_io->at(map_index).hdus.at(k)->addKey("BEAMMAP." + key, 0.0, key
                                                                               + " (" + calib.apt_header_units[key] + ")");
                                    }
                                }
                                else {
                                    f_io->at(map_index).hdus.at(k)->addKey("BEAMMAP." + key, flag2(i), key
                                                                           + " (" + calib.apt_header_units[key] + ")");
                                }
                            }
                            // increment hdu layer
                            k = k + step;
                        }
                    }
                }

                logger->info("maps have been written to:");
                for (Eigen::Index i=0; i<f_io->size(); ++i) {
                    logger->info("{}.fits",f_io->at(i).filepath);
                }
            };

            if (split_by_flag_mode) {
                if (!mb->kernel.empty()) {
                    timestream::log_kernel_map_diag(
                        logger,
                        "map output " + dir_name + " split-by-flag before write",
                        mb->kernel);
                }
                std::set<int> split_values(beammap_split_flag_values.begin(), beammap_split_flag_values.end());
                Eigen::Index n_selected_maps = 0;
                for (Eigen::Index i = 0; i < n_maps; ++i) {
                    const int det_flag = static_cast<int>(std::lround(calib.apt["flag"](i)));
                    if (split_values.count(det_flag) > 0) {
                        n_selected_maps++;
                    }
                }

                if (n_selected_maps <= 0) {
                    logger->warn("beammap split_fits_by_flag selected no detector maps; using standard map output");
                    write_standard_maps();
                }
                else {
                    std::vector<std::string> base_filepaths;
                    base_filepaths.reserve(f_io->size());
                    for (const auto &fio : *f_io) {
                        base_filepaths.push_back(fio.filepath);
                    }

                    std::vector<std::string> base_noise_filepaths;
                    base_noise_filepaths.reserve(n_io->size());
                    for (const auto &nio : *n_io) {
                        base_noise_filepaths.push_back(nio.filepath);
                    }

                    // close and remove the default unsplit files before writing split outputs
                    f_io->clear();
                    n_io->clear();
                    for (const auto &path : base_filepaths) {
                        const auto fits_path = path + ".fits";
                        try {
                            if (fs::exists(fits_path)) {
                                fs::remove(fits_path);
                            }
                        }
                        catch (const std::exception &e) {
                            logger->warn("unable to remove unsplit beammap file {}: {}", fits_path, e.what());
                        }
                    }
                    for (const auto &path : base_noise_filepaths) {
                        const auto fits_path = path + ".fits";
                        try {
                            if (fs::exists(fits_path)) {
                                fs::remove(fits_path);
                            }
                        }
                        catch (const std::exception &e) {
                            logger->warn("unable to remove unsplit beammap noise file {}: {}", fits_path, e.what());
                        }
                    }

                    using split_io_t = fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>;

                    for (const auto flag_value : beammap_split_flag_values) {
                        Eigen::Index n_flag_maps = 0;
                        for (Eigen::Index i = 0; i < n_maps; ++i) {
                            const int det_flag = static_cast<int>(std::lround(calib.apt["flag"](i)));
                            if (det_flag == flag_value) {
                                n_flag_maps++;
                            }
                        }

                        if (n_flag_maps <= 0) {
                            logger->warn("beammap split_fits_by_flag: no detector maps found with flag={}; skipping", flag_value);
                            continue;
                        }

                        std::string split_suffix = "_flag" + std::to_string(flag_value);
                        if (flag_value == 0) {
                            split_suffix += "_good";
                        }
                        else if (flag_value == 1) {
                            split_suffix += "_bad";
                        }

                        std::vector<split_io_t> split_f_io_vec;
                        std::vector<split_io_t> split_n_io_vec;
                        split_f_io_vec.reserve(base_filepaths.size());
                        for (const auto &path : base_filepaths) {
                            split_f_io_vec.emplace_back(path + split_suffix);
                        }
                        if (!mb->noise.empty()) {
                            split_n_io_vec.reserve(base_noise_filepaths.size());
                            for (const auto &path : base_noise_filepaths) {
                                split_n_io_vec.emplace_back(path + split_suffix);
                            }
                        }

                        auto split_f_io = &split_f_io_vec;
                        auto split_n_io = &split_n_io_vec;

                        tula::logging::progressbar pb(
                            [&](const auto &msg) { logger->info("{}", msg); }, 100,
                            "output progress (flag=" + std::to_string(flag_value) + ") ");

                        for (Eigen::Index i = 0; i < split_f_io->size(); ++i) {
                            logger->debug("adding primary header to split file {} flag={}", i, flag_value);
                            add_phdu(split_f_io, mb, i);
                            split_f_io->at(i).pfits->pHDU().addKey("BEAMMAP.SPLIT_BY", "flag",
                                                                    "Beammap detector split criterion");
                            split_f_io->at(i).pfits->pHDU().addKey("BEAMMAP.SPLIT_VALUE", flag_value,
                                                                    "Beammap detector flag value in this file");

                            if (!mb->noise.empty()) {
                                logger->debug("adding primary header to split noise file {} flag={}", i, flag_value);
                                add_phdu(split_n_io, mb, i);
                                split_n_io->at(i).pfits->pHDU().addKey("BEAMMAP.SPLIT_BY", "flag",
                                                                        "Beammap detector split criterion");
                                split_n_io->at(i).pfits->pHDU().addKey("BEAMMAP.SPLIT_VALUE", flag_value,
                                                                        "Beammap detector flag value in this file");
                            }
                        }

                        Eigen::Index step = 2;
                        if (!mb->kernel.empty()) {
                            step++;
                        }
                        if (!mb->coverage.empty()) {
                            step++;
                        }

                        std::vector<Eigen::Index> hdu_layer(split_f_io->size(), 0);

                        for (Eigen::Index i = 0; i < n_maps; ++i) {
                            const int det_flag = static_cast<int>(std::lround(calib.apt["flag"](i)));
                            if (det_flag != flag_value) {
                                continue;
                            }

                            pb.count(n_flag_maps, 1);
                            logger->debug("adding split map for detector {} flag={}", i, flag_value);
                            write_maps(split_f_io, split_n_io, mb, i);

                            if (map_grouping == "detector") {
                                if constexpr (map_type == mapmaking::RawObs) {
                                    const Eigen::Index map_index = arrays_to_maps(i);
                                    const Eigen::Index k = hdu_layer.at(map_index);

                                    logger->debug("adding split beammap header keys");
                                    for (auto const &key : calib.apt_header_keys) {
                                        if (key != "flag2") {
                                            try {
                                                split_f_io->at(map_index).hdus.at(k)->addKey(
                                                    "BEAMMAP." + key, calib.apt[key](i),
                                                    key + " (" + calib.apt_header_units[key] + ")");
                                            }
                                            catch (...) {
                                                split_f_io->at(map_index).hdus.at(k)->addKey(
                                                    "BEAMMAP." + key, 0.0,
                                                    key + " (" + calib.apt_header_units[key] + ")");
                                            }
                                        }
                                        else {
                                            split_f_io->at(map_index).hdus.at(k)->addKey(
                                                "BEAMMAP." + key, flag2(i),
                                                key + " (" + calib.apt_header_units[key] + ")");
                                        }
                                    }
                                    hdu_layer.at(map_index) = hdu_layer.at(map_index) + step;
                                }
                            }
                        }

                        logger->info("beammap split maps (flag={}) have been written to:", flag_value);
                        for (Eigen::Index i = 0; i < split_f_io->size(); ++i) {
                            logger->info("{}.fits", split_f_io->at(i).filepath);
                        }
                    }
                }
            }
            else {
                write_standard_maps();
            }
        }

        // clear fits file vectors to ensure its closed.
        f_io->clear();
        n_io->clear();

        if (map_grouping!="detector") {
            // write psd and histogram files
            logger->debug("writing psds");
            write_psd<map_type>(mb, dir_name);
            logger->debug("writing histograms");
            write_hist<map_type>(mb, dir_name);
            logger->debug("writing map diagnostics");
            write_mapdiag<map_type>(mb, dir_name);
        }
    }
}
