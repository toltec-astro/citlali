#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

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

    calib.apt_meta["beammap_phase_split_enabled"] = beammap_phase_split_enabled;
    calib.apt_meta["beammap_locator_iter"] = beammap_locator_iter;
    calib.apt_meta["beammap_measurement_start_iter"] = beammap_measurement_start_iter;

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
