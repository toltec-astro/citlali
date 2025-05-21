# pragma once

#include <citlali/core/mapmaking/wcs.h>

namespace citlali::config::options {
    std::string redu_type, exec_mode, output_directory_base;
    int n_threads;
    bool verbose, use_subdir;
    std::string subdir_name, tod_type;

    double delta_f_min_hz;
    std::string map_grouping, map_method;
    // string instead of bool to allow "auto" option
    std::string ignore_hwpr;

    std::string units;
    std::map<std::string, double> interface_offset_map = {
        {"toltec0", 0.0}, {"toltec1", 0.0}, {"toltec2", 0.0},
        {"toltec3", 0.0}, {"toltec4", 0.0}, {"toltec5", 0.0},
        {"toltec6", 0.0}, {"toltec7", 0.0}, {"toltec8", 0.0},
        {"toltec9", 0.0}, {"toltec10", 0.0}, {"toltec11", 0.0},
        {"toltec12", 0.0}, {"hwpr", 0.0}
    };

    // rtc
    bool run_kernel;
    bool run_despike;
    bool run_tod_filter;
    bool run_downsample;
    bool run_flux_calib;
    bool run_extinction;
    bool run_polarization;
    // ptc
    bool run_pca_clean;
    // tod output?
    bool run_tod_output_rtc, run_tod_output_ptc;
    // mapmaking
    bool run_mapmaking;
    bool run_noise_maps;
    // post processing
    bool run_map_coadd;
    bool run_map_filter;
    // maps
    WCS config_wcs;
    double pix_size_arcsec, pix_size_radians;
    int n_noise_maps;
    bool randomize_dets;
    // beammap
    Eigen::Matrix<bool, Eigen::Dynamic, 1> converged;
    Eigen::VectorXi convergence_iter;
    int bmp_iter_max;
    double bmp_iter_tolerance;
    int bmp_reference_det;
    bool bmp_subtract_reference_det;
    bool bmp_derotate_apt;
    std::vector<double> bmp_fwhm_lower, bmp_fwhm_upper, bmp_sig2noise_lower, bmp_sig2noise_upper, bmp_dist_max_arcsec;
    std::vector<double> bmp_sens_factors;
    std::string bmp_source_name;
    double bmp_ra_radians, bmp_dec_radians;
    std::map<std::string, double> bmp_flux_mJy_beam, bmp_err_mJy_beam;
    // fruit loops
    bool run_fruit_loops;
    int fruit_iters;
    bool save_all_fruit_iters;
    std::string fruit_path, fruit_type;

    template <typename ConfigType>
    void get_runtime_configs(ConfigType &config) {
        config.get(redu_type, std::tuple{"runtime", "reduction_type"});
        config.get(output_directory_base, std::tuple{"runtime", "output_dir"});
        config.get(exec_mode, std::tuple{"runtime", "parallel_policy"});
        config.get(n_threads, std::tuple{"runtime", "n_threads"});
        config.get(verbose, std::tuple{"runtime", "verbose"});
        config.get(use_subdir, std::tuple{"runtime", "use_subdir"});
        config.get(subdir_name, std::tuple{"runtime", "subdir_name"});
        config.get(tod_type, std::tuple{"timestream", "type"});
        config.get(delta_f_min_hz, std::tuple{"timestream", "raw_time_chunk", "flagging", "delta_f_min_Hz"});
        config.get(map_grouping, std::tuple{"mapmaking", "grouping"});
        config.get(map_method, std::tuple{"mapmaking", "method"});
        config.get(ignore_hwpr, std::tuple{"timestream", "polarimetry", "ignore_hwpr"});
        config.get(units, std::tuple{"mapmaking", "cunit"});

        for (auto& [key, value]: interface_offset_map) {
            config.get(value, std::tuple{"interface_sync_offset", key});
        }
    }

    template <typename ConfigType>
    void get_reduction_configs(ConfigType &config) {
        config.get(run_polarization, std::tuple{"timestream", "polarimetry", "enabled"});
        config.get(run_kernel, std::tuple{"timestream", "raw_time_chunk", "kernel", "enabled"});
        config.get(run_despike, std::tuple{"timestream", "raw_time_chunk", "despike", "enabled"});
        config.get(run_tod_filter, std::tuple{"timestream", "raw_time_chunk", "filter", "enabled"});
        config.get(run_downsample, std::tuple{"timestream", "raw_time_chunk", "downsample", "enabled"});
        config.get(run_flux_calib, std::tuple{"timestream", "raw_time_chunk", "flux_calibration", "enabled"});
        config.get(run_extinction, std::tuple{"timestream", "raw_time_chunk", "extinction_correction", "enabled"});
        config.get(run_tod_output_rtc, std::tuple{"timestream", "raw_time_chunk", "output", "enabled"});
        config.get(run_pca_clean, std::tuple{"timestream", "processed_time_chunk", "clean", "enabled"});
        config.get(run_tod_output_ptc, std::tuple{"timestream", "processed_time_chunk", "output", "enabled"});
        config.get(run_mapmaking, std::tuple{"mapmaking", "enabled"});
        config.get(run_noise_maps, std::tuple{"noise_maps", "enabled"});
        config.get(run_map_coadd, std::tuple{"coadd", "enabled"});
        config.get(run_map_filter, std::tuple{"post_processing", "map_filtering", "enabled"});
        config.get(run_fruit_loops, std::tuple{"timestream", "fruit_loops", "enabled"});

        config.get(pix_size_arcsec, std::tuple{"mapmaking", "pixel_size_arcsec"});
        config.get(n_noise_maps, std::tuple{"noise_maps", "n_noise_maps"});
        config.get(randomize_dets, std::tuple{"noise_maps", "randomize_dets"});


        double wcs_double;
        // get wcs naxis
        std::vector<std::string> naxis_keys = {"x_size_pix", "y_size_pix"};
        for (const auto &key: naxis_keys) {
            config.get(wcs_double, std::tuple{"mapmaking", key});
            config_wcs.naxis.push_back(wcs_double);
        }
        // get wcs crpix
        std::vector<std::string> crpix_keys = {"crpix1", "crpix2"};
        for (const auto &key: crpix_keys) {
            config.get(wcs_double, std::tuple{"mapmaking", key});
            config_wcs.crpix.push_back(wcs_double);
        }
        // get wcs crval
        std::vector<std::string> crval_keys = {"crval1_J2000", "crval2_J2000"};
        for (const auto &key: crval_keys) {
            config.get(wcs_double, std::tuple{"mapmaking", key});
            config_wcs.crval.push_back(wcs_double);
        }
    }

    // run once at start
    template <typename ConfigType>
    void get_beammap_configs(ConfigType &config) {
        config.get(bmp_iter_max, std::tuple{"beammap", "iter_max"});
        config.get(bmp_iter_tolerance, std::tuple{"beammap", "iter_tolerance"});
        config.get(bmp_reference_det, std::tuple{"beammap", "reference_det"});
        config.get(bmp_derotate_apt, std::tuple{"beammap", "derotate"});
        config.get(bmp_fwhm_lower, std::tuple{"beammap", "flagging", "array_lower_fwhm_arcsec"});
        config.get(bmp_fwhm_upper, std::tuple{"beammap", "flagging", "array_upper_fwhm_arcsec"});
        config.get(bmp_sig2noise_lower, std::tuple{"beammap", "flagging", "array_lower_sig2noise"});
        config.get(bmp_sig2noise_upper, std::tuple{"beammap", "flagging", "array_upper_sig2noise"});
        config.get(bmp_dist_max_arcsec, std::tuple{"beammap", "flagging", "array_max_dist_arcsec"});
        config.get(bmp_sens_factors, std::tuple{"beammap", "flagging", "sens_factors"});
    }

    // run for each observation
    template <typename ConfigType>
    void get_photometry_configs(ConfigType &config) {
        // get logger
        std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");
        // check if config file has pointing_offsets
        if (!config.has("beammap_source")) {
            throw std::runtime_error("beammap_source not found in config");
        }

        logger->debug("{}", config);

        try {
            // retrieve fluxes and errors for each array
            const auto& fluxes_node = config.get_node(std::tuple{"beammap_source", "fluxes"});
            for (int i = 0; i < fluxes_node.size(); ++i) {
                auto array_name = config.get_str(std::tuple{"beammap_source", "fluxes", i, "array_name"});
                bmp_flux_mJy_beam[array_name] =
                    config.template get_typed<double>(std::tuple{"beammap_source", "fluxes", i, "value_mJy"});
                bmp_err_mJy_beam[array_name] =
                    config.template get_typed<double>(std::tuple{"beammap_source", "fluxes", i, "uncertainty_mJy"});
            }
        } catch (const std::exception& e) {
            throw std::runtime_error(fmt::format("failed to load photometry config:: {}", e.what()));
        }
    }

    template <typename ConfigType>
    void get_fruit_configs(ConfigType &config) {
        config.get(fruit_iters, std::tuple{"timestream", "fruit_loops", "max_iters"});
        config.get(save_all_fruit_iters, std::tuple{"timestream", "fruit_loops", "save_all_iters"});
        config.get(fruit_path, std::tuple{"timestream", "fruit_loops", "path"});
        config.get(fruit_type, std::tuple{"timestream", "fruit_loops", "type"});
    }
} // controls
