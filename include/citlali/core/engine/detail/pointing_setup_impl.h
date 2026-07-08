#pragma once

// Implementation detail included by pointing.h.

void Pointing::setup() {
    // run obsnum setup
    obsnum_setup();

    // resize the current fit matrix
    params.setZero(n_maps, map_fitter.n_params);
    perrors.setZero(n_maps, map_fitter.n_params);
    fit_valid.setZero(n_maps);

    // units for positions
    std::string pos_units =
        citlali::config::is_radec_map_pixel_axes(telescope.pixel_axes)
            ? "deg"
            : "arcsec";

    // units for ppt header
    ppt_header_units = {
        {"array","N/A"},
        {"amp", omb.sig_unit},
        {"amp_err", omb.sig_unit},
        {"x_t", pos_units},
        {"x_t_err", pos_units},
        {"y_t", pos_units},
        {"y_t_err", pos_units},
        {"a_fwhm", "arcsec"},
        {"a_fwhm_err", "arcsec"},
        {"b_fwhm", "arcsec"},
        {"b_fwhm_err", "arcsec"},
        {"angle", "rad"},
        {"angle_err", "rad"},
        {"sig2noise", "N/A"}
    };

    /* populate ppt meta information */
    ppt_meta.reset();

    auto get_tel_data_mean = [&](const std::string &key, double fallback) {
        auto it = telescope.tel_data.find(key);
        if (it == telescope.tel_data.end() || it->second.size() < 1) {
            logger->warn("tel_data '{}' missing/empty; using fallback {}", key, fallback);
            return fallback;
        }
        const double value = it->second.mean();
        if (!std::isfinite(value)) {
            logger->warn("tel_data '{}' mean non-finite ({}); using fallback {}", key, value, fallback);
            return fallback;
        }
        return value;
    };

    // add obsnum to meta data
    ppt_meta["obsnum"] = obsnum;

    // add source name
    ppt_meta["source"] = telescope.source_name;

    // add project id to meta data
    ppt_meta["project_id"] = telescope.project_id;

    // add date of file creation
    ppt_meta["creation_date"] = engine_utils::current_date_time();

    // add observation date
    ppt_meta["date"] = date_obs.back();

    // mean Modified Julian Date
    ppt_meta["mjd"] = engine_utils::unix_to_modified_julian_date(telescope.tel_data["TelTime"].mean());

    // mean observing geometry
    const double mean_tel_el_rad = get_tel_data_mean("TelElAct", 0.0);
    const double mean_tel_az_rad = get_tel_data_mean("TelAzAct", 0.0);
    const double mean_source_el_rad = get_tel_data_mean("SourceEl", mean_tel_el_rad);
    ppt_meta["MEAN_EL"] = RAD_TO_DEG * mean_tel_el_rad;
    ppt_meta["MEAN_AZ"] = RAD_TO_DEG * mean_tel_az_rad;
    ppt_meta["MEAN_SOURCE_EL"] = RAD_TO_DEG * mean_source_el_rad;

    // reference frame
    ppt_meta["Radesys"] = telescope.pixel_axes;
    const auto &pointing_config = typed_config.pointing;
    ppt_meta["pointing_source_strategy"] =
        std::string(citlali::config::to_string(
            pointing_config.source_strategy));
    ppt_meta["pointing_fit_gaussian_enabled"] = pointing_config.fit_gaussian;
    ppt_meta["fruitloops_source_center_mode"] =
        std::string(citlali::config::to_string(
            pointing_config.fruitloops_center_mode));
    ppt_meta["pointing_header_center_max_radius_arcsec"] =
        pointing_config.header_max_radius_arcsec;
    ppt_meta["pointing_header_center_require_coverage"] =
        pointing_config.header_require_coverage;

    // add array mapping
    for (const auto &[arr_index,arr_name]: toltec_io.array_name_map) {
        ppt_meta["array_order"].push_back(std::to_string(arr_index) + ": " + arr_name);
    }

    // populate ppt meta information
    for (const auto &[param,unit]: ppt_header_units) {
        ppt_meta[param].push_back("units: " + unit);
        // description from apt
        auto description = calib.apt_header_description[unit];
        ppt_meta[param].push_back(description);
    }

    // add point model variables from telescope file
    for (const auto &val: telescope.tel_header) {
        std::size_t found = val.first.find("PointModel");
        if (found!=std::string::npos) {
            ppt_meta[val.first] = val.second(0);
        }
    }
    // add m2 z position
    ppt_meta["Header.M2.ZReq"] = telescope.tel_header["Header.M2.ZReq"](0);
    // add first m1 zernike coefficient
    ppt_meta["Header.M1.ZernikeC"] = telescope.tel_header["Header.M1.ZernikeC"](0);

    for (int i=0; i< telescope.tel_header["Header.M1.ActPos"].size(); ++i) {
        ppt_meta["Header.M1.ActPos"].push_back(telescope.tel_header["Header.M1.ActPos"](i));
        ppt_meta["Header.M1.CmdPos"].push_back(telescope.tel_header["Header.M1.CmdPos"](i));
    }
}
