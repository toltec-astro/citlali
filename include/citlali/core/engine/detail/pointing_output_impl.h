#pragma once

// Implementation detail included by pointing.h.

#include <citlali/core/pipeline/output_policy.h>
#include <citlali/core/pipeline/pointing_fit_table_metrics.h>
#include <citlali/core/pipeline/reduction_config_accessors.h>

Eigen::MatrixXf Pointing::make_pointing_ppt_table(mapmaking::MapBuffer *mb) {
    Eigen::MatrixXf ppt_table(
        map_indices.n_maps,
        citlali::pipeline::pointing_fit_table_column_count(
            map_fitter.n_params));

    for (Eigen::Index i = 0; i < map_indices.n_maps; ++i) {
        ppt_table(i, 0) = map_indices.maps_to_arrays(i);
        const double map_std_dev =
            engine_utils::calc_std_dev(mb->signal[i]);
        const auto metrics =
            citlali::pipeline::pointing_fit_table_metrics(
                params(i, 0), perrors(i, 0), map_std_dev);
        ppt_table(
            i,
            citlali::pipeline::
                pointing_fit_table_legacy_sig2noise_column(
                    map_fitter.n_params)) = metrics.legacy_sig2noise;
        ppt_table(
            i,
            citlali::pipeline::
                pointing_fit_table_peak_over_full_map_rms_column(
                    map_fitter.n_params)) = metrics.peak_over_full_map_rms;
        ppt_table(
            i,
            citlali::pipeline::pointing_fit_table_fit_sig2noise_column(
                map_fitter.n_params)) = metrics.fit_sig2noise;
    }

    Eigen::Index j = 0;
    for (Eigen::Index i = 1; i < 2 * map_fitter.n_params; i += 2) {
        ppt_table.col(i) = params.col(j).cast<float>();
        ppt_table.col(i + 1) = perrors.col(j).cast<float>();
        ++j;
    }

    return ppt_table;
}

void Pointing::add_pointing_fit_header_keys(CCfits::ExtHDU &hdu,
                                            const Eigen::MatrixXf &ppt_table,
                                            Eigen::Index map_row) {
    for (Eigen::Index j = 0; j < ppt_header.size(); ++j) {
        const auto &key = ppt_header[j];
        const std::string comment = key + " (" + ppt_header_units[key] + ")";
        hdu.addKey("POINTING." + key, ppt_table(map_row, j), comment);
    }

    hdu.addKey(
        "POINTING.fit_enabled",
        static_cast<int>(citlali::pipeline::pointing_config(*this).fit_gaussian),
        "Gaussian fit enabled");
    hdu.addKey("POINTING.fit_valid", static_cast<int>(fit_valid(map_row)),
               "Gaussian fit valid");
    hdu.addKey(
        "POINTING.source_strategy",
        std::string(citlali::config::to_string(
            citlali::pipeline::pointing_config(*this).source_strategy)),
        "Pointing source strategy");
    hdu.addKey(
        "POINTING.source_center_mode",
        std::string(citlali::config::to_string(
            citlali::pipeline::pointing_config(*this).fruitloops_center_mode)),
        "Fruit loops source center mode");
}

template <typename FitsIoVector>
void Pointing::write_pointing_map_fits_products(
    FitsIoVector *f_io,
    FitsIoVector *n_io,
    mapmaking::MapBuffer *mb,
    const Eigen::MatrixXf &ppt_table) {
    if (f_io->empty()) {
        return;
    }

    {
        // progress bar
        tula::logging::progressbar pb(
            [&](const auto &msg) { logger->info("{}", msg); }, 100,
            "output progress ");

        for (Eigen::Index i=0; i<f_io->size(); i++) {
            // add primary hdu
            add_phdu(f_io, mb, i);

            if (!mb->noise.empty() && !n_io->empty()) {
                add_phdu(n_io, mb, i);
            }
        }

        Eigen::Index k = 0;

        for (Eigen::Index i=0; i<map_indices.n_maps; i++) {
            // update progress bar
            pb.count(map_indices.n_maps, 1);
            write_maps(f_io,n_io,mb,i);

            Eigen::Index map_index = map_indices.arrays_to_maps(i);

            // check if we move from one file to the next
            // if so go back to first hdu layer
            if (i>0) {
                if (map_index > map_indices.arrays_to_maps(i-1)) {
                    k = 0;
                }
            }
            // get current hdu extension name
            std::string extname = f_io->at(map_index).hdus.at(k)->name();
            // see if this is a signal extension
            std::size_t found = extname.find("signal");

            // find next signal extension
            while (found==std::string::npos && k<f_io->at(map_index).hdus.size()) {
                k = k + 1;
                // get current hdu extension name
                extname = f_io->at(map_index).hdus.at(k)->name();
                // see if this is a signal extension
                found = extname.find("signal");
            }

            add_pointing_fit_header_keys(
                *f_io->at(map_index).hdus.at(k), ppt_table, i);
            ++k; // Move to next extension
        }
    }

    logger->info("maps have been written to:");
    for (const auto& file: *f_io) {
        logger->info("{}.fits", file.filepath);
    }
}

template <mapmaking::MapType map_type>
void Pointing::output(
    citlali::pipeline::StageProfileCollector &stage_profile) {
    (void)stage_profile;
    // pointer to map buffer
    mapmaking::MapBuffer* mb = nullptr;
    // pointer to data file fits vector
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>* f_io = nullptr;
    // pointer to noise file fits vector
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>* n_io = nullptr;

    // directory name
    std::string dir_name;

    // Matrix holds array identity, fit values/errors, the legacy dynamic-range
    // field, and explicit dynamic-range/formal-fit significance fields.
    Eigen::MatrixXf ppt_table(
        map_indices.n_maps,
        citlali::pipeline::pointing_fit_table_column_count(
            map_fitter.n_params));

    // determine pointers and directory name based on map_type
    if constexpr (map_type == mapmaking::RawObs || map_type == mapmaking::FilteredObs) {
        mb = &omb;
        dir_name = output_paths.obsnum_dir_name + (map_type == mapmaking::RawObs ? "raw/" : "filtered/");
        f_io = (map_type == mapmaking::RawObs) ? &map_fits_outputs.obs : &map_fits_outputs.filtered_obs;
        n_io = (map_type == mapmaking::RawObs) ? &map_fits_outputs.obs_noise : &map_fits_outputs.filtered_obs_noise;

        // filename for ppt table
        auto ppt_filename =
            citlali::pipeline::observation_output_filename<
                engine_utils::toltecIO::ppt, engine_utils::toltecIO::map,
                (map_type == mapmaking::RawObs
                     ? engine_utils::toltecIO::raw
                     : engine_utils::toltecIO::filtered)>(
                toltec_io, dir_name,
                citlali::pipeline::runtime_reduction_type(*this), "",
                observation_identity.obsnum, telescope.sim_obs);

        ppt_table = make_pointing_ppt_table(mb);

        // write ppt
        to_ecsv_from_matrix(ppt_filename, ppt_table, ppt_header, ppt_meta);

        if constexpr (map_type == mapmaking::RawObs) {
            // write stats file
            write_stats();
            if (citlali::pipeline::tod_output_files_available(*this)) {
                // add tod header information
                add_tod_header(mb);
            }
        }
    } else if constexpr (map_type == mapmaking::RawCoadd || map_type == mapmaking::FilteredCoadd) {
        mb = &cmb;
        dir_name = output_paths.coadd_dir_name + (map_type == mapmaking::RawCoadd ? "raw/" : "filtered/");
        f_io = (map_type == mapmaking::RawCoadd) ? &map_fits_outputs.coadd : &map_fits_outputs.filtered_coadd;
        n_io = (map_type == mapmaking::RawCoadd) ? &map_fits_outputs.coadd_noise : &map_fits_outputs.filtered_coadd_noise;
    }

    if (citlali::pipeline::mapmaking_outputs_enabled(*this)) {
        write_pointing_map_fits_products(f_io, n_io, mb, ppt_table);

        // clear fits file vectors to ensure its closed.
        f_io->clear();
        n_io->clear();

        // write psd and histogram files
        logger->debug("writing psds");
        write_psd<map_type>(mb, dir_name);
        logger->debug("writing histograms");
        write_hist<map_type>(mb, dir_name);
        logger->debug("writing map diagnostics");
        write_mapdiag<map_type>(mb, dir_name);

        // write source table
        if (citlali::pipeline::source_finding_outputs_enabled(*this)) {
            logger->debug("writing source table");
            write_sources<map_type>(mb, dir_name);
        }
    }
}
