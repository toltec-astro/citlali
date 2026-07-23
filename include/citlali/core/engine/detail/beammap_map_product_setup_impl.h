#pragma once

// Beammap implementation detail.
// Include this only after Beammap has been declared.

void Beammap::add_beammap_detector_map_header(
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *f_io,
    Eigen::Index detector_index,
    Eigen::Index signal_hdu_index,
    const char *breadcrumb,
    int flag_value) {
    try {
        const Eigen::Index map_index =
            map_indices.arrays_to_maps(detector_index);

        logger->debug("adding beammap header keys");
        if (flag_value >= 0) {
            citlali::pipeline::update_map_output_debug_breadcrumb(
                breadcrumb, f_io->at(map_index).filepath.c_str(),
                detector_index, map_index, -1, -1, signal_hdu_index,
                static_cast<Eigen::Index>(f_io->at(map_index).hdus.size()),
                flag_value);
        }
        else {
            citlali::pipeline::update_map_output_debug_breadcrumb(
                breadcrumb, f_io->at(map_index).filepath.c_str(),
                detector_index, map_index, -1, -1, signal_hdu_index,
                static_cast<Eigen::Index>(f_io->at(map_index).hdus.size()));
        }
        beammap_map_product_headers::add_detector_header_keys(
            f_io->at(map_index).hdus.at(signal_hdu_index), calib,
            flag2, detector_index);
        citlali::pipeline::reset_map_output_debug_breadcrumb();
    }
    catch (const CCfits::FitsException &error) {
        citlali::pipeline::reset_map_output_debug_breadcrumb();
        citlali::pipeline::fail_required_output(
            logger,
            fmt::format(
                "beammap detector header write failed: detector_index={} flag={} error={}",
                static_cast<long long>(detector_index), flag_value,
                error.message()));
    }
    catch (const std::exception &error) {
        citlali::pipeline::reset_map_output_debug_breadcrumb();
        citlali::pipeline::fail_required_output(
            logger,
            fmt::format(
                "beammap detector header write failed: detector_index={} flag={} error={}",
                static_cast<long long>(detector_index), flag_value,
                error.what()));
    }
}

template <mapmaking::MapType map_type>
void Beammap::maybe_add_beammap_detector_map_header(
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *f_io,
    Eigen::Index detector_index,
    Eigen::Index signal_hdu_index,
    bool detector_grouping,
    const char *breadcrumb,
    int flag_value) {
    if (!detector_grouping) {
        return;
    }
    if constexpr (map_type == mapmaking::RawObs) {
        add_beammap_detector_map_header(
            f_io, detector_index, signal_hdu_index, breadcrumb, flag_value);
    }
}

void Beammap::add_beammap_map_primary_headers(
    mapmaking::MapBuffer *mb,
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *f_io,
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *n_io,
    citlali::pipeline::StageProfileCollector &stage_profile,
    const std::string &profile_stage_name,
    const std::string &profile_context,
    int flag_value) {
    (void)stage_profile;
    const auto profile_scope =
        citlali::pipeline::profile_stage(stage_profile,
            profile_stage_name.c_str(), logger, profile_context);
    for (Eigen::Index i=0; i<f_io->size(); ++i) {
        if (flag_value >= 0) {
            logger->debug(
                "adding primary header to split file {} flag={}", i,
                flag_value);
        }
        else {
            logger->debug("adding primary header to file {}", i);
        }
        add_phdu(f_io, mb, i);
        if (flag_value >= 0) {
            beammap_map_product_split_helpers::add_split_primary_header(
                *f_io, i, flag_value);
        }

        if (!mb->noise.empty() && !n_io->empty()) {
            if (flag_value >= 0) {
                logger->debug(
                    "adding primary header to split noise file {} flag={}",
                    i, flag_value);
            }
            else {
                logger->debug("adding primary header to noise file {}", i);
            }
            add_phdu(n_io, mb, i);
            if (flag_value >= 0) {
                beammap_map_product_split_helpers::add_split_primary_header(
                    *n_io, i, flag_value);
            }
        }
    }
}

Beammap::BeammapSplitMapOutputFiles
Beammap::prepare_split_beammap_map_output_files(
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *f_io,
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>> *n_io) {
    BeammapSplitMapOutputFiles files;
    files.base_filepaths =
        beammap_map_product_split_helpers::filepaths(*f_io);
    files.base_noise_filepaths =
        beammap_map_product_split_helpers::filepaths(*n_io);

    // Close and remove the default unsplit files before writing split outputs.
    f_io->clear();
    n_io->clear();
    beammap_map_product_split_helpers::remove_fits_files(
        files.base_filepaths, "map", logger);
    beammap_map_product_split_helpers::remove_fits_files(
        files.base_noise_filepaths, "noise", logger);
    return files;
}
