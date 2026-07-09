#pragma once

// Implementation detail included by lali.h.

#include <citlali/core/pipeline/output_policy.h>

template <mapmaking::MapType map_type>
void Lali::output() {
    // pointer to map buffer
    mapmaking::MapBuffer* mb = nullptr;
    // pointer to data file fits vector
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>* f_io = nullptr;
    // pointer to noise file fits vector
    std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>* n_io = nullptr;

    // directory name
    std::string dir_name;

    // set common variables depending on map_type
    if constexpr (map_type == mapmaking::RawObs || map_type == mapmaking::FilteredObs) {
        mb = &omb;
        dir_name = output_paths.obsnum_dir_name + (map_type == mapmaking::RawObs ? "raw/" : "filtered/");
        f_io = (map_type == mapmaking::RawObs) ? &map_fits_outputs.obs : &map_fits_outputs.filtered_obs;
        n_io = (map_type == mapmaking::RawObs) ? &map_fits_outputs.obs_noise : &map_fits_outputs.filtered_obs_noise;

        if constexpr (map_type == mapmaking::RawObs) {
            // write stats file
            write_stats();
            if (citlali::pipeline::tod_output_files_available(*this)) {
                // add tod header information
                add_tod_header(mb);
            }
        }
    }
    else if constexpr (map_type == mapmaking::RawCoadd || map_type == mapmaking::FilteredCoadd) {
        mb = &cmb;
        dir_name = output_paths.coadd_dir_name + (map_type == mapmaking::RawCoadd ? "raw/" : "filtered/");
        f_io = (map_type == mapmaking::RawCoadd) ? &map_fits_outputs.coadd : &map_fits_outputs.filtered_coadd;
        n_io = (map_type == mapmaking::RawCoadd) ? &map_fits_outputs.coadd_noise : &map_fits_outputs.filtered_coadd_noise;
    }

    if (citlali::pipeline::mapmaking_outputs_enabled(*this)) {
        // wiener filtered maps write before this and are deleted from the vector.
        if (!f_io->empty()) {
            {
                // progress bar
                tula::logging::progressbar pb(
                    [&](const auto &msg) { logger->info("{}", msg); }, 100, "output progress ");

                for (Eigen::Index i=0; i<f_io->size(); ++i) {
                    // get the array for the given map
                    // add primary hdu
                    logger->debug("adding primary header to file {}",i);
                    add_phdu(f_io, mb, i);

                    // add primary hdu to noise maps
                    if (!mb->noise.empty() && !n_io->empty()) {
                        logger->debug("adding primary header to noise file {}",i);
                        add_phdu(n_io, mb, i);
                    }
                }

                logger->debug("done adding primary headers");

                // write the maps
                for (Eigen::Index i=0; i<map_indices.n_maps; ++i) {
                    // update progress bar
                    pb.count(map_indices.n_maps, 1);
                    write_maps(f_io,n_io,mb,i);
                }
            }

            logger->info("maps have been written to:");
            for (Eigen::Index i=0; i<f_io->size(); ++i) {
                logger->info("{}.fits",f_io->at(i).filepath);
            }
        }

        std::vector<std::string> data_filepaths;
        data_filepaths.reserve(f_io->size());
        for (const auto &fio : *f_io) {
            data_filepaths.push_back(fio.filepath + ".fits");
        }
        std::vector<std::string> noise_filepaths;
        noise_filepaths.reserve(n_io->size());
        for (const auto &nio : *n_io) {
            noise_filepaths.push_back(nio.filepath + ".fits");
        }
        auto join_filepaths = [](const std::vector<std::string> &paths) {
            std::string joined;
            for (std::size_t idx = 0; idx < paths.size(); ++idx) {
                if (idx > 0) {
                    joined += ", ";
                }
                joined += paths[idx];
            }
            return joined;
        };

        // clear fits file vectors to ensure its closed.
        try {
            std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>().swap(*f_io);
            std::vector<fitsIO<file_type_enum::write_fits, CCfits::ExtHDU*>>().swap(*n_io);
        } catch (const CCfits::FitsError &e) {
            throw std::runtime_error(
                fmt::format("failed while finalizing FITS outputs data_files=[{}] noise_files=[{}]: {}",
                            join_filepaths(data_filepaths),
                            join_filepaths(noise_filepaths),
                            e.message()));
        } catch (const std::exception &e) {
            throw std::runtime_error(
                fmt::format("failed while finalizing FITS outputs data_files=[{}] noise_files=[{}]: {}",
                            join_filepaths(data_filepaths),
                            join_filepaths(noise_filepaths),
                            e.what()));
        }

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
