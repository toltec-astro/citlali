#pragma once

#include <citlali/core/pipeline/mapdiag_labels.h>
#include <citlali/core/pipeline/mapdiag_observation_weight.h>
#include <citlali/core/utils/fits_io.h>
#include <citlali/core/utils/toltec_io.h>

#include <Eigen/Core>

#include <exception>
#include <string>

namespace citlali::engine_detail {

template <class ToltecIO, class LabelStorage>
std::string mapdiag_observation_weight_path(
    ToltecIO &toltec_io, const std::string &redu_dir_name,
    const std::string &redu_type, const LabelStorage &label_storage,
    std::size_t map_index, const std::string &coadd_obsnum, bool sim_obs) {
    const auto obs_dir =
        citlali::pipeline::mapdiag_obs_raw_dir(redu_dir_name, coadd_obsnum);
    return toltec_io
               .template create_filename<engine_utils::toltecIO::toltec,
                                         engine_utils::toltecIO::map,
                                         engine_utils::toltecIO::raw>(
                   obs_dir, redu_type, label_storage.array_names[map_index],
                   coadd_obsnum, sim_obs) +
           ".fits";
}

template <class Context, class MapBuffer, class CoreMask, class ToltecIO,
          class LabelStorage, class ObsTables, class Logger>
void assign_mapdiag_coadd_observation_contributions_for_map(
    const Context &context, Eigen::Index map_index, std::size_t storage_index,
    MapBuffer &mb, const CoreMask &core_mask, ToltecIO &toltec_io,
    const std::string &redu_dir_name, const std::string &redu_type,
    bool sim_obs, const LabelStorage &label_storage, ObsTables obs_tables,
    const Logger &logger) {
    const auto n_obsnums = mb->obsnums.size();
    for (std::size_t obs_idx = 0; obs_idx < n_obsnums; ++obs_idx) {
        const auto &coadd_obsnum = mb->obsnums[obs_idx];
        const auto obs_weight_path =
            mapdiag_observation_weight_path(
                toltec_io, redu_dir_name, redu_type, label_storage,
                storage_index, coadd_obsnum, sim_obs);
        const auto weight_hdu_name =
            citlali::pipeline::mapdiag_weight_hdu_name(
                label_storage.map_names[storage_index],
                label_storage.stokes_names[storage_index]);
        try {
            fitsIO<file_type_enum::read_fits, CCfits::ExtHDU*> obs_fits(
                obs_weight_path);
            const auto obs_weight = obs_fits.get_hdu(weight_hdu_name);
            citlali::pipeline::accumulate_mapdiag_obs_weight(
                map_index, context.n_obsnums, mb->n_rows, mb->n_cols,
                core_mask, obs_weight, obs_idx, obs_tables);
        } catch (const std::exception &e) {
            logger->warn(
                "failed to derive mapdiag contribution from {} [{}]: {}",
                obs_weight_path, weight_hdu_name, e.what());
            citlali::pipeline::zero_mapdiag_obs_entry(
                context, storage_index, obs_idx, obs_tables);
        }
    }
}

template <class Context, class MapBuffer, class CoreMask, class ToltecIO,
          class LabelStorage, class ObsTables, class Logger>
void assign_mapdiag_observation_contributions_for_map(
    const Context &context, Eigen::Index map_index, std::size_t storage_index,
    MapBuffer &mb, const CoreMask &core_mask, double map_weight_sum,
    double map_core_weight_sum, int map_valid_pixels, int map_core_pixels,
    ToltecIO &toltec_io, const std::string &redu_dir_name,
    const std::string &redu_type, bool sim_obs,
    const LabelStorage &label_storage, ObsTables obs_tables,
    const Logger &logger) {
    if (citlali::pipeline::mapdiag_is_single_observation_context(context)) {
        citlali::pipeline::assign_mapdiag_single_obs_entry(
            context, storage_index, map_weight_sum, map_core_weight_sum,
            map_valid_pixels, map_core_pixels, obs_tables);
        return;
    }

    assign_mapdiag_coadd_observation_contributions_for_map(
        context, map_index, storage_index, mb, core_mask, toltec_io,
        redu_dir_name, redu_type, sim_obs, label_storage, obs_tables,
        logger);
}

}  // namespace citlali::engine_detail
