#pragma once

// Engine output implementation detail.
// Include this only after Engine has been declared.

void Engine::cli_summary() {
    const auto &coadd_config = typed_config.coadd;
    const auto &noise_config = typed_config.noise;
    const auto &tod_output_config = typed_config.timestream.output;

    citlali::pipeline::log_reduction_map_summary(
        logger, obsnum, omb, rtcproc.run_polarization);
    const double mb_size_total =
        citlali::pipeline::log_map_memory_summary(
            logger, omb, cmb, coadd_config.enabled, noise_config.enabled);

    logger->info("estimated size of all maps {:.2f} GB", mb_size_total);
    logger->info("number of scans: {}",telescope.scan_indices.cols());
    if (tod_output_config.raw_time_chunk_enabled ||
        tod_output_config.processed_time_chunk_enabled) {
        citlali::pipeline::log_tod_output_selection_summary(
            logger, tod_output_config.type, n_tod_output_scans_rtc,
            rtcproc.tod_output_mini, rtcproc.tod_output_outer,
            n_tod_output_scans_ptc, ptcproc.tod_output_mini);
    }
    citlali::pipeline::log_diagnostics_sidecar_summary(logger);

    // test getting memory usage for fun
    /*struct sysinfo memInfo;
    long long totalPhysMem = memInfo.totalram;
    totalPhysMem *= memInfo.mem_unit;

    logger->info("total physical memory available {} GB", (totalPhysMem/1024)/1e7);*/
    auto phys_memory_kb = engine_utils::get_phys_memory();
    citlali::pipeline::log_physical_memory_summary(logger, phys_memory_kb);
}

template <TCDataKind tc_t>
void Engine::write_chunk_summary(TCData<tc_t, Eigen::MatrixXd> &in) {

    logger->debug("writing summary files for chunk {}",in.index.data);

    const auto filename =
        citlali::pipeline::chunk_summary_filename(in.index.data);

    // write summary log file
    std::ofstream f;
    f.open(citlali::pipeline::summary_log_path(obsnum_dir_name, filename));

    citlali::pipeline::write_chunk_summary_log(
        f, in, CITLALI_GIT_VERSION, KIDSCPP_GIT_VERSION,
        engine_utils::current_date_time(), typed_config.runtime.reduction_type,
        typed_config.timestream.type, omb.sig_unit, rtcproc,
        telescope.outer_scans_chunk,
        (calib.apt["flag"].array()!=0).count(),
        tula::alg::median(in.scans.data),
        engine_utils::calc_std_dev(in.scans.data));

    f.close();
}

template <typename map_buffer_t>
void Engine::write_map_summary(map_buffer_t &mb) {

    logger->debug("writing map summary files");

    const auto filename = citlali::pipeline::map_summary_filename();
    std::ofstream f;
    f.open(citlali::pipeline::summary_log_path(obsnum_dir_name, filename));

    const auto nonfinite_counts =
        citlali::pipeline::count_map_summary_nonfinite(mb);
    citlali::pipeline::write_map_summary_log(
        f, CITLALI_GIT_VERSION, KIDSCPP_GIT_VERSION,
        engine_utils::current_date_time(), typed_config.runtime.reduction_type,
        typed_config.timestream.type, typed_config.mapmaking.grouping, map_indices.n_maps,
        mb, nonfinite_counts);
}

template <mapmaking::MapType map_t, engine_utils::toltecIO::DataType data_t, engine_utils::toltecIO::ProdType prod_t>
auto Engine::setup_filenames(std::string dir_name) {
    return citlali::pipeline::map_output_filename<map_t, data_t, prod_t>(
        toltec_io, dir_name, typed_config.runtime.reduction_type, obsnum,
        telescope.sim_obs);
}

auto Engine::get_map_name(int i) {
    return citlali::pipeline::map_layer_name(
        i, typed_config.mapmaking.grouping, calib);
}
