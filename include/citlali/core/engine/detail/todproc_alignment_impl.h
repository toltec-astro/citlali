#pragma once

// Implementation detail included by todproc.h.

#include <citlali/core/pipeline/timestream_alignment_helpers.h>
#include <citlali/core/pipeline/timestream_alignment_state.h>
#include <citlali/core/pipeline/observation_setup_validation.h>

template <class EngineType>
void TimeOrderedDataProc<EngineType>::align_timestreams(const RawObs &rawobs) {
    using namespace netCDF;
    using namespace netCDF::exceptions;

    citlali::pipeline::clear_alignment_windows(engine().alignment);
    engine().alignment.gaps.clear();

    // vector of network times
    std::vector<Eigen::VectorXd> nw_ts;

    // sample rate
    double fsmp = -1;

    // loop through input files
    for (const RawObs::DataItem &data_item : rawobs.kidsdata()) {
        try {
            // load data file
            NcFile fo(data_item.filepath(), NcFile::read);

            // get roach index for offsets
            int roach_index;
            fo.getVar("Header.Toltec.RoachIndex").getVar(&roach_index);

            // get sample rate
            double fsmp_roach;
            fo.getVar("Header.Toltec.SampleFreq").getVar(&fsmp_roach);

            fsmp = citlali::pipeline::reconcile_sample_rate_hz(
                fsmp, fsmp_roach, roach_index);

            // get dimensions for time matrix
            Eigen::Index n_pts = fo.getVar("Data.Toltec.Ts").getDim(0).getSize();
            Eigen::Index n_time = fo.getVar("Data.Toltec.Ts").getDim(1).getSize();

            // get time matrix
            Eigen::MatrixXi ts(n_time,n_pts);
            fo.getVar("Data.Toltec.Ts").getVar(ts.data());

            // transpose due to row-major order
            ts.transposeInPlace();

            // find gaps
            int gaps = citlali::pipeline::count_packet_counter_gaps(ts);

            // add gaps to engine map
            if (gaps>0) {
                engine().alignment.gaps["Toltec" + std::to_string(roach_index)] = gaps;
            }

            // get fpga frequency
            double fpga_freq;
            fo.getVar("Header.Toltec.FpgaFreq").getVar(&fpga_freq);

            nw_ts.push_back(
                citlali::pipeline::network_time_from_timestream_matrix(
                    ts.cast<double>(), fpga_freq,
                    engine().interface_sync.offsets[
                        "toltec"+std::to_string(roach_index)]));

            fo.close();

        } catch (NcException &e) {
            logger->error("{}", e.what());
            throw DataIOError{fmt::format(
                "failed to load data from netCDF file {}", data_item.filepath())};
        }
    }

    const auto network_overlap =
        citlali::pipeline::find_common_timestream_overlap(
            nw_ts, "align_timestreams");
    double max_t0 = network_overlap.max_start;
    double min_tn = network_overlap.min_end;
    const Eigen::Index max_t0_i = network_overlap.max_start_index;

    // get hwpr timing
    if (engine().calib.run_hwpr) {
        citlali::pipeline::validate_hwpr_alignment_inputs(
            engine().calib.hwpr_recvt, engine().calib.hwpr_angle, "no-gap");
        // if hwpr init time is larger than max start time, replace global max start time
        Eigen::Index hwpr_ts_n_pts = engine().calib.hwpr_recvt.size();
        if (engine().calib.hwpr_recvt(0) > max_t0) {
            max_t0 = engine().calib.hwpr_recvt(0);
        }

        // if hwpr init time is smaller than min end time, replace global min end time
        if (engine().calib.hwpr_recvt(hwpr_ts_n_pts - 1) < min_tn) {
            min_tn = engine().calib.hwpr_recvt(hwpr_ts_n_pts - 1);
        }
    }

    if (!std::isfinite(max_t0) || !std::isfinite(min_tn) || max_t0 > min_tn) {
        throw std::runtime_error(fmt::format(
            "no common time overlap across input timestreams: max_start={} min_end={}",
            max_t0, min_tn));
    }

    const auto sample_window = citlali::pipeline::find_common_sample_window(
        nw_ts, max_t0, min_tn);
    engine().alignment.start_indices = sample_window.start_indices;
    engine().alignment.end_indices = sample_window.end_indices;
    Eigen::Index min_size = sample_window.min_size;

    // if hwpr requested
    if (engine().calib.run_hwpr) {
        // find start index that is larger than max start for hwpr
        Eigen::Index si = citlali::pipeline::find_first_sample_at_or_after(
            engine().calib.hwpr_recvt, max_t0,
            "failed to find aligned HWPR start sample");
        // pushback start index on hwpr start index vector
        engine().alignment.hwpr_start_index = si;

        // find end index that is smaller than min end for hwpr
        Eigen::Index ei = citlali::pipeline::find_last_sample_at_or_before(
            engine().calib.hwpr_recvt, min_tn, si,
            "failed to find aligned HWPR end sample");
        // pushback end index on hwpr end index vector
        engine().alignment.hwpr_end_index = ei;

        // update min_size for all time vectors if hwpr data is shorter (data and hwpr)
        if ((ei - si + 1) < min_size) {
            min_size = ei - si + 1;
        }
    }

    if (min_size <= 0) {
        throw std::runtime_error("aligned common timestream length is not positive");
    }

    // shortest common data time vector
    Eigen::VectorXd xi = nw_ts[max_t0_i].segment(engine().alignment.start_indices[max_t0_i], min_size);

    citlali::pipeline::interpolate_telescope_data_to_common_time(
        engine().telescope.tel_data, xi, false);

    // interpolate hwpr data
    if (engine().calib.run_hwpr) {
        citlali::pipeline::interpolate_hwpr_angle_to_common_time(
            engine().calib.hwpr_angle, engine().calib.hwpr_recvt, xi);
    }
}

// upgraded alignment of tod with telescope
template <class EngineType>
void TimeOrderedDataProc<EngineType>::align_timestreams_gaps(const RawObs &rawobs) {
    using namespace netCDF;
    using namespace netCDF::exceptions;

    citlali::pipeline::clear_gap_alignment_state(engine().alignment);

    const auto kids_data = rawobs.kidsdata();
    std::vector<Eigen::VectorXd> nw_times(kids_data.size());
    std::vector<Eigen::Index> nw_ids(kids_data.size(), -1);

    // loop through networks and build time vectors
    double f_smp_roach = -1.0;
    double fsmp_ref = -1.0;
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(kids_data.size());
         ++i) {
        const RawObs::DataItem &data_item = kids_data[i];
        try {
            // load data file
            NcFile fo(data_item.filepath(), NcFile::read);

            // get roach index for offsets
            int roach_index;
            fo.getVar("Header.Toltec.RoachIndex").getVar(&roach_index);
            nw_ids[i] = roach_index;

            // get roach sample rate and ensure it is consistent across networks
            fo.getVar("Header.Toltec.SampleFreq").getVar(&f_smp_roach);
            fsmp_ref = citlali::pipeline::reconcile_sample_rate_hz(
                fsmp_ref, f_smp_roach, roach_index);

            // get dimensions for time matrix
            Eigen::Index n_pts = fo.getVar("Data.Toltec.Ts").getDim(0).getSize();
            Eigen::Index n_times = fo.getVar("Data.Toltec.Ts").getDim(1).getSize();

            // get time matrix
            Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ts(n_pts, n_times);
            fo.getVar("Data.Toltec.Ts").getVar(ts.data());

            // get fpga frequency
            double fpga_freq;
            fo.getVar("Header.Toltec.FpgaFreq").getVar(&fpga_freq);

            // cast to double
            Eigen::MatrixXd ts_double = ts.cast<double>();

            // find gaps
            int gaps = citlali::pipeline::count_packet_counter_gaps(ts);

            // add gaps to engine map
            if (gaps>0) {
                engine().alignment.gaps["Toltec" + std::to_string(roach_index)] = gaps;
            }

            // store all time vectors
            nw_times[i] =
                citlali::pipeline::network_time_from_timestream_matrix(
                    ts_double, fpga_freq,
                    engine().interface_sync.offsets[
                        "toltec"+std::to_string(roach_index)]);

            fo.close();

        } catch (NcException &e) {
            throw std::runtime_error(fmt::format(
                "unable to open file {}: {}",
                data_item.filepath(), e.what()));
        }
    }

    // get hwpr times if not ignored
    if (engine().calib.run_hwpr) {
        citlali::pipeline::validate_hwpr_alignment_inputs(
            engine().calib.hwpr_recvt, engine().calib.hwpr_angle, "gap");
        logger->debug("calculating hwpr time");
        // hwpr gets added alongside networks
        nw_times.push_back(engine().calib.hwpr_recvt);
    }

    const auto overlap = citlali::pipeline::find_common_timestream_overlap(
        nw_times, "align_timestreams_gaps");
    const double max_init_time = overlap.max_start;
    const double min_final_time = overlap.min_end;

    citlali::pipeline::require_positive_sample_rate_hz(
        fsmp_ref, "align_timestreams_gaps");
    double dt = 1.0 / fsmp_ref;
    Eigen::VectorXd t_common = citlali::pipeline::build_common_gap_time_grid(
        max_init_time, min_final_time, dt, "align_timestreams_gaps");
    double tol = dt / 2.0;

    std::vector<Eigen::VectorXi> masks =
        citlali::pipeline::build_common_time_grid_masks(
            nw_times, t_common, max_init_time, dt, tol, logger);

    // build a network-keyed mask table for downstream flagging
    for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(nw_ids.size()); ++j) {
        if (nw_ids[j] < 0) {
            continue;
        }
        engine().alignment.network_masks[nw_ids[j]] = masks[j];
    }

    citlali::pipeline::interpolate_telescope_data_to_common_time(
        engine().telescope.tel_data, t_common, true);

    // interpolate hwpr
    if (engine().calib.run_hwpr) {
        logger->debug("interpolating hwpr angle");
        int n_times = nw_times.size();
        citlali::pipeline::interpolate_hwpr_angle_to_common_time(
            engine().calib.hwpr_angle, nw_times[n_times - 1], t_common);
    }

    engine().alignment.common_time = t_common;
    engine().alignment.masks = masks;
    engine().alignment.network_times = nw_times;
}
