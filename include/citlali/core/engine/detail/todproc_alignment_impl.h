#pragma once

// Implementation detail included by todproc.h.

#include <citlali/core/pipeline/timestream_alignment_helpers.h>

template <class EngineType>
void TimeOrderedDataProc<EngineType>::align_timestreams(const RawObs &rawobs) {
    using namespace netCDF;
    using namespace netCDF::exceptions;

    // clear start and end indices for each observation
    engine().start_indices.clear();
    engine().end_indices.clear();

    // clear gaps
    engine().gaps.clear();

    // vector of network times
    std::vector<Eigen::VectorXd> nw_ts;
    // start and end times
    std::vector<double> nw_t0, nw_tn;

    // maximum start time
    double max_t0 = -99;

    // minimum end time
    double min_tn = std::numeric_limits<double>::max();
    // indices of max start time and min end time
    Eigen::Index max_t0_i, min_tn_i;

    // set network
    Eigen::Index nw = 0;
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

            // check if sample rate is the same and exit if not
            if (fsmp!=-1 && fsmp_roach!=fsmp) {
                logger->error("mismatched sample rate in toltec{}",roach_index);
                std::exit(EXIT_FAILURE);
            }
            else {
                fsmp = fsmp_roach;
            }

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
                engine().gaps["Toltec" + std::to_string(roach_index)] = gaps;
            }

            // get fpga frequency
            double fpga_freq;
            fo.getVar("Header.Toltec.FpgaFreq").getVar(&fpga_freq);

            nw_ts.push_back(
                citlali::pipeline::network_time_from_timestream_matrix(
                    ts.cast<double>(), fpga_freq,
                    engine().interface_sync_offset[
                        "toltec"+std::to_string(roach_index)]));

            // push back start time
            nw_t0.push_back(nw_ts.back()[0]);

            // push back end time
            nw_tn.push_back(nw_ts.back()[n_pts - 1]);

            // get global max start time and index
            if (nw_t0.back() > max_t0) {
                max_t0 = nw_t0.back();
                max_t0_i = nw;
            }

            // get global min end time and index
            if (nw_tn.back() < min_tn) {
                min_tn = nw_tn.back();
                min_tn_i = nw;
            }

            // increment nw
            nw++;

            fo.close();

        } catch (NcException &e) {
            logger->error("{}", e.what());
            throw DataIOError{fmt::format(
                "failed to load data from netCDF file {}", data_item.filepath())};
        }
    }

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

    // size of smallest data time vector
    Eigen::Index min_size = nw_ts[0].size();

    // loop through time vectors and get the smallest
    for (Eigen::Index i=0; i<nw_ts.size(); ++i) {
        // find start index that is larger than max start
        Eigen::Index si = citlali::pipeline::find_first_sample_at_or_after(
            nw_ts[i], max_t0,
            fmt::format("failed to find aligned start sample for interface index {}", i));
        // pushback start index on start index vector
        engine().start_indices.push_back(si);

        // find end index that is smaller than min end
        Eigen::Index ei = citlali::pipeline::find_last_sample_at_or_before(
            nw_ts[i], min_tn, si,
            fmt::format("failed to find aligned end sample for interface index {}", i));
        // pushback end index on end index vector
        engine().end_indices.push_back(ei);
    }

    // get min size
    for (Eigen::Index i=0; i<nw_ts.size(); ++i) {
        // start indices
        auto si = engine().start_indices[i];
        // end indices
        auto ei = engine().end_indices[i];
        if (ei < si) {
            throw std::runtime_error(fmt::format(
                "invalid aligned sample range for interface index {}: start={} end={}",
                i, si, ei));
        }

        // if smallest length, update min_size
        if ((ei - si + 1) < min_size) {
            min_size = ei - si + 1;
        }
    }

    // if hwpr requested
    if (engine().calib.run_hwpr) {
        // find start index that is larger than max start for hwpr
        Eigen::Index si = citlali::pipeline::find_first_sample_at_or_after(
            engine().calib.hwpr_recvt, max_t0,
            "failed to find aligned HWPR start sample");
        // pushback start index on hwpr start index vector
        engine().hwpr_start_indices = si;

        // find end index that is smaller than min end for hwpr
        Eigen::Index ei = citlali::pipeline::find_last_sample_at_or_before(
            engine().calib.hwpr_recvt, min_tn, si,
            "failed to find aligned HWPR end sample");
        // pushback end index on hwpr end index vector
        engine().hwpr_end_indices = ei;

        // update min_size for all time vectors if hwpr data is shorter (data and hwpr)
        if ((ei - si + 1) < min_size) {
            min_size = ei - si + 1;
        }
    }

    if (min_size <= 0) {
        throw std::runtime_error("aligned common timestream length is not positive");
    }

    // shortest common data time vector
    Eigen::VectorXd xi = nw_ts[max_t0_i].segment(engine().start_indices[max_t0_i], min_size);

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

    // clear start and end indices for each observation
    engine().start_indices.clear();
    engine().end_indices.clear();
    engine().nw_masks.clear();

    std::vector<Eigen::VectorXd> nw_times(rawobs.kidsdata().size());
    std::vector<Eigen::Index> nw_ids(rawobs.kidsdata().size(), -1);

    // clear gaps
    engine().gaps.clear();

    // loop through networks and build time vectors
    int i = 0;
    double f_smp_roach = -1.0;
    double fsmp_ref = -1.0;
    for (const RawObs::DataItem &data_item : rawobs.kidsdata()) {
        try {
            const RawObs::DataItem &data_item = rawobs.kidsdata()[i];
            // load data file
            NcFile fo(data_item.filepath(), NcFile::read);

            // get roach sample rate and ensure it is consistent across networks
            fo.getVar("Header.Toltec.SampleFreq").getVar(&f_smp_roach);
            if (fsmp_ref != -1.0 && f_smp_roach != fsmp_ref) {
                int roach_index_mismatch = -1;
                fo.getVar("Header.Toltec.RoachIndex").getVar(&roach_index_mismatch);
                logger->error("mismatched sample rate in toltec{} ({} vs reference {})",
                              roach_index_mismatch, f_smp_roach, fsmp_ref);
                std::exit(EXIT_FAILURE);
            }
            fsmp_ref = f_smp_roach;

            // get roach index for offsets
            int roach_index;
            fo.getVar("Header.Toltec.RoachIndex").getVar(&roach_index);
            nw_ids[i] = roach_index;

            // get dimensions for time matrix
            Eigen::Index n_pts = fo.getVar("Data.Toltec.Ts").getDim(0).getSize();
            Eigen::Index n_times = fo.getVar("Data.Toltec.Ts").getDim(1).getSize();

            // get time matrix
            //Eigen::MatrixXi ts(n_times, n_pts);
            Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> ts(n_pts, n_times);
            fo.getVar("Data.Toltec.Ts").getVar(ts.data());

            // get fpga frequency
            double fpga_freq;
            fo.getVar("Header.Toltec.FpgaFreq").getVar(&fpga_freq);

            // cast to double
            Eigen::MatrixXd ts_double = ts.cast<double>();

            Eigen::MatrixXi ts_t = ts.transpose();

            // find gaps
            int gaps = citlali::pipeline::count_packet_counter_gaps(ts);

            // add gaps to engine map
            if (gaps>0) {
                engine().gaps["Toltec" + std::to_string(roach_index)] = gaps;
            }

            // store all time vectors
            nw_times[i] =
                citlali::pipeline::network_time_from_timestream_matrix(
                    ts_double, fpga_freq,
                    engine().interface_sync_offset[
                        "toltec"+std::to_string(roach_index)]);
            i++;

            fo.close();

        } catch (NcException &e) {
            throw std::runtime_error(fmt::format("unable to open file : {}", "data_item.filepath()", e.what()));
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

    if (fsmp_ref <= 0.0) {
        logger->error("invalid or missing sample rate in align_timestreams_gaps");
        std::exit(EXIT_FAILURE);
    }
    double dt = 1.0 / fsmp_ref;
    if (!std::isfinite(max_init_time) || !std::isfinite(min_final_time) || max_init_time > min_final_time) {
        throw std::runtime_error(fmt::format(
            "no common time overlap across input timestreams with gap interpolation: max_start={} min_end={}",
            max_init_time, min_final_time));
    }
    Eigen::Index n_samples = static_cast<int>((min_final_time - max_init_time) / dt) + 1;
    if (n_samples <= 0) {
        throw std::runtime_error(fmt::format(
            "invalid common sample count in align_timestreams_gaps: {}", n_samples));
    }
    Eigen::VectorXd t_common = Eigen::VectorXd::LinSpaced(n_samples, max_init_time, max_init_time + dt * (n_samples - 1));
    double tol = dt / 2.0;

    std::vector<Eigen::VectorXi> masks =
        citlali::pipeline::build_common_time_grid_masks(
            nw_times, t_common, max_init_time, dt, tol, logger);

    // build a network-keyed mask table for downstream flagging
    for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(nw_ids.size()); ++j) {
        if (nw_ids[j] < 0) {
            continue;
        }
        engine().nw_masks[nw_ids[j]] = masks[j];
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

    engine().t_common = t_common;
    engine().masks = masks;
    engine().nw_times = nw_times;
}
