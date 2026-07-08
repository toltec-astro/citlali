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
            int gaps = 0;
            if (n_pts > 1) {
                gaps = ((ts.block(1,3,n_pts-1,1).array() - ts.block(0,3,n_pts-1,1).array()).array() > 1).count();
            }

            // add gaps to engine map
            if (gaps>0) {
                engine().gaps["Toltec" + std::to_string(roach_index)] = gaps;
            }

            // get fpga frequency
            double fpga_freq;
            fo.getVar("Header.Toltec.FpgaFreq").getVar(&fpga_freq);

            // ClockTime (sec)
            auto sec0 = ts.cast <double> ().col(0);
            // ClockTimeNanoSec (nsec)
            auto nsec0 = ts.cast <double> ().col(5);
            // PpsCount (pps ticks)
            auto pps = ts.cast <double> ().col(1);
            // ClockCount (clock ticks)
            auto msec = ts.cast <double> ().col(2)/fpga_freq;
            // PacketCount (packet ticks)
            auto count = ts.cast <double> ().col(3);
            // PpsTime (clock ticks)
            auto pps_msec = ts.cast <double> ().col(4)/fpga_freq;
            // get start time
            auto t0 = sec0 + nsec0*1e-9;

            // shift start time (offset determined empirically)
            int start_t = int(t0[0] - 0.5);
            //int start_t = int(t0[0]);

            // convert start time to double
            double start_t_dbl = start_t;
            // clock count - clock ticks
            Eigen::VectorXd dt = msec - pps_msec;
            // remove overflow due to int32
            dt = (dt.array() < 0).select(msec.array() - pps_msec.array() + (pow(2.0,32)-1)/fpga_freq,msec - pps_msec);
            // get network time and add offsets
            nw_ts.push_back(start_t_dbl + pps.array() + dt.array() +
                            engine().interface_sync_offset["toltec"+std::to_string(roach_index)]);

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

    // size of telescope data
    Eigen::Matrix<Eigen::Index,1,1> nd;
    nd << engine().telescope.tel_data["TelTime"].size();

    // shortest common data time vector
    Eigen::VectorXd xi = nw_ts[max_t0_i].segment(engine().start_indices[max_t0_i], min_size);

    // interpolate telescope data
    for (const auto &tel_it : engine().telescope.tel_data) {
        if (tel_it.first !="TelTime") {
            // telescope vector to interpolate
            Eigen::VectorXd yd = engine().telescope.tel_data[tel_it.first];
            // vector to store interpolated outputs in
            Eigen::VectorXd yi(min_size);

            mlinterp::interp(nd.data(), min_size, // nd, ni
                             yd.data(), yi.data(), // yd, yi
                             engine().telescope.tel_data["TelTime"].data(), xi.data()); // xd, xi

            // move back into tel_data vector
            engine().telescope.tel_data[tel_it.first] = std::move(yi);
        }
    }

    // replace telescope time vectors
    engine().telescope.tel_data["TelTime"] = xi;
    engine().telescope.tel_data["TelUTC"] = xi;

    // interpolate hwpr data
    if (engine().calib.run_hwpr) {
        Eigen::Matrix<Eigen::Index,1,1> hwpr_nd;
        hwpr_nd << engine().calib.hwpr_recvt.size();
        Eigen::VectorXd yd = engine().calib.hwpr_angle;
        // vector to store interpolated outputs in
        Eigen::VectorXd yi(min_size);
        mlinterp::interp(hwpr_nd.data(), min_size, // nd, ni
                         yd.data(), yi.data(), // yd, yi
                         engine().calib.hwpr_recvt.data(), xi.data()); // xd, xi

        // move back into hwpr angle
        engine().calib.hwpr_angle = std::move(yi);
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
            int gaps = 0;
            if (n_pts > 1) {
                gaps = ((ts.block(1,3,n_pts-1,1).array() - ts.block(0,3,n_pts-1,1).array()).array() > 1).count();
            }

            // add gaps to engine map
            if (gaps>0) {
                engine().gaps["Toltec" + std::to_string(roach_index)] = gaps;
            }

            auto sec = ts_double.col(0);        // ClockTime (sec)
            auto nsec = ts_double.col(5);       // ClockTimeNanoSec (nsec)
            auto pps = ts_double.col(1);        // PpsCount (pps ticks)
            auto msec = ts_double.col(2) / fpga_freq;  // ClockCount (clock ticks) to seconds
            //Eigen::VectorXd count = ts_double.col(3);      // PacketCount (packet ticks)
            auto pps_msec = ts_double.col(4) / fpga_freq; // PpsTime (clock ticks) to seconds

            // determine start time with empirical offset
            double start_time_dbl = sec[0] + nsec[0] * 1e-9;
            int start_time = int(start_time_dbl - 0.5);
            start_time_dbl = start_time;

            // calculate clock count difference (dt)
            Eigen::VectorXd dt = msec - pps_msec;

            // handle overflow due to int32, using Eigen array logic
            dt = (dt.array() < 0).select(msec.array() - pps_msec.array() + (pow(2.0, 32) - 1) / fpga_freq, msec - pps_msec);

            // build the time vector for the current network
            Eigen::VectorXd nw_time = start_time_dbl + pps.array() + dt.array()
                                        + engine().interface_sync_offset["toltec"+std::to_string(roach_index)];

            // store all time vectors
            nw_times[i] = std::move(nw_time);
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

    // latest starting time of all networks
    double max_init_time = std::numeric_limits<double>::lowest();
    // earliest final time of all networks
    double min_final_time = std::numeric_limits<double>::max();

    // indices for max_init_time and min_final_time
    Eigen::Index max_init_idx = 0;
    Eigen::Index min_final_idx = 0;

    // get global max init and min final times and indices
    for (Eigen::Index i = 0; i < nw_times.size(); ++i) {
        if (nw_times[i].size() == 0) {
            throw std::runtime_error(fmt::format(
                "empty time vector for interface index {} in align_timestreams_gaps", i));
        }
        double initial_time = nw_times[i](0);
        double final_time = nw_times[i](nw_times[i].size() - 1);

        // get latest starting network
        if (initial_time > max_init_time) {
            max_init_time = initial_time;
            max_init_idx = i;
        }
        // get earliest ending network
        if (final_time < min_final_time) {
            min_final_time = final_time;
            min_final_idx = i;
        }
    }

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

    std::vector<Eigen::VectorXi> masks;
    masks.reserve(nw_times.size());

    for (const auto &t : nw_times) {
        Eigen::VectorXi mask = Eigen::VectorXi::Zero(n_samples);

        for (int i = 0; i < t.size(); ++i) {
            double time = t(i);
            int idx = static_cast<int>(std::round((time - max_init_time) / dt));
            if (idx >= 0 && idx < n_samples && std::abs(time - t_common(idx)) <= tol) {
                mask(idx) = 1;
            }
        }

        logger->warn("{}/{} samples were not aligned to the common time grid", mask.size() - mask.sum(), mask.size());

        masks.push_back(std::move(mask));
    }

    // build a network-keyed mask table for downstream flagging
    for (Eigen::Index j = 0; j < static_cast<Eigen::Index>(nw_ids.size()); ++j) {
        if (nw_ids[j] < 0) {
            continue;
        }
        engine().nw_masks[nw_ids[j]] = masks[j];
    }

    // size of telescope data
    Eigen::Matrix<Eigen::Index,1,1> nd;
    nd << engine().telescope.tel_data["TelTime"].size();

    // interpolate telescope data onto data timestream
    for (const auto &tel_it : engine().telescope.tel_data) {
        // don't interpolate telescope time itself
        if (tel_it.first !="TelTime" && tel_it.first !="TelUTC") {
            // telescope vector to interpolate
            Eigen::VectorXd yd = engine().telescope.tel_data.at(tel_it.first);
            // vector to store interpolated outputs in
            Eigen::VectorXd yi(n_samples);

            mlinterp::interp(nd.data(), n_samples, // nd, ni
                                yd.data(), yi.data(), // yd, yi
                                engine().telescope.tel_data.at("TelTime").data(), t_common.data()); // xd, xi

            // move back into data vector
            engine().telescope.tel_data[tel_it.first] = std::move(yi);
        }
    }

    // replace telescope time vectors
    engine().telescope.tel_data.at("TelTime") = t_common;
    engine().telescope.tel_data.at("TelUTC") = t_common;

    // interpolate hwpr
    if (engine().calib.run_hwpr) {
        logger->debug("interpolating hwpr angle");
        int n_times = nw_times.size();
        nd << nw_times[n_times - 1].size();

        // vector to store interpolated outputs in
        Eigen::VectorXd yi(n_samples);

        mlinterp::interp(nd.data(), n_samples, // nd, ni
                            engine().calib.hwpr_angle.data(), yi.data(), // yd, yi
                            nw_times[n_times - 1].data(), t_common.data()); // xd, xi

        engine().calib.hwpr_angle = std::move(yi);
    }

    engine().t_common = t_common;
    engine().masks = masks;
    engine().nw_times = nw_times;
}
