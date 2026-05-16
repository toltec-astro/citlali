#pragma once

#include <thread>
#include <mutex>

#include <citlali/core/timestream/timestream.h>

#include <citlali/core/mapmaking/map.h>
#include <citlali/core/mapmaking/tiled_accumulator.h>
#include <citlali/core/utils/pointing.h>
#include <tula/logging.h>

using timestream::TCData;

// selects the type of TCData
using timestream::TCDataKind;

namespace mapmaking {

class NaiveMapmaker {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");
    std::unique_ptr<std::mutex> naive_mutex = std::make_unique<std::mutex>();

    // toltec array mounting angle
    std::map<int, double> install_ang = {
        {-1,-1},
        {0,pi/2},
        {1,-pi/2},
        {2,-pi/2},
    };

    // toltec detector orientation angles
    std::map<int, double> fgs = {
        {-1,-1},
        {0,0},
        {1,pi/4},
        {2,pi/2},
        {3,3*pi/4}
    };

    // run polarization?
    bool run_polarization;

    // allocate pointing matrix for polarization reduction
    template <class map_buffer_t>
    void allocate_pointing(map_buffer_t &, double, double, double, Eigen::Index, int, int);

    // populate maps with a time chunk (signal, kernel, coverage, and noise)
    template<class map_buffer_t, typename Derived, typename apt_t>
    void populate_maps_naive(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, map_buffer_t &, map_buffer_t &,
                             Eigen::DenseBase<Derived> &, std::string &, apt_t &, double, bool, bool);

    // populate maps with a time chunk (signal, kernel, coverage, and noise)
    template<class map_buffer_t, typename Derived, typename apt_t>
    void populate_maps_naive_parallel(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, map_buffer_t &, map_buffer_t &,
                                      Eigen::DenseBase<Derived> &, std::string &, apt_t &, double, bool, bool);
};

template <class map_buffer_t>
void NaiveMapmaker::allocate_pointing(map_buffer_t &mb, double weight, double cos_2angle, double sin_2angle,
                                      Eigen::Index map_index, int ir, int ic) {
    int pix = mb.n_rows * ic + ir;
    Eigen::MatrixXd& matrix = mb.pointing[map_index];

    // calculate reused expressions
    double weight_cos_2angle = weight * cos_2angle;
    double weight_sin_2angle = weight * sin_2angle;
    double weight_cos2_2angle = weight * std::pow(cos_2angle, 2);
    double weight_sin2_2angle = weight * std::pow(sin_2angle, 2);
    double weight_cos_sin_2angle = weight * cos_2angle * sin_2angle;

    // update pointing matrix
    matrix(pix, 0) += weight;
    matrix(pix, 1) += weight_cos_2angle;
    matrix(pix, 2) += weight_sin_2angle;
    matrix(pix, 3) = matrix(pix, 1);  // previously set to += weight*cos_2angle, then directly assigned here
    matrix(pix, 4) += weight_cos2_2angle;
    matrix(pix, 5) += weight_cos_sin_2angle;
    matrix(pix, 6) = matrix(pix, 2);  // previously set to += weight*sin_2angle, then directly assigned here
    matrix(pix, 7) = matrix(pix, 5);  // reuse the result from matrix(pix,5)
    matrix(pix, 8) += weight_sin2_2angle;
}

template<class map_buffer_t, typename Derived, typename apt_t>
void NaiveMapmaker::populate_maps_naive(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, map_buffer_t &omb,
                                        map_buffer_t &cmb, Eigen::DenseBase<Derived> &map_indices,
                                        std::string &pixel_axes, apt_t &apt, double d_fsmp,
                                        bool run_omb, bool run_noise) {
    tula::logging::scoped_timeit total_timer{"populate_maps_naive total"};

    TiledMapAccumulator signals, weights, kernels, coverages;
    TiledMapAccumulator cmb_signals, cmb_weights, cmb_kernels, cmb_coverages;

    const bool use_cmb = !cmb.noise.empty();
    const bool use_omb = !omb.noise.empty();
    const bool run_kernel = !omb.kernel.empty();
    const bool run_coverage = !omb.coverage.empty();
    const bool run_hwpr = in.hwpr_angle.data.size()!=0;

    if (run_omb) {
        signals.reset(omb.signal.size(), omb.n_rows, omb.n_cols);
        weights.reset(omb.signal.size(), omb.n_rows, omb.n_cols);

        if (run_kernel) {
            kernels.reset(omb.signal.size(), omb.n_rows, omb.n_cols);
        }
        if (run_coverage) {
            coverages.reset(omb.signal.size(), omb.n_rows, omb.n_cols);
        }
    }

    if (run_polarization && !cmb.signal.empty()) {
        cmb_signals.reset(cmb.signal.size(), cmb.n_rows, cmb.n_cols);
        cmb_weights.reset(cmb.signal.size(), cmb.n_rows, cmb.n_cols);

        if (run_kernel) {
            cmb_kernels.reset(cmb.signal.size(), cmb.n_rows, cmb.n_cols);
        }
        if (run_coverage) {
            cmb_coverages.reset(cmb.signal.size(), cmb.n_rows, cmb.n_cols);
        }
    }

    map_buffer_t omb_copy, cmb_copy;
    // pointer to map buffer with noise maps
    map_buffer_t *nmb = nullptr, *nmb_copy = nullptr;

    omb_copy.n_rows = omb.n_rows;
    omb_copy.n_cols = omb.n_cols;

    cmb_copy.n_rows = cmb.n_rows;
    cmb_copy.n_cols = cmb.n_cols;

    if (run_noise) {
        nmb = use_cmb ? &cmb : (use_omb ? &omb : nullptr);
        if (nmb != nullptr) {
            nmb_copy = use_cmb ? &cmb_copy : &omb_copy;
            nmb_copy->noise = nmb->noise;

            for (Eigen::Index i=0; i<nmb_copy->noise.size(); ++i) {
                nmb_copy->noise[i].setZero();
            }
        }
    }

    // step to skip to reach next stokes param
    int step = omb.pointing.size();

    if (!omb.pointing.empty() && run_omb) {
        for (Eigen::Index i=0; i<omb.pointing.size(); ++i) {
            omb_copy.pointing.emplace_back(Eigen::MatrixXd::Zero(omb.pointing[i].rows(), omb.pointing[i].cols()));
        }
    }

    if (!cmb.pointing.empty() && run_omb) {
        for (Eigen::Index i=0; i<cmb.pointing.size(); ++i) {
            cmb_copy.pointing.emplace_back(Eigen::MatrixXd::Zero(cmb.pointing[i].rows(), cmb.pointing[i].cols()));
        }
    }

    // dimensions of data
    Eigen::Index n_pts = in.scans.data.rows();
    Eigen::Index n_dets = in.scans.data.cols();

    // signal, kernel and noise map values
    double signal, kernel, noise_v;

    // noise map indices
    Eigen::Index nmb_ir, nmb_ic;

    // cosine and sine of angles
    double angle, cos_2angle, sin_2angle;

    // add detector to map?
    bool run_det;

    {
        tula::logging::scoped_timeit accumulate_timer{"populate_maps_naive accumulate"};
        for (Eigen::Index i=0; i<n_dets; ++i) {
            // skip fg = -1 if in polarization mode
            if (run_polarization && apt["loc"](i)==-1) {
                run_det = false;
            }
            else {
                run_det = true;
            }

            // skip completely flagged detectors
            if (apt["flag"](i)==0 && (in.flags.data.col(i).array()==0).any() && run_det) {
                // which map to assign detector to
                Eigen::Index map_index = map_indices(i);

                // indices for Q and U maps
                int q_index = map_index + step;
                int u_index = map_index + 2 * step;

                // array index
                Eigen::Index array_index = apt["array"](i);
                // get detector pointing
                auto [lat, lon] = engine_utils::calc_det_pointing(in.tel_data.data, apt["x_t"](i), apt["y_t"](i),
                                                                  pixel_axes, in.pointing_offsets_arcsec.data, omb.map_grouping);

                Eigen::VectorXd alt;

                if (run_polarization) {
                    std::tuple<Eigen::VectorXd,Eigen::VectorXd> altaz_tuple = engine_utils::calc_det_pointing(in.tel_data.data, apt["x_t"](i),
                                                                                                               apt["y_t"](i), "altaz",
                                                                                                               in.pointing_offsets_arcsec.data,
                                                                                                               omb.map_grouping);
                    alt = std::get<0>(altaz_tuple);
                }

                // get map buffer row and col indices for lat and lon vectors
                Eigen::VectorXd omb_irow = lat.array()/omb.pixel_size_rad + (omb.n_rows - 1)/2.;
                Eigen::VectorXd omb_icol = lon.array()/omb.pixel_size_rad + (omb.n_cols - 1)/2.;

                Eigen::VectorXd cmb_irow, cmb_icol;
                if (use_cmb || (run_polarization && !cmb.signal.empty())) {
                    // get coadded map buffer row and col indices for lat and lon vectors
                    cmb_irow = lat.array()/cmb.pixel_size_rad + (cmb.n_rows - 1)/2.;
                    cmb_icol = lon.array()/cmb.pixel_size_rad + (cmb.n_cols - 1)/2.;
                }

                // loop through the samples
                for (Eigen::Index j=0; j<n_pts; ++j) {
                    // check if sample is flagged, ignore if so
                    if (!in.flags.data(j,i)) {
                        Eigen::Index omb_ir = static_cast<Eigen::Index>(std::llround(omb_irow(j)));
                        Eigen::Index omb_ic = static_cast<Eigen::Index>(std::llround(omb_icol(j)));

                        if (run_polarization) {
                            auto fg_index = apt["fg"](i);
                            if (run_hwpr) {
                                angle = 2*in.hwpr_angle.data(j) - (in.angle.data(j) + alt(j) + fgs[fg_index] + install_ang[array_index]);
                            }
                            else {
                                angle = in.angle.data(j) + alt(j) + fgs[fg_index] + install_ang[array_index];
                            }

                            cos_2angle = cos(2.*angle);
                            sin_2angle = sin(2.*angle);
                        }

                        if (run_omb) {
                            // make sure the data point is within the map
                            if ((omb_ir >= 0) && (omb_ir < omb.n_rows) && (omb_ic >= 0) && (omb_ic < omb.n_cols)) {
                                // populate signal map
                                signal = in.scans.data(j,i)*in.weights.data(i);
                                signals.add(map_index, omb_ir, omb_ic, signal);

                                // populate weight map
                                weights.add(map_index, omb_ir, omb_ic, in.weights.data(i));

                                // populate kernel map
                                if (run_kernel) {
                                    kernel = in.kernel.data(j,i)*in.weights.data(i);
                                    kernels.add(map_index, omb_ir, omb_ic, kernel);
                                }

                                // populate coverage map
                                if (run_coverage) {
                                    coverages.add(map_index, omb_ir, omb_ic, 1./d_fsmp);
                                }

                                if (run_polarization) {
                                    // calculate pointing matrix
                                    allocate_pointing(omb_copy, in.weights.data(i), cos_2angle, sin_2angle, map_index, omb_ir, omb_ic);

                                    // update signal map Q and U
                                    signals.add(q_index, omb_ir, omb_ic, signal*cos_2angle);
                                    signals.add(u_index, omb_ir, omb_ic, signal*sin_2angle);

                                    // update kernel map Q and U
                                    if (run_kernel) {
                                        kernels.add(q_index, omb_ir, omb_ic, kernel*cos_2angle);
                                        kernels.add(u_index, omb_ir, omb_ic, kernel*sin_2angle);
                                    }
                                }
                            }
                        }

                        if (run_polarization && !cmb.signal.empty() && run_omb) {
                            Eigen::Index cmb_ir = static_cast<Eigen::Index>(std::llround(cmb_irow(j)));
                            Eigen::Index cmb_ic = static_cast<Eigen::Index>(std::llround(cmb_icol(j)));

                            // make sure the data point is within the map
                            if ((cmb_ir >= 0) && (cmb_ir < cmb.n_rows) && (cmb_ic >= 0) && (cmb_ic < cmb.n_cols)) {
                                // populate signal map
                                signal = in.scans.data(j,i)*in.weights.data(i);
                                cmb_signals.add(map_index, cmb_ir, cmb_ic, signal);

                                // populate weight map
                                cmb_weights.add(map_index, cmb_ir, cmb_ic, in.weights.data(i));

                                // populate kernel map
                                if (run_kernel) {
                                    kernel = in.kernel.data(j,i)*in.weights.data(i);
                                    cmb_kernels.add(map_index, cmb_ir, cmb_ic, kernel);
                                }

                                // populate coverage map
                                if (run_coverage) {
                                    cmb_coverages.add(map_index, cmb_ir, cmb_ic, 1./d_fsmp);
                                }

                                // calculate pointing matrix
                                allocate_pointing(cmb_copy, in.weights.data(i), cos_2angle, sin_2angle, map_index, cmb_ir, cmb_ic);

                                // update signal map Q and U
                                cmb_signals.add(q_index, cmb_ir, cmb_ic, signal*cos_2angle);
                                cmb_signals.add(u_index, cmb_ir, cmb_ic, signal*sin_2angle);

                                // update kernel map Q and U
                                if (run_kernel) {
                                    cmb_kernels.add(q_index, cmb_ir, cmb_ic, kernel*cos_2angle);
                                    cmb_kernels.add(u_index, cmb_ir, cmb_ic, kernel*sin_2angle);
                                }
                            }
                        }

                        // check if noise maps requested
                        if (run_noise && nmb != nullptr && nmb_copy != nullptr) {
                            // if coaddition is enabled
                            if (use_cmb) {
                                nmb_ir = static_cast<Eigen::Index>(std::llround(cmb_irow(j)));
                                nmb_ic = static_cast<Eigen::Index>(std::llround(cmb_icol(j)));
                            }
                            // else make noise maps for obs
                            else {
                                nmb_ir = static_cast<Eigen::Index>(std::llround(omb_irow(j)));
                                nmb_ic = static_cast<Eigen::Index>(std::llround(omb_icol(j)));
                            }

                            // coadd into current noise map
                            if ((nmb_ir >= 0) && (nmb_ir < nmb->n_rows) && (nmb_ic >= 0) && (nmb_ic < nmb->n_cols)) {
                                //if (run_polarization) {
                                    //if (use_cmb) {
                                        // calculate pointing matrix for cmb
                                      //  allocate_pointing(cmb_copy, in.weights.data(i), cos_2angle, sin_2angle, map_index, nmb_ir, nmb_ic);
                                    //}
                                //}
                                // loop through noise maps
                                for (Eigen::Index nn=0; nn<nmb->n_noise; ++nn) {
                                    // randomizing on dets
                                    if (nmb->randomize_dets) {
                                        noise_v = in.noise.data(nn,i)*in.scans.data(j,i)*in.weights.data(i);
                                    }
                                    else {
                                        noise_v = in.noise.data(nn)*in.scans.data(j,i)*in.weights.data(i);
                                    }
                                    // add noise value to current noise map
                                    nmb_copy->noise[map_index](nmb_ir,nmb_ic,nn) += noise_v;

                                    if (run_polarization) {
                                        // update noise map Q
                                        nmb_copy->noise[q_index](nmb_ir,nmb_ic,nn) += noise_v*cos_2angle;
                                        // update noise map U
                                        nmb_copy->noise[u_index](nmb_ir,nmb_ic,nn) += noise_v*sin_2angle;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    {
        tula::logging::scoped_timeit merge_timer{"populate_maps_naive merge"};
        std::scoped_lock<std::mutex> lk(*naive_mutex);
        if (run_omb) {
            signals.merge_into(omb.signal);
            weights.merge_into(omb.weight);

            if (run_kernel) {
                kernels.merge_into(omb.kernel);
            }

            if (run_coverage) {
                coverages.merge_into(omb.coverage);
            }
            if (!omb.pointing.empty()) {
                for (int i=0; i<omb.pointing.size(); ++i) {
                    omb.pointing[i] += omb_copy.pointing[i];
                }
            }
        }

        if (run_polarization && !cmb.signal.empty()) {
            cmb_signals.merge_into(cmb.signal);
            cmb_weights.merge_into(cmb.weight);

            if (run_kernel) {
                cmb_kernels.merge_into(cmb.kernel);
            }

            if (run_coverage) {
                cmb_coverages.merge_into(cmb.coverage);
            }
        }

        if (run_noise && nmb != nullptr && nmb_copy != nullptr) {
            for (Eigen::Index i=0; i<nmb->noise.size(); ++i) {
                nmb->noise[i] += nmb_copy->noise[i];
            }
        }

        if (!cmb.pointing.empty() && run_omb) {
            for (int i=0; i<cmb.pointing.size(); ++i) {
                cmb.pointing[i] += cmb_copy.pointing[i];
            }
        }
    }

    nmb = nullptr;
    nmb_copy = nullptr;
}

template<class map_buffer_t, typename Derived, typename apt_t>
void NaiveMapmaker::populate_maps_naive_parallel(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, map_buffer_t &omb,
                                                 map_buffer_t &cmb, Eigen::DenseBase<Derived> &map_indices,
                                                 std::string &pixel_axes, apt_t &apt, double d_fsmp,
                                                 bool run_omb, bool run_noise) {

    const bool use_cmb = !cmb.noise.empty();
    const bool use_omb = !omb.noise.empty();
    const bool run_kernel = !omb.kernel.empty();
    const bool run_coverage = !omb.coverage.empty();
    const bool run_hwpr = in.hwpr_angle.data.size()!=0;

    // pointer to map buffer with noise maps
    map_buffer_t *nmb;

    if (run_noise) {
        nmb = use_cmb ? &cmb : (use_omb ? &omb : nullptr);
    }

    // step to skip to reach next stokes param
    int step = omb.pointing.size();

    // dimensions of data
    Eigen::Index n_pts = in.scans.data.rows();
    Eigen::Index n_dets = in.scans.data.cols();

    bool unique_map_indices =
        (map_indices.size() == n_dets && omb.signal.size() == static_cast<std::size_t>(n_dets));
    if (unique_map_indices) {
        std::vector<unsigned char> seen(omb.signal.size(), 0);
        for (Eigen::Index i = 0; i < n_dets; ++i) {
            const auto idx = map_indices(i);
            if (idx < 0 || idx >= static_cast<Eigen::Index>(seen.size()) || seen[static_cast<std::size_t>(idx)] != 0) {
                unique_map_indices = false;
                break;
            }
            seen[static_cast<std::size_t>(idx)] = 1;
        }
    }

    if (!unique_map_indices) {
        logger->warn("populate_maps_naive_parallel requires unique map indices; falling back to populate_maps_naive");
        populate_maps_naive(in, omb, cmb, map_indices, pixel_axes, apt, d_fsmp, run_omb, run_noise);
        return;
    }

    // signal, kernel and noise map values
    double signal, kernel, noise_v;

    // noise map indices
    Eigen::Index nmb_ir, nmb_ic;

    // cosine and sine of angles
    double angle, cos_2angle, sin_2angle;

    // add detector to map?
    bool run_det;

    // placeholder vectors of size ndet for grppi maps
    std::vector<int> map_in_vec, map_out_vec;
    map_in_vec.resize(omb.signal.size());
    std::iota(map_in_vec.begin(), map_in_vec.end(), 0);
    map_out_vec.resize(map_in_vec.size());

    // parallelize over detectors
    //for (Eigen::Index i=0; i<n_dets; ++i) {
    grppi::map(tula::grppi_utils::dyn_ex(omb.parallel_policy), map_in_vec, map_out_vec, [&](auto i) {
    //for (Eigen::Index i=0; i<n_dets; ++i) {
        // skip fg = -1 if in polarization mode
        if (run_polarization && apt["loc"](i)==-1) {
            run_det = false;
        }
        else {
            run_det = true;
        }

        // skip completely flagged detectors
        if (apt["flag"](i)==0 && (in.flags.data.col(i).array()==0).any() && run_det) {
            // which map to assign detector to
            Eigen::Index map_index = map_indices(i);

            // indices for Q and U maps
            int q_index = map_index + step;
            int u_index = map_index + 2 * step;

            // array index
            Eigen::Index array_index = apt["array"](i);
            // get detector pointing
            auto [lat, lon] = engine_utils::calc_det_pointing(in.tel_data.data, apt["x_t"](i), apt["y_t"](i),
                                                              pixel_axes, in.pointing_offsets_arcsec.data, omb.map_grouping);

            Eigen::VectorXd alt;

            if (run_polarization) {
                std::tuple<Eigen::VectorXd,Eigen::VectorXd> altaz_tuple = engine_utils::calc_det_pointing(in.tel_data.data, apt["x_t"](i),
                                                                                                           apt["y_t"](i), "altaz",
                                                                                                           in.pointing_offsets_arcsec.data,
                                                                                                           omb.map_grouping);
                alt = std::get<0>(altaz_tuple);
            }

            // get map buffer row and col indices for lat and lon vectors
            Eigen::VectorXd omb_irow = lat.array()/omb.pixel_size_rad + (omb.n_rows - 1)/2.;
            Eigen::VectorXd omb_icol = lon.array()/omb.pixel_size_rad + (omb.n_cols - 1)/2.;

            Eigen::VectorXd cmb_irow, cmb_icol;
            if (use_cmb || (run_polarization && !cmb.signal.empty())) {
                // get coadded map buffer row and col indices for lat and lon vectors
                cmb_irow = lat.array()/cmb.pixel_size_rad + (cmb.n_rows - 1)/2.;
                cmb_icol = lon.array()/cmb.pixel_size_rad + (cmb.n_cols - 1)/2.;
            }

            // loop through the samples
            for (Eigen::Index j=0; j<n_pts; ++j) {
                // check if sample is flagged, ignore if so
                if (!in.flags.data(j,i)) {
                    Eigen::Index omb_ir = static_cast<Eigen::Index>(std::llround(omb_irow(j)));
                    Eigen::Index omb_ic = static_cast<Eigen::Index>(std::llround(omb_icol(j)));

                    if (run_polarization) {
                        auto fg_index = apt["fg"](i);
                        if (run_hwpr) {
                            angle = 2*in.hwpr_angle.data(j) - (in.angle.data(j) + alt(j) + fgs[fg_index] + install_ang[array_index]);
                        }
                        else {
                            angle = in.angle.data(j) + alt(j) + fgs[fg_index] + install_ang[array_index];
                        }

                        cos_2angle = cos(2.*angle);
                        sin_2angle = sin(2.*angle);
                    }

                    if (run_omb) {
                        // make sure the data point is within the map
                        if ((omb_ir >= 0) && (omb_ir < omb.n_rows) && (omb_ic >= 0) && (omb_ic < omb.n_cols)) {
                            // populate signal map
                            signal = in.scans.data(j,i)*in.weights.data(i);
                            omb.signal[map_index](omb_ir, omb_ic) += signal;
                            //signals[map_index].push_back(T(omb_ir,omb_ic,signal));

                            // populate weight map
                            //weights[map_index].push_back(T(omb_ir,omb_ic,in.weights.data(i)));
                            omb.weight[map_index](omb_ir, omb_ic) += in.weights.data(i);

                            // populate kernel map
                            if (run_kernel) {
                                kernel = in.kernel.data(j,i)*in.weights.data(i);
                                //kernels[map_index].push_back(T(omb_ir,omb_ic,kernel));
                                omb.kernel[map_index](omb_ir, omb_ic) += kernel;
                            }

                            // populate coverage map
                            if (run_coverage) {
                                //coverages[map_index].push_back(T(omb_ir,omb_ic,1./d_fsmp));
                                omb.coverage[map_index](omb_ir, omb_ic) += 1./d_fsmp;
                            }

                            if (run_polarization) {
                                // calculate pointing matrix
                                allocate_pointing(omb, in.weights.data(i), cos_2angle, sin_2angle, map_index, omb_ir, omb_ic);

                                // update signal map Q and U
                                //signals[q_index].push_back(T(omb_ir,omb_ic,signal*cos_2angle));
                                //signals[u_index].push_back(T(omb_ir,omb_ic,signal*sin_2angle));

                                omb.signal[q_index](omb_ir, omb_ic) += signal*cos_2angle;
                                omb.signal[u_index](omb_ir, omb_ic) += signal*sin_2angle;


                                // update kernel map Q and U
                                if (run_kernel) {
                                    //kernels[q_index].push_back(T(omb_ir,omb_ic,kernel*cos_2angle));
                                    //kernels[u_index].push_back(T(omb_ir,omb_ic,kernel*sin_2angle));
                                    omb.kernel[q_index](omb_ir, omb_ic) += kernel*cos_2angle;
                                    omb.kernel[u_index](omb_ir, omb_ic) += kernel*sin_2angle;
                                }
                            }
                        }
                    }

                    if (run_polarization && !cmb.signal.empty() && run_omb) {
                        Eigen::Index cmb_ir = static_cast<Eigen::Index>(std::llround(cmb_irow(j)));
                        Eigen::Index cmb_ic = static_cast<Eigen::Index>(std::llround(cmb_icol(j)));

                        // make sure the data point is within the map
                        if ((cmb_ir >= 0) && (cmb_ir < cmb.n_rows) && (cmb_ic >= 0) && (cmb_ic < cmb.n_cols)) {
                            // populate signal map
                            signal = in.scans.data(j,i)*in.weights.data(i);
                            //cmb_signals[map_index].push_back(T(cmb_ir,cmb_ic,signal));
                            cmb.signal[map_index](cmb_ir, cmb_ic) += signal;

                            // populate weight map
                            //cmb_weights[map_index].push_back(T(cmb_ir,cmb_ic,in.weights.data(i)));
                            cmb.weight[map_index](cmb_ir, cmb_ic) += in.weights.data(i);

                            // populate kernel map
                            if (run_kernel) {
                                kernel = in.kernel.data(j,i)*in.weights.data(i);
                                //cmb_kernels[map_index].push_back(T(cmb_ir,cmb_ic,kernel));
                                cmb.kernel[map_index](cmb_ir, cmb_ic) += kernel;
                            }

                            // populate coverage map
                            if (run_coverage) {
                                //cmb_coverages[map_index].push_back(T(cmb_ir,cmb_ic,1./d_fsmp));
                                cmb.coverage[map_index](cmb_ir, cmb_ic) += 1./d_fsmp;
                            }

                            // calculate pointing matrix
                            allocate_pointing(cmb, in.weights.data(i), cos_2angle, sin_2angle, map_index, cmb_ir, cmb_ic);

                            // update signal map Q and U
                            //cmb_signals[q_index].push_back(T(cmb_ir,cmb_ic,signal*cos_2angle));
                            //cmb_signals[u_index].push_back(T(cmb_ir,cmb_ic,signal*sin_2angle));

                            cmb.signal[q_index](cmb_ir, cmb_ic) += signal*cos_2angle;
                            cmb.signal[u_index](cmb_ir, cmb_ic) += signal*sin_2angle;


                            // update kernel map Q and U
                            if (run_kernel) {
                                //cmb_kernels[q_index].push_back(T(cmb_ir,cmb_ic,kernel*cos_2angle));
                                //cmb_kernels[u_index].push_back(T(cmb_ir,cmb_ic,kernel*sin_2angle));
                                cmb.kernel[q_index](cmb_ir, cmb_ic) += kernel*cos_2angle;
                                cmb.kernel[u_index](cmb_ir, cmb_ic) += kernel*sin_2angle;
                            }
                        }
                    }

                    // check if noise maps requested
                    if (run_noise) {
                        // if coaddition is enabled
                        if (use_cmb) {
                            nmb_ir = static_cast<Eigen::Index>(std::llround(cmb_irow(j)));
                            nmb_ic = static_cast<Eigen::Index>(std::llround(cmb_icol(j)));
                        }
                        // else make noise maps for obs
                        else {
                            nmb_ir = static_cast<Eigen::Index>(std::llround(omb_irow(j)));
                            nmb_ic = static_cast<Eigen::Index>(std::llround(omb_icol(j)));
                        }

                        // coadd into current noise map
                        if ((nmb_ir >= 0) && (nmb_ir < nmb->n_rows) && (nmb_ic >= 0) && (nmb_ic < nmb->n_cols)) {
                            //if (run_polarization) {
                                //if (use_cmb) {
                                    // calculate pointing matrix for cmb
                                  //  allocate_pointing(cmb, in.weights.data(i), cos_2angle, sin_2angle, map_index, nmb_ir, nmb_ic);
                                //}
                            //}
                            // loop through noise maps
                            for (Eigen::Index nn=0; nn<nmb->n_noise; ++nn) {
                                // randomizing on dets
                                if (nmb->randomize_dets) {
                                    noise_v = in.noise.data(nn,i)*in.scans.data(j,i)*in.weights.data(i);
                                }
                                else {
                                    noise_v = in.noise.data(nn)*in.scans.data(j,i)*in.weights.data(i);
                                }
                                // add noise value to current noise map
                                nmb->noise[map_index](nmb_ir,nmb_ic,nn) += noise_v;

                                if (run_polarization) {
                                    // update noise map Q
                                    nmb->noise[q_index](nmb_ir,nmb_ic,nn) += noise_v*cos_2angle;
                                    // update noise map U
                                    nmb->noise[u_index](nmb_ir,nmb_ic,nn) += noise_v*sin_2angle;
                                }
                            }
                        }
                    }
                }
            }
        }
        return 0;
    });

    nmb = nullptr;
}
} // namespace mapmaking
