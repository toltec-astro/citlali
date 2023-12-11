#pragma once

#include <boost/random.hpp>
#include <boost/random/random_device.hpp>

#include <thread>

#include <citlali/core/timestream/timestream.h>

#include <citlali/core/mapmaking/map.h>
#include <citlali/core/utils/pointing.h>

using timestream::TCData;

// selects the type of TCData
using timestream::TCDataKind;

namespace mapmaking {

class NaiveMapmaker {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    // run polarization?
    bool run_polarization;

    // allocate pointing matrix for polarization reduction
    template <class map_buffer_t>
    void allocate_pointing(map_buffer_t &, double, double, double, Eigen::Index, int, int);

    // populate maps with a time chunk (signal, kernel, coverage, and noise)
    template<class map_buffer_t, typename Derived, typename apt_t>
    void populate_maps_naive(TCData<TCDataKind::PTC, Eigen::MatrixXd> &, map_buffer_t &, map_buffer_t &,
                             Eigen::DenseBase<Derived> &, Eigen::DenseBase<Derived> &, std::string &,
                             apt_t &, double, bool, bool);
};

template <class map_buffer_t>
void NaiveMapmaker::allocate_pointing(map_buffer_t &mb, double weight, double cos_2angle, double sin_2angle,
                                      Eigen::Index map_index, int ir, int ic) {
    // update pointing matrix
    mb.pointing[map_index](ir,ic,0) += weight;
    mb.pointing[map_index](ir,ic,1) += weight*cos_2angle;
    mb.pointing[map_index](ir,ic,2) += weight*sin_2angle;
    mb.pointing[map_index](ir,ic,3) = mb.pointing[map_index](ir,ic,1);//weight*cos(2*angle);
    mb.pointing[map_index](ir,ic,4) += weight*pow(cos_2angle,2.);
    mb.pointing[map_index](ir,ic,5) += weight*cos_2angle*sin_2angle;
    mb.pointing[map_index](ir,ic,6) = mb.pointing[map_index](ir,ic,2);//weight*sin(2*angle);
    mb.pointing[map_index](ir,ic,7) = mb.pointing[map_index](ir,ic,5);//weight*cos(2*angle)*sin(2*angle);
    mb.pointing[map_index](ir,ic,8) += weight*pow(sin_2angle,2.);
}

template<class map_buffer_t, typename Derived, typename apt_t>
void NaiveMapmaker::populate_maps_naive(TCData<TCDataKind::PTC, Eigen::MatrixXd> &in, map_buffer_t &omb,
                                        map_buffer_t &cmb, Eigen::DenseBase<Derived> &map_indices,
                                        Eigen::DenseBase<Derived> &det_indices, std::string &pixel_axes,
                                        apt_t &apt, double d_fsmp, bool run_omb, bool run_noise) {

    const bool use_cmb = !cmb.noise.empty();
    const bool use_omb = !omb.noise.empty();
    const bool run_kernel = !omb.kernel.empty();
    const bool run_coverage = !omb.coverage.empty();

    // dimensions of data
    Eigen::Index n_pts = in.scans.data.rows();
    Eigen::Index n_dets = in.scans.data.cols();

    // step to skip to reach next stokes param
    int step = omb.pointing.size();

    // pointer to map buffer with noise maps
    ObsMapBuffer* nmb = NULL;

    // matrix to hold random noise value
    Eigen::Matrix<int,Eigen::Dynamic, Eigen::Dynamic> noise;

    if (run_noise) {
        // declare random number generator
        thread_local boost::random::mt19937 eng;

        // boost random number generator (0,1)
        boost::random::uniform_int_distribution<> rands{0,1};

        // set pointer to cmb or omb for noise maps
        nmb = use_cmb ? &cmb : (use_omb ? &omb : nullptr);
        if (nmb) {
            if (nmb->randomize_dets) {
                noise = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>::Zero(nmb->n_noise, n_dets)
                            .unaryExpr([&](int dummy){ return 2 * rands(eng) - 1; });
            } else {
                noise = Eigen::Matrix<int, Eigen::Dynamic, 1>::Zero(nmb->n_noise)
                            .unaryExpr([&](int dummy){ return 2 * rands(eng) - 1; });
            }
        }
    }

    // signal and kernel map values
    double signal, kernel;

    // noise map value
    double noise_v;

    // noise map indices
    Eigen::Index nmb_ir, nmb_ic;

    // cosine and sine of angles
    double cos_2angle, sin_2angle;

    for (Eigen::Index i=0; i<n_dets; ++i) {
        // skip completely flagged detectors
        if ((in.flags.data.col(i).array()==0).any()) {
            // which map to assign detector to
            Eigen::Index map_index = map_indices(i);

            // indices for Q and U maps
            int q_index = map_index + step;
            int u_index = map_index + 2 * step;

            // get detector positions from apt table
            auto det_index = det_indices(i);
            // get detector pointing
            auto [lat, lon] = engine_utils::calc_det_pointing(in.tel_data.data, apt["x_t"](det_index), apt["y_t"](det_index),
                                                              pixel_axes, in.pointing_offsets_arcsec.data, omb.map_grouping);

            // get map buffer row and col indices for lat and lon vectors
            Eigen::VectorXd omb_irow = lat.array()/omb.pixel_size_rad + (omb.n_rows)/2.;
            Eigen::VectorXd omb_icol = lon.array()/omb.pixel_size_rad + (omb.n_cols)/2.;

            Eigen::VectorXd cmb_irow, cmb_icol;
            if (use_cmb) {
                // get coadded map buffer row and col indices for lat and lon vectors
                cmb_irow = lat.array()/cmb.pixel_size_rad + (cmb.n_rows)/2.;
                cmb_icol = lon.array()/cmb.pixel_size_rad + (cmb.n_cols)/2.;
            }

            // loop through the samples
            for (Eigen::Index j=0; j<n_pts; ++j) {
                // check if sample is flagged, ignore if so
                if (!in.flags.data(j,i)) {
                    Eigen::Index omb_ir = omb_irow(j);
                    Eigen::Index omb_ic = omb_icol(j);

                    if (run_polarization) {
                        cos_2angle = cos(2.*in.angle.data(j,i));
                        sin_2angle = sin(2.*in.angle.data(j,i));
                    }

                    if (run_omb) {
                        // make sure the data point is within the map
                        if ((omb_ir >= 0) && (omb_ir < omb.n_rows) && (omb_ic >= 0) && (omb_ic < omb.n_cols)) {
                            // populate signal map
                            signal = in.scans.data(j,i)*in.weights.data(i);
                            omb.signal[map_index](omb_ir,omb_ic) += signal;

                            // populate weight map
                            omb.weight[map_index](omb_ir,omb_ic) += in.weights.data(i);

                            // populate kernel map
                            if (run_kernel) {
                                kernel = in.kernel.data(j,i)*in.weights.data(i);
                                omb.kernel[map_index](omb_ir,omb_ic) += kernel;
                            }

                            // populate coverage map
                            if (run_coverage) {
                                omb.coverage[map_index](omb_ir,omb_ic) += 1./d_fsmp;
                            }

                            if (run_polarization) {
                                // calculate pointing matrix
                                allocate_pointing(omb, in.weights.data(i), cos_2angle, sin_2angle, map_index, omb_ir,omb_ic);

                                // update signal map Q and U
                                omb.signal[q_index](omb_ir,omb_ic) += signal*cos_2angle;
                                omb.signal[u_index](omb_ir,omb_ic) += signal*sin_2angle;

                                // update kernel map Q and U
                                if (run_kernel) {
                                    omb.kernel[q_index](omb_ir,omb_ic) += kernel*cos_2angle;
                                    omb.kernel[u_index](omb_ir,omb_ic) += kernel*sin_2angle;
                                }
                            }
                        }
                    }

                    // check if noise maps requested
                    if (run_noise) {
                        // if coaddition is enabled
                        if (use_cmb) {
                            nmb_ir = cmb_irow(j);
                            nmb_ic = cmb_icol(j);
                        }
                        // else make noise maps for obs
                        else if (use_omb) {
                            nmb_ir = omb_irow(j);
                            nmb_ic = omb_icol(j);
                        }

                        // coadd into current noise map
                        if ((nmb_ir >= 0) && (nmb_ir < nmb->n_rows) && (nmb_ic >= 0) && (nmb_ic < nmb->n_cols)) {
                            if (run_polarization) {
                                if (use_cmb) {
                                    // calculate pointing matrix for cmb
                                    allocate_pointing(cmb, in.weights.data(i), cos_2angle, sin_2angle, map_index, nmb_ir, nmb_ic);
                                }
                            }
                            // loop through noise maps
                            for (Eigen::Index nn=0; nn<nmb->n_noise; ++nn) {
                                // randomizing on dets
                                if (nmb->randomize_dets) {
                                    noise_v = noise(nn,i)*in.scans.data(j,i)*in.weights.data(i);
                                }
                                else {
                                    noise_v = noise(nn)*in.scans.data(j,i)*in.weights.data(i);
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
    }
}
} // namespace mapmaking
