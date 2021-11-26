#pragma once
#include <string>
#include <vector>
#include <map>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include <tula/logging.h>

#include <citlali/core/utils/pointing.h>

using map_dims_t = std::tuple<int, int, Eigen::VectorXd, Eigen::VectorXd>;
using map_extent_t = std::vector<double>;
using map_coord_t = std::vector<Eigen::VectorXd>;
using map_count_t = std::size_t;


class CoaddedMapBuffer {
public:
    Eigen::Index nrows, ncols, nnoise;
    map_count_t map_count;
    double pixel_size;

    // for source fits in coadded maps
    Eigen::MatrixXd pfit;

    // tensor for noise map inclusion (nnoise,nobs,nmaps)
    Eigen::Tensor<double,3> noise_rand;

    // Physical coordinates for rows and cols (radians)
    Eigen::VectorXd rcphys, ccphys;

    // vectors for each map type
    std::vector<Eigen::MatrixXd> signal, weight, kernel, coverage;

    // noise maps (nrows, ncols, nnoise) of length nmaps
    std::vector<Eigen::Tensor<double,3>> noise;

    void setup_maps(std::vector<map_coord_t> map_coords, map_count_t _mc) {
        map_count = _mc;

        double min_row, max_row, min_col, max_col;

        // loop through physical coordinates and get min/max
        for (Eigen::Index i=0; i<map_coords.size(); i++) {
            auto rcp = map_coords.at(i).front();
            auto ccp = map_coords.at(i).back();

            auto rc_npts = rcp.size();
            auto cc_npts = ccp.size();

            // initialze to first obs map's min/max
            if (i == 0) {
                min_row = rcp(0);
                max_row = rcp(rc_npts-1);

                min_col = ccp(0);
                max_col = ccp(cc_npts-1);
            }            
            // see if current min/max is larger than previous
            // and replace if so
            else {
                if (rcp(0) < min_row) {
                    min_row = rcp(0);
                }

                if (rcp(rc_npts-1) > min_row) {
                    max_row = rcp(rc_npts-1);
                }

                if (ccp(0) < min_col) {
                    min_col = ccp(0);
                }

                if (ccp(cc_npts-1) > max_col) {
                    max_col = ccp(cc_npts-1);
                }
            }
        }

        // get number of rows
        Eigen::Index xminpix = ceil(abs(min_row/pixel_size));
        Eigen::Index xmaxpix = ceil(abs(max_row/pixel_size));
        xmaxpix = std::max(xminpix,xmaxpix);
        // always even
        nrows = 2.*xmaxpix + 4;

        // get number of cols
        Eigen::Index yminpix = ceil(abs(min_col/pixel_size));
        Eigen::Index ymaxpix = ceil(abs(max_col/pixel_size));
        ymaxpix = std::max(yminpix,ymaxpix);
        // always even
        ncols = 2.*ymaxpix + 4;

        // set coadded physical coordinate vectors
        rcphys = (Eigen::VectorXd::LinSpaced(nrows,0,nrows-1).array() -
                (nrows)/2.)*pixel_size;
        ccphys = (Eigen::VectorXd::LinSpaced(ncols,0,ncols-1).array() -
                (ncols)/2.)*pixel_size;

        SPDLOG_INFO("coadd map buffer nrows {} ncols {}", nrows, ncols);

        // resize the maps (nobs, [nrows, ncols])
        for (Eigen::Index i=0; i<map_count; i++) {
            signal.push_back(Eigen::MatrixXd::Zero(nrows, ncols));
            weight.push_back(Eigen::MatrixXd::Zero(nrows, ncols));
            kernel.push_back(Eigen::MatrixXd::Zero(nrows, ncols));
            coverage.push_back(Eigen::MatrixXd::Zero(nrows, ncols));
        }

        // resize noise maps (nobs, [nrows, ncols, nnoise])
        for (Eigen::Index i=0; i<map_count; i++) {
            noise.push_back(Eigen::Tensor<double,3>(nrows, ncols, nnoise));
            noise.at(i).setZero();
        }
    }

    template <class MB>
    void coadd(MB &mb, const double fsmp, const bool run_kernel) {
        // offset between CMB and MB physical coordinates
        int deltai = (mb.rcphys(0) - rcphys(0))/pixel_size;
        int deltaj = (mb.ccphys(0) - ccphys(0))/pixel_size;

        // coadd the MB into the CMB
        for (Eigen::Index mi=0; mi<map_count; mi++) {
            // weight += weight
            weight.at(mi).block(deltai, deltaj, mb.nrows, mb.ncols) =
                    weight.at(mi).block(deltai, deltaj, mb.nrows, mb.ncols).array() +
                    mb.weight.at(mi).array();

            // signal += signal*weight
            signal.at(mi).block(deltai, deltaj, mb.nrows, mb.ncols) =
                    signal.at(mi).block(deltai, deltaj, mb.nrows, mb.ncols).array() +
                    (mb.signal.at(mi).array()*mb.signal.at(mi).array()).array();

            if (run_kernel) {
                // kernel += kernel*weight
                kernel.at(mi).block(deltai, deltaj, mb.nrows, mb.ncols) =
                        kernel.at(mi).block(deltai, deltaj, mb.nrows, mb.ncols).array() +
                        (mb.weight.at(mi).array()*mb.kernel.at(mi).array()).array();
            }

            // signal += fsmp*weight
            coverage.at(mi).block(deltai, deltaj, mb.nrows, mb.ncols) +=
                    mb.weight.at(mi)*1./fsmp;
        }
    }

    void normalize_maps(const bool run_kernel) {
        /*for (Eigen::Index mc = 0; mc < map_count; mc++) {
            signal.at(mc) = (weight.at(mc).array() == 0).select(0, signal.at(mc).array() / weight.at(mc).array());
            if (run_kernel) {
                kernel.at(mc) = (weight.at(mc).array() == 0).select(0, kernel.at(mc).array() / weight.at(mc).array());
            }
        }*/

        // normalize signal and kernel
        for (Eigen::Index mc=0; mc<map_count; mc++) {
            for (Eigen::Index i=0; i<nrows; i++) {
                for (Eigen::Index j=0; j<ncols; j++) {
                    auto pixel_weight = weight.at(mc)(i,j);
                    if (pixel_weight != pixel_weight) {
                            SPDLOG_INFO("bad pixel weight {}", pixel_weight);
                    }
                    if (pixel_weight != 0. && pixel_weight == pixel_weight) {
                        signal.at(mc)(i,j) = (signal.at(mc)(i,j)) / pixel_weight;
                        if (run_kernel) {
                            kernel.at(mc)(i,j) = (kernel.at(mc)(i,j)) / pixel_weight;
                        }
                    }
                    else {
                        signal.at(mc)(i,j) = 0;
                        if (run_kernel) {
                            kernel.at(mc)(i,j) = 0;
                        }
                    }
                }
            }
        }

        // normalize noise maps
        for (Eigen::Index k=0;k<nnoise;k++) {
            for (Eigen::Index mc=0; mc<map_count; mc++) {
                for (Eigen::Index i=0; i<nrows; i++) {
                    for (Eigen::Index j=0; j<ncols; j++) {
                        auto pixel_weight = weight.at(mc)(i,j);
                        if (pixel_weight != pixel_weight) {
                                SPDLOG_INFO("bad pixel weight {}", pixel_weight);
                        }
                        if (pixel_weight != 0. && pixel_weight == pixel_weight) {
                            noise.at(mc)(i,j,k) = (noise.at(mc)(i,j,k)) / pixel_weight;
                        }
                        else {
                            noise.at(mc)(i,j,k) = 0;
                        }
                    }
                }
            }
        }
    }
};
