#pragma once

#include <Eigen/Core>
#include <unsupported/Eigen/FFT>
#include <tula/logging.h>

class WienerFilter {
public:
    Eigen::MatrixXd denom;
    Eigen::MatrixXd nume;
    Eigen::MatrixXd rr;
    Eigen::MatrixXd vvq;
    Eigen::MatrixXd tplate;

    bool uniform_weight;

    int nrows;
    int ncols;
    int nx;
    int ny;
    double pixel_size;
    double diffx;
    double diffy;

    template <class CMB>
    WienerFilter(CMB &);

    template <class CMB>
    void setup_template(CMB &);

    template <class CMB>
    void setup_gaussian_template(CMB &);

    template <class CMB>
    void filter_coaddition(CMB &);

    //void filter_noise_maps(NoiseRealizations *nr);

    template <class CMB>
    void calc_rr(CMB &);

    void calc_vvq();

    template <typename Derived>
    void calc_numerator(Eigen::DenseBase<Derived> &);

    void calc_denominator();

    template <class CMB>
    void simple_wiener_filter_2d(CMB &);
};

template <class CMB>
WienerFilter::WienerFilter(CMB &cmb) {
    pixel_size = cmb.pixel_size;
    nrows = cmb.nrows;
    ncols = cmb.ncols;

    nx = 2*(nrows/2);
    ny = 2*(ncols/2);

    int zcount = (cmb.weight.array()==0).count();

    Eigen::VectorXi whz;

    if (zcount > 0) {
        whz.resize(zcount);
        zcount = 0;
        for (Eigen::Index i=0; i<nrows; i++) {
            for (Eigen::Index j=0; j<ncols; j++) {
                if (cmb.weight(i,j) == 0.) {
                    whz(zcount) = ncols*i+j;
                    zcount++;
                }
            }
        }
    }


    diffx = abs(cmb.rcphys(1) - cmb.rcphys(0));
    diffy = abs(cmb.ccphys(1) - cmb.ccphys(0));

    bool get_highpass_only = 0;
    bool get_gaussian_template = 0;

    if (get_highpass_only) {
        tplate.setZero(nx, ny);
        tplate(0, 0) = 1.;
    }
    else if (get_gaussian_template) {
        setup_gaussian_template(cmb);
    }
    else {
        setup_template(cmb);
    }
}

template <class CMB>
void WienerFilter::filter_coaddition(CMB &cmb){
        // here's storage for the filtering
        Eigen::MatrixXd mflt(nx, ny);

        // do the kernel first since it's simpler to calculate
        // the kernel calculation requires uniform weighting
        uniform_weight = 1;

        calc_rr(cmb);
        calc_vvq();

        calc_denominator();

        mflt = cmb.kernel;

        calc_numerator(mflt);
        for (Eigen::Index i=0; i<nx; i++) {
            for (Eigen::Index j=0; j<ny; j++) {
                if (denom(i,j) != 0.0) {
                    cmb.filtered_kernel(i,j)=nume(i,j)/denom(i,j);
                }
                 else {
                    cmb.filtered_kernel(i,j)= 0.0;
                    SPDLOG_INFO("nan at ({},{})",i,j);
                }
            }
        }

        // do the more complex precomputing for the signal map
        uniform_weight=0;
        calc_rr(cmb);
        calc_vvq();
        calc_denominator();

        // here is the signal map to be filtered
        mflt = cmb.signal;

        //calculate the associated numerator
        calc_numerator(mflt);

        // replace the original images with the filtered images
        // note that the filtered weight map = denom
        for (Eigen::Index i=0; i<nx; i++) {
            for(Eigen::Index j=0; j<ny; j++) {
                if (denom(i,j) !=0.0) {
                    cmb.filtered_signal(i,j) = nume(i,j)/denom(i,j);
                }
                else {
                    cmb.filtered_signal(i,j) = 0.0;
                }
                cmb.filtered_weight = denom(i,j);
            }
        }
}

//void WienerFilter::filter_noise_maps() {}

template <typename CMB>
void WienerFilter::calc_rr(CMB &cmb) {
    if (uniform_weight){
        rr.setOnes(nx,ny);
    }
    else {
        rr = sqrt(cmb.weight);
    }
}

