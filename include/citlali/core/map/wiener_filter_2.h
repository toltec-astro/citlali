#pragma once

#include <Eigen/Core>
#include <tula/algorithm/mlinterp/mlinterp.hpp>

#include <citlali/core/utils/constants.h>
#include <citlali/core/utils/utils.h>

class WienerFilter {
public:
    bool run_gaussian_template, run_highpass_only, run_lowpass_only;
    bool normalize_error, uniform_weight;

    bool run_kernel;
    int nloops;

    int nx, ny;
    double diffx, diffy;

    Eigen::MatrixXd rr, vvq, denom, nume;
    Eigen::MatrixXd mflt;
    Eigen::MatrixXd tplate;

    template<class CMB>
    void make_gaussian_template(CMB &cmb, const double);
    void make_symmetric_template();

    template<class CMB>
    void make_template(CMB &cmb, const double gaussian_template_fwhm_rad) {

        // make sure new wiener filtered maps are even dimensioned
        nx = 2*(cmb.nrows/2);
        ny = 2*(cmb.ncols/2);

        // x and y spacing should be equal
        diffx = abs(cmb.rcphys(1) - cmb.rcphys(0));
        diffy = abs(cmb.ccphys(1) - cmb.ccphys(0));

        if (run_highpass_only) {
            tplate.setZero(nx,ny);
            tplate(0,0) = 1;
        }

        else if (run_gaussian_template) {
            make_gaussian_template(cmb, gaussian_template_fwhm_rad);
        }

        else {
            make_symmetric_template();
        }
    }

    template<class CMB>
    void calc_rr(CMB &cmb, const int map_num) {
        if (uniform_weight) {
            rr = Eigen::MatrixXd::Ones(nx, ny);
        }
        else {
            rr = sqrt(cmb.weight.at(map_num).array());
        }
    }

    void calc_vvq();
    void calc_numerator();
    void calc_denominator();

    template<class CMB>
    void run_filter(CMB &cmb, const int map_num) {
        calc_rr(cmb, map_num);
        calc_vvq();
        calc_denominator();
        calc_numerator();
    }

    template<class CMB>
    void filter_coaddition(CMB &cmb, const int map_num) {
        if (run_kernel) {
            mflt = cmb.kernel.at(map_num);
            uniform_weight = true;
            run_filter(cmb, map_num);
            cmb.kernel.at(map_num) = (denom.array() == 0).select(0, nume.array() / denom.array());
        }

        mflt = cmb.signal.at(map_num);
        uniform_weight = false;
        run_filter(cmb, map_num);
        cmb.signal.at(map_num) = (denom.array() == 0).select(0, nume.array() / denom.array());
        cmb.weight.at(map_num) = denom;
    }

    void filter_noise_maps();
};

template<class CMB>
void WienerFilter::make_gaussian_template(CMB &cmb, const double gaussian_template_fwhm_rad) {
    Eigen::VectorXd xgcut = cmb.rcphys.head(nx);
    Eigen::VectorXd ygcut = cmb.ccphys.head(ny);

    Eigen::MatrixXd tem = Eigen::MatrixXd::Zero(nx, ny);

    Eigen::MatrixXd dist(nx, ny);
    dist = (xgcut.replicate(1, ny).array().pow(2.0).matrix()
            + ygcut.replicate(nx, 1).array().pow(2.0).matrix())
               .array()
               .sqrt();

    Eigen::Index xcind = nx / 2;
    Eigen::Index ycind = ny / 2;
    double mindist = 99.;

    mindist = dist.minCoeff(&xcind, &ycind);

    double sig = gaussian_template_fwhm_rad / STD_TO_FWHM;
    tplate = exp(-0.5 * pow(dist.array() / sig, 2.));

    tplate = engine_utils::shift_matrix(tplate, -xcind, -ycind);
}

void WienerFilter::make_symmetric_template() {

}

void WienerFilter::calc_vvq() {
    // size of psd and psd freq vectors
    int npsd;
    // psd and psd freq vectors
    Eigen::VectorXd qf, hp;

    // modify the psd array to take out lowpassing+highpassing
    Eigen::Index maxhpind;
    double qfbreak = 0.;
    double hpbreak = 0.;

    // get max value and index of hp
    auto maxhp = hp.maxCoeff(&maxhpind);

    if (maxhp < -1) {
        maxhp = -1;
        maxhpind = 0;
    }

    // find the frequency where hp falls to 1e-4 its max value
    for (int i = 0; i < npsd; i++)
        if (hp(i) / maxhp < 1.e-4) {
            qfbreak = qf(i);
            break;
        }
    // get the total number of points where the frequency is less than 0.8*qfbreak
    auto count = (qf.array() <= 0.8 * qfbreak).count();

    // set the region of hp corresponding to the frequency range above 0.8*qfbreak to
    // the value of hp at 0.8*qfbreak (i.e. flatten it).  We can do this as a vector
    // since the frequencies are monotonically increasing.
    if (count > 0) {
        if (qfbreak > 0) {
            Eigen::Index start = 0.8 * qfbreak;
            Eigen::Index size = hp.size() - 0.8 * qfbreak;
            hp.segment(start, size).setConstant(hp(start));
        }
    }

    // do the same for the region below the maximum hp
    hp.head(maxhpind).setConstant(maxhp);

    // set up Q-space
    double xsize = nx * diffx;
    double ysize = ny * diffy;
    double diffqx = 1. / xsize;
    double diffqy = 1. / ysize;

    Eigen::VectorXd qx(nx), qy(ny);
    Eigen::MatrixXd qmap(nx, ny);

    /*shift qx */
    qx = Eigen::VectorXd::LinSpaced(nx, -(nx - 1) / 2, nx - (nx - 1) / 2) * diffqx;
    /*shift qy */
    qy = Eigen::VectorXd::LinSpaced(ny, -(ny - 1) / 2, ny - (ny - 1) / 2) * diffqy;

    // make qmap by replicating qy and qx ncols and nrows times respectively.  Faster than a for loop for expected
    // map dimensions
    qmap = (qx.replicate(1, ny).array().pow(2.0).matrix() + qy.replicate(nx, 1).array().pow(2.0).matrix()).array().sqrt();

    Eigen::MatrixXd psdq(nx, ny);

    if (run_lowpass_only) {
        psdq.setOnes();
    }

    else{
        Eigen::Matrix<Eigen::Index, 1, 1> nhp;
        nhp << hp.size();

        // interpolate onto psdq
        Eigen::Index interp_pts = 1;
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                if (qmap(i, j) <= qf(qf.size() - 1) && qmap(i, j) >= qf(0)) {
                    mlinterp::interp(nhp.data(),
                                     interp_pts,
                                     hp.data(),
                                     psdq.data() + nx * j + i,
                                     qf.data(),
                                     qmap.data() + nx * j + i);
                }

                else if (qmap(i, j) > qf(qf.size() - 1)) {
                    psdq(i, j) = hp[hp.size() - 1];
                } else if (qmap(i, j) < qf(0)) {
                    psdq(i, j) = hp(0);
                }
            }
        }
    }

    // find the minimum value of hp
    auto lowval = hp.minCoeff();

    // set all the points in psdq smaller than lowval to lowval
    psdq = (psdq.array() < lowval).select(lowval, psdq);

    // normalize the power spectrum psdq and place into vvq
    vvq = psdq/psdq.sum();
}

void WienerFilter::calc_numerator() {
    // normalization for FFT
    double fftnorm = 1. / nx / ny;

    Eigen::VectorXcd in(nx * ny);
    Eigen::VectorXcd out(nx * ny);

    in.real() = rr * mflt;
    in.imag().setZero();

    out = engine_utils::fft2w<engine_utils::forward>(in, nx, ny);
    out = out * fftnorm;

    in.real() = out.real().array() / vvq.array();
    in.imag() = out.imag().array() / vvq.array();

    out = engine_utils::fft2w<engine_utils::backward>(in, nx, ny);

    in.real() = out.real().array() * rr.array();
    in.imag().setZero();

    out = engine_utils::fft2w<engine_utils::forward>(in, nx, ny);
    out = out * fftnorm;

    // copy of out
    Eigen::VectorXcd qqq = out;

    in.real() = tplate;
    in.imag().setZero();

    out = engine_utils::fft2w<engine_utils::forward>(in, nx, ny);
    out = out * fftnorm;

    in.real() = out.real() * qqq.real() + out.imag() * qqq.imag();
    in.imag() = -out.imag() * qqq.real() + out.real() * qqq.imag();

    out = engine_utils::fft2w<engine_utils::backward>(in, nx, ny);

    // populate numerator
    nume = out.real();
}

void WienerFilter::calc_denominator() {
    double fftnorm = 1. / nx / ny;
    Eigen::VectorXcd in(nx * ny);
    Eigen::VectorXcd out(nx * ny);

    if (uniform_weight) {
        in.real() = tplate;
        in.imag().setZero();

        out = engine_utils::fft2w<engine_utils::forward>(in, nx, ny);
        out = out * fftnorm;

        auto d = ((out.real().array() * out.real().array() + out.imag().array() * out.imag().array()) / vvq.array()).sum();
        denom.setConstant(d);
    }

    else {
        denom.setZero();

        in.real() = pow(vvq.array(),-1.);
        in.imag().setZero();

        out = engine_utils::fft2w<engine_utils::backward>(in, nx, ny);

        Eigen::VectorXd zz2d(nx * ny);
        zz2d = abs(out.real().array());

        Eigen::VectorXd ss_ord;
        ss_ord = zz2d;
        auto sorted = engine_utils::sorter(ss_ord);

        zz2d = out.real();

        // number of iterations for convergence
        nloops = nx * ny / 100;

        // flag for convergence
        bool done = false;

        for (int k = 0; k < nx; k++) {
            for (int l = 0; l < ny; l++) {
                if (!done) {
                    Eigen::VectorXcd in2(nx, ny), out2(nx, ny);

                    /*May need to flip directions due to order of Matrix storage order*/
                    int kk = ny * k + l;
                    if (kk >= nloops) {
                        continue;
                    }

                    auto shifti = std::get<1>(sorted[nx * ny - kk - 1]);

                    double x_shift_n = shifti / ny;
                    double y_shift_n = shifti % ny;

                    Eigen::MatrixXd in_prod = tplate.array() * engine_utils::shift_matrix(tplate, -x_shift_n, -y_shift_n).array();

                    in2.real() = Eigen::Map<Eigen::VectorXd>(
                        (in_prod).data(),
                        nx * ny);
                    in2.imag().setZero();

                    out2 = engine_utils::fft2w<engine_utils::forward>(in2, nx, ny);
                    out2 = out2 * fftnorm;

                    Eigen::VectorXcd ffdq(nx * ny);

                    ffdq.real() = out2.real();
                    ffdq.imag() = out2.imag();

                    in_prod = rr.array() * engine_utils::shift_matrix(tplate, -x_shift_n, -y_shift_n).array();

                    in2.real() = Eigen::Map<Eigen::VectorXd>(in_prod.data(), nx * ny);
                    in2.imag().setZero();

                    out2 = engine_utils::fft2w<engine_utils::forward>(in2, nx, ny);
                    out2 = out2 * fftnorm;

                    in2.real() = ffdq.real() * out2.real() + ffdq.imag() * out2.imag();
                    in2.imag() = -ffdq.imag() * out2.real() + ffdq.real() * out2.imag();

                    out2 = engine_utils::fft2w<engine_utils::backward>(in2, nx, ny);

                    Eigen::MatrixXd updater = zz2d(shifti) * out2.real() * fftnorm;

                    denom = denom + updater;

                    if ((kk % 100) == 1) {
                        double max_ratio = -1;
                        double maxdenom = -999.;

                        maxdenom = denom.maxCoeff();
                        for (int i = 0; i < nx; i++) {
                            for (int j = 0; j < ny; j++) {
                                if (denom(i, j) > 0.01 * maxdenom) {
                                    if (abs(updater(i, j) / denom(i, j)) > max_ratio)
                                        max_ratio = abs(updater(i, j) / denom(i, j));
                                }
                            }
                        }

                        if (((kk >= 500) && (max_ratio < 0.0002)) || max_ratio < 1e-10) {
                            SPDLOG_INFO("Seems like we're done.  max_ratio={}", max_ratio);
                            SPDLOG_INFO("The denom calcualtion required {} iterations", kk);
                            done = true;
                        }
                        else {
                            SPDLOG_INFO("Completed iteration {} of {}. max_ratio={}",
                                        kk, nloops, max_ratio);
                        }
                    }
                }
            }
        }

        double denom_imit = 1.e-4;
        SPDLOG_INFO("Zeroing out any small values (< {}) in denominator", denom_imit);
        denom = (denom.array() < denom_imit).select(0, denom);
    }
}
