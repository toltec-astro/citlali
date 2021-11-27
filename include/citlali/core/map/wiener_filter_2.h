#pragma once

#include <Eigen/Core>
#include <tula/algorithm/mlinterp/mlinterp.hpp>

class WienerFilter {
public:
    bool highpass_only, lowpass_only, normalize_error;
    bool run_kernel;
    double gaussian_template_fwhm_arcsec;

    bool uniform_weight;
    void filter_coaddition() {
        Eigen::MatrixXd mflt;

        if (run_kernel) {
            mflt = kernel;
            uniform_weight = 1;
            run_filter();
            filtered_kernel = (denom == 0).select(0, nume.array() / denom.array());
        }

        mflt = signal;
        uniform_weight = 0;
        run_filter();
        filtered_signal = (denom == 0).select(0, nume.array() / denom.array());
        filtered_weight = denom;
    }

    void filter_noise_maps();

private:

    enum FilterType {
        Kernel = 0,
        Signal = 1
    };

    /* Temporary */
    Eigen::MatrixXd rr, vvq, denom, nume;
    Eigen::MatrixXd signal, weight, kernel;
    Eigen::MatrixXd filtered_signal, filtered_weight, filtered_kernel;
    Eigen::VectorXd rcphys, ccphys;
    Eigen::MatrixXd tplate;

    int nx, ny;
    /* Temporary */

    void calc_rr() {
        if (uniform_weight) {
            rr = Eigen::MatrixXd::Ones(nx, ny);
        } else {
            rr = weight.sqrt();
        }
    }

    void prepare_gaussian_template();
    void prepare_template();
    void calc_vvq();
    void calc_numerator();
    void calc_denominator();

    void run_filter() {
        calc_rr();
        calc_vvq();
        calc_denominator();
        calc_numerator();
    }
};

void WienerFilter::prepare_template() {
    /*Eigen::VectorXd xgcut = rcphys.head(nx);
    Eigen::VectorXd ygcut = ccphys.head(ny);
    Eigen::MatrixXd tem = kernel;
    Eigen::VectorXd pp(6);*/

    /* Fit to Gaussian */

    /*double beamRatio = 30.0 / 8.0;
    beamRatio *= 8.0;
    // Shift the kernel appropriately
    shift(tem, -round(pp(4) / diffx), -round(pp(5) / diffy));
    Eigen::MatrixXd dist(nx, ny);
    dist = (xgcut.replicate(1, ny).array().pow(2.0).matrix()
            + ygcut.replicate(nx, 1).array().pow(2.0).matrix())
               .array()
               .sqrt();
    Eigen::Index xcind = nx / 2;
    Eigen::Index ycind = ny / 2;
    double mindist = 99.;
    mindist = dist.minCoeff(&xcind, &ycind);
    // Create new bins based on diffx
    int nbins = xgcut(nx - 1) / diffx;
    Eigen::VectorXd binlow = Eigen::VectorXd::LinSpaced(nbins, 0, nbins - 1);
    binlow = binlow * diffx;
    Eigen::VectorXd kone = Eigen::VectorXd::Zero(nbins - 1);
    Eigen::VectorXd done = Eigen::VectorXd::Zero(nbins - 1);
*/

}

void WienerFilter::prepare_gaussian_template() {
    Eigen::VectorXd xgcut = rcphys.head(nx);
    Eigen::VectorXd ygcut = ccphys.head(ny);

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

    tplate.resize(nx, ny);

    double fwhm = gaussian_template_fwhm_arcsec;
    double sig = fwhm / 2.3548;
    tplate = exp(-0.5 * pow(dist / sig, 2.));

    shift(tplate, -xcind, -ycind);
}


void WienerFilter::calc_vvq() {
    // size of psd and psd freq vectors
    int npsd;
    // psd and psd freq vectors
    Eigen::VectorXd qf, hp;

    // modify the PSD array to take out lowpassing+highpassing
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
            hp.segment(0.8 * qfbreak, hp.size() - 0.8 * qfbreak) = hp(0.8 * qfbreak);
        }
    }

    // do the same for the region below the maximum hp
    hp.head(maxhpind) = maxhp;

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

    if (lowpass_only) {
        psdq.setOnes();
    }

    else{
        Eigen::Matrix<Eigen::Index, 1, 1> nhp;
        nhp << hp.size();

        // interpolate onto psdq
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                if (qmap(i, j) <= qf[qf.size() - 1] && qmap(i, j) >= qf[0]) {
                    mlinterp::interp(nhp.data(),
                                     interp_pts,
                                     hp.data(),
                                     psdq.data() + nx * j + i,
                                     qf.data(),
                                     qmap.data() + nx * j + i);
                }

                else if (qmap(i, j) > qf[qf.size() - 1]) {
                    psdq(i, j) = hp[hp.size() - 1];
                } else if (qmap(i, j) < qf[0]) {
                    psdq(i, j) = hp[0];
                }
            }
        }
    }

    // find the minimum value of hp
    auto lowval = hp.minCoeff();

    // set all the points in psdq smaller than lowval to lowval
    psdq = (psdq < lowval).select(lowval, psdq);

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

    out = mapmaking::fft2w<forward>(in, nx, ny);
    out = out * fftnorm;

    in.real() = out.real() / vvq;
    in.imag() = out.imag() / vvq;

    out = mapmaking::fft2w<backward>(in, nx, ny);

    in.real() = out.real() * rr;
    in.imag().setZero();

    out = mapmaking::fft2w<forward>(in, nx, ny);
    out = out * fftnorm;

    // Copy of out
    Eigen::VectorXcd qqq = out;

    in.real() = tplate;
    in.imag().setZero();

    out = mapmaking::fft2w<forward>(in, nx, ny);
    out = out * fftnorm;

    in.real() = out.real() * qqq.real() + out.imag() * qqq.imag();
    in.imag() = -out.imag() * qqq.real() + out.real() * qqq.imag();

    out = mapmaking::fft2w<backward>(in, nx, ny);

    // populate numerator
    nume = out.real();
}

void WienerFilter::calc_denominator() {
    double fftnorm = 1. / nx / ny;
    Eigen::VectorXcd in(nx * ny);
    Eigen::VectorXcd out(nx * ny);

    if (uniform_weight) {
        in.real() = tplate();
        in.imag().setZero();

        out = mapmaking::fft2w<forward>(in, nx, ny);
        out = out * fftnorm;

        auto d = ((out.real() * out.real() + out.imag() * out.imag()) / vvq).sum();
        denom.setConstant(d);
    }

    else {
        denom.setZero();

        in.real() = vvq.pow(-1.);
        in.imag().setZero();

        out = mapmaking::fft2w<backward>(in, nx, ny);

        Eigen::VectorXd zz2d(nx * ny);
        zz2d = abs(out.real());

        Eigen::VectorXd ss_ord;
        ss_ord = zz2d;
        auto sorted = eigen_utils::sorter(ss_ord);

        zz2d = out.real();

        // Number of iterations for convergence
        nloops = nx * ny / 100;

        // Flag for convergence
        bool done = false;

        for (int k = 0; k < nx; k++) {
            for (int l = 0; l < ny; l++) {
                if (!done) {
                    Eigen::VectorXcd in2(nx, ny), out2(nx, ny);

                    /*May need to flip directions due to order of Matrix storage order*/
                    int kk = ny * k + l;
                    if (kk >= nloop) {
                        continue;
                    }

                    auto shifti = std::get<1>(sorted[nx * ny - kk - 1]);

                    double xshiftN = shifti / ny;
                    double yshiftN = shifti % ny;

                    in2.real() = Eigen::Map<Eigen::VectorXcd>(
                        (tplate.array() * shift(tplate, -xshiftN, -yshiftN).array()).data(),
                        nx * ny);
                    in2.imag().setZero();

                    out2 = mapmaking::fft2w<forward>(in2, nx, ny);
                    out2 = out2 * fftnorm;

                    Eigen::VectorXcd ffdq(nx * ny);

                    ffdq.real() = out2.real();
                    ffdq.imag() = out2.imag();

                    in2.real() = Eigen::Map<Eigen::VectorXcd>(
                        (rr.array() * shift(tplate, -xshiftN, -yshiftN).array()).data(), nx * ny);
                    in2.imag().setZero();

                    out2 = mapmaking::fft2w<forward>(in2, nx, ny);
                    out2 = out2 * fftnorm;

                    in2.real() = ffdq.real() * out2.real() + ffdq.imag() * out2.imag();
                    in2.imag() = -ffdq.imag() * out2.real() + ffdq.real() * out2.imag();

                    out2 = mapmaking::fft2w<backward>(in2, nx, ny);

                    Eigen::MatrixXd updater = zz2d[shifti] * out2.real() * fftnorm;

                    denom = denom + updater;

                    if ((kk % 100) == 1) {
                        double maxRatio = -1;
                        double maxdenom = -999.;

                        maxdenom = denom.maxCoeff();
                        for (int i = 0; i < nx; i++) {
                            for (int j = 0; j < ny; j++) {
                                if (denom(i, j) > 0.01 * maxdenom) {
                                    if (abs(updater(i, j) / denom(i, j)) > maxRatio)
                                        maxRatio = abs(updater(i, j) / denom(i, j));
                                }
                            }
                        }

                        if (((kk >= 500) && (maxRatio < 0.0002)) || maxRatio < 1e-10) {
                            SPDLOG_INFO("Seems like we're done.  maxRatio={}", maxRatio);
                            SPDLOG_INFO("The denom calcualtion required {} iterations", kk);
                            done = true;
                        } else {
                            SPDLOG_INFO("Completed iteration {} of {}. maxRatio={}",
                                        kk,
                                        nloop,
                                        maxRatio);
                        }
                    }
                }
            }
        }

        double denomLimit = 1.e-4;
        SPDLOG_INFO("Zeroing out any small values (< {}) in denominator", denomLimit);
        denom = (denom.array() < denomLimit).select(0, denom);
    }
}
