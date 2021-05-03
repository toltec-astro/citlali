#pragma once

namespace mapmaking {

class WienerFilter
{
public:
    bool uniformWeight;

    void filterCoaddition()
    {
        Eigen::MatrixXd mflt;

        if (config.get_typed<int>("proc.rtc.kernel")) {
            mflt = kernel;
            uniformWeight = 1;
            runFilter();
            filteredKernel = (Denom == 0).select(0, Nume.array() / Denom.array());
        }

        mflt = signal;
        UniformWeight = 0;
        runFilter();
        filteredSignal = (Denom == 0).select(0, Nume.array() / Denom.array());
        filteredWeight = Denom;

    }

    void filterNoiseMaps();

private:

    enum FilterType {
        Kernel = 0,
        Signal = 1
    };

    /* Temporary */
    Eigen::MatrixXd Rr, Vvq, Denom, Nume;
    Eigen::MatrixXd weight;
    int nx, ny;
    /* Temporary */

    void calcRr()
    {
        if (config->get_typed<int>("uniformWeight")) {
            Rr = Eigen::MatrixXd::Ones(nx, ny);
        } else {
            Rr = weight.sqrt();
        }
    }

    void prepareGaussianTemplate();
    void prepareTemplate();
    void calcVvq();
    void calcNumerator();
    void calcDenominator();

    void runFilter(){
        calcRr();
        calcVvq();
        calcDenominator();
        calcNumerator();
    }
};

void WienerFilter::prepareTemplate()
{
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


void WienerFilter::prepareGaussianTemplate()
{
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

    double fwhm = config->get_typed<double>("GaussianTemplateFWHM");
    double sig = fwhm / 2.3548;
    tplate = exp(-0.5 * pow(dist / sig, 2.));

    shift(tplate, -xcind, -ycind);
}

void WienerFilter::calcVvq()
{
    // Size of PSD and PSD Freq vectors
    int npsd;
    // PSD and PSD Freq vectors
    Eigen::VectorXd qf, hp;

    // Modify the PSD array to take out lowpassing+highpassing
    Eigen::Index maxhpind;
    double qfbreak = 0.;
    double hpbreak = 0.;

    // Get max value and index of hp
    auto maxhp = hp.maxCoeff(&maxhpind);

    if (maxhp < -1) {
        maxhp = -1;
        maxhpind = 0;
    }

    // Find the frequency where hp falls to 1e-4 its max value
    for (int i = 0; i < npsd; i++)
        if (hp(i) / maxhp < 1.e-4) {
            qfbreak = qf(i);
            break;
        }
    // Get the total number of points where the frequency is less than 0.8*qfbreak
    auto count = (qf.array() <= 0.8 * qfbreak).count();

    // Set the region of hp corresponding to the frequency range above 0.8*qfbreak to
    // the value of hp at 0.8*qfbreak (i.e. flatten it).  We can do this as a vector
    // since the frequencies are monotonically increasing.
    if (count > 0) {
        if (qfbreak > 0) {
            hp.segment(0.8 * qfbreak, hp.size() - 0.8 * qfbreak) = hp(0.8 * qfbreak);
        }
    }

    // Do the same for the region below the maximum hp
    hp.head(maxhpind) = maxhp;

    // Set up Q-space
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

    // Make qmap by replicating qy and qx ncols and nrows times respectively.  Faster than a for loop for expected
    // map dimensions
    qmap = (qx.replicate(1, ny).array().pow(2.0).matrix() + qy.replicate(nx, 1).array().pow(2.0).matrix()).array().sqrt();

    Eigen::MatrixXd psdq(nx, ny);

    if (config->get_typed<int>("getLowpassOnly")) {
        psdq.setOnes();
    }

    else{
        Eigen::Matrix<Eigen::Index, 1, 1> nhp;
        nhp << hp.size();

        // Interpolate onto psdq
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

    // Find the minimum value of hp
    auto lowval = hp.minCoeff();

    // Set all the points in psdq smaller than lowval to lowval
    psdq = (psdq < lowval).select(lowval, psdq);

    // Normalize the power spectrum psdq and place into Vvq
    Vvq = psdq/psdq.sum();

}

void WienerFilter::calcNumerator()
{
    // Normalization for FFT
    double fftnorm = 1. / nx / ny;

    Eigen::VectorXcd in(nx * ny);
    Eigen::VectorXcd out(nx * ny);

    in.real() = Rr * mflt;
    in.imag().setZero();

    out = mapmaking::fft2w<forward>(in, nx, ny);
    out = out * fftnorm;

    in.real() = out.real() / Vvq;
    in.imag() = out.imag() / Vvq;

    out = mapmaking::fft2w<backward>(in, nx, ny);

    in.real() = out.real() * Rr;
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

    // Populate Numerator
    Nume = out.real();
}

void WienerFilter::calcDenominator()
{
    double fftnorm = 1. / nx / ny;
    Eigen::VectorXcd in(nx * ny);
    Eigen::VectorXcd out(nx * ny);

    if (config->get_typed<int>("uniformWeight")) {
        in.real() = tplate();
        in.imag().setZero();

        out = mapmaking::fft2w<forward>(in, nx, ny);
        out = out * fftnorm;

        auto d = ((out.real() * out.real() + out.imag() * out.imag()) / Vvq).sum();
        Denom.setConstant(d);
    }

    else {
        Denom.setZero();

        in.real() = Vvq.pow(-1.);
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
                        (Rr.array() * shift(tplate, -xshiftN, -yshiftN).array()).data(), nx * ny);
                    in2.imag().setZero();

                    out2 = mapmaking::fft2w<forward>(in2, nx, ny);
                    out2 = out2 * fftnorm;

                    in2.real() = ffdq.real() * out2.real() + ffdq.imag() * out2.imag();
                    in2.imag() = -ffdq.imag() * out2.real() + ffdq.real() * out2.imag();

                    out2 = mapmaking::fft2w<backward>(in2, nx, ny);

                    Eigen::MatrixXd updater = zz2d[shifti] * out2.real() * fftnorm;

                    Denom = Denom + updater;

                    if ((kk % 100) == 1) {
                        double maxRatio = -1;
                        double maxDenom = -999.;

                        maxDenom = Denom.maxCoeff();
                        for (int i = 0; i < nx; i++) {
                            for (int j = 0; j < ny; j++) {
                                if (Denom(i, j) > 0.01 * maxDenom) {
                                    if (abs(updater(i, j) / Denom(i, j)) > maxRatio)
                                        maxRatio = abs(updater(i, j) / Denom(i, j));
                                }
                            }
                        }

                        if (((kk >= 500) && (maxRatio < 0.0002)) || maxRatio < 1e-10) {
                            SPDLOG_INFO("Seems like we're done.  maxRatio={}", maxRatio);
                            SPDLOG_INFO("The Denom calcualtion required {} iterations", kk);
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
        SPDLOG_INFO("Zeroing out any small values (< {}) in Denominator", denomLimit);
        Denom = (Denom.array() < denomLimit).select(0, Denom);
    }
}

} //namespace
