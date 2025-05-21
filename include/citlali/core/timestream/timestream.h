# pragma once

enum TimestreamFlags {
    Good = 0,
    D21FitsBetter   = 1 << 0,
    D21LargeOffset  = 1 << 1,
    D21NotConverged = 1 << 2,
    D21OutOfRange   = 1 << 3,
    D21QrOutOfRange = 1 << 4,
    LargeOffset     = 1 << 5,
    NotConverged    = 1 << 6,
    OutOfRange      = 1 << 7,
    QrOutOfRange    = 1 << 8,
    LowGain         = 1 << 9,
    APT             = 1 << 10,
    Spike           = 1 << 11,
    Freq            = 1 << 12
};

struct TCData {
    // default constructor
    TCData() {
        // set seed
        rng = gsl_rng_alloc(gsl_rng_mt19937);
    }

    void set_seed(int seed) {
        // auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        // auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
        // unsigned long seed = static_cast<unsigned long>(now ^ tid);

        if (rng) {
            gsl_rng_set(rng, static_cast<unsigned long>(seed));
        }
    }

    Eigen::MatrixXd signal, kernel; // data and kernel
    std::optional<Eigen::MatrixXd> signal_q, signal_u; // matrices for q and u timestreams
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> flag; // bad sample flags
    Eigen::VectorXd weight; // detector weights
    std::optional<Eigen::VectorXd> weight_q, weight_u; // vectors for q and u weights
    Eigen::VectorXd fcf;  // a copy of the apt fcf to account for per chunk extinction

    // copy of apt flags to be set by weight and inverse variance flagging
    Eigen::VectorXd apt_flag;

    // chunk number in observation (0 is the first chunk)
    Eigen::Index chunk;
    // indices where chunks start and end
    Eigen::Matrix<Eigen::Index,Eigen::Dynamic,1> chunk_indices;
    // telescope boresight vectors for current chunk
    std::map<std::string, Eigen::VectorXd> tel_data;
    // HWPR and detector orientation angle
    Eigen::VectorXd hwpr_theta, det_theta;

    // number of spikes per detector
    Eigen::VectorXd n_spikes;

    // data sampling rate (updated when downsampling)
    double data_fs_hz;

    // function to get the number of data points (rows)
    Eigen::Index n_pts() const {
        return signal.rows();
    }
    // function to get the number of detectors (columns)
    Eigen::Index n_dets() const {
        return signal.cols();
    }

    std::tuple<Eigen::Index, Eigen::Index> dims() {
        return std::make_tuple(n_pts(), n_dets());
    }

    // get random +1/-1 for a vector or matrix
    template <typename Derived>
    void random_sign(Eigen::MatrixBase<Derived>& output) {
        output = output.unaryExpr([&](typename Derived::Scalar) {
            return gsl_rng_uniform(rng) < 0.5 ? -1 : 1;
        });
    }

    void gsl_free() {
        gsl_rng_free(rng);
    }

    auto get_good_indices(const int start, const int end) {
        // array to store good indices
        Eigen::ArrayXi good_indices;

        int n_good = 0;

        // find number of good detectors
        for (int k = start; k <= end; ++k) {
            if (!apt_flag(k) && (flag.col(k).array() == false).any()) {
                n_good++;
            }
        }

        good_indices.resize(n_good);

        // populate good indices
        int m = 0;
        for (int k = start; k <= end; ++k) {
            if (!apt_flag(k) && (flag.col(k).array() == false).any()) {
                good_indices(m) = k;
                m++;
            }
        }

        return good_indices;
    }

    void shrink(int start, int size) {
        int nd = n_dets();

        Eigen::MatrixXd signal_copy = signal.block(start, 0, size, nd);
        signal = signal_copy;

        if (signal_q.has_value()) {
            Eigen::MatrixXd signal_copy = signal_q->block(start, 0, size, nd);
            signal_q = signal_copy;
        }

        if (signal_u.has_value()) {
            Eigen::MatrixXd signal_copy = signal_u->block(start, 0, size, nd);
            signal_u = signal_copy;
        }

        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> flag_copy = flag.block(start, 0, size, nd);
        flag = flag_copy;

        if (kernel.size() > 0) {
            Eigen::MatrixXd kernel_copy = kernel.block(start, 0, size, nd);
            kernel = kernel_copy;
        }

        for (auto& [key, val] : tel_data) {
            Eigen::VectorXd tel_copy = val.segment(start, size);
            val = tel_copy;
        }

        if (hwpr_theta.size() > 0) {
            Eigen::VectorXd hwpr_theta_copy = hwpr_theta;
            hwpr_theta = hwpr_theta_copy.segment(start, size);
        }
    }

private:
    // random number generator for noise maps
    // hopefully thread safe
    gsl_rng *rng = nullptr;
};
