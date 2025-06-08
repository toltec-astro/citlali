#pragma once

class ArrayPropertyTable : public PropertyTable {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    // bitwise flags
    enum AptFlags {
        Good         = 0,
        BadFit       = 1 << 0,
        AzFWHM       = 1 << 1,
        ElFWHM       = 1 << 2,
        Sig2Noise    = 1 << 3,
        Sens         = 1 << 4,
        Position     = 1 << 5,
    };

    // number of dets, nws, and arrays
    Eigen::Index n_dets, n_nws, n_arrays;
    // nw, array, and fg indices from apt
    Eigen::VectorXi nws, arrays, fgs;
    // average fwhms
    std::map<int, double> nw_fwhms, array_fwhms;
    // conversions from mJy/beam to other units for each array
    std::map<std::string, std::map<int, double>> unit_conversions;

    // start and end of each nw and array for convenience
    std::vector<std::pair<int, int>> nw_indices, array_indices;

    // constructor that initializes default keys, units, and descriptions
    ArrayPropertyTable() {
        add_column("uid", "N/A", "unique id");
        add_column("tone_freq", "Hz", "tone frequency");
        add_column("array", "N/A", "array index");
        add_column("nw", "N/A", "network index");
        add_column("fg", "N/A", "frequency group");
        add_column("pg", "N/A", "polarization group");
        add_column("ori", "N/A", "orientation");
        add_column("loc", "N/A", "location");
        add_column("responsivity", "N/A", "responsivity");
        add_column("flxscale", "mJy/beam/xs", "flux conversion scale");
        add_column("sens", "mJy/beam x s^0.5", "sensitivity");
        add_column("derot_elev", "rad", "derotation elevation angle");
        add_column("amp", "xs", "fitted amplitude");
        add_column("amp_err", "xs", "fitted amplitude error");
        add_column("x_t", "arcsec", "fitted azimuthal offset");
        add_column("x_t_err", "arcsec", "fitted azimuthal offset error");
        add_column("y_t", "arcsec", "fitted altitude offset");
        add_column("y_t_err", "arcsec", "fitted altitude offset error");
        add_column("a_fwhm", "arcsec", "fitted azimuthal FWHM");
        add_column("a_fwhm_err", "arcsec", "fitted azimuthal FWHM error");
        add_column("b_fwhm", "arcsec", "fitted altitude FWHM");
        add_column("b_fwhm_err", "arcsec", "fitted altitude FWHM error");
        add_column("angle", "rad", "fitted rotation angle");
        add_column("angle_err", "rad", "fitted rotation angle error");
        add_column("converge_iter", "N/A", "beammap convergence iteration");
        add_column("flag", "N/A", "bad detector");
        add_column("sig2noise", "N/A", "signal to noise");
        add_column("x_t_raw", "arcsec", "raw azimuthal offset");
        add_column("y_t_raw", "arcsec", "raw altitude offset");
        add_column("x_t_derot", "arcsec", "derot azimuthal offset");
        add_column("y_t_derot", "arcsec", "derot altitude offset");
    }

    // filter detectors based on a key and value and return a new APT
    ArrayPropertyTable filter_dets(const std::string& column_key, const double value) const {
        ArrayPropertyTable filtered_apt;

        const Eigen::VectorXd& column = columns.at(column_key).data;

        // collect the indices of rows that match the given value
        std::vector<int> matching_indices;
        for (int i = 0; i < column.size(); ++i) {
            if (column(i) == value) {
                matching_indices.push_back(i);
            }
        }

        // filter each column by the matching indices and populate the new APT
        for (const auto& [key, data_column] : columns) {
            Eigen::VectorXd filtered_column(matching_indices.size());
            for (size_t i = 0; i < matching_indices.size(); ++i) {
                filtered_column(i) = data_column.data(matching_indices[i]);
            }
            filtered_apt.set_data(key, filtered_column);
        }

        filtered_apt.init();

        return filtered_apt;
    }

    void load(const std::string&, const Eigen::VectorXi&);
    void write(const std::string);
    void init();
    void rescale_fcf(const std::string, const double);
    std::pair<double, double> find_reference(int);

    template <typename Derived>
    void rotate(Eigen::DenseBase<Derived>&, std::pair<double, double>&);
    auto calc_median(std::string, std::string);
};

void ArrayPropertyTable::load(const std::string& _filepath, const Eigen::VectorXi& interfaces) {

    filepath = _filepath;

    auto [table, header, meta_data] = from_ecsv(filepath);

    // vector to hold any missing header keys
    std::vector<std::string> missing_header_keys, empty_header_keys;

    // find if there are any missing or empty columns, otherwise set data
    for (const auto& [key, column] : columns) {
        if (!(std::find(header.begin(), header.end(), key) != header.end())) {
            missing_header_keys.push_back(key);
        } else if (table[key].size() == 0) {
            empty_header_keys.push_back(key);
        } else {
            set_data(key, table[key]);
        }
    }

    // throw an exception if any keys are missing
    if (!missing_header_keys.empty()) {
        throw std::runtime_error(fmt::format("APT table is missing required columns: {}", missing_header_keys));
    }

    // throw an exception if any keys are empty
    if (!empty_header_keys.empty()) {
        throw std::runtime_error(fmt::format("APT table columns are empty: {}", empty_header_keys));
    }

    // throw an exception if the APT reference frame is not altaz
    if (meta_data["Radesys"] != "altaz") {
        throw std::runtime_error("APT table is not in altaz reference frame");
    }

    // number of detectors in APT that are also in raw files
    Eigen::Index n_dets_temp = std::accumulate(interfaces.data(), interfaces.data() + interfaces.size(), 0,
                                               [&](Eigen::Index sum, Eigen::Index interface) {
                                                   return sum + (columns["nw"].data.cast<int>().array() == interface).count();
                                               });

    // populate apt with data only with corresponding data files
    Eigen::VectorXd networks_temp = columns["nw"].data;
    for (const auto& [key, column] : columns) {
        table[key].setZero(n_dets_temp);
        int i = 0;
        for (int j = 0; j < networks_temp.size(); ++j) {
            if ((networks_temp(j) == interfaces.array()).any()) {
                table[key](i) = columns[key].data(j);
                i++;
            }
        }
        set_data(key, table[key]);
    }

    // run apt initialization
    init();
}

void ArrayPropertyTable::write(const std::string filepath) {
    Eigen::MatrixXd table(n_dets, columns.size());
    std::vector<std::string> header;

    // add date of file creation
    meta["creation_date"] = citlali::utils::timing::current_date_time();

    int i = 0;
    for (const auto& key : column_order) {
        table.col(i) = columns[key].data;
        header.emplace_back(key);
        i++;
    }

    to_ecsv_from_matrix(filepath, table, header, meta);
}

void ArrayPropertyTable::init() {
    // get number of detectors
    n_dets = columns["uid"].data.size();

    // find unique arrays
    arrays = find_unique_elements<Eigen::VectorXd, Eigen::VectorXi>(columns["array"].data);
    n_arrays = arrays.size();

    // find unique networks
    nws = find_unique_elements<Eigen::VectorXd, Eigen::VectorXi>(columns["nw"].data);
    n_nws = nws.size();

    // find unique frequency groups
    fgs = find_unique_elements<Eigen::VectorXd, Eigen::VectorXi>(columns["fg"].data);

    // find network edges
    nw_indices = find_edges(columns["nw"].data);
    // find array edges
    array_indices = find_edges(columns["array"].data);

    // get nw average fwhms
    int i = 0;
    for (const auto& pair : nw_indices) {
        nw_fwhms[nws(i)] = calculate_average_fwhms(
            columns["a_fwhm"].data(Eigen::seq(pair.first, pair.second - 1)),
            columns["b_fwhm"].data(Eigen::seq(pair.first, pair.second - 1)),
            columns["flag"].data(Eigen::seq(pair.first, pair.second - 1))
            );
        i++;
    }

    // get array average fwhms
    i = 0;
    for (const auto& pair : array_indices) {
        array_fwhms[arrays(i)] = calculate_average_fwhms(
            columns["a_fwhm"].data(Eigen::seq(pair.first, pair.second - 1)),
            columns["b_fwhm"].data(Eigen::seq(pair.first, pair.second - 1)),
            columns["flag"].data(Eigen::seq(pair.first, pair.second - 1))
            );
        i++;
    }

    // for (const auto& array : arrays) {
    //     // array average beam area
    //     auto beam_area_arcsec = 2.*pi*pow(array_fwhms[array] * FWHM_TO_STD, 2);

    //     unit_conversions["MJy/Sr"][array] = mJY_ASEC_to_MJY_SR / beam_area_arcsec;
    //     unit_conversions["Jy/px"][array] = 1e-3 / (beam_area_arcsec * std::pow(ASEC_TO_RAD * pix_size_radians, 2));
    //     unit_conversions["uK_cmb"][array]
    // }
}

void ArrayPropertyTable::rescale_fcf(const std::string units, const double pix_size_radians) {
    if (units == "MJy/Sr") {
        for (int det = 0; det < n_dets; ++det) {
            auto array = (*this)["array"].data(det);

            // array average beam area
            auto beam_area_arcsec = 2.*pi*pow(array_fwhms[array] * FWHM_TO_STD, 2);

            // FCF x mJy/beam -> MJy/Sr
            (*this)["flxscale"].data(det) *= mJY_ASEC_to_MJY_SR / beam_area_arcsec;
        }
    } else if (units == "Jy/px") {
        for (int det = 0; det < n_dets; ++det) {
            auto array = (*this)["array"].data(det);

            // array average beam area
            auto beam_area_pix = 2.*pi*pow(array_fwhms[array] * FWHM_TO_STD * ASEC_TO_RAD * pix_size_radians, 2);

            (*this)["flxscale"].data(det) *= 1e-3 / beam_area_pix;
        }
    } else if (units == "uK_cmb") {
        for (int det = 0; det < n_dets; ++det) {
            // auto array = (*this)["array"].data(det);
            // // frequency of band
            // auto nu_Hz = 1e9 * toltec.array_index_to_freq_GHZ[array];

            // // get B_nu(T_cmb)
            // double B_nu_T_cmb = planck_nu(nu_Hz, T_cmb_K);
            // double x = (h_J_s * nu_Hz) / (kB_J_K * T_cmb_K);
            // // conversion from T_cmb_K to mJy/beam
            // double K_mJy_beam = 1e26 * 1e3 * B_nu_T_cmb * std::exp(x) / (std::exp(x - 1) * x / std::pow(T_cmb_K, 2.));

            // (*this)["flxscale"].data(det) *= 1e6 / K_mJy_beam;
        }
    }
}

std::pair<double, double> ArrayPropertyTable::find_reference(int uid) {
    // if uid is a row in the apt, use that detector
    if (uid >= 0) {
        return std::make_pair((*this)["x_t_raw"].data(uid), (*this)["y_t_raw"].data(uid));
    }
    else {
        // otherwise find the closest unflagged detector to (0,0)
        // array to store good indices
        Eigen::ArrayXi good_indices;

        int n_good = 0;

        for (int det = 0; det < n_dets; ++det) {
            if (!(*this)["flag"].data(det)) {
                n_good++;
            }
        }

        good_indices.resize(n_good);

        // populate good indices
        int j = 0;
        for (int det = 0; det < n_dets; ++det) {
            if (!(*this)["flag"].data(det)) {
                good_indices(j) = det;
                j++;
            }
        }

        Eigen::Index uid;
        auto distance = ((*this)["x_t_raw"].data(good_indices).array().pow(2) +
                         (*this)["y_t_raw"].data(good_indices).array().pow(2)).minCoeff(&uid);

        return std::make_pair((*this)["x_t_raw"].data(uid), (*this)["y_t_raw"].data(uid));
    }
}

template <typename Derived>
void ArrayPropertyTable::rotate(Eigen::DenseBase<Derived>& theta, std::pair<double, double>& reference_coord) {
    // copy theta
    (*this)["derot_elev"].data = theta;

    auto cos_theta = theta.derived().array().cos();
    auto sin_theta = theta.derived().array().sin();

    // subtract reference detector
    auto x_t = (*this)["x_t_raw"].data.array() - reference_coord.first;
    auto y_t = (*this)["y_t_raw"].data.array() - reference_coord.second;

    // rotate by theta around (0,0)
    (*this)["x_t"].data = x_t * cos_theta.array() - y_t * sin_theta.array();
    (*this)["y_t"].data = x_t * sin_theta.array() + y_t * cos_theta.array();

    // copy to derotated keys
    (*this)["x_t_derot"].data = (*this)["x_t"].data;
    (*this)["y_t_derot"].data = (*this)["y_t"].data;
}

auto ArrayPropertyTable::calc_median(std::string key, std::string grouping) {
    std::map<int, double> median;
    auto unique_values = find_unique_elements<Eigen::VectorXd, Eigen::VectorXi>((*this)[grouping].data);

    auto key_good = filter_by_condition((*this)[key].data, (*this)["flag"].data, false);
    auto group_good = filter_by_condition((*this)[grouping].data, (*this)["flag"].data, false);

    for (const auto& val : unique_values) {
        auto filtered = filter_by_condition(key_good, group_good, val);
        median[val] = tula::alg::median(filtered);
    }

    return median;
}
