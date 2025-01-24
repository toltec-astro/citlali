# pragma once


class PointingPropertyTable : public PropertyTable {
public:
    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    // constructor that initializes default keys, units, and descriptions
    PointingPropertyTable() {
        add_column("array", "N/A", "array index");
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
        add_column("flag", "N/A", "bad detector");
        add_column("sig2noise", "N/A", "signal to noise");
    }

    void load(const std::string&, const Eigen::VectorXi&);
    void write(const std::string);
};

void PointingPropertyTable::write(const std::string filepath) {
    Eigen::MatrixXd table(columns["array"].data.size(), columns.size());
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
