#pragma once

#include <tula/algorithm/ei_stats.h>
#include <tula/algorithm/index.h>
#include <tula/container.h>
#include <tula/datatable.h>
#include <tula/filename.h>

// create Eigen::Matrix from ecsv file
inline auto to_matrix_from_ecsv(std::string filepath) {
    namespace fs = std::filesystem;
    std::vector<std::string> header;
    Eigen::MatrixXd table;

    YAML::Node meta_;

    try {
        table = datatable::read<double, datatable::Format::ecsv>(
            filepath, &header, &meta_);

    } catch (datatable::ParseError &e) {
        SPDLOG_WARN("unable to read apt table file as ECSV {}: {}", filepath,
                    e.what());
        try {
            table = datatable::read<double, datatable::Format::ascii>(filepath,
                                                                      &header);
        } catch (datatable::ParseError &e) {
            SPDLOG_WARN("unable to read apt table file as ASCII {}: {}",
                        filepath, e.what());
            throw e;
        }
    }
    return std::tuple {table, header, meta_};
}

// create ecsv file from Eigen::Matrix
template <typename Derived>
inline void to_ecsv_from_matrix(std::string filepath, Eigen::DenseBase<Derived> &table, std::vector<std::string> header, YAML::Node meta) {
    namespace fs = std::filesystem;

    try {
        YAML::Node meta_;
        datatable::write<datatable::Format::ecsv>(filepath + ".ecsv", table, header, std::vector<int>{}, meta);

    } catch (datatable::ParseError &e) {
        SPDLOG_WARN("unable to read apt table file as ECSV {}: {}", filepath,
                    e.what());
        try {
            datatable::write<datatable::Format::ascii>(filepath + ".ascii", table, header, std::vector<int>{});

        } catch (datatable::ParseError &e) {
            SPDLOG_WARN("unable to write apt table file as ASCII {}: {}",
                        filepath, e.what());
            throw e;
        }
    }
}
