#pragma once

#include <utils/algorithm/ei_stats.h>
#include <utils/algorithm/index.h>
#include <utils/container.h>
#include <utils/datatable.h>
#include <utils/filename.h>
#include <utils/grppiex.h>
#include <utils/ecsv.h>

template <typename Config>
auto get_aptable_from_ecsv(std::string filepath, const Config &config) {
    namespace fs = std::filesystem;
    std::vector<std::string> header;
    Eigen::MatrixXd table;

    try {
        YAML::Node meta_;
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
    return table;
}

template <typename Config, typename Derived>
void write_aptable_to_ecsv(Eigen::DenseBase<Derived> &data, const std::string &filepath, Config &config) {
    try {
        constexpr auto format = datatable::Format::ecsv;
        YAML::Node meta;
        std::vector<std::string> colnames {
            {"array_name"},
            {"nw"},
            {"S/N"},
            {"x_t"},
            {"y_t"},
            {"a_fwhm"},
            {"b_fwhm"},
            {"ang"}};

        datatable::write<format>(filepath, data, colnames, std::vector<int>{});
        SPDLOG_INFO("finished writing file {}", filepath);
    } catch (const datatable::DumpError &e) {
        SPDLOG_ERROR("unable to write to file {}: {}", filepath, e.what());
    }
}
