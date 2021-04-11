#pragma once

#include <utils/algorithm/ei_stats.h>
#include <utils/algorithm/index.h>
#include <utils/container.h>
#include <utils/datatable.h>
#include <utils/filename.h>
#include <utils/grppiex.h>

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
        SPDLOG_WARN("unable to read fitreport file as ECSV {}: {}", filepath,
                    e.what());
        try {
            table = datatable::read<double, datatable::Format::ascii>(filepath,
                                                                      &header);
        } catch (datatable::ParseError &e) {
            SPDLOG_WARN("unable to read fitreport file as ASCII {}: {}",
                        filepath, e.what());
            throw e;
        }
    }
    return table;
}
