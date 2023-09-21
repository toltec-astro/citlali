#pragma once

#include <tula/algorithm/ei_stats.h>
#include <tula/algorithm/index.h>
#include <tula/container.h>
#include <tula/datatable.h>
#include <tula/filename.h>

#include <tula/ecsv/core.h>
#include <csv_parser/parser.hpp>
#include <sstream>
#include <tula/ecsv/table.h>
#include <tula/formatter/container.h>
#include <tula/formatter/matrix.h>
#include <yaml-cpp/node/emit.h>

// create Eigen::Matrix from ecsv file
inline auto to_matrix_from_ecsv(std::string filepath) {
    namespace fs = std::filesystem;

    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    std::vector<std::string> header;
    Eigen::MatrixXd table;

    YAML::Node meta_;

    try {
        table = datatable::read<double, datatable::Format::ecsv>(
            filepath, &header, &meta_);

    } catch (datatable::ParseError &e) {
        logger->warn("unable to read apt table file as ECSV {}: {}", filepath,
                    e.what());
        try {
            table = datatable::read<double, datatable::Format::ascii>(filepath,
                                                                      &header);
        } catch (datatable::ParseError &e) {
            logger->warn("unable to read apt table file as ASCII {}: {}",
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

    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    try {
        YAML::Node meta_;
        datatable::write<datatable::Format::ecsv>(filepath + ".ecsv", table, header, std::vector<int>{}, meta);

    } catch (datatable::ParseError &e) {
        logger->warn("unable to read apt table file as ECSV {}: {}", filepath,
                    e.what());
        try {
            datatable::write<datatable::Format::ascii>(filepath + ".ascii", table, header, std::vector<int>{});

        } catch (datatable::ParseError &e) {
            logger->warn("unable to write apt table file as ASCII {}: {}",
                        filepath, e.what());
            throw e;
        }
    }
}

inline auto to_map_from_ecsv_mixted_type(std::string filepath) {
    using namespace tula::ecsv;

    // get logger
    std::shared_ptr<spdlog::logger> logger = spdlog::get("citlali_logger");

    // vector to hold header
    std::vector<std::string> header;

    // std map for holding data
    std::map<std::string, Eigen::VectorXd> table;

    // hold str meta
    std::map<std::string, std::string> map_with_strs;

    // to hold meta data
    YAML::Node meta{};

    std::ifstream fo(filepath);
    try {
        // read in header
        auto hdr = ECSVHeader::read(fo);
        // create table
        auto tbl = ECSVTable(hdr);
        // parse the contents
        auto parser = aria::csv::CsvParser(fo).delimiter(tbl.header().delimiter());
        // load rows
        tbl.load_rows(parser);

        // get header colnames
        for (Eigen::Index i=0; i<tbl.header().colnames().size(); i++) {
            header.push_back(tbl.header().colnames()[i]);
        }

        const auto map_with_bools =
            meta_to_map<std::string, bool>(hdr.meta(), &meta);

        map_with_strs =
            meta_to_map<std::string, std::string>(meta, &meta);

        // get ints
        auto int_colnames = tbl.array_data<int>().colnames();
        for (auto & col : int_colnames) {
            table[col] = tbl.col<int>(col).template cast<double> ();
        }

        // get int16
        auto int16_colnames = tbl.array_data<int16_t>().colnames();
        for (auto & col : int16_colnames) {
            table[col] = tbl.col<int16_t>(col).template cast<double> ();
        }

        // get int64
        auto int64_colnames = tbl.array_data<int64_t>().colnames();
        for (auto & col : int64_colnames) {
            table[col] = tbl.col<int64_t>(col).template cast<double> ();
        }

        // get bools
        auto bool_colnames = tbl.array_data<bool>().colnames();
        for (auto & col : bool_colnames) {
            table[col] = tbl.col<bool>(col).template cast<double> ();
        }

        // get floats
        auto float_colnames = tbl.array_data<float>().colnames();
        for (auto & col : float_colnames) {
            table[col] = tbl.col<float>(col).template cast<double> ();
        }

        // get doubles
        auto dbl_colnames = tbl.array_data<double>().colnames();
        for (auto & col : dbl_colnames) {
            table[col] = tbl.col<double>(col);
        }
    }
    catch(...) {
        logger->error("cannot open input table");
        std::exit(EXIT_FAILURE);
    }

    // return map and header
    return std::tuple {table, header, map_with_strs};
}
