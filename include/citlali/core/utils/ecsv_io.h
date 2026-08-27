#pragma once

#include <citlali/core/error/error.h>

#include <tula/algorithm/ei_stats.h>
#include <tula/algorithm/index.h>
#include <tula/container.h>
#include <tula/datatable.h>
#include <tula/filename.h>

#include <tula/ecsv/core.h>
#include <csv_parser/parser.hpp>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <tula/ecsv/table.h>
#include <tula/formatter/container.h>
#include <tula/formatter/matrix.h>
#include <yaml-cpp/node/emit.h>

template <typename T, typename Table>
inline auto ecsv_numeric_column_data(Table &table, const std::string &column) {
#ifdef CITLALI_TULA_V3
    return table.template col<T>(column).data;
#else
    return table.template col<T>(column);
#endif
}

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
    const fs::path final_path(filepath + ".ecsv");
    const fs::path temp_path(final_path.string() + ".tmp");
    std::error_code ec;
    fs::remove(temp_path, ec);
    try {
        datatable::write<datatable::Format::ecsv>(
            temp_path.string(), table, header, std::vector<int>{}, meta);
        ec.clear();
        fs::remove(final_path, ec);
        ec.clear();
        fs::rename(temp_path, final_path, ec);
        if (ec) {
            throw citlali::error::output(
                "failed to publish ECSV temp file " + temp_path.string() +
                " -> " + final_path.string() + ": " + ec.message());
        }
    } catch (const std::exception &e) {
        ec.clear();
        fs::remove(temp_path, ec);
        throw citlali::error::output(
            "failed to write required ECSV output " + final_path.string() +
            ": " + e.what());
    } catch (...) {
        ec.clear();
        fs::remove(temp_path, ec);
        throw citlali::error::output(
            "failed to write required ECSV output " + final_path.string());
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
            table[col] = ecsv_numeric_column_data<int>(tbl, col)
                             .template cast<double>();
        }

        // get int16
        auto int16_colnames = tbl.array_data<int16_t>().colnames();
        for (auto & col : int16_colnames) {
            table[col] = ecsv_numeric_column_data<int16_t>(tbl, col)
                             .template cast<double>();
        }

        // get int64
        auto int64_colnames = tbl.array_data<int64_t>().colnames();
        for (auto & col : int64_colnames) {
            table[col] = ecsv_numeric_column_data<int64_t>(tbl, col)
                             .template cast<double>();
        }

        // get bools
        auto bool_colnames = tbl.array_data<bool>().colnames();
        for (auto & col : bool_colnames) {
            table[col] = ecsv_numeric_column_data<bool>(tbl, col)
                             .template cast<double>();
        }

        // get floats
        auto float_colnames = tbl.array_data<float>().colnames();
        for (auto & col : float_colnames) {
            table[col] = ecsv_numeric_column_data<float>(tbl, col)
                             .template cast<double>();
        }

        // get doubles
        auto dbl_colnames = tbl.array_data<double>().colnames();
        for (auto & col : dbl_colnames) {
            table[col] = ecsv_numeric_column_data<double>(tbl, col);
        }
    }
    catch(const std::exception &error) {
        throw citlali::error::io(
            "cannot open input table " + filepath + ": " + error.what());
    }
    catch(...) {
        throw citlali::error::io("cannot open input table " + filepath);
    }

    // return map and header
    return std::tuple {table, header, map_with_strs};
}
