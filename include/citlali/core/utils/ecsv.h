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

// function to calculate average FWHMs
double calculate_average_fwhms(const Eigen::VectorXd& a_fwhm, const Eigen::VectorXd& b_fwhm,
                               const Eigen::VectorXd& flags) {

    // create a mask for unflagged detectors
    Eigen::Array<bool, Eigen::Dynamic, 1> mask = (flags.array() == 0);

    // select the unflagged values from a_fwhm and b_fwhm using the mask
    double sum_a_fwhm = (a_fwhm.array() * mask.cast<double>()).sum();
    double sum_b_fwhm = (b_fwhm.array() * mask.cast<double>()).sum();

    // count the number of unflagged detectors
    Eigen::Index n_good_det = mask.count();

    // calculate the overall average of a_fwhm and b_fwhm for unflagged detectors
    double avg_fwhm = (sum_a_fwhm + sum_b_fwhm) / (2 * n_good_det);

    return avg_fwhm;
}

// function to find unique elements
template <typename InputType, typename OutputType>
OutputType find_unique_elements(const InputType& data) {
    std::set<double> unique_elements(data.data(), data.data() + data.size());

    OutputType result(unique_elements.size());
    int i = 0;
    for (const auto& elem : unique_elements) {
        result(i++) = elem;
    }

    return result;
}

std::map<int, Eigen::VectorXi> find_corresponding_unique_elements(const Eigen::VectorXi& vec1, const Eigen::VectorXi& vec2) {
    // map to store unique values from vec2 for each value in vec1
    std::map<int, std::set<int>> unique_values_map;

    // iterate over both vectors
    for (int i = 0; i < vec1.size(); ++i) {
        int key = vec1(i);
        int value = vec2(i);

        // insert the value into the set associated with the key in the map
        unique_values_map[key].insert(value);
    }

    // map to store the unique values
    std::map<int, Eigen::VectorXi> unique_values_eigen_map;

    // convert each set of unique values
    for (const auto& [key, values] : unique_values_map) {
        Eigen::VectorXi eigen_values(values.size());
        int idx = 0;
        for (int val : values) {
            eigen_values(idx++) = val;
        }
        unique_values_eigen_map[key] = eigen_values;
    }

    return unique_values_eigen_map;
}

// function to find edges
std::vector<std::pair<int, int>> find_edges(const Eigen::VectorXd& data) {
    std::vector<std::pair<int, int>> edges;
    int start = 0;
    int n = data.size();

    for (int i = 1; i < n; ++i) {
        if (data(i) != data(i - 1)) {
            edges.push_back({start, i - 1});
            start = i;
        }
    }

    // add the last range
    edges.push_back({start, n - 1});

    return edges;
}

template <typename Derived>
Eigen::VectorXd filter_by_condition(const Eigen::DenseBase<Derived>& source, const Eigen::DenseBase<Derived>& condition,
                                    int target_value) {
    // Count how many elements match the condition
    int count = (condition.derived().array() == target_value).count();

    // Allocate the filtered vector
    Eigen::VectorXd filtered(count);

    // Fill the filtered vector
    int index = 0;
    for (int i = 0; i < condition.size(); ++i) {
        if (condition(i) == target_value) {
            filtered(index++) = source(i);
        }
    }

    return filtered;
}

class DataColumn {
public:
    // data associated with this column (e.g., Eigen vector)
    Eigen::VectorXd data;

    // unit and description of the column
    std::string unit;
    std::string description;

    DataColumn() {}

    // constructor
    DataColumn(std::string unit, std::string description)
        : unit(unit), description(description) {}

    // set the data
    void set_data(const Eigen::VectorXd& new_data) {
        data = new_data;
    }

    // method to get the size of the data
    Eigen::Index size() const {
        return data.size();
    }
};

class PropertyTable {
public:
    std::string filepath;

    // meta information
    YAML::Node meta;

    // vector to store the order of keys
    std::vector<std::string> column_order;

    // unordered map to store columns by key
    std::unordered_map<std::string, DataColumn> columns;

    // add a column with its key, unit, and description
    void add_column(const std::string& key, const std::string& unit, const std::string& description) {
        if (columns.find(key) == columns.end()) {
            column_order.push_back(key);
        }
        columns[key] = DataColumn(unit, description);
    }

    // overload the [] operator to access columns by key
    DataColumn& operator[](const std::string& key) {
        return columns.at(key);
    }

    // method to set data for a specific key
    void set_data(const std::string& key, const Eigen::VectorXd& data) {
        columns.at(key).data = data;
    }
};

// helper function to load columns of a specific type from the table
template <typename T>
void load_columns(tula::ecsv::ECSVTable& tbl, std::map<std::string, Eigen::VectorXd>& table) {
    auto colnames = tbl.array_data<T>().colnames();
    for (const auto& col : colnames) {
        table[col] = tbl.col<T>(col).template cast<double>();
    }
}

// load an ecsv table
auto from_ecsv(const std::string& filepath) {
    using namespace tula::ecsv;

    // get logger
    auto logger = spdlog::get("citlali_logger");

    // std::vector to hold header
    std::vector<std::string> header;

    // std::map for holding data
    std::map<std::string, Eigen::VectorXd> table;

    // hold string metadata
    std::map<std::string, std::string> meta_data;

    // YAML node for other metadata
    YAML::Node meta;

    try {
        std::ifstream fo(filepath);
        // read header
        auto hdr = ECSVHeader::read(fo);
        auto tbl = ECSVTable(hdr);

        // parse the contents
        auto parser = aria::csv::CsvParser(fo).delimiter(tbl.header().delimiter());
        tbl.load_rows(parser);

        // get header colnames
        for (int i = 0; i < tbl.header().colnames().size(); ++i) {
            header.push_back(tbl.header().colnames()[i]);
        }

        // needed to get meta data strings
        const auto map_with_bools =
            meta_to_map<std::string, bool>(hdr.meta(), &meta);
        // Extract metadata
        meta_data = meta_to_map<std::string, std::string>(meta, &meta);

        // load columns of different types
        load_columns<int>(tbl, table);
        load_columns<int16_t>(tbl, table);
        load_columns<int64_t>(tbl, table);
        load_columns<bool>(tbl, table);
        load_columns<float>(tbl, table);
        load_columns<double>(tbl, table);

    } catch (const std::exception& e) {
        throw std::runtime_error(fmt::format("unable to open file {}: {}", filepath, e.what()));
    }

    // return map, header, and string metadata
    return std::tuple{table, header, meta_data};
}

// create ECSV file from Eigen::Matrix
template <typename Derived>
void to_ecsv_from_matrix(const std::string& filepath,
                         const Eigen::DenseBase<Derived>& table,
                         const std::vector<std::string>& header,
                         const YAML::Node& meta) {
    namespace fs = std::filesystem;

    try {
        // write ECSV file
        datatable::write<datatable::Format::ecsv>(filepath + ".ecsv", table, header, std::vector<int>{}, meta);
    } catch (const datatable::ParseError& e) {
        try {
            // fallback to ASCII format
            datatable::write<datatable::Format::ascii>(filepath + ".ascii", table, header, std::vector<int>{});
        } catch (const datatable::ParseError& e) {
            throw std::runtime_error("failed to write ECSV and ASCII files: " + std::string(e.what()));
        }
    }
}
