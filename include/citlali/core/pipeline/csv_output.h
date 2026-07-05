#pragma once

#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace citlali::pipeline {

inline std::string csv_escaped(const std::string &value) {
    std::string escaped = "\"";
    for (const char ch : value) {
        if (ch == '"') {
            escaped += "\"\"";
        }
        else {
            escaped += ch;
        }
    }
    escaped += "\"";
    return escaped;
}

template <class Value>
std::string csv_text(const Value &value) {
    std::ostringstream stream;
    stream << value;
    return stream.str();
}

inline void write_csv_row(
    std::ostream &out, const std::vector<std::string> &row) {
    for (std::size_t i = 0; i < row.size(); ++i) {
        if (i > 0) {
            out << ',';
        }
        out << row[i];
    }
    out << '\n';
}

}  // namespace citlali::pipeline
