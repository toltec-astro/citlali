#include <citlali/core/config/timestream_enums.h>

#include <iostream>
#include <stdexcept>

auto main() -> int {
    const auto parsed = citlali::config::parse_tod_type("xs");
    if (!parsed || *parsed != citlali::config::TodType::xs ||
        citlali::config::to_string(*parsed) != "xs") {
        throw std::runtime_error{"installed Citlali enum API failed"};
    }
    std::cout << "citlali installed consumer: xs\n";
    return 0;
}
