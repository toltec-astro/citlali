#include <citlali/core/mapmaking/jinc_contract.h>

namespace {

[[maybe_unused]] auto jinc_contract_header_smoke() {
    return mapmaking::finalize_jinc_accumulators(
        1.0, 1.0, 1.0, 1.0, 1);
}

}  // namespace
