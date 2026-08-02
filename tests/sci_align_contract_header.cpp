#include <citlali/core/pipeline/sci_align_contract.h>

void sci_align_contract_header_compiles_in_isolation() {
    (void)citlali::pipeline::sci_align::rate_multiplier(
        citlali::pipeline::sci_align::NativeRateFactor::one);
}
