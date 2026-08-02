#include <citlali/core/pipeline/sci_align_scan_contract.h>

void sci_align_scan_contract_header_compiles_in_isolation() {
    (void)citlali::pipeline::sci_align::HalfOpenInterval{0, 1}.size();
}
