#include <citlali/core/pipeline/sci_align_field_registry.h>

void sci_align_field_registry_header_compiles_in_isolation() {
    (void)citlali::pipeline::sci_align::active_field_registry.size();
}
