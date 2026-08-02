#include <citlali/core/utils/slot_assignment.h>

void slot_assignment_header_compiles_in_isolation() {
    (void)citlali::utils::round_half_up_slot(0.5);
}
