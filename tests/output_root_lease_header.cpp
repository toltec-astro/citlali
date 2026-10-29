#include <citlali/core/pipeline/staged_output_directory.h>
#include <citlali/core/pipeline/output_root_lease.h>

const char *output_root_lease_header_lock_name() {
    return citlali::pipeline::output_root_lock_filename;
}
