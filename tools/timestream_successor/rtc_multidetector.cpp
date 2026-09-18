// Retained comparison entry point; the implementation is shared with the default CLI.
#include <citlali/core/cli/successor_development_execution.h>
#include <iostream>
int main(int argc, char **argv) {
  if(argc!=3 && argc!=4) {
    std::cerr<<"expected input JSON, NEW output directory, optional reviewed-selection YAML\n";
    return 1;
  }
  return citlali::cli::run_successor_rtc(argv[1],argv[2],
      citlali::cli::RtcInvocation::comparison,
      argc==4 ? std::optional<std::filesystem::path>{argv[3]} : std::nullopt);
}
