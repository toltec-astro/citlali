# Citlali Refactor Design Scaffolding - 2026-06-29

This note sketches the first structural interfaces to implement after baseline
validation is available. It is design scaffolding only; no runtime code is
introduced here.

## Design Constraints

- Preserve current science behavior and YAML compatibility.
- Keep the CLI thin: parse/help/version/log bootstrap/exit codes.
- Keep hot loops free of virtual dispatch, YAML lookups, string maps, and new
  allocations.
- Add typed boundaries outside hot paths first.
- Make each implementation PR reviewable and Unity-testable.

## Proposed Module Boundaries

```text
include/citlali/core/config/
  config_error.h
  runtime_config.h
  mapmaking_config.h
  timestream_config.h
  beammap_config.h
  pointing_config.h
  reduction_config.h
  profile_expander.h

include/citlali/core/pipeline/
  pipeline_runner.h
  reduction_session.h
  observation_context.h
  output_layout.h
  reduction_result.h

include/citlali/core/error/
  error.h
  result.h

src/citlali/core/config/
src/citlali/core/pipeline/
src/citlali/core/error/
```

Actual file names can be adjusted during implementation. The key is that
config, orchestration, and error handling become library concepts rather than
living inside `src/citlali/cli/main.cpp` and `Engine`.

## Typed Config Scaffolding

The first typed config implementation should parse the current full YAML
schema. Compact profile configs should come later, after full-schema parsing can
round-trip defaults.

Sketch:

```cpp
namespace citlali::config {

struct Diagnostic {
    enum class Severity { warning, error };
    Severity severity;
    std::vector<std::string> path;
    std::string message;
};

struct ValidationReport {
    std::vector<Diagnostic> diagnostics;

    bool ok() const;
    std::vector<Diagnostic> errors() const;
    std::string format_for_cli() const;
};

enum class ReductionType { science, pointing, beammap };
enum class ParallelPolicy { seq, omp };
enum class MapMethod { naive, jinc, maximum_likelihood };
enum class MapGrouping { automatic, detector, nw, array, fg };
enum class TodKind { xs, rs, is, qs };

struct RuntimeConfig {
    bool verbose = false;
    bool interp_over_gaps = true;
    int n_threads = 1;
    std::string output_dir;
    ParallelPolicy parallel_policy = ParallelPolicy::seq;
    ReductionType reduction_type = ReductionType::science;
    bool use_subdir = true;
};

struct MapmakingConfig {
    bool enabled = true;
    std::string unit = "mJy/beam";
    MapGrouping grouping = MapGrouping::automatic;
    MapMethod method = MapMethod::naive;
    std::string pixel_axes = "radec";
    double pixel_size_arcsec = 1.0;
    std::array<double, 2> crval_j2000 = {0.0, 0.0};
    std::array<int, 2> explicit_size_pix = {0, 0};
    double coverage_cut = 0.0;
};

struct NoiseConfig {
    bool enabled = false;
    int n_noise_maps = 1;
    bool randomize_dets = true;
    bool write_realizations = false;
    bool products_enabled = true;
    bool apply_empirical_weights = true;
};

struct ReductionConfig {
    RuntimeConfig runtime;
    MapmakingConfig mapmaking;
    NoiseConfig noise;
    // Add coadd, timestream, RTC, PTC, fruitloops, pointing, beammap,
    // post-processing, and wiener filter sections incrementally.
    ValidationReport validation;
};

ReductionConfig parse_full_yaml(const tula::config::YamlConfig &yaml);
tula::config::YamlConfig to_legacy_yaml(const ReductionConfig &config);

}  // namespace citlali::config
```

Implementation strategy:

1. Add enums and structs with no pipeline wiring.
2. Add parsing helpers that read existing YAML keys and produce path-rich
   diagnostics.
3. Add tests for missing, invalid, enum, range, and fixed-vector cases.
4. Populate legacy `Engine` fields from typed config in one section at a time.
5. Only after full-schema parsing is covered, add compact profile expansion.

Do not make config structs read YAML during processing. All YAML access should
be a parse-time concern.

## Config Validation API

Use a small set of typed readers:

```cpp
template <class T>
T require(const YamlConfig &yaml, ConfigPath path, ValidationReport &report);

template <class T>
T optional(const YamlConfig &yaml, ConfigPath path, T fallback,
           ValidationReport &report);

template <class T>
void check_range(T value, ConfigPath path, T min, T max,
                 ValidationReport &report);

template <class Enum>
Enum parse_enum(std::string_view value, ConfigPath path,
                std::span<const EnumName<Enum>> names,
                ValidationReport &report);
```

The CLI can still print the current `missing keys=[...]` and
`invalid keys=[...]` shape at first, but internally we should retain richer
messages.

## PipelineRunner Scaffolding

The first runner should mostly move existing `main.cpp` code without changing
behavior.

Sketch:

```cpp
namespace citlali::pipeline {

struct RunnerOptions {
    std::vector<std::filesystem::path> config_files;
    std::shared_ptr<spdlog::logger> logger;
};

struct ReductionResult {
    int exit_code = EXIT_SUCCESS;
    std::string output_dir;
    std::vector<std::filesystem::path> products;
};

class PipelineRunner {
public:
    explicit PipelineRunner(RunnerOptions options);

    ReductionResult run();

private:
    tula::config::YamlConfig load_and_merge_configs();
    ReductionResult run_science(const tula::config::YamlConfig &);
    ReductionResult run_pointing(const tula::config::YamlConfig &);
    ReductionResult run_beammap(const tula::config::YamlConfig &);
};

}  // namespace citlali::pipeline
```

CLI after extraction:

```cpp
int main(int argc, char *argv[]) {
    try {
        auto cli = parse_cli(argc, argv);
        if (cli.dump_config) {
            print_default_config();
            return EXIT_SUCCESS;
        }
        auto runner = citlali::pipeline::PipelineRunner{cli.runner_options()};
        return runner.run().exit_code;
    } catch (const citlali::error::Error &e) {
        SPDLOG_CRITICAL("{}", e.what());
        return EXIT_FAILURE;
    }
}
```

First extraction rule: move code mechanically. Keep output directory naming,
logger names, config-copy behavior, fruitloops iteration behavior, map
normalization/filtering/coadd order, and return codes unchanged.

## ReductionSession Scaffolding

Once the runner exists, split long-lived runtime state from per-observation and
per-iteration state.

```cpp
namespace citlali::pipeline {

struct ObservationContext {
    std::size_t input_index = 0;
    std::string obsnum;
    std::filesystem::path obs_output_dir;
    double sample_rate_hz = 0.0;
    double processed_sample_rate_hz = 0.0;
};

struct IterationContext {
    int fruit_iter = 0;
    bool learning_source_model_available = false;
    bool save_outputs = true;
};

struct OutputLayout {
    std::filesystem::path root;
    std::filesystem::path reduction_dir;
    std::filesystem::path obsnum_dir(const std::string &obsnum) const;
    std::filesystem::path coadd_dir() const;
};

template <class ReductionEngine>
class ReductionSession {
public:
    ReductionSession(ReductionEngine &engine, SeqIOCoordinator &coordinator,
                     OutputLayout output);

    ReductionResult run();

private:
    void preflight_inputs();
    void run_iteration(IterationContext &iteration);
    void run_observation(IterationContext &, ObservationContext &);
    void finalize_iteration(IterationContext &);
};

}  // namespace citlali::pipeline
```

The initial `ReductionSession` can still use existing `TimeOrderedDataProc` and
`Engine` internals. Its job is to make orchestration explicit before state is
split.

## Error Boundary Scaffolding

Current static inventory found 144 direct exit calls, mostly in library
headers. The target is a typed library error boundary.

Sketch:

```cpp
namespace citlali::error {

enum class ErrorCode {
    config,
    data_io,
    invalid_input,
    invariant,
    unsupported,
    runtime,
};

class Error : public std::runtime_error {
public:
    Error(ErrorCode code, std::string message);
    ErrorCode code() const noexcept;
};

class ConfigError : public Error {
public:
    explicit ConfigError(std::string message);
};

class DataIOError : public Error {
public:
    explicit DataIOError(std::string message);
};

class InvalidInputError : public Error {
public:
    explicit InvalidInputError(std::string message);
};

class InvariantError : public Error {
public:
    explicit InvariantError(std::string message);
};

}  // namespace citlali::error
```

Migration order:

1. CLI boundary catches `citlali::error::Error`, `std::exception`, and fallback
   unknown exceptions.
2. Convert config parse failures first.
3. Convert file/metadata preflight failures next.
4. Convert allocation/product setup failures.
5. Convert RTC/PTC/mapmaking runtime invariant failures only with tests.
6. Leave hot-loop internal checks alone until a performance-neutral strategy is
   clear.

## Header Movement Scaffolding

The inventory found seven natural `.cpp` files currently commented in CMake:

- `src/citlali/core/engine/todproc.cpp`
- `src/citlali/core/engine/kidsproc.cpp`
- `src/citlali/core/engine/engine.cpp`
- `src/citlali/core/mapmaking/wiener_filter.cpp`
- `src/citlali/core/engine/lali.cpp`
- `src/citlali/core/engine/pointing.cpp`
- `src/citlali/core/engine/beammap.cpp`

Proposed movement rule:

- First move only non-template, non-hot functions.
- Enable one `.cpp` source per PR.
- Do not combine source-boundary movement with behavior edits.
- Confirm Unity compile before moving to the next boundary.

Early low-risk candidates should come from config/preflight/output helpers, not
from mapmaking kernels.

## First Implementation PRs After Baseline

1. Add typed error classes and CLI catch branch without changing thrown sites.
2. Add typed config scaffolding with parse-only tests, no pipeline wiring.
3. Extract `PipelineRunner` from `main.cpp` mechanically.
4. Convert first config/preflight `std::exit` paths to typed errors.
5. Move one small non-template implementation group into an existing `.cpp`.

Each step should be reviewed with a manifest comparison whenever runtime paths
are touched.
