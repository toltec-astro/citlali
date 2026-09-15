// Bounded diagnostic adapter: reuse existing pointing, PCA and ordinary
// Stokes-I map projection. No production route or downstream algorithm change.
#define main unused_identity_acceptance_main
#include "identity_route_acceptance.cpp"
#undef main
#include <citlali/core/mapmaking/naive_mm.h>
#include <citlali/core/pipeline/map_buffer_allocation.h>
#include <citlali/core/timestream/ptc/clean.h>
namespace {
Eigen::MatrixXd matrix(const fs::path &p, Eigen::Index n, Eigen::Index d) {
  require(n > 0 && d > 0 && fs::file_size(p) == std::uintmax_t(n * d) * 8,
          "matrix cardinality mismatch");
  std::ifstream f(p, std::ios::binary);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> m(n,
                                                                           d);
  f.read(reinterpret_cast<char *>(m.data()), n * d * 8);
  require(bool(f) && m.allFinite(),
          "nonfinite or incomplete diagnostic matrix");
  return m;
}
void save(const fs::path &p, const Eigen::MatrixXd &m) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> r = m;
  std::ofstream f(p, std::ios::binary);
  f.write(reinterpret_cast<const char *>(r.data()), r.size() * 8);
  f.close();
  require(bool(f), "diagnostic output failed");
}
fs::path checked(const YAML::Node &n) {
  fs::path p = n["path"].as<std::string>();
  require(citlali::utils::sha256_file(p) == n["sha256"].as<std::string>(),
          "changed input binding");
  return p;
}
} // namespace
int main(int argc, char **argv) {
  try {
    require(argc == 3, "expected bound specification and new output directory");
    auto [logger, logs] = configure_logging();
    const auto c = YAML::LoadFile(argv[1]);
    fs::path out = argv[2];
    require(!fs::exists(out), "preserve diagnostic output");
    fs::create_directories(out);
    const auto mode = c["mode"].as<std::string>();
    if (mode == "geometry") {
      const auto v =
          apt::verify_bundle_filesystem(checked(c["manifest"]), true);
      require(v.apt.observation.observation == 152390,
              "unexpected observation");
      auto time =
          matrix(checked(c["time"]), c["rows"].as<int>(), 1).col(0).eval();
      netCDF::NcFile file(checked(c["telescope"]).string(),
                          netCDF::NcFile::read);
      pipeline::NativeTelescopeData raw;
      for (const auto *key : {"TelTime", "TelAzAct", "TelElAct", "ActParAng",
                              "SourceRaAct", "SourceDecAct"}) {
        auto var = file.getVar(std::string("Data.TelescopeBackend.") + key);
        require(!var.isNull() && var.getDimCount() == 1,
                "missing telescope vector");
        Eigen::VectorXd x(var.getDim(0).getSize());
        var.getVar(x.data());
        raw[key] = std::move(x);
      }
      require((raw.at("TelTime").tail(raw.at("TelTime").size() - 1) -
               raw.at("TelTime").head(raw.at("TelTime").size() - 1))
                      .maxCoeff() < .1,
              "telescope gap not admitted");
      const auto telescope = pipeline::evaluate_raw_telescope_trajectory_at(
          pipeline::RawTelescopeTrajectory(raw), time);
      auto tel = telescope;
      double ra[2], dec[2];
      file.getVar("Header.Source.Ra").getVar(ra);
      file.getVar("Header.Source.Dec").getVar(dec);
      Eigen::VectorXd xp(time.size()), yp(time.size());
      engine_utils::gnomonic_projection(
          tel.at("SourceRaAct"), tel.at("SourceDecAct"), ra[0], dec[0], xp, yp);
      tel["ra_phys"] = xp;
      tel["dec_phys"] = yp;
      pipeline::NativePointingOffsetsArcsec source;
      for (auto key : {"az", "alt"}) {
        Eigen::VectorXd x(2);
        x << c["offsets"][key][0].as<double>(),
            c["offsets"][key][1].as<double>();
        source[key] = x;
      }
      Eigen::VectorXd ot(2);
      ot << c["offsets"]["unix"][0].as<double>(),
          c["offsets"]["unix"][1].as<double>();
      const auto offsets =
          pipeline::NativePointingOffsetModel(source, ot).evaluate_at(time);
      Eigen::MatrixXd td(time.size(), 6);
      int j = 0;
      for (auto key : {"TelElAct", "ActParAng", "ra_phys", "dec_phys"})
        td.col(j++) = tel.at(key);
      td.col(4) = offsets.at("az");
      td.col(5) = offsets.at("alt");
      save(out / "telescope.f64", td);
      Eigen::MatrixXd fields = Eigen::MatrixXd::Constant(491, 6, NAN);
      for (const auto &row : v.apt.rows)
        if (row.network == 12) {
          require(row.channel >= 0 && row.channel < 491,
                  "APT channel out of range");
          int k = 0;
          for (auto key : {"x_t", "y_t", "flxscale", "sens", "flag"}) {
            const auto &x = row.fields.at(key);
            double value = NAN;
            if (const auto *p = std::get_if<double>(&x))
              value = *p;
            if (const auto *p = std::get_if<std::int64_t>(&x))
              value = double(*p);
            fields(row.channel, k++) = value;
          }
          fields(row.channel, 5) = double(row.uid);
        }
      save(out / "apt-declared.f64", fields);
      std::ofstream excluded(out / "unavailable-detectors.txt");
      for (int d = 0; d < 491; ++d) {
        if (!fields.row(d).allFinite() || fields(d, 2) <= 0 ||
            fields(d, 3) <= 0) {
          excluded
              << d
              << " missing or unusable diagnostic pointing/calibration/weight "
                 "field; excluded, no production flag change\n";
          fields(d, 0) = 0;
          fields(d, 1) = 0;
          fields(d, 2) = 1;
          fields(d, 3) = 1;
          fields(d, 4) = 1;
        }
      }
      excluded.close();
      require(fields.allFinite(), "APT identity missing");
      save(out / "apt.f64", fields);
      // Export own-detector tangent-plane coordinates in radians, actual native
      // rows.
      for (int d = 0; d < 491; ++d) {
        auto tcopy = tel, ocopy = offsets;
        auto [lat, lon] = engine_utils::calc_det_pointing(
            tcopy, fields(d, 0), fields(d, 1), "radec", ocopy, "array");
        Eigen::MatrixXd xy(time.size(), 2);
        xy.col(0) = lon;
        xy.col(1) = lat;
        save(out / ("pointing-" + std::to_string(d) + ".f64"), xy);
      }
    } else if (mode == "downstream") {
      const auto aptm = matrix(checked(c["apt"]), 491, 6);
      for (const auto &job : c["jobs"]) {
        const auto name = job["name"].as<std::string>();
        require(name.find('/') == std::string::npos, "invalid job name");
        auto dir = out / name;
        fs::create_directories(dir);
        const auto n = job["rows"].as<int>();
        auto data = matrix(checked(job["data"]), n, 491);
        const auto mask = matrix(checked(job["good"]), n, 491);
        require(((mask.array() == 0) || (mask.array() == 1)).all(),
                "malformed paired support");
        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> flags =
            (mask.array() == 0);
        Eigen::VectorXd af = aptm.col(4);
        for (int d = 0; d < 491; ++d)
          if (af(d) != 0)
            flags.col(d).setConstant(true);
        if (job["pca"].as<bool>()) {
          // Same masked-mean precondition as PTCProc::subtract_mean; no
          // replacement or restored mean. Actual Cleaner owns
          // covariance/eigenvalue subtraction.
          for (int d = 0; d < 491; ++d) {
            double sum = 0;
            int count = 0;
            for (int i = 0; i < n; ++i)
              if (!flags(i, d)) {
                sum += data(i, d);
                ++count;
              }
            if (count)
              data.col(d).array() -= sum / count;
          }
          timestream::Cleaner cleaner;
          cleaner.stddev_limit = 0;
          cleaner.n_calc = 0;
          auto [eval, evec] =
              cleaner.calc_eig_values<timestream::Cleaner::EigenBackend>(
                  data, flags, af, 10);
          Eigen::MatrixXd cleaned(n, 491);
          cleaner.remove_eig_values<timestream::Cleaner::EigenBackend>(
              data, flags, eval, evec, cleaned, 10, -1, "nw", 12, 2);
          data = std::move(cleaned);
          save(dir / "eigenvalues.f64", eval);
          save(dir / "cut-eigenvectors.f64", evec.leftCols(10));
        }
        save(dir / "cleaned.f64", data);
        const auto tel = matrix(checked(job["telescope"]), n, 6);
        std::map<std::string, Eigen::VectorXd> apt;
        for (auto key : {"x_t", "y_t", "flag", "uid", "array"})
          apt[key] = Eigen::VectorXd::Zero(491);
        apt["x_t"] = aptm.col(0);
        apt["y_t"] = aptm.col(1);
        apt["flag"] = aptm.col(4);
        apt["uid"] = aptm.col(5);
        apt["array"].setConstant(2);
        for (const auto &support : job["map_supports"]) {
          timestream::TCData<timestream::TCDataKind::PTC, Eigen::MatrixXd> in;
          in.scans.data = data;
          auto gm = matrix(checked(support["good"]), n, 491);
          require(((gm.array() == 0) || (gm.array() == 1)).all() &&
                      (gm.array() <= mask.array()).all(),
                  "map support adds excluded samples");
          in.flags.data = (gm.array() == 0);
          in.weights.data = aptm.col(3).array().square().inverse();
          in.index.data = 0;
          int k = 0;
          for (auto key : {"TelElAct", "ActParAng", "ra_phys", "dec_phys"})
            in.tel_data.data[key] = tel.col(k++);
          in.pointing_offsets_arcsec.data["az"] = tel.col(4);
          in.pointing_offsets_arcsec.data["alt"] = tel.col(5);
          mapmaking::MapBuffer omb{"omb"}, cmb{"cmb"};
          omb.n_rows = c["map_pixels"].as<int>();
          omb.n_cols = omb.n_rows;
          omb.pixel_size_rad = c["pixel_arcsec"].as<double>() * ASEC_TO_RAD;
          omb.map_grouping = "array";
          omb.parallel_policy = "seq";
          pipeline::allocate_map_matrices(omb, 1, false, false, true, false,
                                          "conditional experiment", false);
          Eigen::VectorXi indices = Eigen::VectorXi::Zero(491);
          std::string axes = "radec";
          mapmaking::NaiveMapmaker mm;
          mm.run_polarization = false;
          mm.populate_maps_naive(in, omb, cmb, indices, axes, apt,
                                 c["output_hz"].as<double>(), true, false);
          auto label = support["name"].as<std::string>();
          require(label.find('/') == std::string::npos, "bad support name");
          save(dir / (label + "-sum.f64"), omb.signal[0]);
          save(dir / (label + "-weight.f64"), omb.weight[0]);
          save(dir / (label + "-coverage.f64"), omb.coverage[0]);
        }
      }
    } else
      throw std::invalid_argument("unknown diagnostic mode");
    std::ofstream r(out / "receipt.json");
    r << "{\"status\":\"PASS-conditional-diagnostic\",\"compiled_revision\":\""
      << CITLALI_GIT_REVISION << "\",\"production\":false}";
    r.close();
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
