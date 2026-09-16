// Evidence-only scalar replay of the already selected centered finite stages.
// No masks, coefficients, boundaries, or replacement values are selected here.
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
template <class T> std::vector<T> read(const char *p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("missing input");
  auto bytes = f.tellg();
  if (bytes < 0 || bytes % sizeof(T))
    throw std::runtime_error("shape");
  std::vector<T> v(bytes / sizeof(T));
  f.seekg(0);
  f.read(reinterpret_cast<char *>(v.data()), bytes);
  if (!f)
    throw std::runtime_error("read failed");
  return v;
}
int main(int argc, char **argv) {
  try {
    if (argc != 7)
      throw std::runtime_error(
          "raw-r lp-coeff notch-coeff run-pairs output skip-zero");
    auto raw = read<double>(argv[1]), lp = read<double>(argv[2]),
         notch = read<double>(argv[3]);
    auto runs = read<std::int64_t>(argv[4]);
    bool skip = std::string(argv[6]) == "1";
    if (lp.empty() || lp.size() % 2 != 1 ||
        (!notch.empty() && notch.size() % 2 != 1) || runs.size() % 2)
      throw std::runtime_error("invalid shape");
    const auto lh = lp.size() / 2, nh = notch.size() / 2, half = lh + nh;
    std::vector<double> intermediate(raw.size(), NAN), out(raw.size(), NAN);
    for (std::size_t k = 0; k < runs.size(); k += 2) {
      std::int64_t a = runs[k], b = runs[k + 1];
      if (a < 0 || b < a || b > static_cast<std::int64_t>(raw.size()))
        throw std::runtime_error("range");
      for (auto q = a + static_cast<std::int64_t>(nh);
           q + static_cast<std::int64_t>(nh) < b; ++q) {
        if (notch.empty()) {
          intermediate[q] = raw[q];
          continue;
        }
        double v = 0;
        for (std::size_t j = 0; j < notch.size(); ++j) {
          if (skip && notch[j] == 0)
            continue;
          v = std::fma(notch[j], raw[q + j - nh], v);
        }
        intermediate[q] = v;
      }
      for (auto q = a + static_cast<std::int64_t>(half);
           q + static_cast<std::int64_t>(half) < b; ++q) {
        double v = 0;
        for (std::size_t j = 0; j < lp.size(); ++j) {
          if (skip && lp[j] == 0)
            continue;
          v = std::fma(lp[j], intermediate[q + j - lh], v);
        }
        out[q] = v;
      }
    }
    std::ofstream f(argv[5], std::ios::binary);
    f.write(reinterpret_cast<const char *>(out.data()), out.size() * 8);
    f.close();
    if (!f)
      throw std::runtime_error("write failed");
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
