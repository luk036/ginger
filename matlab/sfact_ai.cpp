#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

using ComplexNum = std::complex<double>;

// Function to calculate the Euclidean norm (magnitude) of a complex number
double magnitude(const ComplexNum &c) {
  return std::sqrt(c.real() * c.real() + c.imag() * c.imag());
}

std::vector<ComplexNum> lejaOrder(std::vector<ComplexNum> points) {
  // Check if input is empty
  if (points.empty()) {
    throw std::runtime_error("Input must be a non-empty vector of points.");
  }

  // Sort points by magnitude initially
  std::sort(points.begin(), points.end(),
            [](const ComplexNum &a, const ComplexNum &b) {
              return magnitude(a) < magnitude(b);
            });

  std::vector<ComplexNum> lejaOrderedPoints;
  lejaOrderedPoints.push_back(
      points.front());          // Start with the smallest magnitude point
  points.erase(points.begin()); // Remove this point from further consideration

  while (!points.empty()) {
    // Find the point with the maximum minimum distance to the current Leja
    // sequence
    size_t nextIdx = 0;
    double maxMinDist = 0.0;
    for (size_t i = 0; i < points.size(); ++i) {
      double minDist = std::numeric_limits<double>::max();
      for (const auto &p : lejaOrderedPoints) {
        minDist = std::min(minDist, magnitude(points[i] - p));
      }
      if (minDist > maxMinDist) {
        maxMinDist = minDist;
        nextIdx = i;
      }
    }

    // Append this point to the Leja ordered sequence
    lejaOrderedPoints.push_back(points[nextIdx]);
    // Remove this point from further consideration
    points.erase(points.begin() + nextIdx);
  }

  return lejaOrderedPoints;
}

std::vector<std::complex<double>> seprts(const std::vector<double> &p) {
  constexpr double SN = 0.0001;
  std::vector<std::complex<double>> rts;
  rts.reserve(p.size());
  for (size_t i = 0; i < p.size(); ++i) {
    rts.emplace_back(p[i], 0.0); // Assuming real polynomial coefficients
  }

  std::vector<std::complex<double>> irts, orts;
  for (const auto &rt : rts) {
    if (std::abs(rt) < (1.0 - SN)) {
      irts.push_back(rt);
    } else if ((std::abs(rt) >= (1.0 - SN)) && (std::abs(rt) <= (1.0 + SN))) {
      orts.push_back(rt);
    }
  }

  if (orts.size() % 2 != 0) {
    std::cout << "Sorry, but there is a problem (1) in seprts function"
              << std::endl;
    return {};
  }

  // Sorting roots on the unit circle by angle is not trivial and is omitted
  // here. In practice, you would need to implement a comparison function for
  // angles.

  orts.resize(
      orts.size() /
      2); // Simplified logic to select every second root, assuming sorting

  irts.insert(irts.end(), orts.begin(), orts.end());
  return irts;
}

std::pair<std::vector<double>, std::vector<std::complex<double>>>
sfact(const std::vector<double> &p) {
  if (p.size() == 1) {
    return {p, {}};
  }

  auto r = seprts(p);
  auto r_leja_ordered = lejaOrder(std::vector<ComplexNum>(r.begin(), r.end()));
  std::vector<double> h;
  for (const auto &re : r_leja_ordered) {
    h.push_back(re);
  }

  double max_p = *std::max_element(p.begin(), p.end());
  double norm_factor =
      std::sqrt(max_p / std::inner_product(
                            h.begin(), h.end(), h.begin(), 0.0, std::plus<>(),
                            [](double a, double b) { return a * a + b * b; }));
  for (auto &coeff : h) {
    coeff *= norm_factor;
  }

  return std::make_pair(h, r_leja_ordered);
}
