#ifndef GINGER_AUTOCORR_HPP
#define GINGER_AUTOCORR_HPP

#include <vector>
#include "vector2.hpp"
#include "rootfinding.hpp"

namespace ginger {

// Calculates initial guesses for autocorrelation roots using coefficient analysis
std::vector<Vector2> initial_autocorr(const std::vector<double>& coeffs);

// Implements Bairstow's method for polynomial root finding with autocorrelation
std::tuple<std::vector<Vector2>, int, bool> pbairstow_autocorr(
    const std::vector<double>& coeffs, 
    std::vector<Vector2>& vrs, 
    const Options& options = Options());

// Normalizes quadratic factors to ensure roots within unit circle
Vector2 extract_autocorr(const Vector2& vr);

} // namespace ginger

#endif // GINGER_AUTOCORR_HPP