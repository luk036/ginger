#ifndef GINGER_ROOTFINDING_HPP
#define GINGER_ROOTFINDING_HPP

#include <complex>
#include <vector>
#include "vector2.hpp"
#include "matrix2.hpp"

namespace ginger {

using Complex = std::complex<double>;
using Num = double;

class Options {
public:
    int max_iters = 2000;
    double tolerance = 1e-12;
    double tol_ind = 1e-15;
};

// Polynomial evaluation using Horner's method
template<typename T>
T horner_eval_f(const std::vector<T>& coeffs, const T& zval) {
    T result = coeffs[0];
    for (size_t i = 1; i < coeffs.size(); ++i) {
        result = result * zval + coeffs[i];
    }
    return result;
}

// Horner evaluation with intermediate coefficients
template<typename T>
std::pair<T, std::vector<T>> horner_eval(const std::vector<T>& coeffs, const T& zval) {
    std::vector<T> intermediate;
    intermediate.reserve(coeffs.size());
    
    T result = coeffs[0];
    intermediate.push_back(result);
    
    for (size_t i = 1; i < coeffs.size(); ++i) {
        result = result * zval + coeffs[i];
        intermediate.push_back(result);
    }
    
    return {result, intermediate};
}

// Quadratic polynomial evaluation (x² - r·x - q)
Vector2 horner(std::vector<double>& coeffs, int degree, const Vector2& vr);

// Calculate adjustment vector for Bairstow's method
Vector2 delta(const Vector2& vA, const Vector2& vr, const Vector2& vp);

// Zero suppression for Bairstow's method
std::pair<Vector2, Vector2> suppress(const Vector2& vA, const Vector2& vA1, 
                                     const Vector2& vri, const Vector2& vrj);

// Generate initial root estimates for Bairstow's method
std::vector<Vector2> initial_guess(const std::vector<double>& coeffs);

// Parallel Bairstow's method for even-degree polynomials
std::tuple<std::vector<Vector2>, int, bool> pbairstow_even(
    const std::vector<double>& coeffs, 
    std::vector<Vector2>& vrs, 
    const Options& options = Options());

// Solve quadratic equation x² - r·x - q = 0
std::pair<Complex, Complex> find_rootq(const Vector2& vr);

} // namespace ginger

#endif // GINGER_ROOTFINDING_HPP