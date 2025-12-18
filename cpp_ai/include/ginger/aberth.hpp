#ifndef GINGER_ABERTH_HPP
#define GINGER_ABERTH_HPP

#include <complex>
#include <vector>
#include "rootfinding.hpp"

namespace ginger {

using Complex = std::complex<double>;

// Backward polynomial evaluation using Horner's method for root refinement
Complex horner_backward(std::vector<double>& coeffs1, int degree, const Complex& alpha);

// Generate initial root guesses using geometric distribution
std::vector<Complex> initial_aberth(const std::vector<double>& coeffs);

// Original implementation of initial guess generation using trigonometric distribution
std::vector<Complex> initial_aberth_orig(const std::vector<double>& coeffs);

// Core implementation of Aberth's root-finding algorithm
std::tuple<std::vector<Complex>, int, bool> aberth(
    const std::vector<double>& coeffs, 
    std::vector<Complex>& zs, 
    const Options& options = Options());

// Multithreaded implementation of Aberth's method
std::tuple<std::vector<Complex>, int, bool> aberth_mt(
    const std::vector<double>& coeffs, 
    std::vector<Complex>& zs, 
    const Options& options = Options());

// Generate initial guesses for autocorrelation polynomials
std::vector<Complex> initial_aberth_autocorr(const std::vector<double>& coeffs);

// Original trigonometric implementation for autocorrelation polynomials
std::vector<Complex> initial_aberth_autocorr_orig(const std::vector<double>& coeffs);

// Aberth's method variant for autocorrelation polynomials
std::tuple<std::vector<Complex>, int, bool> aberth_autocorr(
    const std::vector<double>& coeffs, 
    std::vector<Complex>& zs, 
    const Options& options = Options());

// Worker function for multithreaded autocorrelation Aberth method
std::tuple<double, int, Complex> aberth_autocorr_job(
    const std::vector<double>& coeffs,
    int i,
    const std::vector<Complex>& zsc);

// Multithreaded version of autocorrelation Aberth's method
std::tuple<std::vector<Complex>, int, bool> aberth_autocorr_mt(
    const std::vector<double>& coeffs, 
    std::vector<Complex>& zs, 
    const Options& options = Options());

} // namespace ginger

#endif // GINGER_ABERTH_HPP