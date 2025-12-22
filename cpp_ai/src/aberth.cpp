#include "ginger/aberth.hpp"
#include <cmath>
#include <algorithm>
#include <thread>
#include <future>
#include <vector>

namespace ginger {

Complex horner_backward(std::vector<double>& coeffs1, int degree, const Complex& alpha) {
    // Convert to complex for the computation
    std::vector<Complex> complex_coeffs(coeffs1.begin(), coeffs1.end());
    
    for (int i = 2; i < degree + 2; ++i) {
        complex_coeffs[complex_coeffs.size() - i] -= complex_coeffs[complex_coeffs.size() - (i - 1)];
        complex_coeffs[complex_coeffs.size() - i] /= -alpha;
    }
    
    // Convert result back to double coefficients (taking real part)
    for (size_t i = 0; i < coeffs1.size(); ++i) {
        coeffs1[i] = complex_coeffs[i].real();
    }
    
    return complex_coeffs[complex_coeffs.size() - (degree + 1)];
}

std::vector<Complex> initial_aberth(const std::vector<double>& coeffs) {
    int degree = static_cast<int>(coeffs.size()) - 1;
    double center = -coeffs[1] / (degree * coeffs[0]);
    
    // Convert coefficients to complex for evaluation
    std::vector<Complex> complex_coeffs(coeffs.begin(), coeffs.end());
    Complex poly_c = horner_eval_f(complex_coeffs, Complex(center, 0));
    Complex radius = std::pow(-poly_c, 1.0 / degree);
    
    std::vector<Complex> result;
    result.reserve(degree);
    
    // Simple circle distribution (simplified from Circle generator)
    for (int i = 0; i < degree; ++i) {
        double angle = 2.0 * M_PI * i / degree;
        double x = std::cos(angle);
        double y = std::sin(angle);
        result.push_back(center + radius * Complex(x, y));
    }
    
    return result;
}

std::vector<Complex> initial_aberth_orig(const std::vector<double>& coeffs) {
    int degree = static_cast<int>(coeffs.size()) - 1;
    double center = -coeffs[1] / (degree * coeffs[0]);
    
    // Convert coefficients to complex for evaluation
    std::vector<Complex> complex_coeffs(coeffs.begin(), coeffs.end());
    Complex poly_c = horner_eval_f(complex_coeffs, Complex(center, 0));
    Complex radius = std::pow(-poly_c, 1.0 / degree);
    
    std::vector<Complex> result;
    result.reserve(degree);
    
    double k = 2.0 * M_PI / degree;
    for (int i = 0; i < degree; ++i) {
        double theta = k * (0.25 + i);
        result.push_back(center + radius * Complex(std::cos(theta), std::sin(theta)));
    }
    
    return result;
}

std::tuple<std::vector<Complex>, int, bool> aberth(
    const std::vector<double>& coeffs, 
    std::vector<Complex>& zs, 
    const Options& options) {
    
    // Convert coefficients to complex once
    std::vector<Complex> complex_coeffs(coeffs.begin(), coeffs.end());
    
    for (int niter = 0; niter < options.max_iters; ++niter) {
        double tolerance = 0.0;
        
        for (size_t i = 0; i < zs.size(); ++i) {
            Complex zi = zs[i];
            auto [p_eval, coeffs1] = horner_eval(complex_coeffs, zi);
            double tol_i = std::abs(p_eval);
            
            auto [p1_eval, _] = horner_eval<Complex>(coeffs1, zi);
            tolerance = std::max(tol_i, tolerance);
            
            for (size_t j = 0; j < zs.size(); ++j) {
                if (i != j) {
                    p1_eval -= p_eval / (zi - zs[j]);
                }
            }
            
            zs[i] = zi - p_eval / p1_eval;
        }
        
        if (tolerance < options.tolerance) {
            return {zs, niter, true};
        }
    }
    
    return {zs, options.max_iters, false};
}

std::tuple<std::vector<Complex>, int, bool> aberth_mt(
    const std::vector<double>& coeffs, 
    std::vector<Complex>& zs, 
    const Options& options) {
    
    auto aberth_job = [&](int i) -> std::tuple<double, int, Complex> {
        Complex zi = zs[i];
        auto [p_eval, coeffs1] = horner_eval<Complex>(coeffs, zi);
        double tol_i = std::abs(p_eval);
        
        auto [p1_eval, _] = horner_eval<Complex>(coeffs1, zi);
        
        for (size_t j = 0; j < zs.size(); ++j) {
            if (static_cast<int>(i) != static_cast<int>(j)) {
                p1_eval -= p_eval / (zi - zs[j]);
            }
        }
        
        zi -= p_eval / p1_eval;
        return {tol_i, i, zi};
    };
    
    for (int niter = 0; niter < options.max_iters; ++niter) {
        double tolerance = 0.0;
        std::vector<std::future<std::tuple<double, int, Complex>>> futures;
        
        for (size_t i = 0; i < zs.size(); ++i) {
            futures.push_back(std::async(std::launch::async, aberth_job, i));
        }
        
        for (auto& future : futures) {
            auto [tol_i, i, zi] = future.get();
            if (tol_i > tolerance) {
                tolerance = tol_i;
            }
            zs[i] = zi;
        }
        
        if (tolerance < options.tolerance) {
            return {zs, niter, true};
        }
    }
    
    return {zs, options.max_iters, false};
}

std::vector<Complex> initial_aberth_autocorr(const std::vector<double>& coeffs) {
    int degree = static_cast<int>(coeffs.size()) - 1; // assume even
    double center = -coeffs[1] / (degree * coeffs[0]);
    
    // Convert coefficients to complex for evaluation
    std::vector<Complex> complex_coeffs(coeffs.begin(), coeffs.end());
    Complex poly_c = horner_eval_f(complex_coeffs, Complex(center, 0));
    Complex radius = std::pow(-poly_c, 1.0 / degree);
    
    if (std::abs(radius) > 1.0) {
        radius = 1.0 / radius;
    }
    
    std::vector<Complex> result;
    result.reserve(degree / 2);
    
    // Simple circle distribution for half the roots
    for (int i = 0; i < degree / 2; ++i) {
        double angle = 2.0 * M_PI * i / (degree / 2);
        double x = std::cos(angle);
        double y = std::sin(angle);
        result.push_back(center + radius * Complex(x, y));
    }
    
    return result;
}

std::vector<Complex> initial_aberth_autocorr_orig(const std::vector<double>& coeffs) {
    int degree = static_cast<int>(coeffs.size()) - 1;
    double center = -coeffs[1] / (degree * coeffs[0]);
    
    // Convert coefficients to complex for evaluation
    std::vector<Complex> complex_coeffs(coeffs.begin(), coeffs.end());
    Complex poly_c = horner_eval_f(complex_coeffs, Complex(center, 0));
    double radius = std::pow(std::abs(poly_c), 1.0 / degree);
    
    if (std::abs(radius) > 1.0) {
        radius = 1.0 / radius;
    }
    
    degree /= 2;
    std::vector<Complex> result;
    result.reserve(degree);
    
    double k = 2.0 * M_PI / degree;
    for (int i = 0; i < degree; ++i) {
        double theta = k * (0.25 + i);
        result.push_back(center + radius * Complex(std::cos(theta), std::sin(theta)));
    }
    
    return result;
}

std::tuple<std::vector<Complex>, int, bool> aberth_autocorr(
    const std::vector<double>& coeffs, 
    std::vector<Complex>& zs, 
    const Options& options) {
    
    for (int niter = 0; niter < options.max_iters; ++niter) {
        double tolerance = 0.0;
        
        for (size_t i = 0; i < zs.size(); ++i) {
            Complex zi = zs[i];
            auto [p_eval, coeffs1] = horner_eval<Complex>(coeffs, zi);
            double tol_i = std::abs(p_eval);
            
            auto [p1_eval, _] = horner_eval<Complex>(coeffs1, zi);
            tolerance = std::max(tol_i, tolerance);
            
            for (size_t j = 0; j < zs.size(); ++j) {
                if (i == j) continue;
                p1_eval -= p_eval / (zi - zs[j]);
                p1_eval -= p_eval / (zi - 1.0 / std::conj(zs[j]));
            }
            
            zs[i] = zi - p_eval / p1_eval;
        }
        
        if (tolerance < options.tolerance) {
            return {zs, niter, true};
        }
    }
    
    return {zs, options.max_iters, false};
}

std::tuple<double, int, Complex> aberth_autocorr_job(
    const std::vector<double>& coeffs,
    int i,
    const std::vector<Complex>& zsc) {
    
    Complex zi = zsc[i];
    auto [p_eval, coeffs1] = horner_eval<Complex>(coeffs, zi);
    double tol_i = std::abs(p_eval);
    
    auto [p1_eval, _] = horner_eval<Complex>(coeffs1, zi);
    
    for (size_t j = 0; j < zsc.size(); ++j) {
        if (static_cast<int>(i) != static_cast<int>(j)) {
            p1_eval -= p_eval / (zi - zsc[j]);
            p1_eval -= p_eval / (zi - 1.0 / std::conj(zsc[j]));
        }
    }
    
    zi -= p_eval / p1_eval;
    return {tol_i, i, zi};
}

std::tuple<std::vector<Complex>, int, bool> aberth_autocorr_mt(
    const std::vector<double>& coeffs, 
    std::vector<Complex>& zs, 
    const Options& options) {
    
    for (int niter = 0; niter < options.max_iters; ++niter) {
        double tolerance = 0.0;
        std::vector<std::future<std::tuple<double, int, Complex>>> futures;
        
        for (size_t i = 0; i < zs.size(); ++i) {
            futures.push_back(std::async(std::launch::async, aberth_autocorr_job, coeffs, i, zs));
        }
        
        for (auto& future : futures) {
            auto [tol_i, i, zi] = future.get();
            if (tol_i > tolerance) {
                tolerance = tol_i;
            }
            zs[i] = zi;
        }
        
        if (tolerance < options.tolerance) {
            return {zs, niter, true};
        }
    }
    
    return {zs, options.max_iters, false};
}

} // namespace ginger