#include "ginger/autocorr.hpp"
#include <cmath>
#include <algorithm>

namespace ginger {

std::vector<Vector2> initial_autocorr(const std::vector<double>& coeffs) {
    int degree = static_cast<int>(coeffs.size()) - 1;
    double radius = std::pow(std::abs(coeffs.back()), 1.0 / degree);
    
    if (radius < 1.0) {
        radius = 1.0 / radius;
    }
    
    degree /= 2;
    double k = M_PI / degree;
    
    double m = radius * radius;
    std::vector<Vector2> result;
    result.reserve(degree / 2);
    
    for (int i = 1; i < degree; i += 2) {
        result.emplace_back(2 * radius * std::cos(k * i), -m);
    }
    
    return result;
}

std::tuple<std::vector<Vector2>, int, bool> pbairstow_autocorr(
    const std::vector<double>& coeffs, 
    std::vector<Vector2>& vrs, 
    const Options& options) {
    
    int M = static_cast<int>(vrs.size());
    int degree = static_cast<int>(coeffs.size()) - 1;
    std::vector<bool> converged(M, false);
    
    for (int niter = 0; niter < options.max_iters; ++niter) {
        double tolerance = 0.0;
        
        for (int i = 0; i < M; ++i) {
            if (converged[i]) continue;
            
            std::vector<double> coeffs1 = coeffs;
            Vector2 vA = horner(coeffs1, degree, vrs[i]);
            
            double tol_i = std::max(std::abs(vA.x()), std::abs(vA.y()));
            if (tol_i < options.tol_ind) {
                converged[i] = true;
                continue;
            }
            
            tolerance = std::max(tolerance, tol_i);
            Vector2 vA1 = horner(coeffs1, degree - 2, vrs[i]);
            
            // Suppress influence of other factors and their reciprocals
            for (int j = 0; j < M; ++j) {
                if (i == j) continue;
                
                Vector2 vrj = vrs[j];
                std::tie(vA, vA1) = suppress(vA, vA1, vrs[i], vrj);
                
                // Handle reciprocal roots
                Vector2 vrn = Vector2(-vrj.x(), 1.0) / vrj.y();
                std::tie(vA, vA1) = suppress(vA, vA1, vrs[i], vrn);
            }
            
            vrs[i] = vrs[i] - delta(vA, vrs[i], vA1);
        }
        
        if (tolerance < options.tolerance) {
            return {vrs, niter, true};
        }
    }
    
    return {vrs, options.max_iters, false};
}

Vector2 extract_autocorr(const Vector2& vr) {
    double r = vr.x(), q = vr.y();
    double hr = r / 2.0;
    double d = hr * hr + q;
    
    if (d < 0.0) {
        if (q < -1.0) {
            return Vector2(-r, 1.0) / q;
        }
    } else {
        double a1 = hr + (hr >= 0.0 ? std::sqrt(d) : -std::sqrt(d));
        double a2 = -q / a1;
        
        if (std::abs(a1) > 1.0) {
            a1 = 1.0 / a1;
            if (std::abs(a2) > 1.0) {
                a2 = 1.0 / a2;
            }
            return Vector2(a1 + a2, -a1 * a2);
        } else if (std::abs(a2) > 1.0) {
            a2 = 1.0 / a2;
            return Vector2(a1 + a2, -a1 * a2);
        }
    }
    
    return vr;
}

} // namespace ginger