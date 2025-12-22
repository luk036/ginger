#include "ginger/rootfinding.hpp"
#include <cmath>
#include <algorithm>

namespace ginger {

Vector2 horner(std::vector<double>& coeffs, int degree, const Vector2& vr) {
    double r = vr.x();
    double q = vr.y();
    
    for (int i = 0; i < degree - 1; ++i) {
        coeffs[i + 1] += coeffs[i] * r;
        coeffs[i + 2] += coeffs[i] * q;
    }
    
    return Vector2(coeffs[degree - 1], coeffs[degree]);
}

Vector2 delta(const Vector2& vA, const Vector2& vr, const Vector2& vp) {
    double r = vr.x(), q = vr.y();
    double p = vp.x(), s = vp.y();
    
    Matrix2 mp(Vector2(s, -p), Vector2(-p * q, p * r + s));
    return mp.mdot(vA) / mp.det();
}

std::pair<Vector2, Vector2> suppress(const Vector2& vA, const Vector2& vA1, 
                                     const Vector2& vri, const Vector2& vrj) {
    Vector2 vp = vri - vrj;
    double r = vri.x(), q = vri.y();
    double p = vp.x(), s = vp.y();
    
    Matrix2 m_adjoint(Vector2(s, -p), Vector2(-p * q, p * r + s));
    double e = m_adjoint.det();
    Vector2 va = m_adjoint.mdot(vA);
    Vector2 vc = vA1 * e - va;
    
    // Adjust vc.y
    double vc_y = vc.y() - va.x() * p;
    Vector2 vc_adj(vc.x(), vc_y);
    
    va = va * e;
    Vector2 va1 = m_adjoint.mdot(vc_adj);
    
    return {va, va1};
}

std::vector<Vector2> initial_guess(const std::vector<double>& coeffs) {
    int degree = static_cast<int>(coeffs.size()) - 1;
    double center = -coeffs[1] / (degree * coeffs[0]);
    
    // Convert coefficients to complex for evaluation
    std::vector<Complex> complex_coeffs(coeffs.begin(), coeffs.end());
    Complex poly_c = horner_eval_f(complex_coeffs, Complex(center, 0));
    double radius = std::pow(std::abs(poly_c), 1.0 / degree);
    double m = center * center + radius * radius;
    
    degree /= 2;
    degree *= 2; // make even
    
    std::vector<Vector2> result;
    result.reserve(degree / 2);
    
    // Simple cosine distribution (simplified from VdCorput)
    for (int i = 1; i < degree; i += 2) {
        double theta = M_PI * i / degree;
        double t = radius * std::cos(theta);
        result.emplace_back(2 * (center + t), -(m + 2 * center * t));
    }
    
    return result;
}

std::tuple<std::vector<Vector2>, int, bool> pbairstow_even(
    const std::vector<double>& coeffs, 
    std::vector<Vector2>& vrs, 
    const Options& options) {
    
    int M = static_cast<int>(vrs.size());
    int degree = static_cast<int>(coeffs.size()) - 1;
    std::vector<bool> converged(M, false);
    
    // Simple round-robin iterator (simplified from Robin class)
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
            
            Vector2 vA1 = horner(coeffs1, degree - 2, vrs[i]);
            tolerance = std::max(tol_i, tolerance);
            
            // Suppress influence of other factors
            for (int j = 0; j < M; ++j) {
                if (i == j) continue;
                std::tie(vA, vA1) = suppress(vA, vA1, vrs[i], vrs[j]);
            }
            
            vrs[i] = vrs[i] - delta(vA, vrs[i], vA1);
        }
        
        if (tolerance < options.tolerance) {
            return {vrs, niter, true};
        }
    }
    
    return {vrs, options.max_iters, false};
}

std::pair<Complex, Complex> find_rootq(const Vector2& vr) {
    double hr = vr.x() / 2.0;
    double d = hr * hr + vr.y();
    
    Complex x1, x2;
    if (d < 0.0) {
        x1 = Complex(hr, std::sqrt(-d));
    } else {
        x1 = Complex(hr + (hr >= 0 ? std::sqrt(d) : -std::sqrt(d)), 0.0);
    }
    
    x2 = Complex(-vr.y() / x1.real(), 0.0);
    if (std::abs(x1.imag()) > 1e-10) {
        x2 = Complex(-vr.y()) / x1;
    }
    
    return {x1, x2};
}

} // namespace ginger