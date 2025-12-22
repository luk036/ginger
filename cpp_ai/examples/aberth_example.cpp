#include <iostream>
#include <vector>
#include <complex>
#include "ginger/aberth.hpp"

int main() {
    using namespace ginger;
    
    // Example polynomial: x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
    std::vector<double> coeffs = {1.0, -6.0, 11.0, -6.0};
    
    // Generate initial guesses
    auto initial_guesses = initial_aberth(coeffs);
    
    std::cout << "Polynomial: x^3 - 6x^2 + 11x - 6" << std::endl;
    std::cout << "Initial guesses:" << std::endl;
    for (const auto& z : initial_guesses) {
        std::cout << "  " << z << std::endl;
    }
    
    // Run Aberth's method
    Options options;
    options.tolerance = 1e-8;
    
    auto [roots, iterations, found] = aberth(coeffs, initial_guesses, options);
    
    std::cout << "\nResults:" << std::endl;
    std::cout << "Converged: " << (found ? "Yes" : "No") << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << "Roots:" << std::endl;
    for (const auto& root : roots) {
        std::cout << "  " << root << std::endl;
    }
    
    return 0;
}