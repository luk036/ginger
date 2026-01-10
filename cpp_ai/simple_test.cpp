#include <iostream>
#include "ginger/vector2.hpp"
#include "ginger/matrix2.hpp"

int main() {
    using namespace ginger;

    // Test Vector2
    std::cout << "Testing Vector2..." << std::endl;
    Vector2 v1(1.0, 2.0);
    Vector2 v2(3.0, 4.0);

    std::cout << "v1 = " << v1 << std::endl;
    std::cout << "v2 = " << v2 << std::endl;
    std::cout << "v1.dot(v2) = " << v1.dot(v2) << std::endl;
    std::cout << "v1 - v2 = " << (v1 - v2) << std::endl;
    std::cout << "v1 * 2.0 = " << (v1 * 2.0) << std::endl;
    std::cout << "v1 / 2.0 = " << (v1 / 2.0) << std::endl;

    // Test Matrix2
    std::cout << "\nTesting Matrix2..." << std::endl;
    Matrix2 m(v1, v2);
    std::cout << "m = " << m << std::endl;
    std::cout << "m.det() = " << m.det() << std::endl;
    std::cout << "m.mdot(v1) = " << m.mdot(v1) << std::endl;

    std::cout << "\nAll basic tests passed!" << std::endl;
    return 0;
}
