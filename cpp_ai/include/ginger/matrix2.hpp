#ifndef GINGER_MATRIX2_HPP
#define GINGER_MATRIX2_HPP

#include "vector2.hpp"

namespace ginger {

class Matrix2 {
private:
    Vector2 x_;
    Vector2 y_;

public:
    Matrix2(const Vector2& x, const Vector2& y) : x_(x), y_(y) {}

    const Vector2& x() const { return x_; }
    const Vector2& y() const { return y_; }

    Vector2 mdot(const Vector2& rhs) const {
        return Vector2(x_.dot(rhs), y_.dot(rhs));
    }

    double det() const {
        double a11 = x_.x(), a12 = x_.y();
        double a21 = y_.x(), a22 = y_.y();
        return a11 * a22 - a12 * a21;
    }

    Matrix2 operator/(double alpha) const {
        return Matrix2(x_ / alpha, y_ / alpha);
    }

    friend std::ostream& operator<<(std::ostream& os, const Matrix2& m) {
        os << "Matrix2(" << m.x_ << ", " << m.y_ << ")";
        return os;
    }
};

} // namespace ginger

#endif // GINGER_MATRIX2_HPP