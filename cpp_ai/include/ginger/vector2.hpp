#ifndef GINGER_VECTOR2_HPP
#define GINGER_VECTOR2_HPP

#include <cmath>
#include <iostream>
#include <string>

namespace ginger {

class Vector2 {
private:
    double x_;
    double y_;

public:
    Vector2(double x = 0.0, double y = 0.0) : x_(x), y_(y) {}

    double x() const { return x_; }
    double y() const { return y_; }

    double dot(const Vector2& rhs) const {
        return x_ * rhs.x_ + y_ * rhs.y_;
    }

    Vector2& operator-=(const Vector2& rhs) {
        x_ -= rhs.x_;
        y_ -= rhs.y_;
        return *this;
    }

    Vector2 operator-(const Vector2& rhs) const {
        return Vector2(x_ - rhs.x_, y_ - rhs.y_);
    }

    Vector2& operator*=(double alpha) {
        x_ *= alpha;
        y_ *= alpha;
        return *this;
    }

    Vector2 operator*(double alpha) const {
        return Vector2(x_ * alpha, y_ * alpha);
    }

    Vector2 operator/(double alpha) const {
        return Vector2(x_ / alpha, y_ / alpha);
    }

    friend Vector2 operator*(double alpha, const Vector2& v) {
        return v * alpha;
    }

    friend std::ostream& operator<<(std::ostream& os, const Vector2& v) {
        os << "<" << v.x_ << ", " << v.y_ << ">";
        return os;
    }

    std::string to_string() const {
        return "<" + std::to_string(x_) + ", " + std::to_string(y_) + ">";
    }
};

} // namespace ginger

#endif // GINGER_VECTOR2_HPP
