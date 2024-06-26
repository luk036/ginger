extern crate num_complex;
use num_complex::Complex;
use rand::Rng;

fn horner_eval_f(coeffs: &[f64], zval: f64) -> f64 {
    // Placeholder implementation for demonstration purposes
    coeffs.iter().rev().enumerate().fold(0.0, |acc, (i, &coeff)| acc + coeff * zval.powi(i as i32))
}

struct Circle {
    rng: rand::rngs::ThreadRng,
}

impl Circle {
    fn new() -> Self {
        Circle { rng: rand::thread_rng() }
    }

    fn reseed(&mut self, seed: u64) {
        self.rng = rand::rngs::StdRng::seed_from_u64(seed).into();
    }

    fn pop(&mut self) -> (f64, f64) {
        // Generate random points on a unit circle
        let angle = self.rng.gen_range(0.0..std::f64::consts::TAU);
        (angle.cos(), angle.sin())
    }
}

fn initial_aberth(coeffs: &[f64]) -> Vec<Complex<f64>> {
    let degree = coeffs.len() - 1;
    let center = -coeffs[1] / (coeffs[0] * degree as f64);
    let p_center = horner_eval_f(coeffs, center);
    let re = Complex::from_polar(&-p_center, 1.0 / degree as f64);
    let mut c_gen = Circle::new();
    c_gen.reseed(1);
    let mut zs = Vec::with_capacity(degree);
    for _ in 0..degree {
        let (x, y) = c_gen.pop();
        zs.push(center + re * Complex::new(x, y));
    }
    zs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_aberth() {
        let h = vec![5.0, 2.0, 9.0, 6.0, 2.0];
        let z0s = initial_aberth(&h);
        // Depending on the test strategy, you might want to check properties of z0s
        // rather than exact values due to randomness in the Circle generator.
        assert_eq!(z0s.len(), 4); // Degree was 4, so we expect 4 roots
    }
}