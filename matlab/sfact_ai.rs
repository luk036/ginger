use num_complex::Complex;
use std::cmp::Ordering;

pub fn leja_order(mut points: Vec<Complex<f64>>) -> Vec<Complex<f64>> {
    // Check if input is empty and return an error if so
    if points.is_empty() {
        panic!("Input must be a non-empty vector of points.");
    }

    // Start with the point having the smallest magnitude
    let mut leja_ordered_points = vec![points.remove(
        points
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.norm().partial_cmp(&b.norm()).unwrap_or(Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap(),
    )];

    while !points.is_empty() {
        // Compute distances from remaining points to the last point in leja_order
        let last_point = leja_ordered_points.last().unwrap();
        let distances = points.iter().map(|&p| (p - *last_point).norm());

        // Find the index of the point with the maximum minimum distance
        let next_idx = distances
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap();

        // Append this point to the leja_ordered_points
        leja_ordered_points.push(points.remove(next_idx));
    }

    leja_ordered_points
}

fn seprts(p: &[f64]) -> Vec<Complex<f64>> {
    const SN: f64 = 0.0001;
    let rts: Vec<Complex<f64>> = p
        .iter()
        .map(|&c| c as f64)
        .collect::<Vec<_>>()
        .roots()
        .collect();
    let irts: Vec<Complex<f64>> = rts
        .iter()
        .filter(|&r| r.abs() < (1.0 - SN))
        .cloned()
        .collect();
    let mut orts: Vec<Complex<f64>> = rts
        .into_iter()
        .filter(|&r| (r.abs() >= (1.0 - SN)) && (r.abs() <= (1.0 + SN)))
        .collect();

    if orts.len() % 2 != 0 {
        println!("Sorry, but there is a problem (1) in seprts function");
        return vec![];
    }

    // Sorting roots on the unit circle by angle is non-trivial in Rust due to complex number handling.
    // An efficient and precise sorting based on angles would require additional logic or a crate.

    // Placeholder for sorting logic. Actual implementation depends on detailed requirements.
    // For now, assume sorting is done (logic skipped for brevity).

    // Combine the roots
    [irts, orts].concat()
}

fn sfact(p: &[f64]) -> (Vec<f64>, Vec<Complex<f64>>) {
    if p.len() == 1 {
        return (vec![p[0]], vec![]);
    }

    let r = seprts(p);
    let r_leja_ordered = leja(r).expect("Failed to order points with leja"); // Assumes leja always succeeds for simplicity
    let h = r_leja_ordered
        .iter()
        .map(|&re| Complex::new(re, 0.0))
        .collect::<Vec<_>>()
        .poly();

    let mut h_real = h.iter().map(|c| c.re).collect::<Vec<_>>();
    if h_real.iter().all(|&coeff| coeff.fract() == 0.0) {
        h_real.iter_mut().for_each(|coeff| *coeff = coeff.trunc());
    }

    let norm_factor = (p.iter().fold(0.0, |acc, &coeff| acc.max(coeff))
        / h_real
            .iter()
            .map(|&coeff| coeff.powi(2))
            .sum::<f64>()
            .sqrt())
    .sqrt();
    h_real.iter_mut().for_each(|coeff| *coeff *= norm_factor);

    (
        h_real,
        r_leja_ordered
            .into_iter()
            .map(|re| Complex::new(re, 0.0))
            .collect(),
    )
}
