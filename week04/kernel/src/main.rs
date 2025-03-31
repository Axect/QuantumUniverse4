use peroxide::fuga::*;
use rayon::prelude::*;
use rugfield::{grf_with_rng, Kernel::SquaredExponential};
use std::f64::consts::PI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = stdrng_from_seed(42);
    let n = 300;
    let x = linspace(0.0, 1.0, n);
    let y = grf_with_rng(&mut rng, n, SquaredExponential(0.1));
    let normal = Normal(0.0, 0.2);
    let y_noisy = y.add_v(&normal.sample(n));

    let data = Data::new(x.clone(), y_noisy.clone());
    let kernel_regression1 = KernelRegression::new(data.clone(), Kernel::Epanechnikov, 0.05);
    let kernel_regression2 = KernelRegression::new(data.clone(), Kernel::TriCube, 0.05);
    let kernel_regression3 = KernelRegression::new(data.clone(), Kernel::Gaussian, 0.05);
    let x_domain = linspace(0.0, 1.0, 1000);
    let y_pred = x_domain.par_iter()
        .map(|&xi| {
            (kernel_regression1.predict(xi), kernel_regression2.predict(xi), kernel_regression3.predict(xi))
        })
        .collect::<Vec<(f64, f64, f64)>>();
    let y_pred: (Vec<f64>, (Vec<f64>, Vec<f64>)) = y_pred.into_iter()
        .map(|(a, b, c)| (a, (b, c)))
        .unzip();

    let mut df = DataFrame::new(vec![]);
    df.push("x", Series::new(x_domain));
    df.push("y1", Series::new(y_pred.0));
    df.push("y2", Series::new(y_pred.1.0));
    df.push("y3", Series::new(y_pred.1.1));
    df.print();
    df.write_parquet("kernel_regression.parquet", CompressionOptions::Snappy)?;

    let mut dg = DataFrame::new(vec![]);
    dg.push("x", Series::new(x));
    dg.push("y", Series::new(y_noisy));
    dg.print();
    dg.write_parquet("kernel_true.parquet", CompressionOptions::Snappy)?;

    Ok(())
}

#[derive(Debug, Copy, Clone)]
enum Kernel {
    Epanechnikov,
    TriCube,
    Gaussian,
}

impl Kernel {
    fn kernel(&self, t: f64) -> f64 {
        match self {
            Kernel::Epanechnikov => {
                if t.abs() <= 1.0 {
                    0.75 * (1.0 - t.powi(2))
                } else {
                    0.0
                }
            }
            Kernel::TriCube => {
                if t.abs() <= 1.0 {
                    70.0 / 81.0 * (1.0 - t.abs().powi(3)).powi(3)
                } else {
                    0.0
                }
            }
            Kernel::Gaussian => {
                (-t.powi(2) / 2.0).exp() / (2.0 * PI).sqrt()
            }
        }
    }
}

#[derive(Debug, Clone)]
struct Data {
    x: Vec<f64>,
    y: Vec<f64>,
}

impl Data {
    fn new(x: Vec<f64>, y: Vec<f64>) -> Self {
        Data { x, y }
    }

    fn x(&self) -> &Vec<f64> {
        &self.x
    }

    fn y(&self) -> &Vec<f64> {
        &self.y
    }
}

struct KernelRegression {
    data: Data,
    kernel: Kernel,
    bandwidth: f64,
}

impl KernelRegression {
    fn new(data: Data, kernel: Kernel, bandwidth: f64) -> Self {
        KernelRegression { data, kernel, bandwidth }
    }

    fn predict(&self, x: f64) -> f64 {
        let numerator = self.data.x.iter().zip(self.data.y.iter())
            .map(|(xi, yi)| {
                self.kernel.kernel((x - xi) / self.bandwidth) * yi
            }).sum::<f64>();
        let denominator = self.data.x.iter()
            .map(|xi| {
                self.kernel.kernel((x - xi) / self.bandwidth)
            }).sum::<f64>();
        numerator / denominator
    }
}
