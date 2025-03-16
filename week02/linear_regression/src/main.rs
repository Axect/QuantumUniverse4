use std::env::args;

use peroxide::fuga::*;

fn main() {
    let trial = args().nth(1).unwrap().parse::<usize>().unwrap();
    let eta = args().nth(2).unwrap().parse::<f64>().unwrap();

    let mut rng = stdrng_from_seed(42);
    let uniform = Uniform(0.0, 1.0);
    let normal = Normal(0.0, 0.1); 

    let x = uniform.sample_with_rng(&mut rng, 100);
    let error = normal.sample_with_rng(&mut rng, 100);
    let y = x.zip_with(|x, e| 2.0 * x + 1.0 + e, &error);

    let mut linreg = LinearRegression::new();
    let mut w0 = vec![0f64; trial];
    let mut w1 = vec![0f64; trial];
    let mut loss = vec![0f64; trial];
    loss[0] = linreg.map(&x).zip_with(|y, y_hat| (y - y_hat).powi(2), &y).mean();

    for i in 1..trial {
        linreg.gradient_descent(&x, &y, eta);
        w0[i] = linreg.w0;
        w1[i] = linreg.w1;
        loss[i] = linreg.map(&x).zip_with(|y, y_hat| (y - y_hat).powi(2), &y).mean();
    }

    let mut df = DataFrame::new(vec![]);
    df.push("x", Series::new(x));
    df.push("y", Series::new(y));
    df.print();
    df.write_parquet("results/linreg_data.parquet", CompressionOptions::Snappy).unwrap();

    let mut df = DataFrame::new(vec![]);
    df.push("w0", Series::new(w0));
    df.push("w1", Series::new(w1));
    df.push("loss", Series::new(loss));
    df.print();
    df.write_parquet(&format!("results/linreg_{}_{}.parquet", trial, eta), CompressionOptions::Snappy).unwrap();
}

#[derive(Debug, Clone, Copy)]
struct LinearRegression {
    w0: f64,
    w1: f64,
}

impl LinearRegression {
    fn new() -> Self {
        LinearRegression {
            w0: 0.0,
            w1: 0.0,
        }
    }

    fn map(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&x| self.w0 + self.w1 * x).collect()
    }

    #[allow(non_snake_case)]
    fn gradient_descent(&mut self, x: &[f64], y: &[f64], eta: f64) {
        let N = x.len() as f64;
        let y_hat = self.map(x);

        let mut dLdw0 = 0f64;
        let mut dLdw1 = 0f64;

        for i in 0..x.len() {
            let error = y[i] - y_hat[i];
            dLdw0 += -2.0 * error;
            dLdw1 += -2.0 * error * x[i];
        }

        self.w0 -= eta * dLdw0 / N;
        self.w1 -= eta * dLdw1 / N;
    }
}
