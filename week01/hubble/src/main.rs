use peroxide::fuga::*;

#[allow(non_snake_case)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Read CSV
    let mut df = DataFrame::read_csv("data/hubble.csv", ',')?;
    df.as_types(vec![F64, F64]);
    df.print();

    // 2. Linear Regression
    let x: Vec<f64> = df["distance"].to_vec().fmap(|x| x / 1e+6);
    let y: Vec<f64> = df["velocity"].to_vec();
    let H0 = x.dot(&y) / x.dot(&x);
    println!("[Linear Regression] H0: {} km/s/Mpc", H0);

    // 3. Levenberg Marquardt
    let H0_init = vec![1f64];
    let data = column_stack(&[x.clone(), y.clone()])?;
    
    let mut opt = Optimizer::new(data, linear);
    let p = opt.set_init_param(H0_init)
        .set_max_iter(10)
        .set_method(LevenbergMarquardt)
        .set_lambda_init(1e-3)
        .set_lambda_max(1e+3)
        .optimize();
    let H0 = p[0];
    println!("[Levenberg-Marquardt] H0: {} km/s/Mpc", H0);

    Ok(())
}

fn linear(x: &Vec<f64>, n: Vec<AD>) -> Option<Vec<AD>> {
    Some(
        x.clone().into_iter()
            .map(|t| AD1(t, 0f64))
            .map(|t| t * n[0])
            .collect()
    )
}
