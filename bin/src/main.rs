use trading_lib;
mod fxcm;
use rand::{SeedableRng, Rng, seq::SliceRandom};

fn run_linear_regression() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(1138);
    let mut x : Vec<f32> = (0..100).map(|_s| rng.gen::<f32>()).collect();
    x.shuffle(&mut rng);
    let y : Vec<f32> = x.iter().map(|v| 1.0 + 2.0 * v + 0.1 * rng.gen::<f32>()).collect();

    let x_train = &x[0..80];
    let y_train = &y[0..80];
    let x_val = &x[80..100];
    let y_val = &y[80..100];

    // println!("{:?}", x_train);
    // println!("{:?}", x_val);

    let x_train_tensor = tch::Tensor::of_slice(&x_train);
    let y_train_tensor = tch::Tensor::of_slice(&y_train);

    // let x_vec : Vec<f32> = Vec::from(&x_train_tensor);
    // let y_vec : Vec<f32> = Vec::from(&y_train_tensor);
    // for i in 0..x_vec.len() {
    //     println!("{},{}", x_vec[i], y_vec[i]);
    // }

    // x_train_tensor.print();
    // y_train_tensor.print();

    tch::manual_seed(1138);
    let a = tch::Tensor::randn(&[1], (tch::Kind::Float, tch::Device::Cpu));
    let mut a = a.set_requires_grad(true);
    let b = tch::Tensor::randn(&[1], (tch::Kind::Float, tch::Device::Cpu));
    let mut b = b.set_requires_grad(true);
    a.print();
    b.print();
    println!("====================");

    let learning_rate = 1e-2;
    for _i in 0..1000 {

        let yhat = &a + &b * &x_train_tensor;
        let error = &y_train_tensor - yhat;
        let loss = error.pow(2).mean(tch::Kind::Float);
        loss.print();

        loss.backward();

        // println!("{:?}", a.grad());
        // println!("{:?}", b.grad());

        tch::no_grad(|| {
            a -= learning_rate * &a.grad();
            b -= learning_rate * &b.grad();
        });

        a.zero_grad();
        b.zero_grad();
    }
    println!("====================");

    a.print();
    b.print();
}

fn main() -> anyhow::Result<()> {
    // let mut service = fxcm::service::FxcmTradingService::create("api-demo.fxcm.com", "4979200962b698e88aa1492f4e62f6e30e338a27")?;

    // trading_lib::run(&mut service)

    run_linear_regression();
    Ok(())
}
