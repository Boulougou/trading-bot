use trading_lib;
use tch::nn::{Module, OptimizerConfig};
use rand::{SeedableRng, Rng, seq::SliceRandom};

pub struct PyTorchModel {
    input_window_size : u32,
    prediction_window_size : u32
}

impl PyTorchModel {
    pub fn new(input_window_size : u32, prediction_window_size : u32) -> PyTorchModel {
        PyTorchModel{ input_window_size, prediction_window_size }
    }

    fn prepare_input_data(&self, history : &Vec<trading_lib::HistoryStep>,
            min_ever_bid_price : f32, max_ever_bid_price : f32) -> (Vec<f32>, Vec<f32>, u32, u32) {
        let normalize = |v : f32| (v - min_ever_bid_price) / (max_ever_bid_price - min_ever_bid_price);
        let input_size = history.len() - self.prediction_window_size as usize - self.input_window_size as usize;
        
        let mut input = Vec::new();
        let mut expected_output = Vec::new();
        for i in 0..input_size {
            let input_end = i + self.input_window_size as usize;
            let input_steps = &history[i..input_end];
            let future_steps = &history[input_end..input_end + self.prediction_window_size as usize];

            for s in input_steps {
                input.push(normalize(s.bid_candle.price_low));
                input.push(normalize(s.bid_candle.price_high));
                input.push(normalize(s.bid_candle.price_open));
                input.push(normalize(s.bid_candle.price_close));
            }

            let mut min_bid_price = f32::INFINITY;
            let mut max_bid_price = f32::NEG_INFINITY;
            for s in future_steps {
                min_bid_price = min_bid_price.min(s.bid_candle.price_low);
                max_bid_price = max_bid_price.max(s.bid_candle.price_high);
            }
            expected_output.push(normalize(min_bid_price));
            expected_output.push(normalize(max_bid_price));
        }

        let input_per_history_step : u32 = 4;
        let output_per_history_step : u32 = 2;
        (input, expected_output, input_per_history_step, output_per_history_step)

    }

    fn split_input<'a>(&self, input : &'a Vec<f32>, expected_output : &'a Vec<f32>, input_stride : u32,
            output_stride : u32, split_point : f32) -> (&'a[f32], &'a[f32], &'a[f32], &'a[f32]) {
        let input_stride = self.input_window_size * input_stride;
        let train_input_size = (split_point * (input.len() / input_stride as usize) as f32) as usize * input_stride as usize;
        let train_output_size = (split_point * (expected_output.len() / output_stride as usize) as f32) as usize * output_stride as usize;

        (&input[..train_input_size], &expected_output[..train_output_size],
         &input[train_input_size..], &expected_output[train_output_size..])
    }

    fn find_min_max_prices(history : &Vec<trading_lib::HistoryStep>) -> (f32, f32) {
        let mut min_bid_price = f32::INFINITY;
        let mut max_bid_price = f32::NEG_INFINITY;
        for step in history {
            min_bid_price = min_bid_price.min(step.bid_candle.price_low);
            max_bid_price = max_bid_price.max(step.bid_candle.price_high);
        }

        (min_bid_price, max_bid_price)
    }
}

impl trading_lib::TradingModel for PyTorchModel {

    fn train(&mut self, history : &Vec<trading_lib::HistoryStep>) -> anyhow::Result<()> {
        let (min_ever_bid_price, max_ever_bid_price) = PyTorchModel::find_min_max_prices(history);
        let bid_price_range = max_ever_bid_price - min_ever_bid_price;

        let (input, expected_output, input_per_history_step, output_per_history_step) = self.prepare_input_data(
            history, min_ever_bid_price, max_ever_bid_price);
        let (train_input, train_expected_output, test_input, test_expected_output) = self.split_input(
            &input, &expected_output, input_per_history_step, output_per_history_step, 0.8);

        let input_layer_size : i64 = input_per_history_step as i64 * self.input_window_size as i64;
        let hidden_layer_size : i64 = input_layer_size / 2;
        let output_layer_size : i64 = output_per_history_step as i64;

        let var_store = tch::nn::VarStore::new(tch::Device::Cpu);
        let neural_net = tch::nn::seq()
            .add(tch::nn::linear(&var_store.root() / "layer1", input_layer_size, hidden_layer_size, Default::default()))
            // .add_fn(|xs| xs.sigmoid())
            .add(tch::nn::linear(var_store.root(), hidden_layer_size, output_layer_size, Default::default()));

        let mut optimizer = tch::nn::Adam::default().build(&var_store, 1e-3).unwrap();

        let input_tensor = tch::Tensor::
            of_slice(&train_input).
            reshape(&[train_input.len() as i64 / input_layer_size, input_layer_size]);
        let expected_output_tensor = tch::Tensor::
            of_slice(&train_expected_output).
            reshape(&[train_expected_output.len() as i64 / output_layer_size, output_layer_size]);

        // input_tensor.print();
        // expected_output_tensor.print();

        let test_input_tensor = tch::Tensor::
            of_slice(&test_input).
            reshape(&[test_input.len() as i64 / input_layer_size, input_layer_size]);
        let test_expected_output_tensor = tch::Tensor::
            of_slice(&test_expected_output).
            reshape(&[test_expected_output.len() as i64 / output_layer_size, output_layer_size]);

        for epoch in 0..2600 {
            // println!("AAAAAAA: {:?} => {:?}", input_tensor, expected_output_tensor);
            let output_tensor = neural_net.forward(&input_tensor);
            // let error = output_tensor - expected_output_tensor;
            let loss_tensor = output_tensor.mse_loss(&expected_output_tensor, tch::Reduction::Mean);

            optimizer.backward_step(&loss_tensor);

            let test_output_tensor = neural_net.forward(&test_input_tensor);
            let test_loss_tensor = test_output_tensor.mse_loss(&test_expected_output_tensor, tch::Reduction::Mean);

            if epoch == 2599 {
            // test_input_tensor.print();
                let expected = (&test_expected_output_tensor * bid_price_range as f64) + min_ever_bid_price as f64;
                let actual = (&test_output_tensor * bid_price_range as f64) + min_ever_bid_price as f64;
                expected.print();
                actual.print();
            }
            
            println!(
                "epoch: {:4} train loss: {:8.5} test loss: {:8.5}",
                epoch,
                f64::from(&loss_tensor),
                f64::from(&test_loss_tensor),
                // 100. * f64::from(&test_accuracy),
            );
        }

        // Make predictions

        Ok(())
    }
}


fn _run_linear_regression() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(1138);
    let mut x : Vec<f32> = (0..100).map(|_s| rng.gen::<f32>()).collect();
    x.shuffle(&mut rng);
    let y : Vec<f32> = x.iter().map(|v| 1.0 + 2.0 * v + 0.1 * rng.gen::<f32>()).collect();

    let x_train = &x[0..80];
    let y_train = &y[0..80];
    // let x_val = &x[80..100];
    // let y_val = &y[80..100];

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

fn _run_neural_network() {
    let var_store = tch::nn::VarStore::new(tch::Device::Cpu);

    let num_inputs = 3;
    let num_hidden_nodes = 3;
    let num_outputs = 1;
    let neural_net = tch::nn::seq()
        .add(tch::nn::linear(&var_store.root() / "layer1", num_inputs, num_hidden_nodes, Default::default()))
        // .add_fn(|xs| xs.sigmoid())
        .add(tch::nn::linear(var_store.root(), num_hidden_nodes, num_outputs, Default::default()));

    let mut optimizer = tch::nn::Adam::default().build(&var_store, 1e-2).unwrap();

    for epoch in 1..400 {
        let input_tensor = tch::Tensor::
            of_slice(&[1.0f32, 6.0, 3.0, 9.0, 2.0, 7.0, 3.0, 5.0, 3.0]).
            reshape(&[3, num_inputs]);
        let expected_output_tensor = tch::Tensor::
            of_slice(&[10.0f32, 18.0, 11.0]).
            reshape(&[3, num_outputs]);
        // println!("AAAAAAA: {:?} => {:?}", input_tensor, expected_output_tensor);
        let output_tensor = neural_net.forward(&input_tensor);
        // let error = output_tensor - expected_output_tensor;
        let loss_tensor = output_tensor.mse_loss(&expected_output_tensor, tch::Reduction::Mean);

        optimizer.backward_step(&loss_tensor);
        
        println!(
            "epoch: {:4} train loss: {:8.5}",
            epoch,
            f64::from(&loss_tensor),
            // 100. * f64::from(&test_accuracy),
        );
    }

    let input_tensor = tch::Tensor::
        of_slice(&[5.0f32, 2.0, 1.0, 8.0, 8.0, 9.0, 5.0, 5.0, 3.0]).
        reshape(&[3, num_inputs]);
    let output_tensor = neural_net.forward(&input_tensor);
    output_tensor.print();
}