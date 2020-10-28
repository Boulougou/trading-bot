use trading_lib;
use tch::nn::{Module, OptimizerConfig};
use anyhow::anyhow;
use serde::{Deserialize, Serialize};
use trading_lib::HistoryTimeframe;
use rand::prelude::SliceRandom;
use rand::SeedableRng;
use chrono::{Utc, TimeZone, Datelike, Timelike};
use tempfile::NamedTempFile;

#[derive(Debug, Deserialize, Serialize)]
struct TrainingMetadata {
    input_window : u32,
    prediction_window : u32,
    input_per_history_step : u32,
    output_per_history_step : u32,
    hidden_layer_size : u32,
    train_loss : f32,
    test_loss : f32,
    total_epochs : u32,
    train_data_size : u32,
    min_input_value : f32,
    max_input_value : f32,
    symbol : String,
    timeframe : HistoryTimeframe
}

#[derive(Debug)]
struct TrainingArtifacts {
    var_store : tch::nn::VarStore,
    neural_net : tch::nn::Sequential,
    metadata : TrainingMetadata
}

pub struct PyTorchModel {
    training_artifacts : Option<TrainingArtifacts>
}

impl PyTorchModel {
    pub fn new() -> PyTorchModel {
        PyTorchModel{ training_artifacts : None }
    }

    fn load_from_disk(&mut self, name : &str) -> anyhow::Result<(String, HistoryTimeframe, u32, u32)> {
        let file = std::fs::File::open(format!("{}/training_metadata.json", name))?;
        let training_metadata : TrainingMetadata = ::serde_json::from_reader(&file)?;

        let input_layer_size : i64 = training_metadata.input_per_history_step as i64 * training_metadata.input_window as i64;
        let output_layer_size : i64 = training_metadata.output_per_history_step as i64;

        let mut var_store = tch::nn::VarStore::new(tch::Device::Cpu);
        let neural_net = PyTorchModel::build_neural_net(input_layer_size,
                    training_metadata.hidden_layer_size as i64, output_layer_size, &mut var_store);
        var_store.load(format!("{}/model.var", name))?;

        let training_artifacts = TrainingArtifacts { var_store, neural_net, metadata : training_metadata };
        let return_value = (training_artifacts.metadata.symbol.clone(),
                            training_artifacts.metadata.timeframe, training_artifacts.metadata.input_window,
                            training_artifacts.metadata.prediction_window);
        self.training_artifacts = Some(training_artifacts);

        Ok(return_value)
    }

    fn prepare_input_data(history : &Vec<trading_lib::HistoryStep>, input_window : u32, prediction_window : u32,
            min_ever_bid_price : f32, max_ever_bid_price : f32) -> anyhow::Result<(Vec<f32>, Vec<f32>, u32, u32)> {
        let mut input_events = trading_lib::utils::extract_input_and_prediction_windows(history, input_window, prediction_window)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(1138);
        input_events.shuffle(&mut rng);

        let normalize = |v : f32| PyTorchModel::normalize_value_f(v, min_ever_bid_price, max_ever_bid_price);
        let mut input = Vec::new();
        let mut expected_output = Vec::new();
        let input_per_history_step : u32 = 12;
        let output_per_history_step : u32 = 2;
        for (input_steps, future_steps) in input_events {
            for s in input_steps {
                let datetime = Utc.timestamp(s.timestamp as i64, 0);
                input.push(PyTorchModel::normalize_value_f(datetime.month0() as f32, 0.0, 11.0));
                input.push(PyTorchModel::normalize_value_f(datetime.day0() as f32, 0.0, 30.0));
                input.push(PyTorchModel::normalize_value_f(datetime.weekday() as i32 as f32, 0.0, 6.0));
                input.push(PyTorchModel::normalize_value_f(datetime.hour() as f32, 0.0, 23.0));

                input.push(normalize(s.bid_candle.price_low));
                input.push(normalize(s.bid_candle.price_high));
                input.push(normalize(s.bid_candle.price_open));
                input.push(normalize(s.bid_candle.price_close));
                input.push(normalize(s.ask_candle.price_low));
                input.push(normalize(s.ask_candle.price_high));
                input.push(normalize(s.ask_candle.price_open));
                input.push(normalize(s.ask_candle.price_close));
            }

            let (min_bid_price, max_bid_price) = trading_lib::utils::find_bid_price_range(&future_steps);
            expected_output.push(normalize(min_bid_price));
            expected_output.push(normalize(max_bid_price));
        }

        Ok((input, expected_output, input_per_history_step, output_per_history_step))
    }

    fn normalize_value(&self, value : f32) -> f32 {
        let training_metadata = &self.training_artifacts.as_ref().unwrap().metadata;
        let min_value = training_metadata.min_input_value;
        let max_value = training_metadata.max_input_value;
        PyTorchModel::normalize_value_f(value, min_value, max_value)
    }

    fn normalize_value_f(value : f32, min_value : f32, max_value : f32) -> f32 {
        (value - min_value) / (max_value - min_value) - 0.5
    }

    fn split_input<'a>(input : &'a Vec<f32>, expected_output : &'a Vec<f32>, input_window : u32, input_stride : u32,
            output_stride : u32, split_point : f32) -> (&'a[f32], &'a[f32], &'a[f32], &'a[f32]) {
        let input_stride = input_window * input_stride;
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

    fn build_neural_net(input_layer_size: i64, hidden_layer_size : i64, output_layer_size: i64, var_store: &mut tch::nn::VarStore) -> tch::nn::Sequential {
        if hidden_layer_size == 0 {
            let neural_net = tch::nn::seq()
                .add(tch::nn::linear(&var_store.root() / "layer1", input_layer_size, output_layer_size, Default::default()));

            neural_net
        }
        else {
            let neural_net = tch::nn::seq()
                .add(tch::nn::linear(&var_store.root() / "layer1", input_layer_size, hidden_layer_size, Default::default()))
                .add_fn(|xs| xs.relu())
                .add(tch::nn::linear(&var_store.root() / "output_layer", hidden_layer_size, output_layer_size, Default::default()));

            neural_net
        }
    }

    fn convert_to_one_dimension(&self, predictions: &[(f32, f32)]) -> Vec<f32> {
        let mut pred_one_dim = Vec::new();
        for (l, r) in predictions {
            pred_one_dim.push(self.normalize_value(*l));
            pred_one_dim.push(self.normalize_value(*r));
        }
        pred_one_dim
    }

    fn calculate_loss_tensor(output : &tch::Tensor, expected_output : &tch::Tensor) -> tch::Tensor {
        // let diff = output - expected_output;
        // // println!("diff: ");
        // // diff.print();
        //
        // let prelu = diff.prelu(&tch::Tensor::from(0.95f32));
        // // println!("prelu: ");
        // // prelu.print();
        //
        // let square = prelu.square();
        // // println!("square: ");
        // // square.print();
        //
        // let mean = square.mean(tch::Kind::Float);
        // // println!("mean: ");
        // // mean.print();
        //
        // mean

        // let diff = output - expected_output;
        // let low_output = output.chunk();
        // let range_loss = output.mse_loss(expected_output, tch::Reduction::Mean);
        //
        let accuracy_loss = output.mse_loss(expected_output, tch::Reduction::Mean);

        accuracy_loss// + 0.25 * range_loss
    }
}

impl trading_lib::TradingModel for PyTorchModel {

    type TrainingParams = (f32, u32, u32);

    fn train(&mut self, history : &Vec<trading_lib::HistoryStep>,
             history_metadata : &trading_lib::HistoryMetadata,
             input_window : u32, prediction_window : u32,
             &(learning_rate, num_iterations, hidden_layer_factor) : &Self::TrainingParams) -> anyhow::Result<()> {
        let (min_ever_bid_price, max_ever_bid_price) = PyTorchModel::find_min_max_prices(history);

        let (input, expected_output, input_per_history_step, output_per_history_step) = PyTorchModel::prepare_input_data(
            history, input_window, prediction_window, min_ever_bid_price, max_ever_bid_price)?;
        let (train_input, train_expected_output, test_input, test_expected_output) = PyTorchModel::split_input(
            &input, &expected_output, input_window, input_per_history_step, output_per_history_step, 0.8);

        let input_layer_size : i64 = input_per_history_step as i64 * input_window as i64;
        let hidden_layer_size = if hidden_layer_factor == 0 { 0 } else { input_layer_size / hidden_layer_factor as i64 };
        let output_layer_size : i64 = output_per_history_step as i64;

        let mut var_store = tch::nn::VarStore::new(tch::Device::Cpu);
        let neural_net = PyTorchModel::build_neural_net(input_layer_size,
            hidden_layer_size, output_layer_size, &mut var_store);

        let mut optimizer = tch::nn::Adam::default().build(&var_store, learning_rate as f64).unwrap();

        if let Some(artifacts) = &self.training_artifacts {
            if artifacts.metadata.hidden_layer_size != hidden_layer_size as u32 {
                return Err(anyhow!("Invalid hidden layer size passed, loaded models has {} while passed size is {}",
                    artifacts.metadata.hidden_layer_size, hidden_layer_size));
            }
            var_store.copy(&artifacts.var_store)?;
        }

        let input_tensor = tch::Tensor::
            of_slice(&train_input).
            reshape(&[train_input.len() as i64 / input_layer_size, input_layer_size]);
        let expected_output_tensor = tch::Tensor::
            of_slice(&train_expected_output).
            reshape(&[train_expected_output.len() as i64 / output_layer_size, output_layer_size]);

        let test_input_tensor = tch::Tensor::
            of_slice(&test_input).
            reshape(&[test_input.len() as i64 / input_layer_size, input_layer_size]);
        let test_expected_output_tensor = tch::Tensor::
            of_slice(&test_expected_output).
            reshape(&[test_expected_output.len() as i64 / output_layer_size, output_layer_size]);

        let mut train_loss : f32 = 0.0;
        let mut test_loss : f32  = 0.0;
        let mut min_train_loss = f32::INFINITY;
        let mut min_test_loss = f32::INFINITY;
        let mut prev_test_loss = f32::INFINITY;
        let mut backup_weights : Option<NamedTempFile> = None;
        for epoch in 0..num_iterations {
            let output_tensor = neural_net.forward(&input_tensor);
            // let error = output_tensor - expected_output_tensor;
            let loss_tensor = PyTorchModel::calculate_loss_tensor(&output_tensor, &expected_output_tensor);

            optimizer.backward_step(&loss_tensor);

            let _no_grad_guard = tch::no_grad_guard();
            let test_output_tensor = neural_net.forward(&test_input_tensor);
            let test_loss_tensor = PyTorchModel::calculate_loss_tensor(&test_output_tensor, &test_expected_output_tensor);

            if test_loss > prev_test_loss && test_loss < min_test_loss {
                min_test_loss = test_loss;
                min_train_loss = train_loss;

                let temp_file = tempfile::Builder::new()
                    .prefix(&format!("epoch{}_", epoch))
                    .suffix(".var")
                    .rand_bytes(5)
                    .tempfile()?;

                var_store.save(&temp_file)?;
                backup_weights = Some(temp_file);
            }
            prev_test_loss = test_loss;

            train_loss = f64::from(&loss_tensor) as f32;
            test_loss = f64::from(&test_loss_tensor) as f32;
            println!(
                "epoch: {:4} train loss: {:8.10} test loss: {:8.10}",
                epoch,
                train_loss,
                test_loss,
            );
        }

        if test_loss > min_test_loss {
            println!("Using backup weights with train loss: {:8.10} test lost: {:8.10}", min_train_loss, min_test_loss);
            train_loss = min_train_loss;
            test_loss = min_test_loss;
            var_store.load(&backup_weights.unwrap())?;
        }

        let train_data_size = train_input.len() as u32 / input_layer_size as u32 + test_input.len() as u32 / input_layer_size as u32;
        let training_metadata = TrainingMetadata { input_window, prediction_window,
            min_input_value : min_ever_bid_price, max_input_value : max_ever_bid_price,
            timeframe : history_metadata.timeframe, symbol : history_metadata.symbol.clone(),
            input_per_history_step, output_per_history_step, total_epochs : num_iterations,
            train_loss, test_loss, train_data_size, hidden_layer_size : hidden_layer_size as u32
        };
        self.training_artifacts = Some(TrainingArtifacts { var_store, neural_net, metadata : training_metadata });

        Ok(())
    }

    fn predict(&mut self, history : &Vec<trading_lib::HistoryStep>) -> anyhow::Result<(f32, f32)> {
        let training_artifacts = self.training_artifacts.as_ref().ok_or(anyhow!("Model has not been trained yet"))?;
        let training_metadata = &training_artifacts.metadata;

        if history.len() as u32 != training_metadata.input_window {
            return Err(anyhow!("Passed history length should fit exactly the model input size"));
        }

        let (input, _expected_output, input_per_history_step, _output_per_history_step) = PyTorchModel::prepare_input_data(
            history, training_metadata.input_window, 0,
            training_metadata.min_input_value, training_metadata.max_input_value)?;

        let input_layer_size : i64 = input_per_history_step as i64 * training_metadata.input_window as i64;

        let input_tensor = tch::Tensor::
            of_slice(&input).
            reshape(&[1, input_layer_size]);
        let output_tensor = training_artifacts.neural_net.forward(&input_tensor);

        let denormalize = |v : &f64| (*v as f32 + 0.5) * (training_metadata.max_input_value - training_metadata.min_input_value) + training_metadata.min_input_value;
        let output_values : Vec<f64> = From::from(output_tensor);
        let output_values = output_values.iter().map(denormalize).collect::<Vec<_>>();
        Ok((output_values[0] as f32, output_values[1] as f32))
    }

    fn calculate_loss(&mut self, predictions : &[(f32, f32)], expectations : &[(f32, f32)]) -> f32 {
        let pred_one_dim = self.convert_to_one_dimension(predictions);
        let exp_one_dim = self.convert_to_one_dimension(expectations);

        let pred_tensor = tch::Tensor::of_slice(&pred_one_dim).
            reshape(&[pred_one_dim.len() as i64 / 2, 2]);
        let exp_tensor = tch::Tensor::of_slice(&exp_one_dim).
            reshape(&[exp_one_dim.len() as i64 / 2, 2]);

        let loss_tensor = PyTorchModel::calculate_loss_tensor(&pred_tensor, &exp_tensor);

        f32::from(loss_tensor)
    }

    fn save(&mut self, output_name : &str) -> anyhow::Result<()> {
        let training_artifacts = self.training_artifacts.as_ref().ok_or(anyhow!("Model has not been trained yet"))?;

        std::fs::create_dir_all(format!("{}", output_name))?;

        training_artifacts.var_store.save(format!("{}/model.var", output_name))?;
        let file = std::fs::File::create(format!("{}/training_metadata.json", output_name))?;
        ::serde_json::to_writer(&file, &training_artifacts.metadata)?;
        Ok(())
    }

    fn load(&mut self, name : &str) -> anyhow::Result<(String, HistoryTimeframe, u32, u32)> {
        self.load_from_disk(name)
    }
}
