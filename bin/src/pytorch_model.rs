use trading_lib;
use tch::nn::{Module, OptimizerConfig};
use anyhow::anyhow;
use serde::{Deserialize, Serialize};
use trading_lib::HistoryTimeframe;

#[derive(Debug, Deserialize, Serialize)]
struct TrainingMetadata {
    input_window : u32,
    prediction_window : u32,
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

    fn load_from_disk(&mut self, name : &str) -> anyhow::Result<(String, HistoryTimeframe)> {
        let file = std::fs::File::open(format!("{}/training_metadata.json", name))?;
        let training_metadata : TrainingMetadata = ::serde_json::from_reader(&file)?;

        let input_per_history_step : u32 = 8;
        let output_per_history_step : u32 = 2;
        let input_layer_size : i64 = input_per_history_step as i64 * training_metadata.input_window as i64;
        let output_layer_size : i64 = output_per_history_step as i64;

        let mut var_store = tch::nn::VarStore::new(tch::Device::Cpu);
        let neural_net = tch::nn::seq()
            .add(tch::nn::linear(&var_store.root() / "layer1", input_layer_size, output_layer_size, Default::default()));
        var_store.load(format!("{}/model.var", name))?;

        let training_artifacts = TrainingArtifacts { var_store, neural_net, metadata : training_metadata };
        let return_value = (training_artifacts.metadata.symbol.clone(), training_artifacts.metadata.timeframe);
        self.training_artifacts = Some(training_artifacts);

        Ok(return_value)
    }

    fn prepare_input_data(history : &Vec<trading_lib::HistoryStep>, input_window : u32, prediction_window : u32,
            min_ever_bid_price : f32, max_ever_bid_price : f32) -> (Vec<f32>, Vec<f32>, u32, u32) {
        let normalize = |v : f32| (v - min_ever_bid_price) / (max_ever_bid_price - min_ever_bid_price) - 0.5;
        let input_size = history.len() - prediction_window as usize - input_window as usize + 1;
        
        let mut input = Vec::new();
        let mut expected_output = Vec::new();
        for i in 0..input_size {
            let input_end = i + input_window as usize;
            let input_steps = &history[i..input_end];
            let future_steps = &history[input_end..input_end + prediction_window as usize];

            for s in input_steps {
                input.push(normalize(s.bid_candle.price_low));
                input.push(normalize(s.bid_candle.price_high));
                input.push(normalize(s.bid_candle.price_open));
                input.push(normalize(s.bid_candle.price_close));
                input.push(normalize(s.ask_candle.price_low));
                input.push(normalize(s.ask_candle.price_high));
                input.push(normalize(s.ask_candle.price_open));
                input.push(normalize(s.ask_candle.price_close));
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

        let input_per_history_step : u32 = 8;
        let output_per_history_step : u32 = 2;
        (input, expected_output, input_per_history_step, output_per_history_step)

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
}

impl trading_lib::TradingModel for PyTorchModel {

    type TrainingParams = (f32, u32);

    fn train(&mut self, history : &Vec<trading_lib::HistoryStep>,
             history_metadata : &trading_lib::HistoryMetadata,
             input_window : u32, prediction_window : u32,
             &(learning_rate, num_iterations) : &Self::TrainingParams) -> anyhow::Result<()> {
        let (min_ever_bid_price, max_ever_bid_price) = PyTorchModel::find_min_max_prices(history);

        let (input, expected_output, input_per_history_step, output_per_history_step) = PyTorchModel::prepare_input_data(
            history, input_window, prediction_window, min_ever_bid_price, max_ever_bid_price);
        let (train_input, train_expected_output, test_input, test_expected_output) = PyTorchModel::split_input(
            &input, &expected_output, input_window, input_per_history_step, output_per_history_step, 0.8);

        let input_layer_size : i64 = input_per_history_step as i64 * input_window as i64;
        // let hidden_layer_size : i64 = input_layer_size / 2;
        let output_layer_size : i64 = output_per_history_step as i64;

        let mut var_store = tch::nn::VarStore::new(tch::Device::Cpu);
        let neural_net = tch::nn::seq()
            .add(tch::nn::linear(&var_store.root() / "layer1", input_layer_size, output_layer_size, Default::default()));

        if let Some(artifacts) = &self.training_artifacts {
            var_store.copy(&artifacts.var_store)?;
        }

        let mut optimizer = tch::nn::Adam::default().build(&var_store, learning_rate as f64).unwrap();

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

        for epoch in 0..num_iterations {
            let output_tensor = neural_net.forward(&input_tensor);
            // let error = output_tensor - expected_output_tensor;
            let loss_tensor = output_tensor.mse_loss(&expected_output_tensor, tch::Reduction::Mean);

            optimizer.backward_step(&loss_tensor);

            let test_output_tensor = neural_net.forward(&test_input_tensor);
            let test_loss_tensor = test_output_tensor.mse_loss(&test_expected_output_tensor, tch::Reduction::Mean);

            println!(
                "epoch: {:4} train loss: {:8.5} test loss: {:8.5}",
                epoch,
                f64::from(&loss_tensor),
                f64::from(&test_loss_tensor),
            );
        }

        let training_metadata = TrainingMetadata { input_window, prediction_window,
            min_input_value : min_ever_bid_price, max_input_value : max_ever_bid_price,
            timeframe : history_metadata.timeframe, symbol : history_metadata.symbol.clone()
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
            training_metadata.min_input_value, training_metadata.max_input_value);

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

    fn get_input_window(&self) -> anyhow::Result<u32> {
        self.training_artifacts.as_ref().map(|a| a.metadata.input_window).ok_or(anyhow!("Model has not been trained yet"))
    }

    fn save(&mut self, output_name : &str) -> anyhow::Result<()> {
        let training_artifacts = self.training_artifacts.as_ref().ok_or(anyhow!("Model has not been trained yet"))?;

        std::fs::create_dir_all(format!("{}", output_name))?;

        training_artifacts.var_store.save(format!("{}/model.var", output_name))?;
        let file = std::fs::File::create(format!("{}/training_metadata.json", output_name))?;
        ::serde_json::to_writer(&file, &training_artifacts.metadata)?;
        Ok(())
    }

    fn load(&mut self, name : &str) -> anyhow::Result<(String, HistoryTimeframe)> {
        self.load_from_disk(name)
    }
}