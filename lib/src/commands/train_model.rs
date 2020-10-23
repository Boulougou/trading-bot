use crate::trading_model::*;
use crate::storage::*;
use strum::EnumString;
use anyhow::{anyhow, Context};

#[derive(Debug, PartialEq, PartialOrd, Clone, Copy, EnumString)]
pub enum TrainingOutputMode {
    OverwriteModel,
    ContinueTraining
}

pub fn train_model<T : TradingModel>(model : &mut T,
                   storage : &mut impl Storage,
                   input_name : &str,
                   input_window : u32,
                   prediction_window : u32,
                   extra_training_params : &T::TrainingParams,
                   model_name : &str,
                   output_mode : TrainingOutputMode) -> anyhow::Result<()> {
    let (history, metadata) = storage.load_symbol_history(input_name)?;

    if output_mode == TrainingOutputMode::ContinueTraining {
        let (model_symbol, model_timeframe) = model.load(model_name).context("Failed to load existing model")?;
        if model_symbol != metadata.symbol {
            return Err(anyhow!("Existing model has different symbol, {} != {}", model_symbol, metadata.symbol));
        }
        if model_timeframe != metadata.timeframe {
            return Err(anyhow!("Existing model has different timeframe, {:?} != {:?}", model_timeframe, metadata.timeframe));
        }
    }

    model.train(&history, &metadata, input_window, prediction_window, extra_training_params)?;
    model.save(model_name)
}
