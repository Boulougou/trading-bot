use crate::trading_model::*;
use crate::storage::*;

pub fn train_model<T : TradingModel>(model : &mut T,
                   storage : &mut impl Storage,
                   input_name : &str,
                   input_window : u32,
                   prediction_window : u32,
                   extra_training_params : &T::TrainingParams,
                   output_name : &str) -> anyhow::Result<()> {

    let (history, metadata) = storage.load_symbol_history(input_name)?;
    model.train(&history, &metadata, input_window, prediction_window, extra_training_params)?;
    model.save(output_name)
}
