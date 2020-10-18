use crate::trading_service::HistoryStep;

#[cfg(test)]
use mockall::{automock};

#[cfg_attr(test, automock(type TrainingParams = f32;))]
pub trait TradingModel {
    type TrainingParams;
    
    fn train(&mut self, input : &Vec<HistoryStep>, input_window : u32, prediction_window : u32, extra_params : &Self::TrainingParams) -> anyhow::Result<()>;
    fn predict(&mut self, input : &Vec<HistoryStep>) -> anyhow::Result<(f32, f32)>;
    
    fn get_input_window(&self) -> anyhow::Result<u32>;

    fn save(&mut self, name : &str) -> anyhow::Result<()>;
    fn load(&mut self, name : &str) -> anyhow::Result<()>;
}
