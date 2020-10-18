use crate::trading_service::*;

#[cfg(test)]
use mockall::{automock};

#[cfg_attr(test, automock)]
pub trait Storage {
    fn save_symbol_history(&mut self, name : &str, history : &Vec<HistoryStep>) -> anyhow::Result<()>;
    fn load_symbol_history(&mut self, name : &str) -> anyhow::Result<Vec<HistoryStep>>;
}
