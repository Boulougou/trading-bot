use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use strum::EnumString;

#[cfg(test)]
use mockall::{automock};

pub type TradeId = String;

#[derive(Debug, PartialEq, Clone, Deserialize, Serialize)]
pub struct Candlestick {
    pub price_open : f32,
    pub price_close : f32,
    pub price_high : f32,
    pub price_low : f32
}

#[derive(Debug, PartialEq, Clone, Deserialize, Serialize)]
pub struct HistoryStep {
    pub timestamp : u32,
    pub bid_candle : Candlestick,
    pub ask_candle : Candlestick
}

#[derive(Debug, PartialEq, Clone, Deserialize, Serialize)]
pub struct HistoryMetadata {
    pub symbol : String,
    pub timeframe : HistoryTimeframe,
    pub from_date : DateTime<Utc>,
    pub to_date : DateTime<Utc>
}

#[derive(Debug, PartialEq, PartialOrd, Clone, Copy, EnumString, Deserialize, Serialize)]
pub enum HistoryTimeframe {
    Min1,
    Min5,
    Min15,
    Min30,
    Hour1,
    Hour2,
    Hour3,
    Hour4,
    Hour6,
    Hour8,
    Day1,
    Week1,
    Month1
}

impl HistoryTimeframe {
    pub fn in_minutes(&self) -> u64 {
        match self {
            HistoryTimeframe::Min1 => 1,
            HistoryTimeframe::Min5 => 5,
            HistoryTimeframe::Min15 => 15,
            HistoryTimeframe::Min30 => 30,
            HistoryTimeframe::Hour1 => 60,
            HistoryTimeframe::Hour2 => 2 * 60,
            HistoryTimeframe::Hour3 => 3 * 60,
            HistoryTimeframe::Hour4 => 4 * 60,
            HistoryTimeframe::Hour6 => 6 * 60,
            HistoryTimeframe::Hour8 => 8 * 60,
            HistoryTimeframe::Day1 => 24 * 60,
            HistoryTimeframe::Week1 => 7 * 24 * 60,
            HistoryTimeframe::Month1 => 30 * 24 * 60
        }
    }
}

#[derive(Debug, Default, PartialEq)]
pub struct TradeOptions {
    pub limit : Option<f32>,
    pub stop : Option<f32>
}

#[cfg_attr(test, automock)]
pub trait TradingService {
    fn get_trade_symbols(&mut self) -> anyhow::Result<Vec<String>>;
    fn get_symbol_history(&mut self, symbol : &str, timeframe : HistoryTimeframe,
        since_date : &DateTime<Utc>, to_date : &DateTime<Utc>) -> anyhow::Result<Vec<HistoryStep>>;
    fn max_history_steps_per_call(&mut self) -> anyhow::Result<u32>;

    fn get_market_update(&mut self, symbol : &str) -> anyhow::Result<(f32, f32)>;

    fn open_buy_trade(&mut self, symbol : &str, amount_in_lots : u32, options : &TradeOptions) -> anyhow::Result<TradeId>;
    fn open_sell_trade(&mut self, symbol : &str, amount_in_lots : u32, options : &TradeOptions) -> anyhow::Result<TradeId>;
    fn close_trade(&mut self, trade_id : &TradeId) -> anyhow::Result<()>;
}