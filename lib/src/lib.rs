use chrono::{DateTime, Utc, Duration};
use std::ops::Add;
use serde::{Deserialize, Serialize};
use anyhow::{Context};
use strum::EnumString;

#[cfg(test)]
use mockall::{automock, predicate::*};
#[cfg(test)]
use chrono::{TimeZone};

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

#[derive(Debug, PartialEq, PartialOrd, Clone, Copy, EnumString)]
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
    fn in_minutes(&self) -> u64 {
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

#[cfg_attr(test, automock)]
pub trait TradingService {
    fn get_trade_symbols(&mut self) -> anyhow::Result<Vec<String>>;
    fn get_symbol_history(&mut self, symbol : &str, timeframe : HistoryTimeframe,
        since_date : &DateTime<Utc>, to_date : &DateTime<Utc>) -> anyhow::Result<Vec<HistoryStep>>;
    fn max_history_steps_per_call(&mut self) -> anyhow::Result<u32>;
    fn open_buy_trade(&mut self, symbol : &str, amount_in_lots : u32) -> anyhow::Result<TradeId>;
    fn open_sell_trade(&mut self, symbol : &str, amount_in_lots : u32) -> anyhow::Result<TradeId>;
    fn close_trade(&mut self, trade_id : &TradeId) -> anyhow::Result<()>;
}

#[cfg_attr(test, automock)]
pub trait TradingModel {
    fn train(&mut self, input : &Vec<HistoryStep>) -> anyhow::Result<()>;
}

#[cfg_attr(test, automock)]
pub trait Storage {
    fn save_symbol_history(&mut self, name : &str, history : &Vec<HistoryStep>) -> anyhow::Result<()>;
    fn load_symbol_history(&mut self, name : &str) -> anyhow::Result<Vec<HistoryStep>>;
}

pub fn fetch_symbol_history(service : &mut impl TradingService,
                            storage : &mut impl Storage,
                            symbol : &str,
                            timeframe : HistoryTimeframe,
                            since_date : &DateTime<Utc>,
                            to_date : &DateTime<Utc>) -> anyhow::Result<()> {
    let max_steps_per_call = service.max_history_steps_per_call()?;
    let minutes_per_step = timeframe.in_minutes() as i64;

    let duration = to_date.signed_duration_since(*since_date);
    let steps_needed = duration.num_minutes() / minutes_per_step;
    let requests_needed = steps_needed / max_steps_per_call as i64;

    let mut history = Vec::new();
    for i in 0..requests_needed {
        let request_from_date = since_date.add(Duration::minutes(i * minutes_per_step * max_steps_per_call as i64));
        let request_to_date = request_from_date.add(Duration::minutes(minutes_per_step * max_steps_per_call as i64));
        let mut steps = service.get_symbol_history(symbol, timeframe, &request_from_date, &request_to_date)?;
        history.append(&mut steps);
    }

    let last_request_steps = steps_needed % max_steps_per_call as i64;
    if last_request_steps > 0 {
        let request_from_date = since_date.add(Duration::minutes(minutes_per_step * requests_needed * max_steps_per_call as i64));
        let request_to_date = request_from_date.add(Duration::minutes(minutes_per_step * last_request_steps));
        let mut steps = service.get_symbol_history(symbol, timeframe, &request_from_date, &request_to_date)?;
        history.append(&mut steps);
    }

    let entry_name = format!("{}_{:?}_{}_{}",
        symbol.replace("/", "_"), timeframe,
        since_date.format("%Y%m%d%H%M"),
        to_date.format("%Y%m%d%H%M"));
    storage.save_symbol_history(&entry_name, &history)?;

    Ok(())
}

pub fn train_model(model : &mut impl TradingModel,
                   storage : &mut impl Storage,
                   input_entry : &str) -> anyhow::Result<()> {

    let history = storage.load_symbol_history(input_entry)?;
    model.train(&history)
}

/*
pub fn run(service : &mut impl TradingService, model : &mut impl TradingModel) -> anyhow::Result<()> {

    let trade_symbols_result = service.get_trade_symbols();
    match trade_symbols_result {
        Ok(symbols) => println!("Retrieved trade symbols, {:?}", symbols),
        Err(msg) => println!("Failed to retrieve trade symbols, {:?}", msg)
    }

    let open_buy_result = service.open_buy_trade("EUR/CAD", 1);
    let buy_trade_id = match open_buy_result {
        Ok(trade_id) => trade_id,
        Err(error_msg) => { println!("Failed {:?}", error_msg); return Err(error_msg); }
    };

    let open_sell_result = service.open_sell_trade("EUR/CAD", 1);
    let sell_trade_id = match open_sell_result {
        Ok(trade_id) => trade_id,
        Err(error_msg) => { println!("Failed {:?}", error_msg); return Err(error_msg); }
    };

    println!("Opened buy trade successfully {}", buy_trade_id);
    println!("Opened sell trade successfully {}", sell_trade_id);
    std::thread::sleep(std::time::Duration::from_secs(10));

    let close_result = service.close_trade(&sell_trade_id);
    match close_result {
        Ok(()) => println!("Successfully closed trade with id, {}", sell_trade_id),
        Err(error) => println!("Failed to close trade with id {}, error: {}", sell_trade_id, error),
    }

    let close_result = service.close_trade(&buy_trade_id);
    match close_result {
        Ok(()) => { println!("Successfully closed trade with id, {}", buy_trade_id); Ok(()) }
        Err(error) => { println!("Failed to close trade with id {}, error: {}", buy_trade_id, error); Err(error) }
    }
}*/


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fetch_history_fitting_in_one_request() -> anyhow::Result<()> {
        let mut service = MockTradingService::new();
        let mut storage = MockStorage::new();

        let max_history_steps_per_call : u32 = 100;
        service.expect_max_history_steps_per_call()
            .return_once(move || Ok(max_history_steps_per_call));

        let since_date = Utc.ymd(2019, 1, 1).and_hms(21, 30, 0);
        let to_date = since_date.add(Duration::minutes(max_history_steps_per_call as i64));
        service.expect_get_symbol_history()
            .with(eq("EUR/USD"), eq(HistoryTimeframe::Min1), eq(since_date), eq(to_date))
            .times(1)
            .return_once(move |_, _, _, _| Ok(build_history(max_history_steps_per_call)));

        storage.expect_save_symbol_history()
            .with(eq("EUR_USD_Min1_201901012130_201901012310"), eq(build_history(max_history_steps_per_call)))
            .times(1)
            .return_once(|_, _| Ok(()));

        fetch_symbol_history(&mut service, &mut storage, "EUR/USD", HistoryTimeframe::Min1, &since_date, &to_date)?;

        Ok(())
    }

    #[test]
    fn fetch_history_not_fitting_in_one_request() -> anyhow::Result<()> {
        let mut service = MockTradingService::new();
        let mut storage = MockStorage::new();

        let max_history_steps_per_call : u32 = 100;
        service.expect_max_history_steps_per_call()
            .return_once(move || Ok(max_history_steps_per_call));

        let since_date = Utc.ymd(2019, 1, 1).and_hms(21, 30, 0);
        let to_date = since_date.add(Duration::minutes(2 * max_history_steps_per_call as i64 + 1));
        service.expect_get_symbol_history()
            .with(eq("EUR/CAN"), eq(HistoryTimeframe::Min1), eq(since_date), eq(since_date.add(Duration::minutes(max_history_steps_per_call as i64))))
            .times(1)
            .return_once(move |_, _, _, _| Ok(build_history(max_history_steps_per_call)));
        service.expect_get_symbol_history()
            .with(eq("EUR/CAN"), eq(HistoryTimeframe::Min1), eq(since_date.add(Duration::minutes(max_history_steps_per_call as i64))),
                eq(since_date.add(Duration::minutes(2 * max_history_steps_per_call as i64))))
            .times(1)
            .return_once(move |_, _, _, _| Ok(build_history(max_history_steps_per_call)));
        service.expect_get_symbol_history()
            .with(eq("EUR/CAN"), eq(HistoryTimeframe::Min1), eq(since_date.add(Duration::minutes(2 * max_history_steps_per_call as i64))),
                eq(to_date))
            .times(1)
            .return_once(|_, _, _, _| Ok(build_history(1)));

        storage.expect_save_symbol_history()
            .with(eq("EUR_CAN_Min1_201901012130_201901020051"), eq(build_history(2 * max_history_steps_per_call + 1)))
            .times(1)
            .return_once(|_, _| Ok(()));

        fetch_symbol_history(&mut service, &mut storage, "EUR/CAN", HistoryTimeframe::Min1, &since_date, &to_date)?;

        Ok(())
    }

    #[test]
    fn fetch_history_using_min5_timeframe() -> anyhow::Result<()> {
        let mut service = MockTradingService::new();
        let mut storage = MockStorage::new();

        let max_history_steps_per_call : u32 = 10;
        service.expect_max_history_steps_per_call()
            .return_once(move || Ok(max_history_steps_per_call));

        let since_date = Utc.ymd(2019, 1, 1).and_hms(21, 30, 0);
        let to_date = since_date.add(Duration::minutes(5 * max_history_steps_per_call as i64 + 5));
        service.expect_get_symbol_history()
            .with(eq("EUR/CAN"), eq(HistoryTimeframe::Min5), eq(since_date), eq(since_date.add(Duration::minutes(5 * max_history_steps_per_call as i64))))
            .times(1)
            .return_once(move |_, _, _, _| Ok(build_history(max_history_steps_per_call)));
        service.expect_get_symbol_history()
            .with(eq("EUR/CAN"), eq(HistoryTimeframe::Min5), eq(since_date.add(Duration::minutes(5 * max_history_steps_per_call as i64))),
                eq(to_date))
            .times(1)
            .return_once(|_, _, _, _| Ok(build_history(1)));

        storage.expect_save_symbol_history()
            .with(eq("EUR_CAN_Min5_201901012130_201901012225"), eq(build_history(max_history_steps_per_call + 1)))
            .times(1)
            .return_once(|_, _| Ok(()));

        fetch_symbol_history(&mut service, &mut storage, "EUR/CAN", HistoryTimeframe::Min5, &since_date, &to_date)?;

        Ok(())
    }

    #[test]
    fn fetch_history_using_hour1_timeframe() -> anyhow::Result<()> {
        let mut service = MockTradingService::new();
        let mut storage = MockStorage::new();

        let max_history_steps_per_call : u32 = 10;
        service.expect_max_history_steps_per_call()
            .return_once(move || Ok(max_history_steps_per_call));

        let since_date = Utc.ymd(2019, 1, 1).and_hms(21, 30, 0);
        let to_date = since_date.add(Duration::minutes(60 * max_history_steps_per_call as i64 + 60));
        service.expect_get_symbol_history()
            .with(eq("EUR/CAN"), eq(HistoryTimeframe::Hour1), eq(since_date), eq(since_date.add(Duration::minutes(60 * max_history_steps_per_call as i64))))
            .times(1)
            .return_once(move |_, _, _, _| Ok(build_history(max_history_steps_per_call)));
        service.expect_get_symbol_history()
            .with(eq("EUR/CAN"), eq(HistoryTimeframe::Hour1), eq(since_date.add(Duration::minutes(60 * max_history_steps_per_call as i64))),
                eq(to_date))
            .times(1)
            .return_once(|_, _, _, _| Ok(build_history(1)));

        storage.expect_save_symbol_history()
            .with(eq("EUR_CAN_Hour1_201901012130_201901020830"), eq(build_history(max_history_steps_per_call + 1)))
            .times(1)
            .return_once(|_, _| Ok(()));

        fetch_symbol_history(&mut service, &mut storage, "EUR/CAN", HistoryTimeframe::Hour1, &since_date, &to_date)?;

        Ok(())
    }

    fn build_history(num_steps : u32) -> Vec<HistoryStep> {
        let step = HistoryStep {
            timestamp : 32432,
            bid_candle : Candlestick { price_open : 0.5, price_close : 0.4, price_high : 0.7, price_low : 0.1 },
            ask_candle : Candlestick { price_open : 0.5, price_close : 0.4, price_high : 0.7, price_low : 0.1 }
        };

        let mut history = Vec::new();
        for _ in 0..num_steps {
            history.push(step.clone());
        }

        history
    }
}