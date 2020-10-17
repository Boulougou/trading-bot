use std::ops::Sub;
use chrono::{DateTime, Utc, Duration};
use std::ops::Add;
use serde::{Deserialize, Serialize};
use strum::EnumString;
use anyhow::anyhow;

#[cfg(test)]
use mockall::{automock, Sequence, predicate::*};
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

#[cfg_attr(test, automock(type TrainingParams = f32;))]
pub trait TradingModel {
    type TrainingParams;
    
    fn train(&mut self, input : &Vec<HistoryStep>, input_window : u32, prediction_window : u32, extra_params : &Self::TrainingParams) -> anyhow::Result<()>;
    fn predict(&mut self, input : &Vec<HistoryStep>) -> anyhow::Result<(f32, f32)>;
    
    fn get_input_window(&self) -> anyhow::Result<u32>;

    fn save(&mut self, name : &str) -> anyhow::Result<()>;
    fn load(&mut self, name : &str) -> anyhow::Result<()>;
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

pub fn train_model<T : TradingModel>(model : &mut T,
                   storage : &mut impl Storage,
                   input_name : &str,
                   input_window : u32,
                   prediction_window : u32,
                   extra_training_params : &T::TrainingParams,
                   output_name : &str) -> anyhow::Result<()> {

    let history = storage.load_symbol_history(input_name)?;
    model.train(&history, input_window, prediction_window, extra_training_params)?;
    model.save(output_name)
}

pub fn evaluate_model(model : &mut impl TradingModel,
                      storage : &mut impl Storage,
                      model_name : &str,
                      input_name : &str) -> anyhow::Result<Vec<(f32, f32)>> {
    model.load(model_name)?;

    let input_size = model.get_input_window()? as usize;
    let history = storage.load_symbol_history(input_name)?;

    let num_predictions = 1 + history.len() as i64 - input_size as i64;
    if num_predictions <= 0 {
        return Err(anyhow!("History size must be at least {}", input_size));
    }

    let mut results = Vec::new();
    for i in 0..num_predictions as usize {
        let current_input = Vec::from(&history[i..i + input_size]);
        let prediction = model.predict(&current_input)?;
        results.push(prediction);
    }

    // TODO Now we just return predictions without evaluating the model
    // either rename the existing method or change it to also do the evaluation based on the expected output
    Ok(results)
}

pub fn open_trade(service : &mut impl TradingService,
                  model : &mut impl TradingModel,
                  symbol : &str,
                  ammount : u32,
                  timeframe : HistoryTimeframe,
                  current_time : &DateTime<Utc>,
                  model_name : &str) -> anyhow::Result<(TradeId, TradeOptions)> {
    open_trade_with_profit(service, model, symbol, ammount, 0.0, timeframe, current_time, model_name)
}

pub fn open_trade_with_profit(service : &mut impl TradingService,
                  model : &mut impl TradingModel,
                  symbol : &str,
                  ammount : u32,
                  min_percent_profit : f32,
                  timeframe : HistoryTimeframe,
                  current_time : &DateTime<Utc>,
                  model_name : &str) -> anyhow::Result<(TradeId, TradeOptions)> {
    model.load(model_name)?;

    let input_size = model.get_input_window()? as usize;
    let request_to_date = current_time;
    let history_buffer = 5;
    let request_from_date = request_to_date.sub(Duration::minutes(history_buffer + input_size as i64));
    let history = service.get_symbol_history(symbol, timeframe, &request_from_date, &request_to_date)?;

    if history.len() < input_size {
        return Err(anyhow!("History size must be at least {} (was {})", input_size, history.len()));
    }

    let input = Vec::from(&history[history.len() - input_size..]);
    let (min_bid_price, max_bid_price) = model.predict(&input)?;

    let (current_bid_price, current_ask_price) = service.get_market_update(symbol)?;
    let cost = current_ask_price - current_bid_price;
    let possible_income = max_bid_price - current_bid_price;
    let possible_profit = possible_income - cost;

    if possible_profit / cost > min_percent_profit {
        let trade_options = TradeOptions { stop : None, limit : Some(max_bid_price) };
        let trade_id = service.open_buy_trade(symbol, ammount, &trade_options)?;
        Ok((trade_id, trade_options))
    }
    else {
        Err(anyhow!("Profit not possible: Cost({}) > Profit({}), (Bid({}), Ask({})), (MinBid({}), MaxBid({}))",
            current_bid_price, current_ask_price, min_bid_price, max_bid_price, cost, possible_profit))
    }
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
            .return_once(move |_, _, _, _| Ok(build_history_offset(0, max_history_steps_per_call)));
        service.expect_get_symbol_history()
            .with(eq("EUR/CAN"), eq(HistoryTimeframe::Min1), eq(since_date.add(Duration::minutes(max_history_steps_per_call as i64))),
                eq(since_date.add(Duration::minutes(2 * max_history_steps_per_call as i64))))
            .times(1)
            .return_once(move |_, _, _, _| Ok(build_history_offset(max_history_steps_per_call, max_history_steps_per_call)));
        service.expect_get_symbol_history()
            .with(eq("EUR/CAN"), eq(HistoryTimeframe::Min1), eq(since_date.add(Duration::minutes(2 * max_history_steps_per_call as i64))),
                eq(to_date))
            .times(1)
            .return_once(move |_, _, _, _| Ok(build_history_offset(2 * max_history_steps_per_call, 1)));

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
            .return_once(move |_, _, _, _| Ok(build_history_offset(0, max_history_steps_per_call)));
        service.expect_get_symbol_history()
            .with(eq("EUR/CAN"), eq(HistoryTimeframe::Min5), eq(since_date.add(Duration::minutes(5 * max_history_steps_per_call as i64))),
                eq(to_date))
            .times(1)
            .return_once(move |_, _, _, _| Ok(build_history_offset(max_history_steps_per_call, 1)));

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
            .return_once(move |_, _, _, _| Ok(build_history_offset(0, max_history_steps_per_call)));
        service.expect_get_symbol_history()
            .with(eq("EUR/CAN"), eq(HistoryTimeframe::Hour1), eq(since_date.add(Duration::minutes(60 * max_history_steps_per_call as i64))),
                eq(to_date))
            .times(1)
            .return_once(move |_, _, _, _| Ok(build_history_offset(max_history_steps_per_call, 1)));

        storage.expect_save_symbol_history()
            .with(eq("EUR_CAN_Hour1_201901012130_201901020830"), eq(build_history(max_history_steps_per_call + 1)))
            .times(1)
            .return_once(|_, _| Ok(()));

        fetch_symbol_history(&mut service, &mut storage, "EUR/CAN", HistoryTimeframe::Hour1, &since_date, &to_date)?;

        Ok(())
    }

    #[test]
    fn evaluate_model_with_history_smaller_than_input_size() {
        let mut model = MockTradingModel::new();
        let mut storage = MockStorage::new();

        let mut seq = Sequence::new();
        model.expect_load()
            .times(1)
            .with(eq("modelA"))
            .in_sequence(&mut seq)
            .return_once(|_| Ok(()));
        model.expect_get_input_window()
            .times(1)
            .in_sequence(&mut seq)
            .return_once(|| Ok(10));
        storage.expect_load_symbol_history()
            .with(eq("EUR_CAN_Hour1"))
            .times(1)
            .return_once(|_| Ok(build_history(9)));

        let result = evaluate_model(&mut model, &mut storage, "modelA", "EUR_CAN_Hour1");

        assert!(result.is_err());
    }

    #[test]
    fn evaluate_model_with_history_equal_to_input_size() -> anyhow::Result<()> {
        let mut model = MockTradingModel::new();
        let mut storage = MockStorage::new();

        let mut seq = Sequence::new();
        model.expect_load()
            .times(1)
            .with(eq("modelA"))
            .in_sequence(&mut seq)
            .return_once(|_| Ok(()));
        model.expect_get_input_window()
            .times(1)
            .in_sequence(&mut seq)
            .return_once(|| Ok(10));

        storage.expect_load_symbol_history()
            .with(eq("EUR_CAN_Hour1"))
            .times(1)
            .return_once(|_| Ok(build_history(10)));

        model.expect_predict()
            .with(eq(build_history(10)))
            .times(1)
            .return_once(|_| Ok((0.2, 0.5)));

        let result = evaluate_model(&mut model, &mut storage, "modelA", "EUR_CAN_Hour1")?;

        assert_eq!(result, vec!((0.2, 0.5)));
        Ok(())
    }

    #[test]
    fn evaluate_model_with_history_larger_from_input_size() -> anyhow::Result<()> {
        let mut model = MockTradingModel::new();
        let mut storage = MockStorage::new();

        let mut seq = Sequence::new();
        model.expect_load()
            .times(1)
            .with(eq("modelA"))
            .in_sequence(&mut seq)
            .return_once(|_| Ok(()));
        model.expect_get_input_window()
            .times(1)
            .in_sequence(&mut seq)
            .return_once(|| Ok(10));

        storage.expect_load_symbol_history()
            .with(eq("EUR_CAN_Hour1"))
            .times(1)
            .return_once(|_| Ok(build_history(12)));

        model.expect_predict()
            .with(eq(build_history_offset(0, 10)))
            .times(1)
            .return_once(|_| Ok((0.2, 0.5)));
        model.expect_predict()
            .with(eq(build_history_offset(1, 10)))
            .times(1)
            .return_once(|_| Ok((0.1, 0.6)));
        model.expect_predict()
            .with(eq(build_history_offset(2, 10)))
            .times(1)
            .return_once(|_| Ok((-0.2, -0.5)));

        let result = evaluate_model(&mut model, &mut storage, "modelA", "EUR_CAN_Hour1")?;

        assert_eq!(result, vec!((0.2, 0.5), (0.1, 0.6), (-0.2, -0.5)));
        Ok(())
    }

    #[test]
    fn evaluate_model_failing_to_load_model() {
        let mut model = MockTradingModel::new();
        let mut storage = MockStorage::new();

        model.expect_load()
            .times(1)
            .with(eq("modelB"))
            .return_once(|_| Err(anyhow!("Failed")));
        let result = evaluate_model(&mut model, &mut storage, "modelB", "EUR_CAN_Hour1");

        assert!(result.is_err());
    }

    #[test]
    fn evaluate_model_failing_to_get_input_window() {
        let mut model = MockTradingModel::new();
        let mut storage = MockStorage::new();

        model.expect_load()
            .times(1)
            .with(eq("modelB"))
            .return_once(|_| Ok(()));
        model.expect_get_input_window()
            .times(1)
            .return_once(|| Err(anyhow!("Failed")));
        let result = evaluate_model(&mut model, &mut storage, "modelB", "EUR_CAN_Hour1");

        assert!(result.is_err());
    }

    #[test]
    fn evaluate_model_failing_to_load_input() {
        let mut model = MockTradingModel::new();
        let mut storage = MockStorage::new();

        model.expect_load()
            .times(1)
            .with(eq("modelB"))
            .return_once(|_| Ok(()));
        model.expect_get_input_window()
            .times(1)
            .return_once(|| Ok(10));
        storage.expect_load_symbol_history()
            .with(eq("EUR_CAN_Hour1"))
            .times(1)
            .return_once(|_| Err(anyhow!("Failed")));
        let result = evaluate_model(&mut model, &mut storage, "modelB", "EUR_CAN_Hour1");

        assert!(result.is_err());
    }

    #[test]
    fn evaluate_model_failing_run_prediction() {
        let mut model = MockTradingModel::new();
        let mut storage = MockStorage::new();

        model.expect_load()
            .times(1)
            .with(eq("modelB"))
            .return_once(|_| Ok(()));
        model.expect_get_input_window()
            .times(1)
            .return_once(|| Ok(10));
        storage.expect_load_symbol_history()
            .with(eq("EUR_CAN_Hour1"))
            .times(1)
            .return_once(|_| Ok(build_history(10)));

        model.expect_predict()
            .with(eq(build_history(10)))
            .times(1)
            .return_once(|_| Err(anyhow!("Failed")));
        let result = evaluate_model(&mut model, &mut storage, "modelB", "EUR_CAN_Hour1");

        assert!(result.is_err());
    }

    #[test]
    fn open_trade_when_prediction_range_covers_current_spread() -> anyhow::Result<()> {
        let mut model = MockTradingModel::new();
        let mut service = MockTradingService::new();

        let input_window = 10;
        let mut seq = Sequence::new();
        model.expect_load()
            .with(eq("trained_model"))
            .times(1)
            .in_sequence(&mut seq)
            .return_once(|_| Ok(()));
        model.expect_get_input_window()
            .times(1)
            .in_sequence(&mut seq)
            .return_once(move || Ok(input_window));

        let current_date = Utc.ymd(2019, 1, 1).and_hms(21, 30, 0);
        let since_date = current_date.sub(Duration::minutes(input_window as i64 + 5));
        service.expect_get_symbol_history()
            .with(eq("EUR/USD"), eq(HistoryTimeframe::Min1), eq(since_date), eq(current_date))
            .times(1)
            .return_once(move |_, _, _, _| Ok(build_history(input_window + 6)));

        model.expect_predict()
            .with(eq(build_history_offset(6, input_window)))
            .times(1)
            .return_once(|_| Ok((1.0, 1.9)));

        service.expect_get_market_update()
            .with(eq("EUR/USD"))
            .times(1)
            .return_once(|_| Ok((1.4, 1.6)));
        service.expect_open_buy_trade()
            .with(eq("EUR/USD"), eq(12), eq(TradeOptions { stop : None, limit : Some(1.9) }))
            .times(1)
            .return_once(|_, _, _| Ok(String::from("trade#1138")));
        let (trade_id, trade_options) = open_trade(&mut service, &mut model, "EUR/USD", 12, HistoryTimeframe::Min1, &current_date, "trained_model")?;

        assert_eq!(trade_id, "trade#1138");
        assert_eq!(trade_options, TradeOptions { stop : None, limit : Some(1.9) });
        Ok(())
    }

    #[test]
    fn do_not_open_trade_when_prediction_does_not_cover_current_spread() {
        let mut model = MockTradingModel::new();
        let mut service = MockTradingService::new();

        let input_window = 10;
        let mut seq = Sequence::new();
        model.expect_load()
            .with(eq("trained_model"))
            .times(1)
            .in_sequence(&mut seq)
            .return_once(|_| Ok(()));
        model.expect_get_input_window()
            .times(1)
            .in_sequence(&mut seq)
            .return_once(move || Ok(input_window));

        let current_date = Utc.ymd(2019, 1, 1).and_hms(21, 30, 0);
        let since_date = current_date.sub(Duration::minutes(input_window as i64 + 5));
        service.expect_get_symbol_history()
            .with(eq("EUR/USD"), eq(HistoryTimeframe::Min1), eq(since_date), eq(current_date))
            .times(1)
            .return_once(move |_, _, _, _| Ok(build_history(input_window + 6)));

        model.expect_predict()
            .with(eq(build_history_offset(6, input_window)))
            .times(1)
            .return_once(|_| Ok((1.0, 1.7)));

        service.expect_get_market_update()
            .with(eq("EUR/USD"))
            .times(1)
            .return_once(|_| Ok((1.6, 1.8)));
        service.expect_open_buy_trade().never();
        let result = open_trade(&mut service, &mut model, "EUR/USD", 12, HistoryTimeframe::Min1, &current_date, "trained_model");

        assert!(result.is_err());
    }

    #[test]
    fn open_trade_when_prediction_range_covers_desired_profit() -> anyhow::Result<()> {
        let mut model = MockTradingModel::new();
        let mut service = MockTradingService::new();

        let input_window = 10;
        let mut seq = Sequence::new();
        model.expect_load()
            .with(eq("trained_model"))
            .times(1)
            .in_sequence(&mut seq)
            .return_once(|_| Ok(()));
        model.expect_get_input_window()
            .times(1)
            .in_sequence(&mut seq)
            .return_once(move || Ok(input_window));

        let current_date = Utc.ymd(2019, 1, 1).and_hms(21, 30, 0);
        let since_date = current_date.sub(Duration::minutes(input_window as i64 + 5));
        service.expect_get_symbol_history()
            .with(eq("EUR/USD"), eq(HistoryTimeframe::Min1), eq(since_date), eq(current_date))
            .times(1)
            .return_once(move |_, _, _, _| Ok(build_history(input_window + 6)));

        model.expect_predict()
            .with(eq(build_history_offset(6, input_window)))
            .times(1)
            .return_once(|_| Ok((1.0, 2.0)));

        service.expect_get_market_update()
            .with(eq("EUR/USD"))
            .times(1)
            .return_once(|_| Ok((1.4, 1.6)));
        service.expect_open_buy_trade()
            .with(eq("EUR/USD"), eq(12), eq(TradeOptions { stop : None, limit : Some(2.0) }))
            .times(1)
            .return_once(|_, _, _| Ok(String::from("trade#1138")));
        let (trade_id, trade_options) = open_trade_with_profit(&mut service, &mut model,
            "EUR/USD", 12, 0.25, HistoryTimeframe::Min1, &current_date, "trained_model")?;

        assert_eq!(trade_id, "trade#1138");
        assert_eq!(trade_options, TradeOptions { stop : None, limit : Some(2.0) });
        Ok(())
    }

    #[test]
    fn do_not_open_trade_when_prediction_range_does_not_cover_desired_profit() {
        let mut model = MockTradingModel::new();
        let mut service = MockTradingService::new();

        let input_window = 10;
        let mut seq = Sequence::new();
        model.expect_load()
            .with(eq("trained_model"))
            .times(1)
            .in_sequence(&mut seq)
            .return_once(|_| Ok(()));
        model.expect_get_input_window()
            .times(1)
            .in_sequence(&mut seq)
            .return_once(move || Ok(input_window));

        let current_date = Utc.ymd(2019, 1, 1).and_hms(21, 30, 0);
        let since_date = current_date.sub(Duration::minutes(input_window as i64 + 5));
        service.expect_get_symbol_history()
            .with(eq("EUR/USD"), eq(HistoryTimeframe::Min1), eq(since_date), eq(current_date))
            .times(1)
            .return_once(move |_, _, _, _| Ok(build_history(input_window + 6)));

        model.expect_predict()
            .with(eq(build_history_offset(6, input_window)))
            .times(1)
            .return_once(|_| Ok((1.0, 2.0)));

        service.expect_get_market_update()
            .with(eq("EUR/USD"))
            .times(1)
            .return_once(|_| Ok((1.4, 1.6)));
        service.expect_open_buy_trade().never();
        let result = open_trade_with_profit(&mut service, &mut model,
            "EUR/USD", 12, 2.01, HistoryTimeframe::Min1, &current_date, "trained_model");

        assert!(result.is_err());
    }

    fn build_history(num_steps : u32) -> Vec<HistoryStep> {
        build_history_offset(0, num_steps)
    }

    fn build_history_offset(offset : u32, num_steps : u32) -> Vec<HistoryStep> {
        let step = HistoryStep {
            timestamp : 32432,
            bid_candle : Candlestick { price_open : 0.5, price_close : 0.4, price_high : 0.7, price_low : 0.1 },
            ask_candle : Candlestick { price_open : 0.5, price_close : 0.4, price_high : 0.7, price_low : 0.1 }
        };

        let mut history = Vec::new();
        for i in 0..num_steps {
            let current_step = HistoryStep { timestamp : i + offset, ..step.clone() };
            history.push(current_step);
        }

        history
    }
}