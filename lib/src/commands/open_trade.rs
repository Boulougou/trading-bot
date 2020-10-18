use std::ops::Sub;
use chrono::{DateTime, Utc, Duration};
use anyhow::anyhow;

use crate::trading_service::*;
use crate::trading_model::*;

#[cfg(test)]
use mockall::{Sequence, predicate::*};
#[cfg(test)]
use chrono::{TimeZone};

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


#[cfg(test)]
mod tests {
    use super::*;

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