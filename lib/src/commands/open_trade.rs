use std::ops::Sub;
use chrono::{DateTime, Utc, Duration, TimeZone};
use anyhow::anyhow;
use getset::{Setters};

use crate::trading_service::*;
use crate::trading_model::*;
use crate::commands::utils;

#[derive(Debug, Setters)]
#[getset(set = "pub")]
pub struct OpenTradeOptions {
    pub min_profit_percent : f32,
    pub max_profit_percent : f32,
    pub max_used_margin_percent : f32
}

impl Default for OpenTradeOptions {
    fn default() -> Self {
        OpenTradeOptions { min_profit_percent : 0.0, max_profit_percent : f32::INFINITY, max_used_margin_percent : 1.0}
    }
}

pub fn open_trade(service : &mut impl TradingService,
                  model : &mut impl TradingModel,
                  amount : u32,
                  current_time : &DateTime<Utc>,
                  model_name : &str) -> anyhow::Result<(TradeId, Option<f32>, Option<f32>)> {
    open_trade_with_options(service, model, amount, current_time, model_name, &OpenTradeOptions::default())
}

pub fn open_trade_with_options(service : &mut impl TradingService,
                  model : &mut impl TradingModel,
                  amount : u32,
                  current_time : &DateTime<Utc>,
                  model_name : &str,
                  options : &OpenTradeOptions) -> anyhow::Result<(TradeId, Option<f32>, Option<f32>)> {
    let (symbol, timeframe) = model.load(model_name)?;

    let input_size = model.get_input_window()? as usize;
    let request_to_date = current_time;
    let history_buffer = 100;
    let request_from_date = request_to_date.sub(Duration::minutes(timeframe.in_minutes() as i64 * (history_buffer + input_size as i64)));
    let mut history = service.get_symbol_history(&symbol, timeframe, &request_from_date, &request_to_date)?;

    if timeframe > HistoryTimeframe::Hour8 {
        let last_timestamp = history.last().map(|s| s.timestamp).ok_or(anyhow!("Fetched history was empty"))?;
        let hour8_history = service.get_symbol_history(&symbol, HistoryTimeframe::Hour8,
                                                      &Utc.timestamp(last_timestamp as i64, 0), &request_to_date)?;
        history.push(utils::consolidate_history(&hour8_history));
    }
    else if timeframe > HistoryTimeframe::Min1 {
        let last_timestamp = history.last().map(|s| s.timestamp).ok_or(anyhow!("Fetched history was empty"))?;
        let min1_history = service.get_symbol_history(&symbol, HistoryTimeframe::Min1,
                                     &Utc.timestamp(last_timestamp as i64, 0), &request_to_date)?;
        history.push(utils::consolidate_history(&min1_history));
    }

    if history.len() < input_size {
        return Err(anyhow!("History size must be at least {} (was {})", input_size, history.len()));
    }

    let input = Vec::from(&history[history.len() - input_size..]);
    let (min_bid_price, max_bid_price) = model.predict(&input)?;

    let account_summary = service.get_account_summary()?;
    let (current_bid_price, current_ask_price) = service.get_market_update(&symbol)?;

    let used_maint_margin_percent = account_summary.used_maintenance_margin / account_summary.equity;
    if used_maint_margin_percent > options.max_used_margin_percent {
        return Err(anyhow!("Cannot surpass max allowed used margin, {} / {} = {} > {}",
            account_summary.used_maintenance_margin, account_summary.equity,
            used_maint_margin_percent, options.max_used_margin_percent));
    }

    let cost = current_ask_price - current_bid_price;
    let possible_income = max_bid_price - current_bid_price;
    let possible_profit = possible_income - cost;

    let profit_percent = possible_profit / cost;
    if profit_percent > options.min_profit_percent {
        let trade_limit = if profit_percent > options.max_profit_percent {
            current_bid_price + cost + options.max_profit_percent * cost
        }
        else {
            max_bid_price
        };

        let trade_options = TradeOptions { stop : None, limit : Some(trade_limit) };
        let trade_id = service.open_buy_trade(&symbol, amount, &trade_options)?;
        Ok((trade_id, trade_options.stop, trade_options.limit))
    }
    else {
        Err(anyhow!("Profit not possible: Cost({}) > Profit({}), (Bid({}), Ask({})), (MinBid({}), MaxBid({}))",
            current_bid_price, current_ask_price, min_bid_price, max_bid_price, cost, possible_profit))
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::utils;
    use crate::commands::utils::tests::*;
    use mockall::{Sequence, predicate::*};
    use chrono::{TimeZone};

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
            .return_once(|_| Ok((String::from("EUR/USD"), HistoryTimeframe::Min1)));
        model.expect_get_input_window()
            .times(1)
            .in_sequence(&mut seq)
            .return_once(move || Ok(input_window));

        let current_date = Utc.ymd(2019, 1, 1).and_hms(21, 30, 0);
        let since_date = current_date.sub(Duration::minutes(input_window as i64 + 100));
        service.expect_get_symbol_history()
            .with(eq("EUR/USD"), eq(HistoryTimeframe::Min1), eq(since_date), eq(current_date))
            .times(1)
            .return_once(move |_, _, _, _| Ok(build_history(input_window + 6)));

        model.expect_predict()
            .with(eq(build_history_offset(6, input_window)))
            .times(1)
            .return_once(|_| Ok((1.0, 1.9)));

        service.expect_get_account_summary()
            .times(1)
            .return_once(|| Ok(build_account_summary(1000.0, 10.0)));
        service.expect_get_market_update()
            .with(eq("EUR/USD"))
            .times(1)
            .return_once(|_| Ok((1.4, 1.6)));
        service.expect_open_buy_trade()
            .with(eq("EUR/USD"), eq(12), eq(TradeOptions { stop : None, limit : Some(1.9) }))
            .times(1)
            .return_once(|_, _, _| Ok(String::from("trade#1138")));
        let (trade_id, stop, limit) = open_trade(&mut service, &mut model, 12, &current_date, "trained_model")?;

        assert_eq!(trade_id, "trade#1138");
        assert_eq!(stop, None);
        assert_eq!(limit, Some(1.9));
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
            .return_once(|_| Ok((String::from("EUR/USD"), HistoryTimeframe::Min1)));
        model.expect_get_input_window()
            .times(1)
            .in_sequence(&mut seq)
            .return_once(move || Ok(input_window));

        let current_date = Utc.ymd(2019, 1, 1).and_hms(21, 30, 0);
        let since_date = current_date.sub(Duration::minutes(input_window as i64 + 100));
        service.expect_get_symbol_history()
            .with(eq("EUR/USD"), eq(HistoryTimeframe::Min1), eq(since_date), eq(current_date))
            .times(1)
            .return_once(move |_, _, _, _| Ok(build_history(input_window + 6)));

        model.expect_predict()
            .with(eq(build_history_offset(6, input_window)))
            .times(1)
            .return_once(|_| Ok((1.0, 1.7)));

        service.expect_get_account_summary()
            .times(1)
            .return_once(|| Ok(build_account_summary(1000.0, 10.0)));
        service.expect_get_market_update()
            .with(eq("EUR/USD"))
            .times(1)
            .return_once(|_| Ok((1.6, 1.8)));
        service.expect_open_buy_trade().never();
        let result = open_trade(&mut service, &mut model, 12, &current_date, "trained_model");

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
            .return_once(|_| Ok((String::from("EUR/USD"), HistoryTimeframe::Min1)));
        model.expect_get_input_window()
            .times(1)
            .in_sequence(&mut seq)
            .return_once(move || Ok(input_window));

        let current_date = Utc.ymd(2019, 1, 1).and_hms(21, 30, 0);
        let since_date = current_date.sub(Duration::minutes(input_window as i64 + 100));
        service.expect_get_symbol_history()
            .with(eq("EUR/USD"), eq(HistoryTimeframe::Min1), eq(since_date), eq(current_date))
            .times(1)
            .return_once(move |_, _, _, _| Ok(build_history(input_window + 6)));

        model.expect_predict()
            .with(eq(build_history_offset(6, input_window)))
            .times(1)
            .return_once(|_| Ok((1.0, 2.0)));

        service.expect_get_account_summary()
            .times(1)
            .return_once(|| Ok(build_account_summary(1000.0, 10.0)));
        service.expect_get_market_update()
            .with(eq("EUR/USD"))
            .times(1)
            .return_once(|_| Ok((1.4, 1.6)));
        service.expect_open_buy_trade()
            .with(eq("EUR/USD"), eq(12), eq(TradeOptions { stop : None, limit : Some(2.0) }))
            .times(1)
            .return_once(|_, _, _| Ok(String::from("trade#1138")));
        let (trade_id, stop, limit) = open_trade_with_options(&mut service, &mut model,
            12, &current_date, "trained_model",
            &OpenTradeOptions::default().set_min_profit_percent(0.25))?;

        assert_eq!(trade_id, "trade#1138");
        assert_eq!(stop, None);
        assert_eq!(limit, Some(2.0));
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
            .return_once(|_| Ok((String::from("EUR/USD"), HistoryTimeframe::Min1)));
        model.expect_get_input_window()
            .times(1)
            .in_sequence(&mut seq)
            .return_once(move || Ok(input_window));

        let current_date = Utc.ymd(2019, 1, 1).and_hms(21, 30, 0);
        let since_date = current_date.sub(Duration::minutes(input_window as i64 + 100));
        service.expect_get_symbol_history()
            .with(eq("EUR/USD"), eq(HistoryTimeframe::Min1), eq(since_date), eq(current_date))
            .times(1)
            .return_once(move |_, _, _, _| Ok(build_history(input_window + 6)));

        model.expect_predict()
            .with(eq(build_history_offset(6, input_window)))
            .times(1)
            .return_once(|_| Ok((1.0, 2.0)));

        service.expect_get_account_summary()
            .times(1)
            .return_once(|| Ok(build_account_summary(1000.0, 10.0)));
        service.expect_get_market_update()
            .with(eq("EUR/USD"))
            .times(1)
            .return_once(|_| Ok((1.4, 1.6)));
        service.expect_open_buy_trade().never();
        let result = open_trade_with_options(&mut service, &mut model,
            12, &current_date, "trained_model",
            &OpenTradeOptions::default().set_min_profit_percent(2.01));

        assert!(result.is_err());
    }

    #[test]
    fn open_trade_with_reduced_limit_when_passing_profit_limit() -> anyhow::Result<()> {
        let mut model = MockTradingModel::new();
        let mut service = MockTradingService::new();

        let input_window = 10;
        let mut seq = Sequence::new();
        model.expect_load()
            .with(eq("trained_model"))
            .times(1)
            .in_sequence(&mut seq)
            .return_once(|_| Ok((String::from("EUR/USD"), HistoryTimeframe::Min1)));
        model.expect_get_input_window()
            .times(1)
            .in_sequence(&mut seq)
            .return_once(move || Ok(input_window));

        let current_date = Utc.ymd(2019, 1, 1).and_hms(21, 30, 0);
        let since_date = current_date.sub(Duration::minutes(input_window as i64 + 100));
        service.expect_get_symbol_history()
            .with(eq("EUR/USD"), eq(HistoryTimeframe::Min1), eq(since_date), eq(current_date))
            .times(1)
            .return_once(move |_, _, _, _| Ok(build_history(input_window + 6)));

        model.expect_predict()
            .with(eq(build_history_offset(6, input_window)))
            .times(1)
            .return_once(|_| Ok((1.0, 2.0)));

        service.expect_get_account_summary()
            .times(1)
            .return_once(|| Ok(build_account_summary(1000.0, 10.0)));
        service.expect_get_market_update()
            .with(eq("EUR/USD"))
            .times(1)
            .return_once(|_| Ok((1.4, 1.6)));
        service.expect_open_buy_trade()
            .with(eq("EUR/USD"), eq(12), eq(TradeOptions { stop : None, limit : Some(1.7) }))
            .times(1)
            .return_once(|_, _, _| Ok(String::from("trade#1138")));
        let (trade_id, stop, limit) = open_trade_with_options(&mut service, &mut model,
                                                              12, &current_date, "trained_model",
                                                              &OpenTradeOptions::default().set_max_profit_percent(0.5))?;

        assert_eq!(trade_id, "trade#1138");
        assert_eq!(stop, None);
        assert_eq!(limit, Some(1.7));
        Ok(())
    }

    #[test]
    fn do_not_open_trade_when_used_margin_is_above_limit() {
        let mut model = MockTradingModel::new();
        let mut service = MockTradingService::new();

        let input_window = 10;
        let mut seq = Sequence::new();
        model.expect_load()
            .with(eq("trained_model"))
            .times(1)
            .in_sequence(&mut seq)
            .return_once(|_| Ok((String::from("EUR/USD"), HistoryTimeframe::Min1)));
        model.expect_get_input_window()
            .times(1)
            .in_sequence(&mut seq)
            .return_once(move || Ok(input_window));

        let current_date = Utc.ymd(2019, 1, 1).and_hms(21, 30, 0);
        let since_date = current_date.sub(Duration::minutes(input_window as i64 + 100));
        service.expect_get_symbol_history()
            .with(eq("EUR/USD"), eq(HistoryTimeframe::Min1), eq(since_date), eq(current_date))
            .times(1)
            .return_once(move |_, _, _, _| Ok(build_history(input_window + 6)));

        model.expect_predict()
            .with(eq(build_history_offset(6, input_window)))
            .times(1)
            .return_once(|_| Ok((2.0, 2.5)));

        service.expect_get_account_summary()
            .times(1)
            .return_once(|| Ok(build_account_summary(1000.0, 901.0)));
        service.expect_get_market_update()
            .with(eq("EUR/USD"))
            .times(1)
            .return_once(|_| Ok((1.7, 2.0)));
        service.expect_open_buy_trade().never();
        let result =
            open_trade_with_options(&mut service, &mut model,
                                    10, &current_date, "trained_model",
                                    &OpenTradeOptions::default().set_max_used_margin_percent(0.9));

        assert!(result.is_err());
    }

    #[test]
    fn open_trade_fetching_min1_candlesticks_when_passed_timeframe_not_granular_enough() -> anyhow::Result<()> {
        let mut model = MockTradingModel::new();
        let mut service = MockTradingService::new();

        let input_window = 10;
        let mut seq = Sequence::new();
        model.expect_load()
            .with(eq("trained_model"))
            .times(1)
            .in_sequence(&mut seq)
            .return_once(|_| Ok((String::from("EUR/USD"), HistoryTimeframe::Hour8)));
        model.expect_get_input_window()
            .times(1)
            .in_sequence(&mut seq)
            .return_once(move || Ok(input_window));

        let current_date = Utc.ymd(2019, 1, 1).and_hms(21, 30, 0);
        let since_date = current_date.sub(Duration::hours(8 * (input_window as i64 + 100)));
        let hour8_history = build_history(input_window + 6);
        let min1_history = build_history(100);
        let mut concatenated_history = build_history_offset(7, input_window - 1);
        concatenated_history.push(utils::consolidate_history(&min1_history));

        let last_hour8_date = Utc.timestamp(hour8_history.last().unwrap().timestamp as i64, 0);
        service.expect_get_symbol_history()
            .with(eq("EUR/USD"), eq(HistoryTimeframe::Hour8), eq(since_date), eq(current_date))
            .times(1)
            .return_once(move |_, _, _, _| Ok(hour8_history));
        service.expect_get_symbol_history()
            .with(eq("EUR/USD"), eq(HistoryTimeframe::Min1), eq(last_hour8_date), eq(current_date))
            .times(1)
            .return_once(move |_, _, _, _| Ok(min1_history));

        model.expect_predict()
            .with(eq(concatenated_history))
            .times(1)
            .return_once(|_| Ok((1.0, 1.9)));

        service.expect_get_account_summary()
            .times(1)
            .return_once(|| Ok(build_account_summary(1000.0, 10.0)));
        service.expect_get_market_update()
            .with(eq("EUR/USD"))
            .times(1)
            .return_once(|_| Ok((1.4, 1.6)));
        service.expect_open_buy_trade()
            .with(eq("EUR/USD"), eq(12), eq(TradeOptions { stop : None, limit : Some(1.9) }))
            .times(1)
            .return_once(|_, _, _| Ok(String::from("trade#1138")));
        open_trade(&mut service, &mut model, 12,
                   &current_date, "trained_model")?;

        Ok(())
    }

    #[test]
    fn open_trade_fetching_hour8_candlesticks_when_passed_timeframe_not_granular_enough() -> anyhow::Result<()> {
        let mut model = MockTradingModel::new();
        let mut service = MockTradingService::new();

        let input_window = 10;
        let mut seq = Sequence::new();
        model.expect_load()
            .with(eq("trained_model"))
            .times(1)
            .in_sequence(&mut seq)
            .return_once(|_| Ok((String::from("EUR/USD"), HistoryTimeframe::Month1)));
        model.expect_get_input_window()
            .times(1)
            .in_sequence(&mut seq)
            .return_once(move || Ok(input_window));

        let current_date = Utc.ymd(2019, 1, 1).and_hms(21, 30, 0);
        let since_date = current_date.sub(Duration::days(30 * (input_window as i64 + 100)));
        let month1_history = build_history(input_window + 6);
        let hour8_history = build_history(100);
        let mut concatenated_history = build_history_offset(7, input_window - 1);
        concatenated_history.push(utils::consolidate_history(&hour8_history));

        let last_month1_date = Utc.timestamp(month1_history.last().unwrap().timestamp as i64, 0);
        service.expect_get_symbol_history()
            .with(eq("EUR/USD"), eq(HistoryTimeframe::Month1), eq(since_date), eq(current_date))
            .times(1)
            .return_once(move |_, _, _, _| Ok(month1_history));
        service.expect_get_symbol_history()
            .with(eq("EUR/USD"), eq(HistoryTimeframe::Hour8), eq(last_month1_date), eq(current_date))
            .times(1)
            .return_once(move |_, _, _, _| Ok(hour8_history));

        model.expect_predict()
            .with(eq(concatenated_history))
            .times(1)
            .return_once(|_| Ok((1.0, 1.9)));

        service.expect_get_account_summary()
            .times(1)
            .return_once(|| Ok(build_account_summary(1000.0, 10.0)));
        service.expect_get_market_update()
            .with(eq("EUR/USD"))
            .times(1)
            .return_once(|_| Ok((1.4, 1.6)));
        service.expect_open_buy_trade()
            .with(eq("EUR/USD"), eq(12), eq(TradeOptions { stop : None, limit : Some(1.9) }))
            .times(1)
            .return_once(|_, _, _| Ok(String::from("trade#1138")));
        open_trade(&mut service, &mut model, 12,
                   &current_date, "trained_model")?;

        Ok(())
    }
    
    fn build_account_summary(equity : f32, used_maintenance_margin : f32) -> AccountSummary {
        AccountSummary { equity, used_margin : 0.0, usable_margin : 0.0,
            used_maintenance_margin, usable_maintenance_margin : 0.0 }
    }
}