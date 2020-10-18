use chrono::{DateTime, Utc, Duration};
use std::ops::Add;

use crate::trading_service::*;
use crate::storage::*;

#[cfg(test)]
use mockall::{predicate::*};
#[cfg(test)]
use chrono::{TimeZone};

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