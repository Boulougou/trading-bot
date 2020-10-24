use crate::trading_service::*;
use anyhow::anyhow;

pub fn consolidate_history(min1_history: &Vec<HistoryStep>) -> HistoryStep {
    let first_step = min1_history.first().unwrap();
    let last_step = min1_history.last().unwrap();
    let mut consolidated_step = HistoryStep {
        timestamp : first_step.timestamp,
        bid_candle : Candlestick { price_open : first_step.bid_candle.price_open, price_close : last_step.bid_candle.price_close,
            price_high : f32::NEG_INFINITY, price_low : f32::INFINITY },
        ask_candle : Candlestick { price_open : first_step.ask_candle.price_open, price_close : last_step.ask_candle.price_close,
            price_high : f32::NEG_INFINITY, price_low : f32::INFINITY },
    };

    for s in min1_history {
        if s.bid_candle.price_high > consolidated_step.bid_candle.price_high {
            consolidated_step.bid_candle.price_high = s.bid_candle.price_high;
        }

        if s.ask_candle.price_high > consolidated_step.ask_candle.price_high {
            consolidated_step.ask_candle.price_high = s.ask_candle.price_high;
        }
        if s.bid_candle.price_low < consolidated_step.bid_candle.price_low {
            consolidated_step.bid_candle.price_low = s.bid_candle.price_low;
        }

        if s.ask_candle.price_low < consolidated_step.ask_candle.price_low {
            consolidated_step.ask_candle.price_low = s.ask_candle.price_low;
        }
    }

    consolidated_step
}

pub fn extract_input_and_prediction_windows(history : &[HistoryStep], input_window : u32, prediction_window : u32)
                                        -> anyhow::Result<Vec<(Vec<HistoryStep>, Vec<HistoryStep>)>> {
    let input_size = history.len() as i64 - prediction_window as i64 - input_window as i64 + 1;
    if input_size <= 0 {
        return Err(anyhow!("History size {} is not big enough for input window {} and prediction window {}",
            history.len(), input_window, prediction_window));
    }
    let input_size = input_size as usize;

    let mut input_events = Vec::new();
    for i in 0..input_size {
        let input_end = i + input_window as usize;
        let input_steps = Vec::from(&history[i..input_end]);
        let future_steps = Vec::from(&history[input_end..input_end + prediction_window as usize]);

        input_events.push((input_steps, future_steps));
    }

    Ok(input_events)
}

pub fn find_bid_price_range(history : &[HistoryStep]) -> (f32, f32) {
    let mut min_bid_price = f32::INFINITY;
    let mut max_bid_price = f32::NEG_INFINITY;
    for s in history {
        min_bid_price = min_bid_price.min(s.bid_candle.price_low);
        max_bid_price = max_bid_price.max(s.bid_candle.price_high);
    }

    (min_bid_price, max_bid_price)
}

pub fn adjust_prediction((current_bid_price, current_ask_price) : (f32, f32),
                     (predicted_min_bid_price, predicted_max_bid_price) : (f32, f32),
                     min_profit_percent : f32,
                     max_profit_percent : f32,
                     max_loss_percent : f32) -> anyhow::Result<(Option<f32>, Option<f32>)> {
    let cost = current_ask_price - current_bid_price;
    let possible_income = predicted_max_bid_price - current_bid_price;
    let possible_profit = possible_income - cost;

    let profit_percent = possible_profit / cost;
    if profit_percent <= min_profit_percent {
        return Err(anyhow!("Profit not possible: Profit({}) / Cost({}) = {} < MinProfit({}), (Bid({}), Ask({})), (MinBid({}), MaxBid({}))",
            possible_profit, cost, profit_percent, min_profit_percent,
            current_bid_price, current_ask_price, predicted_min_bid_price, predicted_max_bid_price))
    }

    let trade_limit = if profit_percent > max_profit_percent {
        current_bid_price + cost + max_profit_percent * cost
    }
    else {
        predicted_max_bid_price
    };

    let trade_stop = current_bid_price - (max_loss_percent - 1.0) * cost;
    let maybe_trade_stop = if trade_stop > 0.0 { Some(trade_stop) } else { None };

    Ok((maybe_trade_stop, Some(trade_limit)))
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn from_min1_history_to_hour8() {
        let mut history = Vec::new();
        for i in 0..4 {
            let bid_base = 2.0 + i as f32;
            let ask_base = -(i as f32) - 2.0;
            let step = HistoryStep {
                timestamp : i + 100,
                bid_candle : Candlestick { price_open : bid_base, price_close : bid_base, price_high : bid_base + 100.0, price_low : -bid_base - 100.0 },
                ask_candle : Candlestick { price_open : ask_base, price_close : ask_base, price_high : ask_base + 100.0, price_low : -ask_base - 100.0 },
            };
            history.push(step);
        }

        let result = consolidate_history(&history);
        let expected_result = HistoryStep {
            timestamp : 100,
            bid_candle : Candlestick { price_open : 2.0, price_close : 5.0, price_high : 105.0, price_low : -105.0 },
            ask_candle : Candlestick { price_open : -2.0, price_close : -5.0, price_high : 98.0, price_low : -98.0 },
        };
        assert_eq!(result, expected_result);
    }

    pub fn build_history(num_steps : u32) -> Vec<HistoryStep> {
        build_history_offset(0, num_steps)
    }

    pub fn build_history_offset(offset : u32, num_steps : u32) -> Vec<HistoryStep> {

        let mut history = Vec::new();
        for i in 0..num_steps {
            let base = i as f32 + offset as f32 + 1.0;
            let step = HistoryStep {
                timestamp : i + offset,
                bid_candle : Candlestick { price_open : base, price_close : -base, price_high : 2.0 * base, price_low : -2.0 * base },
                ask_candle : Candlestick { price_open : base, price_close : -base, price_high : 3.0 * base, price_low : -3.0 * base }
            };
            history.push(step);
        }

        history
    }
}