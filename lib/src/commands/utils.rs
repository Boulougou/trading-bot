use crate::trading_service::*;

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