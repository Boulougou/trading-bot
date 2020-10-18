#[cfg(test)]
pub mod tests {
    use crate::trading_service::*;

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