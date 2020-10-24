use crate::trading_model::*;
use crate::storage::*;
use crate::{utils, HistoryStep};
use getset::{Setters};
use anyhow::Context;

#[derive(Debug, Setters)]
#[getset(set = "pub")]
pub struct PredictionEvaluationOptions {
    pub min_profit_percent : f32,
    pub max_profit_percent : f32,
    pub max_loss_percent : f32
}

impl Default for PredictionEvaluationOptions {
    fn default() -> Self {
        PredictionEvaluationOptions { min_profit_percent : 0.0, max_profit_percent : f32::INFINITY,
            max_loss_percent : f32::INFINITY }
    }
}

pub fn evaluate_model(model : &mut impl TradingModel,
                      storage : &mut impl Storage,
                      model_name : &str,
                      input_name : &str) -> anyhow::Result<(f32, f32)> {
    let (_symbol, _timeframe, input_window, prediction_window) = model.load(model_name)?;

    let (history, _metadata) = storage.load_symbol_history(input_name)?;

    let input_events = utils::extract_input_and_prediction_windows(&history, input_window as u32, prediction_window)?;

    let mut predictions = Vec::new();
    let mut expectations = Vec::new();
    let mut profit_or_loss = 0.0;
    for (input_steps, future_steps) in input_events {
        let prediction = model.predict(&input_steps)?;
        predictions.push(prediction);

        let expectation = utils::find_bid_price_range(&future_steps);
        expectations.push(expectation);

        let current_profit_or_loss = evaluate_prediction(prediction, &future_steps)?;
        profit_or_loss += current_profit_or_loss;
    }

    let model_loss = model.calculate_loss(&predictions, &expectations);
    Ok((model_loss, profit_or_loss))
}

fn evaluate_prediction((min_predicted_price, max_predicted_price) : (f32, f32), future_steps : &[HistoryStep]) -> anyhow::Result<f32> {
    let first_step = future_steps.first().context("There should be at least 1 future step")?;
    let pos_ask_price = first_step.ask_candle.price_open;
    let pos_bid_price = first_step.bid_candle.price_open;
    let pos_cost = pos_ask_price - pos_bid_price;

    for s in future_steps {
        if s.bid_candle.price_low < min_predicted_price {
            return Ok(-(pos_bid_price - min_predicted_price) - pos_cost);
        }

        if s.bid_candle.price_high > max_predicted_price {
            return Ok(max_predicted_price - pos_bid_price - pos_cost);
        }
    }

    Ok(-(pos_bid_price - min_predicted_price) - pos_cost)
}


#[cfg(test)]
mod tests {
    use anyhow::anyhow;
    use super::*;
    use crate::utils::tests::*;
    use mockall::{Sequence, predicate::*};
    use crate::trading_service::{HistoryMetadata, HistoryStep, HistoryTimeframe};
    use chrono::{Utc};
    use crate::Candlestick;

    #[test]
    fn evaluate_model_with_history_smaller_than_required_size() {
        let mut model = MockTradingModel::new();
        let mut storage = MockStorage::new();

        let mut seq = Sequence::new();
        model.expect_load()
            .times(1)
            .with(eq("modelA"))
            .in_sequence(&mut seq)
            .return_once(|_| Ok((String::from("EUR/CAN"), HistoryTimeframe::Hour1, 8, 2)));
        storage.expect_load_symbol_history()
            .with(eq("EUR_CAN_Hour1"))
            .times(1)
            .return_once(|_| Ok(build_history_and_meta(9, "EUR/CAN", HistoryTimeframe::Hour1)));

        let result = evaluate_model(&mut model, &mut storage, "modelA", "EUR_CAN_Hour1");

        assert!(result.is_err());
    }

    #[test]
    fn evaluate_model_with_history_equal_to_required_size() -> anyhow::Result<()> {
        let mut model = MockTradingModel::new();
        let mut storage = MockStorage::new();

        let mut seq = Sequence::new();
        model.expect_load()
            .times(1)
            .with(eq("modelA"))
            .in_sequence(&mut seq)
            .return_once(|_| Ok((String::from("EUR/CAN"), HistoryTimeframe::Hour1, 10, 5)));

        storage.expect_load_symbol_history()
            .with(eq("EUR_CAN_Hour1"))
            .times(1)
            .return_once(|_| Ok(build_history_and_meta(15, "EUR/CAN", HistoryTimeframe::Hour1)));

        model.expect_predict()
            .with(eq(build_history(10)))
            .times(1)
            .return_once(|_| Ok((0.2, 0.5)));
        model.expect_calculate_loss()
            .withf(|predictions, expectations| predictions == vec!((0.2, 0.5)) && expectations == vec!((-30.0, 30.0)))
            .times(1)
            .return_once(|_, _| 0.564);

        let (model_loss, _profit_or_loss) = evaluate_model(&mut model, &mut storage, "modelA", "EUR_CAN_Hour1")?;

        assert_eq!(model_loss, 0.564);
        Ok(())
    }

    #[test]
    fn evaluate_model_with_history_larger_than_input_size() -> anyhow::Result<()> {
        let mut model = MockTradingModel::new();
        let mut storage = MockStorage::new();

        let mut seq = Sequence::new();
        model.expect_load()
            .times(1)
            .with(eq("modelA"))
            .in_sequence(&mut seq)
            .return_once(|_| Ok((String::from("EUR/CAN"), HistoryTimeframe::Hour1, 10, 5)));

        storage.expect_load_symbol_history()
            .with(eq("EUR_CAN_Hour1"))
            .times(1)
            .return_once(|_| Ok(build_history_and_meta(17, "EUR/CAN", HistoryTimeframe::Hour1)));

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

        model.expect_calculate_loss()
            .withf(|predictions, expectations| predictions == vec!((0.2, 0.5), (0.1, 0.6), (-0.2, -0.5)) &&
                expectations == vec!((-30.0, 30.0), (-32.0, 32.0), (-34.0, 34.0)))
            .times(1)
            .return_once(|_, _| 0.478);

        let (model_loss, _profit_or_loss) = evaluate_model(&mut model, &mut storage, "modelA", "EUR_CAN_Hour1")?;

        assert_eq!(model_loss, 0.478);
        Ok(())
    }

    #[test]
    fn return_profit_when_max_predicted_price_is_reached() -> anyhow::Result<()> {
        let mut model = MockTradingModel::new();
        let mut storage = MockStorage::new();

        model.expect_load()
            .times(1)
            .return_once(|_| Ok((String::from("EUR/CAN"), HistoryTimeframe::Hour1, 2, 2)));

        let history = vec!(
            HistoryStep {
                timestamp : 1,
                bid_candle : Candlestick { price_open : 1.0, price_close : 1.0, price_high : 1.0, price_low : 1.0 },
                ask_candle : Candlestick { price_open : 1.0, price_close : 1.0, price_high : 1.0, price_low : 1.0 },
            },
            HistoryStep {
                timestamp : 2,
                bid_candle : Candlestick { price_open : 1.0, price_close : 1.0, price_high : 1.0, price_low : 1.0 },
                ask_candle : Candlestick { price_open : 1.0, price_close : 1.5, price_high : 1.0, price_low : 1.0 },
            },
            HistoryStep {
                timestamp : 3,
                bid_candle : Candlestick { price_open : 1.0, price_close : 1.0, price_high : 1.0, price_low : 1.0 },
                ask_candle : Candlestick { price_open : 1.5, price_close : 1.0, price_high : 1.0, price_low : 1.0 },
            },
            HistoryStep {
                timestamp : 4,
                bid_candle : Candlestick { price_open : 1.0, price_close : 1.0, price_high : 1.7, price_low : 1.0 },
                ask_candle : Candlestick { price_open : 1.0, price_close : 1.0, price_high : 1.0, price_low : 1.0 },
            });

        let history_meta = build_history_metadata("EUR/CAN", HistoryTimeframe::Hour1);
        storage.expect_load_symbol_history()
            .times(1)
            .return_once(|_| Ok((history, history_meta)));

        model.expect_predict()
            .times(1)
            .return_once(|_| Ok((0.2, 1.6)));
        model.expect_calculate_loss()
            .times(1)
            .return_once(|_, _| 0.777);

        let (_model_loss, profit_or_loss) = evaluate_model(&mut model, &mut storage, "modelA", "EUR_CAN_Hour1")?;

        assert!((profit_or_loss - 0.1).abs() < 0.00001);
        Ok(())
    }

    #[test]
    fn return_loss_when_max_predicted_price_is_never_reached() -> anyhow::Result<()> {
        let mut model = MockTradingModel::new();
        let mut storage = MockStorage::new();

        model.expect_load()
            .times(1)
            .return_once(|_| Ok((String::from("EUR/CAN"), HistoryTimeframe::Hour1, 2, 2)));

        let history = vec!(
            HistoryStep {
                timestamp : 1,
                bid_candle : Candlestick { price_open : 1.0, price_close : 1.0, price_high : 1.0, price_low : 1.0 },
                ask_candle : Candlestick { price_open : 1.0, price_close : 1.0, price_high : 1.0, price_low : 1.0 },
            },
            HistoryStep {
                timestamp : 2,
                bid_candle : Candlestick { price_open : 1.0, price_close : 1.0, price_high : 1.0, price_low : 1.0 },
                ask_candle : Candlestick { price_open : 1.0, price_close : 1.5, price_high : 1.0, price_low : 1.0 },
            },
            HistoryStep {
                timestamp : 3,
                bid_candle : Candlestick { price_open : 1.0, price_close : 1.0, price_high : 1.0, price_low : 1.0 },
                ask_candle : Candlestick { price_open : 1.5, price_close : 1.0, price_high : 1.0, price_low : 1.0 },
            },
            HistoryStep {
                timestamp : 4,
                bid_candle : Candlestick { price_open : 1.0, price_close : 1.0, price_high : 1.5, price_low : 1.0 },
                ask_candle : Candlestick { price_open : 1.0, price_close : 1.0, price_high : 1.0, price_low : 1.0 },
            });

        let history_meta = build_history_metadata("EUR/CAN", HistoryTimeframe::Hour1);
        storage.expect_load_symbol_history()
            .times(1)
            .return_once(|_| Ok((history, history_meta)));

        model.expect_predict()
            .times(1)
            .return_once(|_| Ok((0.2, 1.6)));
        model.expect_calculate_loss()
            .times(1)
            .return_once(|_, _| 0.777);

        let (_model_loss, profit_or_loss) = evaluate_model(&mut model, &mut storage, "modelA", "EUR_CAN_Hour1")?;

        assert!((profit_or_loss + 1.3).abs() < 0.00001);
        Ok(())
    }

    #[test]
    fn return_loss_when_min_predicted_price_is_reached() -> anyhow::Result<()> {
        let mut model = MockTradingModel::new();
        let mut storage = MockStorage::new();

        model.expect_load()
            .times(1)
            .return_once(|_| Ok((String::from("EUR/CAN"), HistoryTimeframe::Hour1, 2, 2)));

        let history = vec!(
            HistoryStep {
                timestamp : 1,
                bid_candle : Candlestick { price_open : 1.0, price_close : 1.0, price_high : 1.0, price_low : 1.0 },
                ask_candle : Candlestick { price_open : 1.0, price_close : 1.0, price_high : 1.0, price_low : 1.0 },
            },
            HistoryStep {
                timestamp : 2,
                bid_candle : Candlestick { price_open : 1.0, price_close : 1.0, price_high : 1.0, price_low : 1.0 },
                ask_candle : Candlestick { price_open : 1.0, price_close : 1.5, price_high : 1.0, price_low : 1.0 },
            },
            HistoryStep {
                timestamp : 3,
                bid_candle : Candlestick { price_open : 1.0, price_close : 1.0, price_high : 1.7, price_low : 0.1 },
                ask_candle : Candlestick { price_open : 1.5, price_close : 1.0, price_high : 1.0, price_low : 1.0 },
            },
            HistoryStep {
                timestamp : 4,
                bid_candle : Candlestick { price_open : 1.0, price_close : 1.0, price_high : 1.0, price_low : 1.0 },
                ask_candle : Candlestick { price_open : 1.0, price_close : 1.0, price_high : 1.0, price_low : 1.0 },
            });

        let history_meta = build_history_metadata("EUR/CAN", HistoryTimeframe::Hour1);
        storage.expect_load_symbol_history()
            .times(1)
            .return_once(|_| Ok((history, history_meta)));

        model.expect_predict()
            .times(1)
            .return_once(|_| Ok((0.2, 1.6)));
        model.expect_calculate_loss()
            .times(1)
            .return_once(|_, _| 0.777);

        let (_model_loss, profit_or_loss) = evaluate_model(&mut model, &mut storage, "modelA", "EUR_CAN_Hour1")?;

        assert!((profit_or_loss + 1.3).abs() < 0.00001);
        Ok(())
    }

    #[test]
    fn return_accumulated_profit_or_loss_when_many_evaluations() -> anyhow::Result<()> {
        let mut model = MockTradingModel::new();
        let mut storage = MockStorage::new();

        model.expect_load()
            .times(1)
            .return_once(|_| Ok((String::from("EUR/CAN"), HistoryTimeframe::Hour1, 2, 1)));

        let history = vec!(
            HistoryStep {
                timestamp : 1,
                bid_candle : Candlestick { price_open : 1.0, price_close : 1.0, price_high : 1.0, price_low : 1.0 },
                ask_candle : Candlestick { price_open : 1.0, price_close : 1.0, price_high : 1.0, price_low : 1.0 },
            },
            HistoryStep {
                timestamp : 2,
                bid_candle : Candlestick { price_open : 1.0, price_close : 1.0, price_high : 1.0, price_low : 1.0 },
                ask_candle : Candlestick { price_open : 1.0, price_close : 1.5, price_high : 1.0, price_low : 1.0 },
            },
            HistoryStep {
                timestamp : 3,
                bid_candle : Candlestick { price_open : 1.0, price_close : 1.0, price_high : 1.5, price_low : 0.1 },
                ask_candle : Candlestick { price_open : 1.5, price_close : 1.5, price_high : 1.0, price_low : 1.0 },
            },
            HistoryStep {
                timestamp : 4,
                bid_candle : Candlestick { price_open : 1.0, price_close : 1.0, price_high : 1.7, price_low : 1.0 },
                ask_candle : Candlestick { price_open : 1.5, price_close : 1.0, price_high : 1.0, price_low : 1.0 },
            });

        let history_meta = build_history_metadata("EUR/CAN", HistoryTimeframe::Hour1);
        storage.expect_load_symbol_history()
            .times(1)
            .return_once(|_| Ok((history, history_meta)));

        model.expect_predict()
            .times(2)
            .returning(|_| Ok((0.2, 1.6)));
        model.expect_calculate_loss()
            .times(1)
            .return_once(|_, _| 0.777);

        let (_model_loss, profit_or_loss) = evaluate_model(&mut model, &mut storage, "modelA", "EUR_CAN_Hour1")?;

        assert!((profit_or_loss - 0.1 + 1.3).abs() < 0.00001);
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
    fn evaluate_model_failing_to_load_input() {
        let mut model = MockTradingModel::new();
        let mut storage = MockStorage::new();

        model.expect_load()
            .times(1)
            .with(eq("modelB"))
            .return_once(|_| Ok((String::from("EUR/CAN"), HistoryTimeframe::Hour1, 10, 5)));
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
            .return_once(|_| Ok((String::from("EUR/CAN"), HistoryTimeframe::Hour1, 10, 5)));
        storage.expect_load_symbol_history()
            .with(eq("EUR_CAN_Hour1"))
            .times(1)
            .return_once(|_| Ok(build_history_and_meta(15, "EUR/CAN", HistoryTimeframe::Hour1)));

        model.expect_predict()
            .with(eq(build_history(10)))
            .times(1)
            .return_once(|_| Err(anyhow!("Failed")));
        let result = evaluate_model(&mut model, &mut storage, "modelB", "EUR_CAN_Hour1");

        assert!(result.is_err());
    }

    fn build_history_and_meta(num_steps : u32, symbol : &str, timeframe : HistoryTimeframe) -> (Vec<HistoryStep>, HistoryMetadata) {
        (build_history(num_steps), build_history_metadata(symbol, timeframe))
    }

    fn build_history_metadata(symbol : &str, timeframe : HistoryTimeframe) -> HistoryMetadata {
        let history_metadata = HistoryMetadata { symbol : String::from(symbol),
            timeframe, from_date : Utc::now(), to_date : Utc::now()
        };

        history_metadata
    }
}