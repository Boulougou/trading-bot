use crate::trading_model::*;
use crate::storage::*;
use crate::plotter::*;
use crate::{utils, HistoryStep};
use getset::{Setters};
use anyhow::Context;

#[derive(Debug, Setters)]
#[getset(set = "pub")]
pub struct PredictionEvaluationOptions {
    pub min_profit_percent : f32,
    pub max_profit_percent : f32,
    pub max_loss_percent : f32,
    pub prediction_window_multiplier : u32
}

impl Default for PredictionEvaluationOptions {
    fn default() -> Self {
        PredictionEvaluationOptions { min_profit_percent : 0.0, max_profit_percent : f32::INFINITY,
            max_loss_percent : f32::INFINITY, prediction_window_multiplier : 1 }
    }
}

pub fn evaluate_model(model : &mut impl TradingModel,
                      plotter : &mut impl Plotter,
                      storage : &mut impl Storage,
                      model_name : &str,
                      input_name : &str,
                      eval_options : PredictionEvaluationOptions) -> anyhow::Result<(f32, f32)> {
    let (_symbol, _timeframe, input_window, prediction_window) = model.load(model_name)?;

    let (history, _metadata) = storage.load_symbol_history(input_name)?;

    let input_events = utils::extract_input_and_prediction_windows(&history,
        input_window as u32, prediction_window * eval_options.prediction_window_multiplier)?;

    let mut predictions = Vec::new();
    let mut expectations = Vec::new();
    let mut high_delta = Vec::new();
    let mut low_delta = Vec::new();
    let mut profit_history = Vec::new();
    let mut ask_price_history = Vec::new();
    let mut cumulative_profit_history = Vec::new();
    let mut profit_or_loss = 0.0;
    for (input_steps, future_steps) in input_events {
        let prediction = model.predict(&input_steps)?;
        predictions.push(prediction);

        let expectation = utils::find_bid_price_range(&future_steps[0..prediction_window as usize]);
        expectations.push(expectation);

        let (current_bid_price, current_ask_price) = get_current_prices(&future_steps)?;
        let adjustment_result = utils::adjust_prediction((current_bid_price, current_ask_price),
             prediction, eval_options.min_profit_percent,
             eval_options.max_profit_percent, eval_options.max_loss_percent);
        let current_profit_or_loss = match adjustment_result {
            Err(_err) => {
                // println!("NOT POSSIBLE {}", err);
                0.0
            },
            Ok(position) => evaluate_position(
                (current_bid_price, current_ask_price), position, &future_steps)?
        };

        profit_or_loss += current_profit_or_loss;

        low_delta.push(expectation.0 - prediction.0);
        high_delta.push(expectation.1 - prediction.1);
        profit_history.push(current_profit_or_loss);
        ask_price_history.push(current_ask_price);
        cumulative_profit_history.push(profit_or_loss);
    }

    plotter.plot_lines(&vec!((String::from("Low delta"), low_delta.clone())),
                       "Expected Low Bid price - Predicted Low Bid Price", &format!("{}/low_delta", model_name))?;
    plotter.plot_lines(&vec!((String::from("High delta"), high_delta.clone())),
                       "Expected High Bid price - Predicted High Bid Price", &format!("{}/high_delta", model_name))?;
    plotter.plot_lines(&vec!((String::from("Profit"), profit_history.clone())),
                       "Profit or Loss on each step", &format!("{}/profit", model_name))?;
    plotter.plot_lines(&vec!((String::from("Profit"), cumulative_profit_history)),
                       "Cumulative profit", &format!("{}/cumulative_profit", model_name))?;
    plotter.plot_lines(&vec!((String::from("Low delta"), low_delta),
                             (String::from("High delta"), high_delta),
                             (String::from("Profit"), profit_history.clone())),
        "Delta Summary", &format!("{}/summary", model_name))?;
    let predictions_low = predictions.iter().map(|p| p.0).collect();
    let predictions_high = predictions.iter().map(|p| p.1).collect();
    let expected_low = expectations.iter().map(|e| e.0).collect();
    let expected_high = expectations.iter().map(|e| e.1).collect();
    plotter.plot_lines(&vec!((String::from("Low Pred"), predictions_low),
                             (String::from("High Pred"), predictions_high),
                             (String::from("Low Expect"), expected_low),
                             (String::from("High Expect"), expected_high),
                             (String::from("Ask price"), ask_price_history)),
                       "Prices", &format!("{}/prices", model_name))?;

    let model_loss = model.calculate_loss(&predictions, &expectations);
    Ok((model_loss, profit_or_loss))
}

fn evaluate_position((_pos_bid_price, pos_ask_price) : (f32, f32),
                       (maybe_stop, maybe_limit) : (Option<f32>, Option<f32>),
                       future_steps : &[HistoryStep]) -> anyhow::Result<f32> {
    // let min_predicted_price = min_predicted_price - pos_spread;
    for s in future_steps {
        if maybe_stop.is_some() && s.bid_candle.price_low < maybe_stop.unwrap() {
            // println!("LOSS: {}", -(pos_ask_price - maybe_stop.unwrap()));
            return Ok(-(pos_ask_price - maybe_stop.unwrap()));
        }

        if maybe_limit.is_some() && s.bid_candle.price_high > maybe_limit.unwrap() {
            // println!("PROFIT: {}", maybe_limit.unwrap() - pos_ask_price);
            return Ok(maybe_limit.unwrap() - pos_ask_price);
        }
    }

    if maybe_stop.is_some() {
        // println!("INCONCLUSIVE WITH STOP: {}", -(pos_ask_price - maybe_stop.unwrap()));
        Ok(-(pos_ask_price - maybe_stop.unwrap()))
    }
    else {
        let last_bid_price = future_steps.last().unwrap().bid_candle.price_low;
        let penalty = -(pos_ask_price - last_bid_price).abs();
        // println!("INCONCLUSIVE: {}", penalty);
        Ok(penalty)
    }
}

fn get_current_prices(future_steps: &[HistoryStep]) -> anyhow::Result<(f32, f32)> {
    let first_step = future_steps.first().context("There should be at least 1 future step")?;
    let pos_ask_price = first_step.ask_candle.price_open;
    let pos_bid_price = first_step.bid_candle.price_open;
    Ok((pos_bid_price, pos_ask_price))
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
        let mut plotter = MockPlotter::new();
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

        let result = evaluate_model(&mut model, &mut plotter, &mut storage,
                                    "modelA", "EUR_CAN_Hour1",
                                    PredictionEvaluationOptions::default());

        assert!(result.is_err());
    }

    #[test]
    fn evaluate_model_with_history_equal_to_required_size() -> anyhow::Result<()> {
        let mut model = MockTradingModel::new();
        let mut plotter = MockPlotter::new();
        let mut storage = MockStorage::new();

        plotter.expect_plot_lines().returning(|_, _, _| Ok(()));

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

        let (model_loss, _profit_or_loss) = evaluate_model(&mut model, &mut plotter, &mut storage,
                                                           "modelA", "EUR_CAN_Hour1",
                                                           PredictionEvaluationOptions::default())?;

        assert_eq!(model_loss, 0.564);
        Ok(())
    }

    #[test]
    fn evaluate_model_with_history_larger_than_input_size() -> anyhow::Result<()> {
        let mut model = MockTradingModel::new();
        let mut plotter = MockPlotter::new();
        plotter.expect_plot_lines().returning(|_, _, _| Ok(()));
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

        let (model_loss, _profit_or_loss) = evaluate_model(&mut model, &mut plotter, &mut storage,
                                                           "modelA", "EUR_CAN_Hour1",
                                                           PredictionEvaluationOptions::default())?;

        assert_eq!(model_loss, 0.478);
        Ok(())
    }

    #[test]
    fn return_profit_when_max_predicted_price_is_reached() -> anyhow::Result<()> {
        let mut model = MockTradingModel::new();
        let mut plotter = MockPlotter::new();
        plotter.expect_plot_lines().returning(|_, _, _| Ok(()));
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

        let (_model_loss, profit_or_loss) = evaluate_model(&mut model, &mut plotter, &mut storage,
                                                           "modelA", "EUR_CAN_Hour1",
                                                           PredictionEvaluationOptions::default())?;

        assert!((profit_or_loss - 0.1).abs() < 0.00001);
        Ok(())
    }

    #[test]
    fn return_max_loss_when_max_predicted_price_is_never_reached() -> anyhow::Result<()> {
        let mut model = MockTradingModel::new();
        let mut plotter = MockPlotter::new();
        plotter.expect_plot_lines().returning(|_, _, _| Ok(()));
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

        let eval_options = PredictionEvaluationOptions { max_loss_percent : 2.0, ..PredictionEvaluationOptions::default() };
        let (_model_loss, profit_or_loss) = evaluate_model(&mut model, &mut plotter, &mut storage,
                                                           "modelA", "EUR_CAN_Hour1",
                                                           eval_options)?;

        assert!((profit_or_loss + 1.0).abs() < 0.00001);
        Ok(())
    }

    #[test]
    fn return_loss_when_max_loss_percentage_is_reached() -> anyhow::Result<()> {
        let mut model = MockTradingModel::new();
        let mut plotter = MockPlotter::new();
        plotter.expect_plot_lines().returning(|_, _, _| Ok(()));
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

        let eval_options = PredictionEvaluationOptions { max_loss_percent : 2.0, ..PredictionEvaluationOptions::default() };
        let (_model_loss, profit_or_loss) = evaluate_model(&mut model, &mut plotter, &mut storage,
                                                           "modelA", "EUR_CAN_Hour1",
                                                           eval_options)?;

        assert!((profit_or_loss + 1.0).abs() < 0.00001);
        Ok(())
    }

    #[test]
    fn return_zero_profit_when_max_predicted_price_cannot_cover_spread() -> anyhow::Result<()> {
        let mut model = MockTradingModel::new();
        let mut plotter = MockPlotter::new();
        plotter.expect_plot_lines().returning(|_, _, _| Ok(()));
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
                bid_candle : Candlestick { price_open : 1.0, price_close : 1.0, price_high : 1.2, price_low : 0.1 },
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
            .return_once(|_| Ok((0.2, 1.4)));
        model.expect_calculate_loss()
            .times(1)
            .return_once(|_, _| 0.777);

        let (_model_loss, profit_or_loss) = evaluate_model(&mut model, &mut plotter, &mut storage,
                                                           "modelA", "EUR_CAN_Hour1",
                                                           PredictionEvaluationOptions::default())?;

        assert_eq!(profit_or_loss, 0.0);
        Ok(())
    }

    #[test]
    fn return_accumulated_profit_or_loss_when_many_evaluations() -> anyhow::Result<()> {
        let mut model = MockTradingModel::new();
        let mut plotter = MockPlotter::new();
        plotter.expect_plot_lines().returning(|_, _, _| Ok(()));
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

        let eval_options = PredictionEvaluationOptions { max_loss_percent : 2.0, ..PredictionEvaluationOptions::default() };
        let (_model_loss, profit_or_loss) = evaluate_model(&mut model, &mut plotter, &mut storage,
                                                           "modelA", "EUR_CAN_Hour1",
                                                           eval_options)?;

        assert!((profit_or_loss - 0.1 + 1.0).abs() < 0.00001);
        Ok(())
    }

    #[test]
    fn evaluate_model_failing_to_load_model() {
        let mut model = MockTradingModel::new();
        let mut plotter = MockPlotter::new();
        let mut storage = MockStorage::new();

        model.expect_load()
            .times(1)
            .with(eq("modelB"))
            .return_once(|_| Err(anyhow!("Failed")));
        let result = evaluate_model(&mut model, &mut plotter, &mut storage,
                                    "modelB", "EUR_CAN_Hour1",
                                    PredictionEvaluationOptions::default());

        assert!(result.is_err());
    }

    #[test]
    fn evaluate_model_failing_to_load_input() {
        let mut model = MockTradingModel::new();
        let mut plotter = MockPlotter::new();
        let mut storage = MockStorage::new();

        model.expect_load()
            .times(1)
            .with(eq("modelB"))
            .return_once(|_| Ok((String::from("EUR/CAN"), HistoryTimeframe::Hour1, 10, 5)));
        storage.expect_load_symbol_history()
            .with(eq("EUR_CAN_Hour1"))
            .times(1)
            .return_once(|_| Err(anyhow!("Failed")));
        let result = evaluate_model(&mut model, &mut plotter, &mut storage,
                                    "modelB", "EUR_CAN_Hour1",
                                    PredictionEvaluationOptions::default());

        assert!(result.is_err());
    }

    #[test]
    fn evaluate_model_failing_run_prediction() {
        let mut model = MockTradingModel::new();
        let mut plotter = MockPlotter::new();
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
        let result = evaluate_model(&mut model, &mut plotter, &mut storage,
                                    "modelB", "EUR_CAN_Hour1",
                                    PredictionEvaluationOptions::default());

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