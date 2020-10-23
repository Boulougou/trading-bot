use crate::trading_model::*;
use crate::storage::*;
use crate::utils;

pub fn evaluate_model(model : &mut impl TradingModel,
                      storage : &mut impl Storage,
                      model_name : &str,
                      input_name : &str) -> anyhow::Result<f32> {
    let (_symbol, _timeframe, input_window, prediction_window) = model.load(model_name)?;

    let (history, _metadata) = storage.load_symbol_history(input_name)?;

    let input_events = utils::extract_input_and_prediction_windows(&history, input_window as u32, prediction_window)?;

    let mut predictions = Vec::new();
    let mut expectations = Vec::new();
    for (input_steps, future_steps) in input_events {
        let prediction = model.predict(&input_steps)?;
        predictions.push(prediction);

        let expectation = utils::find_bid_price_range(&future_steps);
        expectations.push(expectation);
    }

    Ok(model.calculate_loss(&predictions, &expectations))
}

#[cfg(test)]
mod tests {
    use anyhow::anyhow;
    use super::*;
    use crate::utils::tests::*;
    use mockall::{Sequence, predicate::*};
    use crate::trading_service::{HistoryMetadata, HistoryStep, HistoryTimeframe};
    use chrono::{Utc};

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

        let loss = evaluate_model(&mut model, &mut storage, "modelA", "EUR_CAN_Hour1")?;

        assert_eq!(loss, 0.564);
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

        let loss = evaluate_model(&mut model, &mut storage, "modelA", "EUR_CAN_Hour1")?;

        assert_eq!(loss, 0.478);
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
        let history_metadata = HistoryMetadata { symbol : String::from(symbol),
            timeframe, from_date : Utc::now(), to_date : Utc::now()
        };

        (build_history(num_steps), history_metadata)
    }
}