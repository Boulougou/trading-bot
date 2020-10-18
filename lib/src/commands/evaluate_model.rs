use anyhow::anyhow;

use crate::trading_model::*;
use crate::storage::*;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::utils::tests::*;
    use mockall::{Sequence, predicate::*};

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
}