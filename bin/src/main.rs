use trading_lib;
mod fxcm_service;

fn main() -> Result<(), String> {
    let mut service = fxcm_service::FxcmTradingService::create("4979200962b698e88aa1492f4e62f6e30e338a27")?;

    trading_lib::run(&mut service);
    Ok(())
}
