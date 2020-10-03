use trading_lib;
mod fxcm;

fn main() -> anyhow::Result<()> {
    let mut service = fxcm::service::FxcmTradingService::create("api-demo.fxcm.com", "4979200962b698e88aa1492f4e62f6e30e338a27")?;

    trading_lib::run(&mut service)
}
