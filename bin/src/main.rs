use chrono::*;
use trading_lib;
use clap::arg_enum;
use structopt::StructOpt;

mod fxcm;
mod file_storage;
mod pytorch_model;

arg_enum! {
    #[derive(Debug)]
    #[allow(non_camel_case_types)]
    enum Mode {
        fetch,
        train
    }
}

/// A basic example
#[derive(StructOpt, Debug)]
#[structopt(name = "trading-bot")]
struct Opt {

    #[structopt(possible_values = &Mode::variants())]
    mode : Mode,

    #[structopt(short, long, required_if("mode", "fetch"))]
    symbol : Option<String>,

    #[structopt(short, long, required_if("mode", "fetch"))]
    timeframe : Option<trading_lib::HistoryTimeframe>,

    #[structopt(long, required_if("mode", "fetch"))]
    from_date : Option<String>,

    #[structopt(long)]
    to_date : Option<String>,

    #[structopt(short, long, required_if("mode", "train"))]
    input : Option<String>,
}

fn parse_date(date_str : &str) -> anyhow::Result<DateTime<Utc>> {
    let naive_date = NaiveDateTime::parse_from_str(date_str, "%Y%m%d%H%M")?;
    Ok(Utc.ymd(naive_date.year(), naive_date.month(), naive_date.day()).and_hms(naive_date.hour(), naive_date.minute(), 0))
}

fn main() -> anyhow::Result<()> {
    let opt = Opt::from_args();
    // println!("{:#?}", opt);

    let fxcm_host = "api-demo.fxcm.com";
    let fxcm_token = "4979200962b698e88aa1492f4e62f6e30e338a27";

    match opt.mode {
        Mode::fetch => {
            let from_date = parse_date(&opt.from_date.unwrap())?;
            let to_date = match opt.to_date {
                None => Utc::now(),
                Some(s) => parse_date(&s)?
            };

            let mut service = fxcm::service::FxcmTradingService::create(fxcm_host, fxcm_token)?;
            let mut storage = file_storage::FileStorage::create()?;
            trading_lib::fetch_symbol_history(&mut service, &mut storage, &opt.symbol.unwrap(), opt.timeframe.unwrap(), &from_date, &to_date)?;
        },
        Mode::train => {
            let mut storage = file_storage::FileStorage::create()?;
            let mut model = pytorch_model::PyTorchModel{};
            trading_lib::train_model(&mut model, &mut storage, &opt.input.unwrap())?;
        }
    }

    // pytorch_model::run_linear_regression();
    // pytorch_model::run_neural_network();
    Ok(())
}
