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
        train,
        eval,
        trade
    }
}

/// A basic example
#[derive(StructOpt, Debug)]
#[structopt(name = "trading-bot")]
struct Opt {

    #[structopt(possible_values = &Mode::variants())]
    mode : Mode,

    #[structopt(short, long, required_if("mode", "fetch"), required_if("mode", "trade"))]
    symbol : Option<String>,

    #[structopt(short, long, required_if("mode", "trade"))]
    ammount : Option<u32>,

    #[structopt(short, long, required_if("mode", "fetch"))]
    timeframe : Option<trading_lib::HistoryTimeframe>,

    #[structopt(long, required_if("mode", "fetch"))]
    from_date : Option<String>,

    #[structopt(long)]
    to_date : Option<String>,

    #[structopt(short, long, required_if("mode", "train"), required_if("mode", "eval"))]
    input : Option<String>,

    #[structopt(short, long, required_if("mode", "train"), required_if("mode", "eval"), required_if("mode", "trade"))]
    model : Option<String>,

    #[structopt(long, required_if("mode", "train"))]
    input_window : Option<u32>,

    #[structopt(long, required_if("mode", "train"))]
    pred_window : Option<u32>,

    #[structopt(long, default_value = "1e-3", required_if("mode", "train"))]
    learning_rate : f32,

    #[structopt(long, default_value = "3000", required_if("mode", "train"))]
    learning_iterations : u32,
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
            let mut model = pytorch_model::PyTorchModel::new();
            trading_lib::train_model(&mut model, &mut storage, &opt.input.unwrap(),
                opt.input_window.unwrap(), opt.pred_window.unwrap(),
                &(opt.learning_rate, opt.learning_iterations), &opt.model.unwrap())?;
        },
        Mode::eval => {
            let mut storage = file_storage::FileStorage::create()?;
            let mut model = pytorch_model::PyTorchModel::new();
            let predictions = trading_lib::evaluate_model(&mut model, &mut storage, &opt.model.unwrap(), &opt.input.unwrap())?;
            println!("Predictions: {:?}", predictions);
        },
        Mode::trade => {
            let mut service = fxcm::service::FxcmTradingService::create(fxcm_host, fxcm_token)?;
            let mut model = pytorch_model::PyTorchModel::new();
            let (trade_id, options) = trading_lib::open_trade(&mut service, &mut model, &opt.symbol.unwrap(), opt.ammount.unwrap(), &opt.model.unwrap())?;
            println!("Opened trade {:?}, stop: {:?}, limit: {:?}", trade_id, options.stop, options.limit);
        }
    }

    // pytorch_model::run_linear_regression();
    // pytorch_model::run_neural_network();
    Ok(())
}
