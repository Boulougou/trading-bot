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

    #[structopt(short, long, required_if("mode", "fetch"))]
    symbol : Option<String>,

    #[structopt(short, long, required_if("mode", "trade"))]
    amount: Option<u32>,

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

    #[structopt(long, default_value = "0", required_if("mode", "train"))]
    hidden_layer_factor : u32,

    #[structopt(short, long)]
    continue_training : bool,

    #[structopt(long)]
    min_profit : Option<f32>,

    #[structopt(long)]
    max_profit : Option<f32>,

    #[structopt(long)]
    max_loss : Option<f32>,

    #[structopt(long)]
    max_used_margin : Option<f32>
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
            let mode = if opt.continue_training { trading_lib::TrainingOutputMode::ContinueTraining }
                else { trading_lib::TrainingOutputMode::OverwriteModel };
            trading_lib::train_model(&mut model, &mut storage, &opt.input.unwrap(),
                opt.input_window.unwrap(), opt.pred_window.unwrap(),
                &(opt.learning_rate, opt.learning_iterations, opt.hidden_layer_factor),
                &opt.model.unwrap(), mode)?;
        },
        Mode::eval => {
            let mut storage = file_storage::FileStorage::create()?;
            let mut model = pytorch_model::PyTorchModel::new();
            let (model_loss, profit_or_loss) = trading_lib::evaluate_model(&mut model, &mut storage, &opt.model.unwrap(), &opt.input.unwrap())?;
            println!("Model Loss: {}, Profit/Loss: {}", model_loss, profit_or_loss);
        },
        Mode::trade => {
            let mut service = fxcm::service::FxcmTradingService::create(fxcm_host, fxcm_token)?;
            let mut model = pytorch_model::PyTorchModel::new();

            let mut trade_options = trading_lib::OpenTradeOptions::default();
            if let Some(profit) = opt.min_profit {
                trade_options.set_min_profit_percent(profit);
            }
            if let Some(profit) = opt.max_profit {
                trade_options.set_max_profit_percent(profit);
            }
            if let Some(loss) = opt.max_loss {
                trade_options.set_max_loss_percent(loss);
            }
            if let Some(margin) = opt.max_used_margin {
                trade_options.set_max_used_margin_percent(margin);
            }

            let (trade_id, stop, limit) = trading_lib::open_trade_with_options(
                &mut service, &mut model,opt.amount.unwrap(), &Utc::now(),
                &opt.model.unwrap(), &trade_options)?;
            println!("Opened trade {:?}, stop: {:?}, limit: {:?}", trade_id, stop, limit);
        }
    }

    Ok(())
}
