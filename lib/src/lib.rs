pub type TradeId = String;

#[derive(Debug)]
pub struct Candlestick {
    pub price_open : f32,
    pub price_close : f32,
    pub price_high : f32,
    pub price_low : f32
}

#[derive(Debug)]
pub struct HistoryStep {
    pub timestamp : u32,
    pub bid_candle : Candlestick,
    pub ask_candle : Candlestick
}

pub enum HistoryTimeframe {
    Min1,
    Min5,
    Min15,
    Min30,
    Hour1,
    Hour2,
    Hour3,
    Hour4,
    Hour6,
    Hour8,
    Day1,
    Week1,
    Month1
}

pub trait TradingService {
    fn get_trade_symbols(&mut self) -> anyhow::Result<Vec<String>>;
    fn get_symbol_history(&mut self, symbol : &str, timeframe : HistoryTimeframe, num_entries : u32) -> anyhow::Result<Vec<HistoryStep>>;
    fn open_buy_trade(&mut self, symbol : &str, amount_in_lots : u32) -> anyhow::Result<TradeId>;
    fn open_sell_trade(&mut self, symbol : &str, amount_in_lots : u32) -> anyhow::Result<TradeId>;
    fn close_trade(&mut self, trade_id : &TradeId) -> anyhow::Result<()>;
}

pub fn run(service : &mut impl TradingService) -> anyhow::Result<()> {
    let trade_symbols_result = service.get_symbol_history("EUR/CAD", HistoryTimeframe::Min1, 100);
    match trade_symbols_result {
        Ok(symbols) => println!("get_symbol_history, {:?}", symbols),
        Err(msg) => println!("Failed to get_symbol_history, {:?}", msg)
    }

    Ok(())
/*
    let trade_symbols_result = service.get_trade_symbols();
    match trade_symbols_result {
        Ok(symbols) => println!("Retrieved trade symbols, {:?}", symbols),
        Err(msg) => println!("Failed to retrieve trade symbols, {:?}", msg)
    }

    let open_buy_result = service.open_buy_trade("EUR/CAD", 1);
    let buy_trade_id = match open_buy_result {
        Ok(trade_id) => trade_id,
        Err(error_msg) => { println!("Failed {:?}", error_msg); return Err(error_msg); }
    };

    let open_sell_result = service.open_sell_trade("EUR/CAD", 1);
    let sell_trade_id = match open_sell_result {
        Ok(trade_id) => trade_id,
        Err(error_msg) => { println!("Failed {:?}", error_msg); return Err(error_msg); }
    };

    println!("Opened buy trade successfully {}", buy_trade_id);
    println!("Opened sell trade successfully {}", sell_trade_id);
    std::thread::sleep(std::time::Duration::from_secs(10));

    let close_result = service.close_trade(&sell_trade_id);
    match close_result {
        Ok(()) => println!("Successfully closed trade with id, {}", sell_trade_id),
        Err(error) => println!("Failed to close trade with id {}, error: {}", sell_trade_id, error),
    }

    let close_result = service.close_trade(&buy_trade_id);
    match close_result {
        Ok(()) => { println!("Successfully closed trade with id, {}", buy_trade_id); Ok(()) }
        Err(error) => { println!("Failed to close trade with id {}, error: {}", buy_trade_id, error); Err(error) }
    }*/
}