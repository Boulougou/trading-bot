pub type TradeId = String;

pub type TradingError = String;

pub trait TradingService {
    fn get_trade_symbols(&mut self) -> Result<Vec<String>, TradingError>;
    fn open_buy_trade(&mut self, symbol : &str, amount_in_lots : u32) -> Result<TradeId, TradingError>;
    fn open_sell_trade(&mut self, symbol : &str, amount_in_lots : u32) -> Result<TradeId, TradingError>;
    fn close_trade(&mut self, trade_id : &TradeId) -> Option<TradingError>;
}

pub fn run(service : &mut impl TradingService) {
    let trade_symbols_result = service.get_trade_symbols();
    match trade_symbols_result {
        Ok(symbols) => println!("Retrieved trade symbols, {:?}", symbols),
        Err(msg) => println!("Failed to retrieve trade symbols, {:?}", msg)
    }

    let open_result = service.open_buy_trade("EUR/CAD", 1);

    let trade_id = match open_result {
        Ok(trade_id) => trade_id,
        Err(error_msg) => { println!("Failed {:?}", error_msg); return; }
    };

    println!("Opened buy trade successfully {}", trade_id);
    std::thread::sleep(std::time::Duration::from_secs(10));

    let close_result = service.close_trade(&trade_id);
    match close_result {
        None => println!("Successfully closed trade with id, {}", trade_id),
        Some(error) => println!("Failed to close trade with id {}, error: {}", trade_id, error),
    }
}