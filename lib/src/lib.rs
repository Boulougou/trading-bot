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

    let open_buy_result = service.open_buy_trade("EUR/CAD", 1);
    let buy_trade_id = match open_buy_result {
        Ok(trade_id) => trade_id,
        Err(error_msg) => { println!("Failed {:?}", error_msg); return; }
    };

    let open_sell_result = service.open_sell_trade("EUR/CAD", 1);
    let sell_trade_id = match open_sell_result {
        Ok(trade_id) => trade_id,
        Err(error_msg) => { println!("Failed {:?}", error_msg); return; }
    };

    println!("Opened buy trade successfully {}", buy_trade_id);
    println!("Opened sell trade successfully {}", sell_trade_id);
    std::thread::sleep(std::time::Duration::from_secs(10));

    let close_result = service.close_trade(&sell_trade_id);
    match close_result {
        None => println!("Successfully closed trade with id, {}", sell_trade_id),
        Some(error) => println!("Failed to close trade with id {}, error: {}", sell_trade_id, error),
    }

    let close_result = service.close_trade(&buy_trade_id);
    match close_result {
        None => println!("Successfully closed trade with id, {}", buy_trade_id),
        Some(error) => println!("Failed to close trade with id {}, error: {}", buy_trade_id, error),
    }
}