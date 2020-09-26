pub type TradeId = String;

#[derive(Debug)]
pub enum TradingError {
    Unauthorized
}

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

    // let openResult = service.open_buy_trade("EUR/USD", 100);

    // match openResult {
    //     Ok(trade_id) => println!("Success {}", trade_id),
    //     Err(error_msg) => println!("Failed {:?}", error_msg)
    // }

}