use std::net::TcpStream;
use serde_json::Value;
use trading_lib;
use std::collections::HashMap;
use anyhow::{anyhow, Context};
use chrono::{DateTime, Utc};
use crate::fxcm::utils;

enum FxcmTableType {
    _OpenPosition = 1,
    _ClosedPosition,
    Order,
    _Summary,
    _Account
}

pub struct FxcmTradingService {
    host : String,
    authorization_token : String,
    socket : tungstenite::WebSocket<native_tls::TlsStream<TcpStream>>,
    account_id : String,
    symbol_to_offer_id : HashMap<String, i32>,
    trade_amounts : HashMap<trading_lib::TradeId, u32>
}

impl FxcmTradingService {
    pub fn create(api_host : &str, account_token : &str) -> anyhow::Result<FxcmTradingService> {
        let mut fxcm_service = FxcmTradingService::connect(api_host, account_token).with_context(|| format!("Failed to connect to '{}'", api_host))?;

        println!("Retrieving symbol to offer mapping");
        fxcm_service.symbol_to_offer_id = FxcmTradingService::retrieve_symbol_to_offer_id_map(api_host, &fxcm_service.authorization_token).
            context("Failed to retrieve symbol to offer mapping")?;

        println!("Retrieving account id");
        fxcm_service.account_id = FxcmTradingService::retrieve_account_id(api_host, &fxcm_service.authorization_token).
            context("Failed to retrieve account id")?;

        println!("Subscribing to order table");
        FxcmTradingService::subscribe_to_order_table(api_host, &fxcm_service.authorization_token).
            context("Failed to subscribe to order table")?;

        println!("Successfully established connection to {}", api_host);
        Ok(fxcm_service)
    }

    fn connect(api_host : &str, account_token : &str) -> anyhow::Result<FxcmTradingService> {
        println!("Connecting to {}", api_host);

        let connect_str = format!("{}:443", api_host);
        let stream = TcpStream::connect(&connect_str)?;
        // stream.set_nonblocking(true);
        let connector = native_tls::TlsConnector::new()?;
        let tls_stream = connector.connect(api_host, stream)?;

        // Token validity can be tested in http://restapi101.herokuapp.com/
        let (mut socket, response) = tungstenite::client(
            format!("wss://{}/socket.io/?EIO=3&transport=websocket&access_token={}", api_host, account_token),
            tls_stream)?;

        if response.status() != http::StatusCode::SWITCHING_PROTOCOLS {
            println!("Connected but received an erroneous HTTP response: {:?}", response);
            return Err(anyhow!("Connected but received an erroneous HTTP status: {:?}", response.status()));
        }

        println!("Receiving WebSocket id");
        let v: Value = utils::read_message_from_socket(&mut socket)?;
        let server_socket_id = v["sid"].as_str().ok_or(anyhow!("No 'sid' found in response"))?;
        let bearer_access_token = format!("Bearer {}{}", server_socket_id, account_token);

        println!("Received WebSocket id '{}', now receiving dummy WebSocket message", server_socket_id);
        // Not sure why, but fxcm sends another message right away.
        let dummy_message = socket.read_message().context("Failed to receive first message from web socket")?;
        println!("Received dummy message {:?}", dummy_message);

        let fxcm_service = FxcmTradingService {
            host : String::from(api_host),
            authorization_token : String::from(&bearer_access_token),
            socket : socket,
            symbol_to_offer_id : HashMap::new(),
            account_id : String::new(),
            trade_amounts : HashMap::new()
        };

        Ok(fxcm_service)
    }

    fn retrieve_symbol_to_offer_id_map(api_host : &str, authorization_token : &str) -> anyhow::Result<HashMap<String, i32>> {
        let get_model_params = vec!((String::from("models"), String::from("Offer")));
        let json_root : Value = utils::http_get_json(authorization_token, api_host, "trading/get_model/", &get_model_params)?;

        let offers_array = &json_root["offers"].as_array().ok_or_else(|| anyhow!("No 'offers' array found in {}", json_root))?;
        let mapping = offers_array.iter().fold(HashMap::new(), |mut acc, e| {
            let maybe_symbol = e["currency"].as_str().map(String::from);
            let maybe_offer = e["offerId"].as_i64().map(|x| x as i32);
            if let (Some(symbol), Some(offer_id)) = (maybe_symbol, maybe_offer) {
                acc.insert(symbol, offer_id);
            }
            acc
        });

        Ok(mapping)
    }

    fn retrieve_account_id(api_host : &str, authorization_token : &str) -> anyhow::Result<String> {
        let get_model_params = vec!((String::from("models"), String::from("Account")));
        let json_root : Value = utils::http_get_json(authorization_token, api_host, "trading/get_model/", &get_model_params)?;

        let accounts_array = &json_root["accounts"].as_array().ok_or_else(|| anyhow!("No 'accounts' array found in {}", json_root))?;
        let account_id = accounts_array[0]["accountId"].as_str().ok_or_else(|| anyhow!("No 'accountId' found in {}", json_root))?;
        Ok(String::from(account_id))
    }

    fn subscribe_to_order_table(api_host : &str, authorization_token : &str) -> anyhow::Result<()> {
        let params = vec!((String::from("models"), String::from("Order")));
        utils::http_post_json(authorization_token, api_host, "trading/subscribe/", &params).map(|_| ())
    }

    fn open_trade(&mut self, symbol : &str, amount_in_lots : u32, is_buy : bool) -> anyhow::Result<trading_lib::TradeId> {
        let open_trade_params = vec!(
            (String::from("account_id"), self.account_id.clone()),
            (String::from("is_buy"), is_buy.to_string()),
            (String::from("amount"), amount_in_lots.to_string()),
            (String::from("time_in_force"), String::from("GTC")),
            (String::from("order_type"), String::from("AtMarket")),
            (String::from("symbol"), String::from(symbol)));
        let http_resp_json : Value = utils::http_post_json(&self.authorization_token, &self.host, "trading/open_trade/", &open_trade_params)?;

        println!("Open trade json: {}", http_resp_json);
        let order_id = http_resp_json["data"]["orderId"].as_u64().
            ok_or_else(|| anyhow!("No 'data/orderId' found in response {}", http_resp_json))?;
        println!("Received order id: {:?}", order_id);

        println!("Receiving Trade Id event");
        let v = loop {
            let trade_event_json = utils::read_message_from_socket(&mut self.socket)?;
            let nested_json = trade_event_json.as_array().and_then(|a| a[1].as_str()).
                ok_or_else(|| anyhow!("No nested array found in response {}", trade_event_json))?;
            let v: Value = serde_json::from_str(nested_json)?;

            if v["t"].as_u64() == Some(FxcmTableType::Order as u64) &&
                v["action"].as_str() == Some("I") &&
                v["orderId"].as_str() == Some(&order_id.to_string()) {
                break v;
            }
        };
        
        let trade_id = v["tradeId"].as_str().ok_or_else(|| anyhow!("No 'tradeId' found in response {}", v))?;
        println!("Received trade-id: {}", trade_id);

        self.trade_amounts.insert(String::from(trade_id), amount_in_lots);
        Ok(String::from(trade_id))
    }

    fn create_history_step(json_candle : &serde_json::Value) -> anyhow::Result<trading_lib::HistoryStep> {
        let timestamp = json_candle[0].as_u64().ok_or_else(|| anyhow!("Could not find timestamp in {}", json_candle))? as u32;
        let bid_open = json_candle[1].as_f64().ok_or_else(|| anyhow!("Could not find bid_open in {}", json_candle))? as f32;
        let bid_close = json_candle[2].as_f64().ok_or_else(|| anyhow!("Could not find bid_close in {}", json_candle))? as f32;
        let bid_high = json_candle[3].as_f64().ok_or_else(|| anyhow!("Could not find bid_high in {}", json_candle))? as f32;
        let bid_low = json_candle[4].as_f64().ok_or_else(|| anyhow!("Could not find bid_low in {}", json_candle))? as f32;
        let ask_open = json_candle[5].as_f64().ok_or_else(|| anyhow!("Could not find ask_open in {}", json_candle))? as f32;
        let ask_close = json_candle[6].as_f64().ok_or_else(|| anyhow!("Could not find ask_close in {}", json_candle))? as f32;
        let ask_high = json_candle[7].as_f64().ok_or_else(|| anyhow!("Could not find ask_high in {}", json_candle))? as f32;
        let ask_low = json_candle[8].as_f64().ok_or_else(|| anyhow!("Could not find ask_low in {}", json_candle))? as f32;

        let step = trading_lib::HistoryStep {
            timestamp : timestamp,
            bid_candle : trading_lib::Candlestick { price_open : bid_open, price_close : bid_close, price_high : bid_high, price_low : bid_low },
            ask_candle : trading_lib::Candlestick { price_open : ask_open, price_close : ask_close, price_high : ask_high, price_low : ask_low }
        };

        Ok(step)
    }
}

impl trading_lib::TradingService for FxcmTradingService {
    fn get_trade_symbols(&mut self) -> anyhow::Result<Vec<String>> {
        let json_root : Value = utils::http_get_json(&self.authorization_token, &self.host, "trading/get_instruments/", &Vec::new())?;

        let maybe_instrument_array = &json_root["data"]["instrument"].as_array();
        let maybe_symbols = maybe_instrument_array.map(|a| {
            a.iter().map(|e| e["symbol"].as_str().map(String::from)).flatten().collect::<Vec<String>>()
        });
        maybe_symbols.map_or_else(|| Err(anyhow!("Failed to read symbols from json: {:?}", json_root)), |s| Ok(s))
    }

    fn get_symbol_history(&mut self, symbol : &str, timeframe : trading_lib::HistoryTimeframe,
                          since_date : &DateTime<Utc>, to_date : &DateTime<Utc>) -> anyhow::Result<Vec<trading_lib::HistoryStep>> {
        let offer_id = self.symbol_to_offer_id.get(symbol).ok_or_else(|| anyhow!("Could not find symbol {}", symbol))?;

        let url = format!("candles/{}/{}", offer_id, utils::convert_timeframe(&timeframe));
        let http_params = vec!(
            (String::from("num"), String::from("10")),
            (String::from("from"), /* String::from("1602235500") */since_date.timestamp().to_string()),
            (String::from("to"), /*String::from("1602404400")*/ to_date.timestamp().to_string()));
        let json_root : Value = utils::http_get_json(&self.authorization_token, &self.host, &url, &http_params)?;

        let candles_array = &json_root["candles"].as_array().
            ok_or_else(|| anyhow!("No 'candles' array in response: {}", json_root))?;

        let candles = candles_array.iter().
            map(FxcmTradingService::create_history_step).
            flatten().
            collect::<Vec<trading_lib::HistoryStep>>();

        Ok(candles)
    }
    
    fn max_history_steps_per_call(&mut self) -> anyhow::Result<u32> {
        Ok(10000)
     }

    fn open_buy_trade(&mut self, symbol : &str, amount_in_lots : u32) -> anyhow::Result<trading_lib::TradeId> {
        self.open_trade(&symbol, amount_in_lots, true)
    }

    fn open_sell_trade(&mut self, symbol : &str, amount_in_lots : u32) -> anyhow::Result<trading_lib::TradeId> {
        self.open_trade(&symbol, amount_in_lots, false)
    }

    fn close_trade(&mut self, trade_id : &trading_lib::TradeId) -> anyhow::Result<()> {
        let amount = self.trade_amounts[trade_id];

        let close_trade_params = vec!(
            (String::from("trade_id"), String::from(trade_id)),
            (String::from("amount"), amount.to_string()),
            (String::from("time_in_force"), String::from("GTC")),
            (String::from("order_type"), String::from("AtMarket")));
        let json_root : Value = utils::http_post_json(&self.authorization_token, &self.host, "trading/close_trade/", &close_trade_params)?;

        println!("Received close resp: {}", json_root);
        Ok(())
    }
}