use serde_json::Value;
use trading_bot;
use std::collections::HashMap;

static FXCM_API_HOST: &str = "api-demo.fxcm.com";

struct FxcmTradingService {
    account_token : String,
    socket_id : String,
    authorization_token : String,
    socket : tungstenite::WebSocket<tungstenite::client::AutoStream>
}

impl FxcmTradingService {
    fn create(account_token : &str) -> Result<FxcmTradingService, String> {
        println!("Connecting to {}", FXCM_API_HOST);

        // Token validity can be tested in http://restapi101.herokuapp.com/
        let connect_result = tungstenite::connect(
            format!("wss://{}/socket.io/?EIO=3&transport=websocket&access_token={}", FXCM_API_HOST, account_token));
            
        let (mut socket, response) = match connect_result {
            Ok(result) => result,
            Err(message) => {
                println!("Could not connect to {}: '{}'", FXCM_API_HOST, message);
                return Err(message.to_string());
            }
        };

        if response.status() != http::StatusCode::SWITCHING_PROTOCOLS {
            println!("Connected but received an erroneous HTTP response: {:?}", response);
            return Err(format!("Connected but received an erroneous HTTP status: {:?}", response.status()));
        }

        println!("Receiving WebSocket id");
        let socket_id_message_result = socket.read_message();
        let socket_id_message = match socket_id_message_result {
            Ok(socket_id_message) => socket_id_message,
            Err(message) => {
                println!("Could not receive socket id message form socket: {}", message);
                return Err(message.to_string());
            }
        };

        let socket_id_message_text = socket_id_message.to_text().unwrap();
        let socket_id_message_json_text = socket_id_message_text.strip_prefix("0").unwrap();
        let v: Value = serde_json::from_str(socket_id_message_json_text).unwrap();
        let server_socket_id = format!("{}", v["sid"].as_str().unwrap());
        let bearer_access_token = format!("Bearer {}{}", server_socket_id, account_token);

        println!("Received WebSocket id '{}', now receiving dummy WebSocket message", server_socket_id);
        // Not sure why, but fxcm sends another message right away.
        let dummy_message = socket.read_message();
        println!("Received dummy message {:?}", dummy_message);

        let fxcm_service = FxcmTradingService {
            account_token : String::from(account_token),
            socket_id : String::from(&server_socket_id),
            authorization_token : String::from(&bearer_access_token),
            socket : socket
        };

        println!("Sending a test HTTP request");
        let http_resp_result = fxcm_service.http_get("trading/get_instruments/", &HashMap::new());
        if let Err(message) = http_resp_result {
            println!("Connected, but could not process a simple HTTP request: {}", message);
            return Err(message.to_string());
        }

        println!("Successfully established and tested connection to {}", FXCM_API_HOST);
        Ok(fxcm_service)
    }

    fn http_get(&self, uri : &str, param_map : &HashMap<String, String>) -> reqwest::Result<reqwest::blocking::Response> {
        let mut query_str = String::new();
        for (k, v) in param_map {
            query_str.push_str(k);
            query_str.push_str("=");
            query_str.push_str(v);
        }

        let url_string = format!("https://{}:443/{}?{}", FXCM_API_HOST, uri, query_str);
        let url = url::Url::parse(&url_string).unwrap();

        let client = reqwest::blocking::Client::new();
        let resp = client
            .get(url)
            .header("Authorization", &self.authorization_token)
            .header("Accept", "application/json")
            .header("Host", FXCM_API_HOST)
            .header("User-Agent", "request")
            .header("Content-Type", "application/x-www-form-urlencoded'")
            .header("Connection", "close")
            .send();
        resp
    }
}

impl trading_bot::TradingService for FxcmTradingService {
    fn get_trade_symbols(&mut self) -> Result<Vec<String>, trading_bot::TradingError> {
        let http_get_result = self.http_get("trading/get_instruments/", &HashMap::new());

        let response = match http_get_result {
            Ok(response) => response,
            Err(message) => return Err(message.to_string())
        };

        if response.status() != http::StatusCode::OK {
            return Err(format!("Erroneous HTTP status returned: {}", response.status()));
        }

        let response_body = match response.text() {
            Ok(text) => text,
            Err(message) => return Err(message.to_string())
        };

        let json_root : Value = match serde_json::from_str(&response_body) {
            Ok(json_root) => json_root,
            Err(message) => return Err(format!("Failed to parse response '{}' a json: {}", response_body, message))
        };

        let maybe_instrument_array = &json_root["data"]["instrument"].as_array();
        let maybe_symbols = maybe_instrument_array.map(|a| {
            a.iter().map(|e| e["symbol"].as_str().map(String::from)).flatten().collect::<Vec<String>>()
        });
        maybe_symbols.map_or(Err(format!("Failed to read symbols from json: {:?}", json_root)), |s| Ok(s))
    }

    fn open_buy_trade(&mut self, symbol : &str, amount_in_lots : u32) -> Result<trading_bot::TradeId, trading_bot::TradingError> {
        unimplemented!()
    }

    fn open_sell_trade(&mut self, symbol : &str, amount_in_lots : u32) -> Result<trading_bot::TradeId, trading_bot::TradingError> {
        unimplemented!()
    }

    fn close_trade(&mut self, trade_id : &trading_bot::TradeId) -> Option<trading_bot::TradingError> {
        unimplemented!()
    }
}

fn main() -> Result<(), String> {
    let mut service = FxcmTradingService::create("4979200962b698e88aa1492f4e62f6e30e338a27")?;

    trading_bot::run(&mut service);
    Ok(())
}
