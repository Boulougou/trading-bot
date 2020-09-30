use std::net::TcpStream;
use serde_json::Value;
use trading_lib;
use std::collections::HashMap;

static FXCM_API_HOST: &str = "api-demo.fxcm.com";

pub struct FxcmTradingService {
    account_token : String,
    socket_id : String,
    authorization_token : String,
    socket : tungstenite::WebSocket<native_tls::TlsStream<TcpStream>>,
    account_id : String,
    symbol_to_offer_id : HashMap<String, i32>
}

impl FxcmTradingService {
    pub fn create(account_token : &str) -> Result<FxcmTradingService, String> {
        println!("Connecting to {}", FXCM_API_HOST);

        let stream = TcpStream::connect(format!("{}:443", FXCM_API_HOST)).unwrap();
        // stream.set_nonblocking(true);
        let connector = native_tls::TlsConnector::new().unwrap();
        let tls_stream = connector.connect(FXCM_API_HOST, stream).unwrap();

        // Token validity can be tested in http://restapi101.herokuapp.com/
        let connect_result = tungstenite::client(
            format!("wss://{}/socket.io/?EIO=3&transport=websocket&access_token={}", FXCM_API_HOST, account_token),
            tls_stream);
            
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

        let mut fxcm_service = FxcmTradingService {
            account_token : String::from(account_token),
            socket_id : String::from(&server_socket_id),
            authorization_token : String::from(&bearer_access_token),
            socket : socket,
            symbol_to_offer_id : HashMap::new(),
            account_id : String::new()
        };

        {
            println!("Retrieving offer table snapshot");
            let mut get_model_params = Vec::new();
            get_model_params.push((String::from("models"), String::from("Offer")));
            let json_root : Value = match fxcm_service.http_get_json("trading/get_model/", &get_model_params) {
                Ok(json_root) => json_root,
                Err(message) => return Err(message)
            };

            let maybe_offers_array = &json_root["offers"].as_array();
            fxcm_service.symbol_to_offer_id = maybe_offers_array.map(|offers| offers.iter().fold(HashMap::new(), |mut acc, e| {
                acc.insert(String::from(e["currency"].as_str().unwrap()), e["offerId"].as_i64().unwrap() as i32);
                acc
            } )).unwrap();

            println!("{:?}", fxcm_service.symbol_to_offer_id);
        }

        {
            println!("Retrieving account table snapshot");
            let mut get_model_params = Vec::new();
            get_model_params.push((String::from("models"), String::from("Account")));
            let json_root : Value = match fxcm_service.http_get_json("trading/get_model/", &get_model_params) {
                Ok(json_root) => json_root,
                Err(message) => return Err(message)
            };

            let maybe_accounts_array = &json_root["accounts"].as_array();
            fxcm_service.account_id = maybe_accounts_array.map(|accounts| String::from(accounts[0]["accountId"].as_str().unwrap())).unwrap();

            println!("AccountId: {}", fxcm_service.account_id);
        }

        println!("Successfully established and tested connection to {}", FXCM_API_HOST);
        Ok(fxcm_service)
    }

    fn http_get(&self, uri : &str, param_map : &Vec<(String, String)>) -> reqwest::Result<reqwest::blocking::Response> {
        let query_str = FxcmTradingService::pairs_to_query_string(&param_map);

        let url_string = format!("https://{}:443/{}?{}", FXCM_API_HOST, uri, query_str);
        let url = url::Url::parse(&url_string).unwrap();

        let client = reqwest::blocking::Client::new();
        let resp = client
            .get(url)
            .header("Authorization", &self.authorization_token)
            .header("Accept", "application/json")
            .header("Host", FXCM_API_HOST)
            .header("User-Agent", "request")
            .header("Content-Type", "application/x-www-form-urlencoded")
            .header("Connection", "close")
            .send();
        resp
    }

    fn http_post(&self, uri : &str, param_map : &Vec<(String, String)>) -> reqwest::Result<reqwest::blocking::Response> {
        let query_str = FxcmTradingService::pairs_to_query_string(&param_map);
        println!("Post body:{}", query_str);

        let url_string = format!("https://{}:443/{}", FXCM_API_HOST, uri);
        let url = url::Url::parse(&url_string).unwrap();

        let query_str_bytes = query_str.into_bytes();
        let client = reqwest::blocking::Client::new();
        let resp = client
            .post(url)
            .header("Authorization", &self.authorization_token)
            .header("Host", FXCM_API_HOST)
            .header("User-Agent", "request")
            .header("Content-Type", "application/x-www-form-urlencoded; charset=utf-8")
            .header("Connection", "close")
            .header("Transfer-Encoding", "chunked")
            .body(query_str_bytes)
            .send();
        resp
    }

    fn pairs_to_query_string(param_map : &Vec<(String, String)>) -> String {
        let mut query_str = String::new();
        for (k, v) in param_map {
            query_str.push_str(k);
            query_str.push_str("=");
            query_str.push_str(v);
            query_str.push_str("&");
        };

        query_str
    }

    fn http_get_json(&self, uri : &str, param_map : &Vec<(String, String)>) -> Result<serde_json::Value, String> {
        let http_get_result = self.http_get(uri, param_map);

        FxcmTradingService::http_response_to_json(http_get_result)
    }

    fn http_post_json(&self, uri : &str, param_map : &Vec<(String, String)>) -> Result<serde_json::Value, String> {
        let http_post_result = self.http_post(uri, param_map);

        FxcmTradingService::http_response_to_json(http_post_result)
    }

    fn http_response_to_json(response_result : reqwest::Result<reqwest::blocking::Response>) -> Result<serde_json::Value, String> {
        let response = match response_result {
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

        Ok(json_root)
    }
}

impl trading_lib::TradingService for FxcmTradingService {
    fn get_trade_symbols(&mut self) -> Result<Vec<String>, trading_lib::TradingError> {
        let json_root : Value = match self.http_get_json("trading/get_instruments/", &Vec::new()) {
            Ok(json_root) => json_root,
            Err(message) => return Err(message)
        };

        let maybe_instrument_array = &json_root["data"]["instrument"].as_array();
        let maybe_symbols = maybe_instrument_array.map(|a| {
            a.iter().map(|e| e["symbol"].as_str().map(String::from)).flatten().collect::<Vec<String>>()
        });
        maybe_symbols.map_or(Err(format!("Failed to read symbols from json: {:?}", json_root)), |s| Ok(s))
    }

    fn open_buy_trade(&mut self, symbol : &str, amount_in_lots : u32) -> Result<trading_lib::TradeId, trading_lib::TradingError> {
        let mut open_trade_params = Vec::new();
        open_trade_params.push((String::from("account_id"), self.account_id.clone()));
        open_trade_params.push((String::from("is_buy"), String::from("true")));
        open_trade_params.push((String::from("amount"), amount_in_lots.to_string()));
        open_trade_params.push((String::from("time_in_force"), String::from("GTC")));
        open_trade_params.push((String::from("order_type"), String::from("AtMarket")));
        open_trade_params.push((String::from("symbol"), String::from(symbol)));
        let json_root : Value = match self.http_post_json("trading/open_trade/", &open_trade_params) {
            Ok(json_root) => json_root,
            Err(message) => return Err(message)
        };

        println!("Open trade json: {}", json_root);
        let order_id = json_root["data"]["orderId"].as_u64().unwrap();
        println!("Received order id: {}", order_id);
        Err(String::from("temp implementation"))
    }

    fn open_sell_trade(&mut self, symbol : &str, amount_in_lots : u32) -> Result<trading_lib::TradeId, trading_lib::TradingError> {
        unimplemented!()
    }

    fn close_trade(&mut self, trade_id : &trading_lib::TradeId) -> Option<trading_lib::TradingError> {
        unimplemented!()
    }
}