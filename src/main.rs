//use tungstenite;
//use std::io;
// use std::path::Path;
// use std::fs::File;
// use url::Url;

use serde_json::Value;

fn main() {
    println!("Hello, world!");

    // let path = Path::new("hello.txt");
    // let file_result = File::create(&path);
    // let file = file_result.unwrap();

    println!("Opened file!");

    //let trading_api_url = "https://api-demo.fxcm.com:443";
    //let trading_api_url = "127.0.0.1:9001";

    // Test token validity in http://restapi101.herokuapp.com/
    let login_token = "4979200962b698e88aa1492f4e62f6e30e338a27";
    
    let result = tungstenite::connect(
        format!("wss://api-demo.fxcm.com/socket.io/?EIO=3&transport=websocket&access_token={}", login_token));
    match result {
        Ok((mut socket, _response)) => {
            println!("Success");

            let maybe_message = socket.read_message();
            println!("WebSocket message: {:?}", maybe_message);
            let v: Value = serde_json::from_str(maybe_message.unwrap().to_text().unwrap().strip_prefix("0").unwrap()).unwrap();
            println!("{}", v["sid"]);
            let server_socket_id = format!("{}", v["sid"].as_str().unwrap());
    
            let bearer_access_token = format!("Bearer {}{}", server_socket_id, login_token);
            println!("Bearer: {}", bearer_access_token);

            let maybe_message2 = socket.read_message();
            println!("WebSocket message2: {:?}", maybe_message2);

            // Bearer n5BoWSRFJvYi2GEQAAKy a11f7bc3d6b14ff77f65dd9d21df16ac1b4c41ea
            // Bearer covX2GE44OPZnn9SACma 4053cece3bbe29d6c994691d34b05316c210a137

            let client = reqwest::blocking::Client::new();
            let resp = client
                // .get("https://api-demo.fxcm.com:443/candles/1/m1?num=5")
                .get("https://api-demo.fxcm.com:443/trading/get_instruments/")
                .header("Authorization", bearer_access_token)
                .header("Accept", "application/json")
                .header("Host", "api-demo.fxcm.com")
                // .header("port", "443")
                // .header("path", "application/json")
                .header("User-Agent", "request")
                .header("Content-Type", "application/x-www-form-urlencoded'")
                .header("Connection", "close")
                .send();
            // println!("{:#?}", resp);
            match resp {
                Ok(r) => println!("{:?}", r.text()),
                _ => println!("error")
            }

            println!("Finished!");
        } ,
        Err(err) => {
            println!("Could not connect: '{}'", err);
        }
    };
}
