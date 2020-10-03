use std::net::TcpStream;
use serde_json::Value;
use anyhow::{anyhow, Context};

pub fn http_get_json(authorization_token : &str, host : &str, uri : &str, param_map : &Vec<(String, String)>) -> anyhow::Result<serde_json::Value> {
    let http_resp = http_get(authorization_token, host, uri, param_map).context("Failed to send HTTP GET")?;

    http_response_to_json(http_resp).context("Failed to parse HTTP response")
}

pub fn http_post_json(authorization_token : &str, host : &str, uri : &str, param_map : &Vec<(String, String)>) -> anyhow::Result<serde_json::Value> {
    let http_resp = http_post(authorization_token, host, uri, param_map).context("Failed to send HTTP POST")?;

    http_response_to_json(http_resp).context("Failed to parse HTTP response")
}

fn http_get(authorization_token : &str, host : &str, uri : &str, param_map : &Vec<(String, String)>) -> anyhow::Result<reqwest::blocking::Response> {
    let query_str = pairs_to_query_string(&param_map);

    let url_string = format!("https://{}:443/{}?{}", host, uri, query_str);
    let url = url::Url::parse(&url_string)?;

    let client = reqwest::blocking::Client::new();
    let resp = client
        .get(url)
        .header("Authorization", authorization_token)
        .header("Accept", "application/json")
        .header("Host", host)
        .header("User-Agent", "request")
        .header("Content-Type", "application/x-www-form-urlencoded")
        .header("Connection", "close")
        .send()?;
    Ok(resp)
}

fn http_post(authorization_token : &str, host : &str, uri : &str, param_map : &Vec<(String, String)>) -> anyhow::Result<reqwest::blocking::Response> {
    let query_str = pairs_to_query_string(&param_map);
    println!("Post body:{}", query_str);

    let url_string = format!("https://{}:443/{}", host, uri);
    let url = url::Url::parse(&url_string)?;

    let query_str_bytes = query_str.into_bytes();
    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(url)
        .header("Authorization", authorization_token)
        .header("Host", host)
        .header("User-Agent", "request")
        .header("Content-Type", "application/x-www-form-urlencoded; charset=utf-8")
        .header("Connection", "close")
        .header("Transfer-Encoding", "chunked")
        .body(query_str_bytes)
        .send()?;
    Ok(resp)
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

fn http_response_to_json(response : reqwest::blocking::Response) -> anyhow::Result<serde_json::Value> {
    if response.status() != http::StatusCode::OK {
        return Err(anyhow!("Erroneous HTTP status returned: {}", response.status()));
    }

    let response_body = response.text()?;
    let json_root : Value = serde_json::from_str(&response_body)?;

    Ok(json_root)
}

pub fn read_message_from_socket(socket : &mut tungstenite::WebSocket<native_tls::TlsStream<TcpStream>>,) -> anyhow::Result<serde_json::Value> {
    let message = socket.read_message()?;
    let message_text = message.to_text()?;
    let message_json = message_text.trim_start_matches(|c| c >= '0' && c <= '9');
    let v = serde_json::from_str(message_json)?;
    Ok(v)
}