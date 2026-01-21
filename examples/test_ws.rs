// Quick WebSocket connection test - run with: cargo run --example test_ws
// Save this as examples/test_ws.rs

use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures_util::{SinkExt, StreamExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let url = "ws://127.0.0.1:8765";
    println!("Connecting to ASR server at {}...", url);

    let (mut ws_stream, _) = connect_async(url).await?;
    println!("Connected!");

    // Send a ping message
    let ping_msg = serde_json::json!({"type": "ping"});
    ws_stream.send(Message::Text(ping_msg.to_string())).await?;
    println!("Sent ping message");

    // Wait for response
    if let Some(msg) = ws_stream.next().await {
        match msg? {
            Message::Text(text) => {
                println!("Received: {}", text);
            }
            Message::Ping(_) => {
                println!("Received WS ping");
            }
            other => {
                println!("Received other: {:?}", other);
            }
        }
    }

    // Send start message to begin session
    let start_msg = serde_json::json!({"type": "start"});
    ws_stream.send(Message::Text(start_msg.to_string())).await?;
    println!("Sent start message");

    // Wait for acknowledgement
    if let Some(msg) = ws_stream.next().await {
        match msg? {
            Message::Text(text) => {
                println!("Received: {}", text);
            }
            other => {
                println!("Received: {:?}", other);
            }
        }
    }

    println!("Connection test successful!");
    Ok(())
}
