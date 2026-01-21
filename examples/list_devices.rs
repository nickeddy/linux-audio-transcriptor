// List available audio devices
// Run with: cargo run --example list_devices

use cpal::traits::{DeviceTrait, HostTrait};

fn main() {
    let host = cpal::default_host();

    println!("Host: {:?}\n", host.id());

    println!("=== Input Devices ===");
    match host.input_devices() {
        Ok(devices) => {
            for device in devices {
                if let Ok(name) = device.name() {
                    let is_monitor = name.to_lowercase().contains("monitor");
                    let marker = if is_monitor { " [MONITOR]" } else { "" };
                    println!("  - {}{}", name, marker);
                }
            }
        }
        Err(e) => println!("  Error: {}", e),
    }

    println!("\n=== Output Devices ===");
    match host.output_devices() {
        Ok(devices) => {
            for device in devices {
                if let Ok(name) = device.name() {
                    println!("  - {}", name);
                }
            }
        }
        Err(e) => println!("  Error: {}", e),
    }

    println!("\n=== Default Devices ===");
    if let Some(device) = host.default_input_device() {
        println!("  Default input: {}", device.name().unwrap_or_default());
    }
    if let Some(device) = host.default_output_device() {
        println!("  Default output: {}", device.name().unwrap_or_default());
    }
}
