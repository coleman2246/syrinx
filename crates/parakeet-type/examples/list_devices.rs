use cpal::traits::{DeviceTrait, HostTrait};
fn main() {
    let host = cpal::default_host();
    println!("host: {:?}", host.id());
    for (i, d) in host.input_devices().unwrap().enumerate() {
        let name = d.description().map(|x| x.name().to_string()).unwrap_or("?".into());
        let cfg = d.default_input_config().map(|c| format!("{}Hz {}ch {:?}", c.sample_rate(), c.channels(), c.sample_format())).unwrap_or("-".into());
        println!("  [{i}] {name}  ({cfg})");
    }
}
