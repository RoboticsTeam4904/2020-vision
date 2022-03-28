mod analysis;
mod extraction;
mod pipeline;

use anyhow::{Context, Result};
use serde_json;
use std::{fs::File, time::SystemTime};
use stdvis_opencv::camera::OcvCamera;
use vision_2020::udp::UdpSender;

fn main() -> Result<()> {
    let config_file = File::open("config.json")?;
    let config = serde_json::from_reader(config_file)?;

    let mut camera = OcvCamera::new(config)?;
    camera.set_exposure(18);
    let extractor = extraction::RFTapeContourExtractor::new();
    let analyzer = analysis::WallTapeContourAnalyzer::new();

    let mut pipeline = pipeline::VisionPipeline::new(camera, extractor, analyzer);

    // TODO: Consider moving destination hostname to environment variable.
    // This line will block until the hostname appears on the network.
    println!("Initializing UDP sensor.");
    let sender = UdpSender::new(7698, "10.49.4.9:2746".to_string());
    println!("UDP sensor initialized.");

    loop {
        let target = pipeline.run()?;
        dbg!(&target);
        sender.send((
            target,
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64,
        ));
    }

    Ok(())
}
