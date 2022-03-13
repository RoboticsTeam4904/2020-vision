mod analysis;
mod extraction;
mod pipeline;

use anyhow::{Context, Result};
use serde_json;
use std::{fs::File, io::Write, time::SystemTime};
use stdvis_core::traits::{Camera, ContourExtractor};
use stdvis_opencv::camera::OpenCVCamera;
use vision_2020::udp::UdpSender;

fn main() -> Result<()> {
    let config_file = File::open("config.json")?;
    let config = serde_json::from_reader(config_file)?;

    let mut camera = OpenCVCamera::new(config)?;
    let extractor = extraction::RFTapeContourExtractor::new();
    let analyzer = analysis::WallTapeContourAnalyzer::new();

    let mut pipeline = pipeline::VisionPipeline::new(camera, extractor, analyzer);
    // TODO: Consider moving destination hostname to environment variable.
    // This line will block until the hostname appears on the network.
    let sender = UdpSender::new(4904, "nano2-4904-frc.local:4826".to_string());

    loop {
        let target = pipeline.run()?;
        dbg!(&target);
        sender.send((
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
            target,
        ));

        // let frame = camera
        //     .grab_frame()
        //     .context("Failed to read frame from camera")?;

        // let _contour_groups = extractor
        //     .extract_from(&frame)
        //     .context("Contour extraction failed")?;

        std::thread::sleep(std::time::Duration::from_millis(200));
    }

    Ok(())
}
