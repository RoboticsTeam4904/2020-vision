mod analysis;
mod extraction;
mod pipeline;

use anyhow::{Context, Result};
use serde_json;
use std::{fs::File, io::Write, time::SystemTime};
use stdvis_core::traits::{Camera, ContourExtractor};
use stdvis_opencv::camera::OpenCVCamera;

fn main() -> Result<()> {
    // Update last_run_time file.
    // Used to make sure the vision service is working as intended.
    let mut timestamp_file = File::open("last_run_time.txt")?;
    write!(timestamp_file, "Last run: {:?}", SystemTime::now());

    let config_file = File::open("config.json")?;
    let config = serde_json::from_reader(config_file)?;

    let mut camera = OpenCVCamera::new(config)?;
    let extractor = extraction::RFTapeContourExtractor::new();
    let analyzer = analysis::WallTapeContourAnalyzer::new();

    let mut pipeline = pipeline::VisionPipeline::new(camera, extractor, analyzer);

    loop {
        let target = pipeline.run()?;
        dbg!(target);

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
