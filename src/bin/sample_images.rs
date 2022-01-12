use std::{fs, io::prelude::*, path::PathBuf};

use anyhow::{bail, Context, Result};
use clap::Parser;
use opencv::imgcodecs;
use serde::{Deserialize, Serialize};
use serde_json;
use stdvis_core::{
    traits::Camera,
    types::{CameraConfig, VisionTarget},
};
use stdvis_opencv::{camera::OpenCVCamera, convert::AsMatView};

#[derive(Default, Serialize, Deserialize)]
struct Params {
    label: String,
    target: VisionTarget,
    camera: CameraConfig,
}

#[derive(Serialize, Deserialize)]
struct Metadata {
    images: Vec<ImageMetadata>,
}

#[derive(Serialize, Deserialize)]
struct ImageMetadata {
    index: usize,
    label: String,
    config: CameraConfig,
    exposure: f64,
}

// Captures a set of sample images.
#[derive(Debug, Parser)]
#[clap(about)]
struct Args {
    /// Specifies the amount of time, in ms, to wait between captures.
    #[clap(short, long, name = "delay")]
    delay_ms: Option<u64>,

    /// The path to read input parameters for.
    /// If the target file does not exist, a template will be created upon the first run.
    #[clap(parse(from_os_str))]
    params_file: PathBuf,

    /// The directory to output images and corresponding metadata to.
    #[clap(parse(from_os_str))]
    output_dir: PathBuf,
}

fn main() -> Result<()> {
    const METADATA_FILE: &str = "metadata.json";

    let args = Args::try_parse()?;

    let mut params_file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(args.params_file)
        .context("failed open params file")?;

    let mut params_str = String::new();
    params_file.read_to_string(&mut params_str).unwrap();

    if params_str.is_empty() {
        serde_json::to_writer_pretty(params_file, &Params::default()).unwrap();

        bail!(
            "Please configure input parameters.
            A template file has been created at the specified path."
        );
    }

    let params: Params = match serde_json::from_str(&params_str) {
        Ok(params) => params,
        Err(_) => {
            bail!(
                "Failed to parse input parameters.
                The config file may be malformed or out of date.
                You may want to let the script generate a new template."
            );
        }
    };

    let mut camera = OpenCVCamera::new(params.camera).unwrap();
    let camera_config = camera.config().clone();

    use std::{thread, time::Duration};

    // Allow the camera to "warm up."
    thread::sleep(Duration::from_millis(1000));

    let mut metadata_file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(METADATA_FILE)
        .unwrap();

    let mut metadata_str = String::new();
    metadata_file.read_to_string(&mut metadata_str).unwrap();

    let mut metadata =
        serde_json::from_str(&metadata_str).unwrap_or(Metadata { images: Vec::new() });

    let delay = args.delay_ms.map(|ms| Duration::from_millis(ms));
    let output_dir = args.output_dir;

    for idx in 0..10 {
        camera.set_exposure(-(idx as f64) * 0.1).unwrap();

        let frame = camera.grab_frame().unwrap();
        let image_mat = frame.as_mat_view();

        let index = metadata.images.len();

        imgcodecs::imwrite(
            output_dir
                .join(format!("{}_{}", params.label, index))
                .to_str()
                .unwrap(),
            &*image_mat,
            &opencv::types::VectorOfi32::with_capacity(0),
        )
        .context("writing image to disk")?;

        metadata.images.push(ImageMetadata {
            index,
            label: params.label.clone(),
            config: camera_config.clone(),
            exposure: camera.exposure().unwrap().clone(),
        });

        if let Some(delay) = delay {
            thread::sleep(delay);
        }
    }

    let metadata_file = fs::OpenOptions::new()
        .truncate(true)
        .write(true)
        .open("sample-images/metadata.json")?;

    serde_json::to_writer_pretty(metadata_file, &metadata)?;

    Ok(())
}
