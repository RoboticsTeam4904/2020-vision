use stdvis_core::{
    traits::Camera,
    types::{CameraConfig, Target},
};
use stdvis_opencv::{camera::OpenCVCamera, convert::AsMatView};

use serde::{Deserialize, Serialize};
use serde_json;

#[derive(Default, Serialize, Deserialize)]
struct Input {
    config: CameraConfig,
    target: Target,
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
}

fn main() {
    const IN_FILE: &str = "input.json";
    const OUT_PATH: &str = "sample-images";
    const METADATA_FILE: &str = "sample-images/metadata.json";

    let mut input_file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(IN_FILE)
        .unwrap();

    let mut input_str = String::new();
    input_file.read_to_string(&mut input_str).unwrap();

    let input: Input = match serde_json::from_str(&input_str) {
        Ok(input) => input,
        Err(_) => {
            serde_json::to_writer_pretty(input_file, &Input::default()).unwrap();

            println!("Please configure camera config and target parameters.");
            return;
        }
    };

    let config = input.config;

    let mut cam = OpenCVCamera::new(
        config.id,
        config.pose,
        config.fov,
        config.focal_length,
        config.sensor_width,
        config.sensor_height,
    )
    .unwrap();

    use std::{thread, time};

    // Allow the camera to "warm up."
    thread::sleep(time::Duration::from_millis(1000));

    use std::env;
    use std::fs;
    use std::io::prelude::*;

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

    let args = env::args().collect::<Vec<_>>();
    let label = args.get(1).unwrap();

    for idx in 0..10 {
        let frame = cam.grab_frame().unwrap();
        let image_mat = frame.as_mat_view();

        println!("current exposure: {}", cam.config().exposure);

        use opencv::{imgcodecs, prelude::*};

        println!("{}", cam.set_exposure(-(idx as f64) * 0.1).unwrap());

        let index = metadata.images.len();

        metadata.images.push(ImageMetadata {
            index: idx,
            label: label.clone(),
            config: (*cam.config()).clone(),
        });

        imgcodecs::imwrite(
            format!("{}/{}_{}.png", OUT_PATH, label, index).as_str(),
            &*image_mat,
            &opencv::types::VectorOfint::with_capacity(0),
        )
        .unwrap();
    }

    let metadata_file = fs::OpenOptions::new()
        .truncate(true)
        .write(true)
        .open("sample-images/metadata.json")
        .unwrap();

    serde_json::to_writer_pretty(metadata_file, &metadata).unwrap();
}
