use stdvis_core::{
    traits::Camera,
    types::{CameraConfig, Target},
};
use stdvis_opencv::{camera::OpenCVCamera, convert::AsMatView};

use serde::{Deserialize, Serialize};
use serde_json;

#[derive(Default, Serialize, Deserialize)]
struct Params {
    label: String,
    target: Target,
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

fn main() {
    const IN_FILE: &str = "params.json";
    const METADATA_FILE: &str = "metadata.json";

    let mut args = env::args().skip(1);
    let out_path = args.next().expect("Expected an output directory path");

    let mut params_file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(format!("{}/{}", out_path, IN_FILE))
        .unwrap();

    let mut params_str = String::new();
    params_file.read_to_string(&mut params_str).unwrap();

    if params_str.is_empty() {
        serde_json::to_writer_pretty(params_file, &Params::default()).unwrap();

        println!("Please configure input parameters.");
        return;
    }

    let params: Params = match serde_json::from_str(&params_str) {
        Ok(params) => params,
        Err(_) => {
            panic!("Failed to parse input parameters. The schema used may be out of date.");
        }
    };

    let mut camera = OpenCVCamera::new(params.camera).unwrap();
    let camera_config = camera.config().clone();

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

    for idx in 0..10 {
        camera.set_exposure(-(idx as f64) * 0.1).unwrap();

        let frame = camera.grab_frame().unwrap();
        let image_mat = frame.as_mat_view();

        use opencv::{imgcodecs, prelude::*};

        let index = metadata.images.len();

        imgcodecs::imwrite(
            format!("{}/{}_{}.png", out_path, params.label, index).as_str(),
            &*image_mat,
            &opencv::types::VectorOfi32::with_capacity(0),
        )
        .unwrap();

        metadata.images.push(ImageMetadata {
            index,
            label: params.label.clone(),
            config: camera_config.clone(),
            exposure: camera.exposure().unwrap().clone(),
        });
    }

    let metadata_file = fs::OpenOptions::new()
        .truncate(true)
        .write(true)
        .open("sample-images/metadata.json")
        .unwrap();

    serde_json::to_writer_pretty(metadata_file, &metadata).unwrap();
}
