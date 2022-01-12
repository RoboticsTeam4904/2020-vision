use std::{env, fs, io::prelude::*};

use anyhow::{bail, Context, Result};
use opencv::{
    calib3d,
    core::{Mat, Point3f, Size, TermCriteria, TermCriteria_Type},
    imgcodecs::{self, IMREAD_COLOR},
    prelude::*,
    types::{VectorOfPoint2f, VectorOfPoint3f, VectorOfVectorOfPoint2f, VectorOfVectorOfPoint3f},
};
use serde::{Deserialize, Serialize};
use serde_json;

use stdvis_core::types::CameraConfig;
use stdvis_opencv::convert::AsArrayView;

#[derive(Default, Serialize, Deserialize)]
struct Config {
    square_size_mm: f32,
    board_rows: u8,
    board_cols: u8,
    camera: CameraConfig,
}

fn compute_checkerboard_obj_points(square_size: f32, width: u8, height: u8) -> Vec<Point3f> {
    let mut obj_points = Vec::with_capacity((width * height) as usize);

    for row in 0..height {
        for col in 0..width {
            obj_points.push(Point3f::new(
                square_size * col as f32,
                square_size * row as f32,
                0.,
            ));
        }
    }

    obj_points
}

fn main() -> Result<()> {
    let mut args = env::args().skip(1);

    let config_path = args.next().expect("Expected config path as first arg");

    let image_paths = args.collect::<Vec<_>>();

    if image_paths.len() < 2 {
        panic!("Expected at least 2 images");
    }

    let mut config_file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(&config_path)
        .context("opening config file")?;

    let mut config_str = String::new();
    config_file
        .read_to_string(&mut config_str)
        .context("reading config file")?;

    if config_str.is_empty() {
        serde_json::to_writer_pretty(config_file, &Config::default())
            .context("writing template config file")?;

        bail!("Please provide camera configuration and target parameters.");
    }

    let mut config: Config = match serde_json::from_str(&config_str) {
        Ok(config) => config,
        Err(_) => {
            panic!("Failed to parse camera configuration and target parameters. The schema used may be out of date.");
        }
    };

    let num_images = image_paths.len();
    let template_obj_points = compute_checkerboard_obj_points(
        config.square_size_mm / 1000.,
        config.board_rows,
        config.board_cols,
    );

    let mut obj_points = Vec::<VectorOfPoint3f>::with_capacity(num_images);
    let mut img_points = Vec::<VectorOfPoint2f>::with_capacity(num_images);

    let resolution = config.camera.resolution;
    let image_size = Size::new(resolution.0 as i32, resolution.1 as i32);

    for path in image_paths {
        let image = imgcodecs::imread(&path, IMREAD_COLOR).context("reading image from disk")?;

        if image.size().unwrap() != image_size {
            bail!("Expected all images to be the same size");
        }

        let mut corners = VectorOfPoint2f::new();
        calib3d::find_chessboard_corners_sb(
            &image,
            Size::new(config.board_rows as i32, config.board_cols as i32),
            &mut corners,
            0,
        )
        .context("failed to find checkerboard corners")?;

        obj_points.push(VectorOfPoint3f::from_iter(template_obj_points.clone()));
        img_points.push(corners);
    }

    let mut camera_matrix = Mat::default();
    let mut dist_coeffs = Mat::default();
    let mut rvecs = Mat::default();
    let mut tvecs = Mat::default();

    calib3d::calibrate_camera(
        &VectorOfVectorOfPoint3f::from_iter(obj_points),
        &VectorOfVectorOfPoint2f::from_iter(img_points),
        image_size,
        &mut camera_matrix,
        &mut dist_coeffs,
        &mut rvecs,
        &mut tvecs,
        0,
        // Listed as the default TermCriteria for this method in the OpenCV docs.
        TermCriteria::new(
            TermCriteria_Type::COUNT as i32 + TermCriteria_Type::EPS as i32,
            30,
            std::f64::EPSILON,
        )
        .unwrap(),
    )
    .context("calibrating camera")?;

    config.camera.intrinsic_matrix = camera_matrix
        .as_array_view::<f64>()
        .into_shape((3, 3))
        .context("converting intrinsic_matrix Mat")?
        .to_owned();

    config.camera.distortion_coeffs = dist_coeffs
        .as_array_view::<f64>()
        .into_shape(5)
        .context("converting distortion_coeffs Mat")?
        .to_owned();

    let config_file = fs::OpenOptions::new()
        .truncate(true)
        .write(true)
        .open(&config_path)
        .context("opening config file")?;

    serde_json::to_writer_pretty(config_file, &config).context("writing updated config file")?;

    Ok(())
}
