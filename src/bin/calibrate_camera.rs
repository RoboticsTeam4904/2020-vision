use std::{fs, io::prelude::*, path::PathBuf};

use anyhow::{bail, Context, Result};
use clap::Parser;
use opencv::{
    calib3d::{
        self, CALIB_CB_ACCURACY, CALIB_CB_EXHAUSTIVE, CALIB_CB_LARGER, CALIB_CB_MARKER,
        CALIB_CB_NORMALIZE_IMAGE,
    },
    core::{Mat, Point2f, Point3f, Size, TermCriteria, TermCriteria_Type, Vector},
    imgcodecs::{self, IMREAD_COLOR},
    prelude::*,
};

use stdvis_core::types::CameraConfig;
use stdvis_opencv::convert::AsArrayView;

const MIN_CALIBRATION_IMAGES: usize = 3;

#[derive(Debug, Parser)]
#[clap(about)]
struct Args {
    /// Side length of a checkerboard square in millimeters
    #[clap(short = 's', long = "square-size")]
    square_size_mm: f32,

    /// Number of checkerboard rows to try to detect
    #[clap(short = 'r', long = "board-rows")]
    board_rows: u8,

    /// Number of checkerboard columns to try to detect
    #[clap(short = 'c', long = "board-cols")]
    board_cols: u8,

    /// Path to write resulting camera configuration file
    #[clap(short = 'o', long = "config", parse(from_os_str))]
    camera_config: PathBuf,

    /// If specified, path to directory for writing images with detected checkerboard markers overlaid
    #[clap(short, long, parse(from_os_str))]
    debug_dir: Option<PathBuf>,

    /// Paths to images containing checkerboards to be used for calibration
    #[clap(parse(from_os_str), min_values = MIN_CALIBRATION_IMAGES, required = true)]
    image_paths: Vec<PathBuf>,
}

fn compute_checkerboard_obj_points(square_size: f32, width: u8, height: u8) -> Vector<Point3f> {
    let mut obj_points = Vector::with_capacity((width * height) as usize);

    for col in 0..width {
        for row in 0..height {
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
    let args = Args::parse();

    let image_paths = args.image_paths;

    let board_rows = args.board_rows - 1;
    let board_cols = args.board_cols - 1;

    let num_images = image_paths.len();
    let template_obj_points =
        compute_checkerboard_obj_points(args.square_size_mm / 1000., board_cols, board_rows);

    let mut object_points: Vector<Vector<Point3f>> = Vector::with_capacity(num_images);
    let mut image_points: Vector<Vector<Point2f>> = Vector::with_capacity(num_images);

    let mut image_size = None;
    let pattern_size = Size::new(board_cols as i32, board_rows as i32);

    for path in image_paths {
        let image = imgcodecs::imread(path.to_str().unwrap(), IMREAD_COLOR)
            .context("reading image from disk")?;

        let curr_image_size = image.size().context("expected image to have size")?;

        if image_size.get_or_insert(curr_image_size) != &curr_image_size {
            bail!("Expected all images to be the same size. Failed on image: {path:?}");
        }

        let mut corners: Vector<Point2f> = Vector::new();
        let found = calib3d::find_chessboard_corners_sb(
            &image,
            pattern_size,
            &mut corners,
            CALIB_CB_NORMALIZE_IMAGE
                | CALIB_CB_EXHAUSTIVE
                | CALIB_CB_ACCURACY
                | CALIB_CB_MARKER
                | CALIB_CB_LARGER,
        )
        .context("finding checkerboard corners")?;

        if let Some(ref out_path) = args.debug_dir {
            let mut image_debug = image.clone();

            calib3d::draw_chessboard_corners(&mut image_debug, pattern_size, &corners, found)
                .context("drawing checkerboard corners")?;

            imgcodecs::imwrite(
                out_path.join(path.file_name().unwrap()).to_str().unwrap(),
                &image_debug,
                &Vector::new(),
            )
            .context("writing debug image to disk")?;
        }

        if !found {
            println!("Failed to find checkerboard corners for image: {path:?}");
            continue;
        }

        let mut sharpness = opencv::core::no_array();
        let sharpness_stats = calib3d::estimate_chessboard_sharpness(
            &image,
            pattern_size,
            &corners,
            0.8,
            false,
            &mut sharpness,
        )
        .context("estimating checkerboard sharpness")?;

        println!(
            "avg. sharpness: {}, avg. brightness (min, max): ({}, {}) for image: {path:?}",
            sharpness_stats[0], sharpness_stats[1], sharpness_stats[2]
        );

        object_points.push(template_obj_points.clone());
        image_points.push(corners);
    }

    println!(
        "Successfully performed corner-finding on {} images",
        image_points.len()
    );

    if image_points.len() < MIN_CALIBRATION_IMAGES {
        bail!("Insufficient successful corner-finding results to continue to calibration");
    }

    let image_size = image_size.expect("image_size should be Some");

    // Use the top-right corner of the checkerboard as a fixed point, as recommended by the documentation for `calibrate_camera_ro`.
    let i_fixed_point = (template_obj_points.len() - 2) as i32;

    let mut camera_matrix = Mat::default();
    let mut dist_coeffs = Mat::default();
    let mut rvecs = Mat::default();
    let mut tvecs = Mat::default();
    let mut new_obj_points = opencv::core::no_array();

    let reproj_error = calib3d::calibrate_camera_ro(
        &object_points,
        &image_points,
        image_size,
        i_fixed_point,
        &mut camera_matrix,
        &mut dist_coeffs,
        &mut rvecs,
        &mut tvecs,
        &mut new_obj_points,
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

    println!("Calibration finished with reprojection error: {reproj_error}");

    let mut config_file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(&args.camera_config)
        .context("opening config file")?;

    let mut config_str = String::new();
    config_file
        .read_to_string(&mut config_str)
        .context("reading camera config file")?;

    let mut config = if config_str.is_empty() {
        CameraConfig::default()
    } else {
        match serde_json::from_str(&config_str) {
            Ok(config) => config,
            Err(_) => {
                panic!("Failed to parse camera configuration; the schema used may be out of date");
            }
        }
    };

    config.intrinsic_matrix = camera_matrix
        .as_array_view::<f64>()
        .into_shape((3, 3))
        .context("converting intrinsic_matrix Mat")?
        .to_owned();

    config.distortion_coeffs = dist_coeffs
        .as_array_view::<f64>()
        .into_shape(5)
        .context("converting distortion_coeffs Mat")?
        .to_owned();

    serde_json::to_writer_pretty(config_file, &config).context("writing updated config file")?;

    Ok(())
}
