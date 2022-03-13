use anyhow::{Context, Result};
use opencv::prelude::Mat;
use serde_json;
use std::fs::File;
use stdvis_core::{traits::Camera, types::CameraConfig};
use stdvis_opencv::camera::OpenCVCamera;
use vision_2020::aruco::{analyze_pose_board, extract_markers, find_targets, write_poses};

fn main() -> Result<()> {
    let config_file = File::open("config.json")?;
    let config: CameraConfig = serde_json::from_reader(config_file)?;

    let i = &config.intrinsic_matrix;
    let d = &config.distortion_coeffs;

    let intrinsic_matrix = Mat::from_slice_2d(&[
        &[i[[0, 0]], i[[0, 1]], i[[0, 2]]],
        &[i[[1, 0]], i[[1, 1]], i[[1, 2]]],
        &[i[[2, 0]], i[[2, 1]], i[[2, 2]]],
    ])?;

    let distortion_coeffs = Mat::from_slice(d.as_slice().unwrap())?;

    let mut camera = OpenCVCamera::new(config)?;

    loop {
        let image = camera
            .grab_frame()
            .context("Failed to read frame from camera")?;

        let (corners, ids) = extract_markers(&image, &intrinsic_matrix, &distortion_coeffs)?;
        // let aruco_result = analyze_pose_single(corners, &intrinsic_matrix, &distortion_coeffs)?;
        let aruco_result =
            analyze_pose_board(corners, &ids, &intrinsic_matrix, &distortion_coeffs)?;
        let targets = find_targets(&aruco_result)?;
        write_poses(
            &image,
            &aruco_result,
            &ids,
            &intrinsic_matrix,
            &distortion_coeffs,
        )?;

        dbg!(targets);

        std::thread::sleep(std::time::Duration::from_millis(200));
    }

    Ok(())
}
