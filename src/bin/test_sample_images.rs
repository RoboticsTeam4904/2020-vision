use anyhow::{Context, Result};

use clap::Parser;

use opencv::{imgcodecs, prelude::Mat};

use std::{f64::consts::PI, fs::File, path::PathBuf};

use stdvis_core::{
    traits::{ContourAnalyzer, ContourExtractor, ImageData},
    types::{CameraConfig, Image, VisionTarget},
};

use stdvis_opencv::camera::MatImageData;

use vision_2020::{
    analysis,
    aruco::{analyze_pose_board, extract_markers, find_targets},
    extraction,
};

#[derive(Debug, Parser)]
#[clap(about)]
struct Args {
    #[clap(parse(from_os_str), required = true)]
    image_paths: Vec<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();

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

    let image_paths = args.image_paths;

    let extractor = extraction::RFTapeContourExtractor::new();
    let analyzer = analysis::WallTapeContourAnalyzer::new();

    // -------- IMPORT IMAGES --------

    let mut images = Vec::new();
    for path in image_paths {
        let image_mat = imgcodecs::imread(path.to_str().unwrap(), imgcodecs::IMREAD_COLOR)
            .context("reading image from disk")?;
        images.push((
            Image::new(
                std::time::Instant::now(),
                &config,
                MatImageData::new(image_mat.clone()),
            ),
            path.file_name().unwrap().to_str().unwrap().to_string(),
        ))
    }

    // -------- ARUCO STUFF --------

    let mut aruco_targets = Vec::new();

    for image_data in &images {
        let (image_image, _filename) = image_data;
        let (corners, ids) = extract_markers(image_image, &intrinsic_matrix, &distortion_coeffs)?;
        let aruco_result =
            analyze_pose_board(corners, &ids, &intrinsic_matrix, &distortion_coeffs)?;
        let targets = find_targets(&aruco_result)?;
        if let Some(target) = targets.get(0) {
            aruco_targets.push(target.clone());
        }
    }

    let aruco_target_sum = aruco_targets.iter().fold(
        VisionTarget {
            id: 0,
            beta: 0.,
            theta: 0.,
            dist: 0.,
            height: 0.,
            confidence: 0.,
        },
        |acc, target| VisionTarget {
            id: 0,
            beta: acc.beta + target.beta,
            theta: acc.theta + target.theta,
            dist: acc.dist + target.dist,
            height: acc.height + target.height,
            confidence: acc.confidence + target.confidence,
        },
    );
    let num_targets = aruco_targets.len() as f64;
    let aruco_target_avg = VisionTarget {
        id: 0,
        beta: aruco_target_sum.beta / num_targets,
        theta: aruco_target_sum.theta / num_targets,
        dist: aruco_target_sum.dist / num_targets,
        height: aruco_target_sum.height / num_targets,
        confidence: aruco_target_sum.confidence / num_targets as f32,
    };
    dbg!(&aruco_target_avg);

    // -------- PNP STUFF --------

    for image_data in images {
        let (mut image_image, filename) = image_data;

        dbg!(&filename);

        let contour_groups = extractor
            .extract_from(&image_image)
            .context("Contour extraction failed")?;

        let mut image_mat = image_image.as_raw_mut();

        if contour_groups.len() > 0 {
            let target = analyzer
                .analyze(&contour_groups[0])
                .context("Contour analysis failed")?;

            dbg!(&target);
            dbg!(&target.theta * 180. / PI);

            let diff = VisionTarget {
                id: 0,
                beta: target.beta - aruco_target_avg.beta,
                theta: target.theta - aruco_target_avg.theta,
                dist: target.dist - aruco_target_avg.dist,
                height: target.height - aruco_target_avg.height,
                confidence: target.confidence - aruco_target_avg.confidence,
            };

            // dbg!(diff);

            let pnp_params = analyzer.make_pnp_params(&contour_groups[0]);
            let pnp_result = analyzer.solve_pnp(pnp_params)?;

            opencv::calib3d::draw_frame_axes(
                &mut image_mat,
                &intrinsic_matrix,
                &distortion_coeffs,
                &pnp_result[0].rvec_mat,
                &pnp_result[0].tvec_mat,
                0.3,
                2,
            )?;
        }

        opencv::imgcodecs::imwrite(
            &format!("b/{}", filename),
            image_mat,
            &opencv::types::VectorOfi32::with_capacity(0),
        )?;

        println!("-----");
    }

    Ok(())
}
