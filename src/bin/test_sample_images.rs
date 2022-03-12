use std::fs::File;
use std::path::PathBuf;
use stdvis_core::traits::{ContourAnalyzer, ContourExtractor};
use vision_2020::{analysis, extraction};

use anyhow::{Context, Result};
use clap::Parser;
use opencv::imgcodecs::{self, IMREAD_COLOR};
use opencv::{
    aruco::{self, detect_markers, estimate_pose_board, Board, DetectorParameters},
    calib3d::{decompose_projection_matrix, rodrigues},
    core::{hconcat, no_array, Point2f, Point3f, Vec3d, Vector},
    prelude::Mat,
    types::VectorOfMat,
};
use std::f64::consts::PI;
use stdvis_core::types::VisionTarget;
use stdvis_core::types::{CameraConfig, Image};
use stdvis_opencv::{camera::MatImageData, convert::AsArrayView};

// for markers from theta = 0 clockwise, for each marker, top left corner and going counter clockwise

const ARUCO_BOARD_OBJECT_POINTS_BIG: [[(f32, f32, f32); 4]; 3] = [
    [
        (0.67785, 0.0, 0.124),
        (0.668749, 0.110702, 0.124),
        (0.668749, 0.110702, 0.0128),
        (0.67785, 0.0, 0.0128),
    ],
    [
        (0.626252, 0.259402, 0.124),
        (0.57548, 0.358195, 0.124),
        (0.57548, 0.358195, 0.0128),
        (0.626252, 0.259402, 0.0128),
    ],
    [
        (0.479312, 0.479312, 0.124),
        (0.394599, 0.551155, 0.124),
        (0.394599, 0.551155, 0.0128),
        (0.479312, 0.479312, 0.0128),
    ],
];

const ARUCO_BOARD_IDS_BIG: [i32; 3] = [6, 9, 12];

#[derive(Debug, Parser)]
#[clap(about)]
struct Args {
    #[clap(parse(from_os_str), required = true)]
    image_paths: Vec<PathBuf>,
}
pub struct ArucoPoseResult {
    pub rvecs: Vector<Vec3d>,
    pub tvecs: Vector<Vec3d>,
    pub corners: Vector<Vector<Point2f>>,
}

// find where the markers are in the image
fn extract_markers(
    image: &Mat,
    intrinsic_matrix: &Mat,
    distortion_coeffs: &Mat,
) -> Result<(Vector<Vector<Point2f>>, Vector<i32>)> {
    // let mat_image = image.as_mat_view();
    let dictionary =
        aruco::get_predefined_dictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_50)?;
    let mut corners = Vector::<Vector<Point2f>>::new();
    let mut ids = Vector::<i32>::new();
    let params = DetectorParameters::create()?;
    let mut rejected_img_points = no_array();

    detect_markers(
        &*image,
        &dictionary,
        &mut corners,
        &mut ids,
        &params,
        &mut rejected_img_points,
        &intrinsic_matrix,
        &distortion_coeffs,
    )?;

    Ok((corners, ids))
}

// same thing as `analyze_pose_single` but for a "board" of markers
fn analyze_pose_board(
    corners: Vector<Vector<Point2f>>,
    ids: &Vector<i32>,
    intrinsic_matrix: &Mat,
    distortion_coeffs: &Mat,
) -> Result<ArucoPoseResult> {
    let obj_points = Vector::<Vector<Point3f>>::from_iter(
        ARUCO_BOARD_OBJECT_POINTS_BIG.iter().map(|marker_points| {
            marker_points
                .iter()
                .map(|point| Point3f::new(point.0, point.1, point.2))
                .collect()
        }),
    );
    let dictionary =
        aruco::get_predefined_dictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_50)?;
    let all_ids = Vector::<i32>::from_slice(&ARUCO_BOARD_IDS_BIG);
    let board = Board::create(&obj_points, &dictionary, &all_ids)?;

    let mut rvec = Vec3d::default();
    let mut tvec = Vec3d::default();

    let num_markers = estimate_pose_board(
        &corners,
        ids,
        &board,
        intrinsic_matrix,
        distortion_coeffs,
        &mut rvec,
        &mut tvec,
        false,
    )?;

    if num_markers == 0 {
        return Ok(ArucoPoseResult {
            rvecs: Vector::<Vec3d>::new(),
            tvecs: Vector::<Vec3d>::new(),
            corners,
        });
    }

    Ok(ArucoPoseResult {
        rvecs: Vector::<Vec3d>::from_slice(&[rvec]),
        tvecs: Vector::<Vec3d>::from_slice(&[tvec]),
        corners,
    })
}

// find dist, theta, and yaw from rvecs and tvecs
fn find_targets(aruco_result: &ArucoPoseResult) -> Result<Vec<VisionTarget>> {
    let mut targets = Vec::new();

    for idx in 0..aruco_result.rvecs.len() {
        let rvec_vec = aruco_result.rvecs.get(idx)?;
        let tvec_vec = aruco_result.tvecs.get(idx)?;

        let mut rmat_mat = Mat::default();
        let mut jacobian_mat = Mat::default();
        rodrigues(&rvec_vec, &mut rmat_mat, &mut jacobian_mat)?;

        let tvec_mat = Mat::from_exact_iter(tvec_vec.into_iter())?;
        // let tvec = tvec_mat.as_array_view::<f64>().into_shape((3, 1))?;

        let mut mat_array = VectorOfMat::new();
        mat_array.push(rmat_mat);
        mat_array.push(tvec_mat);

        let mut proj_mat = Mat::default();
        hconcat(&mat_array, &mut proj_mat)?;

        let mut camera_matrix = Mat::default();
        let mut rot_matrix = Mat::default();
        let mut trans_vect = Mat::default();
        let mut rot_matrix_x = Mat::default();
        let mut rot_matrix_y = Mat::default();
        let mut rot_matrix_z = Mat::default();
        let mut euler_angles_mat = Mat::default();
        decompose_projection_matrix(
            &proj_mat,
            &mut camera_matrix,
            &mut rot_matrix,
            &mut trans_vect,
            &mut rot_matrix_x,
            &mut rot_matrix_y,
            &mut rot_matrix_z,
            &mut euler_angles_mat,
        )?;

        let x = tvec_vec.get(0).unwrap();
        let y = tvec_vec.get(1).unwrap();
        let z = tvec_vec.get(2).unwrap();

        let theta = x.atan2(*z) * 180. / PI;

        let euler_angles = euler_angles_mat.as_array_view::<f64>().into_shape((3, 1))?;
        let roll = euler_angles[[2, 0]];
        let pitch = euler_angles[[0, 0]];
        let yaw = euler_angles[[1, 0]];

        let target = VisionTarget {
            id: 0,
            theta: theta,
            beta: yaw,
            dist: *z,
            height: *y,
            confidence: 0.,
        };

        targets.push(target);
    }
    Ok(targets)
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

    for path in image_paths {
        dbg!(&path);
        let mut image_mat = imgcodecs::imread(path.to_str().unwrap(), IMREAD_COLOR)
            .context("reading image from disk")?;

        let image_image = Image::new(
            std::time::Instant::now(),
            &config,
            MatImageData::new(image_mat.clone()),
        );

        let contour_groups = extractor
            .extract_from(&image_image)
            .context("Contour extraction failed")?;

        if contour_groups.len() > 0 {
            let target = analyzer
                .analyze(&contour_groups[0])
                .context("Contour analysis failed")?;

            dbg!(target);

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

        let filename = path.file_name().unwrap().to_str().unwrap();

        opencv::imgcodecs::imwrite(
            &format!("b/{}", filename),
            &mut image_mat,
            &opencv::types::VectorOfi32::with_capacity(0),
        )?;

        println!("-----");
    }

    Ok(())
}
