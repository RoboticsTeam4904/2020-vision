use anyhow::{Context, Result};
use opencv::{
    aruco::{
        self, detect_markers, draw_detected_markers, estimate_pose_board,
        estimate_pose_single_markers, Board, DetectorParameters,
    },
    calib3d::{decompose_projection_matrix, rodrigues},
    core::{hconcat, no_array, Point2f, Point3f, Scalar, Vec3d, Vector},
    prelude::Mat,
    types::VectorOfMat,
};
use serde_json;
use std::{f64::consts::PI, fs::File, ops::DerefMut};
use stdvis_core::{
    traits::Camera,
    types::{CameraConfig, Image, VisionTarget},
};
use stdvis_opencv::{
    camera::{MatImageData, OpenCVCamera},
    convert::{AsArrayView, AsMatView},
};

const ARUCO_SIZE: f32 = 0.0369;

// for markers from theta = 0 clockwise, for each marker, top left corner and going counter clockwise
const ARUCO_BOARD_OBJECT_POINTS_SMALL: [[(f32, f32, f32); 4]; 16] = [
    [
        (0.67785, 0.0, 0.01845),
        (0.676846, 0.036882, 0.01845),
        (0.676846, 0.036882, -0.01845),
        (0.67785, 0.0, -0.01845),
    ],
    [
        (0.626252, 0.259402, 0.01845),
        (0.61121, 0.293092, 0.01845),
        (0.61121, 0.293092, -0.01845),
        (0.626252, 0.259402, -0.01845),
    ],
    [
        (0.479312, 0.479312, 0.01845),
        (0.452523, 0.504682, 0.01845),
        (0.452523, 0.504682, -0.01845),
        (0.479312, 0.479312, -0.01845),
    ],
    [
        (0.259402, 0.626252, 0.01845),
        (0.224943, 0.639438, 0.01845),
        (0.224943, 0.639438, -0.01845),
        (0.259402, 0.626252, -0.01845),
    ],
    [
        (0.0, 0.67785, 0.01845),
        (-0.036882, 0.676846, 0.01845),
        (-0.036882, 0.676846, -0.01845),
        (0.0, 0.67785, -0.01845),
    ],
    [
        (-0.259402, 0.626252, 0.01845),
        (-0.293092, 0.61121, 0.01845),
        (-0.293092, 0.61121, -0.01845),
        (-0.259402, 0.626252, -0.01845),
    ],
    [
        (-0.479312, 0.479312, 0.01845),
        (-0.504682, 0.452523, 0.01845),
        (-0.504682, 0.452523, -0.01845),
        (-0.479312, 0.479312, -0.01845),
    ],
    [
        (-0.626252, 0.259402, 0.01845),
        (-0.639438, 0.224943, 0.01845),
        (-0.639438, 0.224943, -0.01845),
        (-0.626252, 0.259402, -0.01845),
    ],
    [
        (-0.67785, 0.0, 0.01845),
        (-0.676846, -0.036882, 0.01845),
        (-0.676846, -0.036882, -0.01845),
        (-0.67785, 0.0, -0.01845),
    ],
    [
        (-0.626252, -0.259402, 0.01845),
        (-0.61121, -0.293092, 0.01845),
        (-0.61121, -0.293092, -0.01845),
        (-0.626252, -0.259402, -0.01845),
    ],
    [
        (-0.479312, -0.479312, 0.01845),
        (-0.452523, -0.504682, 0.01845),
        (-0.452523, -0.504682, -0.01845),
        (-0.479312, -0.479312, -0.01845),
    ],
    [
        (-0.259402, -0.626252, 0.01845),
        (-0.224943, -0.639438, 0.01845),
        (-0.224943, -0.639438, -0.01845),
        (-0.259402, -0.626252, -0.01845),
    ],
    [
        (-0.0, -0.67785, 0.01845),
        (0.036882, -0.676846, 0.01845),
        (0.036882, -0.676846, -0.01845),
        (-0.0, -0.67785, -0.01845),
    ],
    [
        (0.259402, -0.626252, 0.01845),
        (0.293092, -0.61121, 0.01845),
        (0.293092, -0.61121, -0.01845),
        (0.259402, -0.626252, -0.01845),
    ],
    [
        (0.479312, -0.479312, 0.01845),
        (0.504682, -0.452523, 0.01845),
        (0.504682, -0.452523, -0.01845),
        (0.479312, -0.479312, -0.01845),
    ],
    [
        (0.626252, -0.259402, 0.01845),
        (0.639438, -0.224943, 0.01845),
        (0.639438, -0.224943, -0.01845),
        (0.626252, -0.259402, -0.01845),
    ],
];

const ARUCO_BOARD_IDS_SMALL: [i32; 16] =
    [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48];

const ARUCO_BOARD_OBJECT_POINTS_BIG: [[(f32, f32, f32); 4]; 3] = [
    [
        (0.67785, 0.124, 0.0),
        (0.668749, 0.124, 0.110702),
        (0.668749, 0.0128, 0.110702),
        (0.67785, 0.0128, 0.0),
    ],
    [
        (0.626252, 0.124, 0.259402),
        (0.57548, 0.124, 0.358195),
        (0.57548, 0.0128, 0.358195),
        (0.626252, 0.0128, 0.259402),
    ],
    [
        (0.479312, 0.124, 0.479312),
        (0.394599, 0.124, 0.551155),
        (0.394599, 0.0128, 0.551155),
        (0.479312, 0.0128, 0.479312),
    ],
];

const ARUCO_BOARD_IDS_BIG: [i32; 3] = [6, 9, 12];

pub struct ArucoPoseResult {
    pub rvecs: Vector<Vec3d>,
    pub tvecs: Vector<Vec3d>,
    pub corners: Vector<Vector<Point2f>>,
}

// find where the markers are in the image, returns (corners, ids)
fn extract_markers(
    image: &Image<MatImageData>,
    intrinsic_matrix: &Mat,
    distortion_coeffs: &Mat,
) -> Result<(Vector<Vector<Point2f>>, Vector<i32>)> {
    let mat_image = image.as_mat_view();
    let dictionary =
        aruco::get_predefined_dictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_50)?;
    let mut corners = Vector::<Vector<Point2f>>::new();
    let mut ids = Vector::<i32>::new();
    let params = DetectorParameters::create()?;
    let mut rejected_img_points = no_array();

    detect_markers(
        &*mat_image,
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

// generate rotation and translation vectors from corners for individual markers
fn analyze_pose_single(
    corners: Vector<Vector<Point2f>>,
    intrinsic_matrix: &Mat,
    distortion_coeffs: &Mat,
) -> Result<ArucoPoseResult> {
    let mut rvecs = Vector::<Vec3d>::new();
    let mut tvecs = Vector::<Vec3d>::new();

    estimate_pose_single_markers(
        &corners,
        ARUCO_SIZE,
        intrinsic_matrix,
        distortion_coeffs,
        &mut rvecs,
        &mut tvecs,
        &mut no_array(),
    )?;

    Ok(ArucoPoseResult {
        rvecs,
        tvecs,
        corners,
    })
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

        let theta = (-x).atan2(-z);

        let euler_angles = euler_angles_mat.as_array_view::<f64>().into_shape((3, 1))?;
        let roll = euler_angles[[2, 0]] * PI / 180.;
        let pitch = euler_angles[[0, 0]] * PI / 180.;
        let yaw = euler_angles[[1, 0]] * PI / 180.;

        let target = VisionTarget {
            id: 0,
            theta: theta,
            beta: yaw,
            dist: (x.powi(2) + z.powi(2)).sqrt(),
            height: *y,
            confidence: 0.,
        };

        targets.push(target);
    }
    Ok(targets)
}

fn write_poses(
    image: &Image<MatImageData>,
    aruco_result: &ArucoPoseResult,
    ids: &Vector<i32>,
    intrinsic_matrix: &Mat,
    distortion_coeffs: &Mat,
) -> Result<()> {
    let mut mat_image = image.as_mat_view();
    let mut out_img = mat_image.deref_mut();

    draw_detected_markers(
        &mut out_img,
        &aruco_result.corners,
        ids,
        Scalar::new(255.0, 0.0, 0.0, 0.0),
    )?;

    for idx in 0..aruco_result.rvecs.len() {
        opencv::calib3d::draw_frame_axes(
            out_img,
            intrinsic_matrix,
            distortion_coeffs,
            &aruco_result.rvecs.get(idx)?,
            &aruco_result.tvecs.get(idx)?,
            0.05,
            3,
        )?;
    }

    opencv::imgcodecs::imwrite(
        "aruco.png",
        out_img,
        &opencv::types::VectorOfi32::with_capacity(0),
    )?;

    Ok(())
}

fn find_center(target: &VisionTarget) -> VisionTarget {
    let hoop_rad: f64 = 0.67785;

    let dx = target.dist * target.theta.sin() + hoop_rad * target.beta.sin();
    let dy = target.dist * target.theta.cos() + hoop_rad * target.beta.cos();

    let dist = (dx.powi(2) + dy.powi(2)).sqrt();
    let theta = dy.atan2(dx);

    VisionTarget {
        id: target.id,
        theta: theta,
        beta: 0.,
        dist: dist,
        height: target.height,
        confidence: 0.,
    }
}

fn find_average(targets: &Vec<VisionTarget>) -> VisionTarget {
    let centers: Vec<(f64, f64)> = targets
        .iter()
        .map(|target| {
            (
                find_center(target).dist * (find_center(target).theta).sin(),
                find_center(target).dist * (find_center(target).theta).cos(),
            )
        })
        .collect();

    let sum: (f64, f64) = centers
        .iter()
        .fold((0., 0.), |acc, x| (acc.0 + x.0, acc.1 + x.1));

    let dx = sum.0 / targets.len() as f64;
    let dy = sum.1 / targets.len() as f64;

    let dist = (dx.powi(2) + dy.powi(2)).sqrt();
    let theta = dy.atan2(dx);

    VisionTarget {
        id: 0,
        theta: theta,
        beta: 0.,
        dist: dist,
        height: 0.,
        confidence: 0.,
    }
}

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

        std::thread::sleep(std::time::Duration::from_millis(200));
    }

    Ok(())
}
