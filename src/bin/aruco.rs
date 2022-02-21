use anyhow::{Context, Result};
use opencv::{
    aruco::{
        self, detect_markers, draw_detected_markers, estimate_pose_single_markers,
        DetectorParameters,
    },
    calib3d::{decompose_projection_matrix, rodrigues},
    core::{hconcat, no_array, Point2f, Scalar, Vec3d, Vector},
    prelude::Mat,
    types::VectorOfMat,
};
use serde_json;
use std::{f64::consts::PI, fs::File, ops::DerefMut};
use stdvis_core::{
    traits::Camera,
    types::{Image, VisionTarget},
};
use stdvis_opencv::{
    camera::{MatImageData, OpenCVCamera},
    convert::{AsArrayView, AsMatView},
};
pub struct ArucoPoseResult {
    pub intrinsic_matrix: Mat,
    pub distortion_coeffs: Mat,
    pub rvecs: Vector<Vec3d>,
    pub tvecs: Vector<Vec3d>,
    pub ids: Vector<i32>,
    pub corners: Vector<Vector<Point2f>>,
}

fn find_pose(image: &Image<MatImageData>) -> Result<ArucoPoseResult> {
    let mat_image = image.as_mat_view();
    let dictionary =
        aruco::get_predefined_dictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_50)?;
    let mut corners = Vector::<Vector<Point2f>>::new();
    let mut ids = Vector::<i32>::new();
    let params = DetectorParameters::create()?;
    let mut rejected_img_points = no_array();

    let i = image.camera.intrinsic_matrix.view();
    let d = image.camera.distortion_coeffs.view();

    let intrinsic_matrix = Mat::from_slice_2d(&[
        &[i[[0, 0]], i[[0, 1]], i[[0, 2]]],
        &[i[[1, 0]], i[[1, 1]], i[[1, 2]]],
        &[i[[2, 0]], i[[2, 1]], i[[2, 2]]],
    ])?;

    let distortion_coeffs = Mat::from_slice(d.as_slice().unwrap())?;

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

    let mut rvecs = Vector::<Vec3d>::new();
    let mut tvecs = Vector::<Vec3d>::new();

    estimate_pose_single_markers(
        &corners,
        0.036,
        &intrinsic_matrix,
        &distortion_coeffs,
        &mut rvecs,
        &mut tvecs,
        &mut no_array(),
    )?;

    Ok(ArucoPoseResult {
        intrinsic_matrix,
        distortion_coeffs,
        rvecs,
        tvecs,
        corners,
        ids,
    })
}

fn find_targets(aruco_result: &ArucoPoseResult) -> Result<Vec<VisionTarget>> {
    let mut targets = Vec::new();

    for idx in 0..aruco_result.ids.len() {
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

fn write_poses(image: &Image<MatImageData>, aruco_result: &ArucoPoseResult) -> Result<()> {
    let mut mat_image = image.as_mat_view();
    let mut out_img = mat_image.deref_mut();

    draw_detected_markers(
        &mut out_img,
        &aruco_result.corners,
        &aruco_result.ids,
        Scalar::new(255.0, 0.0, 0.0, 0.0),
    )?;

    for idx in 0..aruco_result.ids.len() {
        opencv::calib3d::draw_frame_axes(
            out_img,
            &aruco_result.intrinsic_matrix,
            &aruco_result.distortion_coeffs,
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

fn find_center(target: VisionTarget) -> VisionTarget {
    let hoop_rad: f64 = 0.61;
    let rad_theta: f64 = target.theta * PI / 180.;
    let rad_beta: f64 = target.beta * PI / 180.;

    let dx = target.dist * rad_theta.sin() + hoop_rad * rad_beta.sin();
    let dy = target.dist * rad_theta.cos() + hoop_rad * rad_beta.cos();
    
    let dist = (dx.powi(2) + dy.powi(2)).sqrt();
    let theta = atan2(dy, dx) * 180. / PI;

    let center = VisionTarget {
        id: target.id,
        theta: theta,
        beta: 0,
        dist: dist,
        height: target.height,
        confidence: 0.,
    };

    center
}

fn find_average(targets: Vec(VisionTarget)) -> VisionTarget {
    let mut sum_x: f64 = 0.;
    let mut sum_y: f64 = 0.; 
    for target in targets.iter() {
        let center = find_center(target);
        let rad_theta: f64 = center.theta * PI / 180.;

        let center_x = center.dist * rad_theta.sin();
        let center_y = center.dist * rad_theta.cos();

        sum_x += center_x;
        sum_y += center_y;
    }
    let dx = sum_x / targets.len();
    let dy = sum_y / targets.len();

    let dist = (dx.powi(2) + dy.powi(2)).sqrt();
    let theta = atan2(dy, dx) * 180. / PI;
    
    let average = VisionTarget {
        id: 0,
        theta: theta,
        beta: 0,
        dist: dist,
        height: targets[0].height,
        confidence: 0.,
    };

}

fn main() -> Result<()> {
    let config_file = File::open("config.json")?;
    let config = serde_json::from_reader(config_file)?;
    let mut camera = OpenCVCamera::new(config)?;

    loop {
        let image = camera
            .grab_frame()
            .context("Failed to read frame from camera")?;

        let aruco_result = find_pose(&image)?;
        let targets = find_targets(&aruco_result)?;
        write_poses(&image, &aruco_result)?;

        dbg!(targets);

        println!("-------------------------------");

        std::thread::sleep(std::time::Duration::from_millis(200));
    }

    Ok(())
}
