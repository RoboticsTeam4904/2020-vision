use opencv::{
    calib3d::{
        draw_chessboard_corners, find_chessboard_corners_sb, rodrigues, solve_pnp, CALIB_CB_FAST_CHECK,
    },
    core::{Mat, Size},
    imgcodecs::{imwrite, IMWRITE_JPEG_QUALITY},
    prelude::*,
    types::{VectorOfPoint2f, VectorOfPoint3f, VectorOfint},
};

use stdvis_core::{
    traits::ImageData,
    types::{CameraConfig, Image},
};

use stdvis_opencv::convert::{AsArrayView, AsMatView};

// use ndarray::prelude::*;

pub fn find_chessboard<I: ImageData>(
    image: &Image<I>,
    obj_points: &VectorOfPoint3f,
    config: &CameraConfig,
    board_size: Size,
) {
    let mut flag = VectorOfint::new();
    flag.push(IMWRITE_JPEG_QUALITY);
    flag.push(95);

    let mut image_mat: Mat = image.as_mat_view().clone().unwrap();
    let mut corners = VectorOfPoint2f::new();

    imwrite("b.jpg", &image_mat, &flag).unwrap();

    let corners_found =
        find_chessboard_corners_sb(&image_mat, board_size, &mut corners, 0).unwrap();

    if !corners_found {
        return;
    }

    draw_chessboard_corners(&mut image_mat, board_size, &corners, corners_found).unwrap();

    let mut rvec = Mat::default().unwrap();
    let mut tvec_mat = Mat::default().unwrap();
    let dist_coeffs = Mat::default().unwrap();

    let fx = config.focal_length * config.resolution.0 as f64 / config.sensor_width;
    let fy = config.focal_length * config.resolution.1 as f64 / config.sensor_height;

    // let fx = config.resolution.0 as f64 / config.sensor_width;
    // let fy = config.resolution.1 as f64 / config.sensor_height;
    let cx = config.resolution.0 as f64 / 2.;
    let cy = config.resolution.1 as f64 / 2.;
    let camera_mat = Mat::from_slice_2d(&[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]).unwrap();

    solve_pnp(
        &obj_points,
        &corners,
        &camera_mat,
        &dist_coeffs,
        &mut rvec,
        &mut tvec_mat,
        false,
        opencv::calib3d::SOLVEPNP_EPNP,
    )
    .unwrap();

    opencv::calib3d::draw_frame_axes(
        &mut image_mat,
        &camera_mat,
        &dist_coeffs,
        &rvec,
        &tvec_mat,
        0.02,
        10,
    )
    .unwrap();

    let mut rmat_mat = Mat::default().unwrap();
    let mut jacobian_mat = Mat::default().unwrap();

    rodrigues(&rvec, &mut rmat_mat, &mut jacobian_mat).unwrap();

    let tvec = tvec_mat.as_array_view::<f64>().into_shape((3, 1)).unwrap();
    let rmat = rmat_mat.as_array_view::<f64>().into_shape((3, 3)).unwrap();
    let rmat_trans = rmat.t();

    let camera_pose = rmat_trans.dot(&tvec);

    let r00 = rmat[[0, 0]];
    let r10 = rmat[[1, 0]];
    let r20 = rmat[[2, 0]];
    let r21 = rmat[[2, 1]];
    let r22 = rmat[[2, 2]];

    let roll = r10.atan2(r00); // actually roll
    let pitch = -r20.atan2((r21.powf(2.) + r22.powf(2.)).sqrt());
    let yaw = r21.atan2(r22); // actually yaw

    println!("---");
    println!("yaw: {}", yaw.to_degrees());
    println!("pitch: {}", pitch.to_degrees());
    println!("roll: {}", roll.to_degrees());

    imwrite("chess_corners.jpg", &image_mat, &flag).unwrap();
}
