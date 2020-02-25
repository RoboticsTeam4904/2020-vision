use opencv::{
    calib3d::{self, draw_chessboard_corners, find_chessboard_corners_sb, rodrigues, solve_pnp},
    core::{Mat, Size},
    imgcodecs::{imwrite, IMWRITE_JPEG_QUALITY},
    prelude::*,
    types::{VectorOfPoint2f, VectorOfPoint3f, VectorOfint},
};

use stdvis_core::{
    traits::ImageData,
    types::Image,
};

use stdvis_opencv::convert::{AsArrayView, AsMatView};

// use ndarray::prelude::*;

pub fn find_chessboard<I: ImageData>(
    image: &Image<I>,
    obj_points: &VectorOfPoint3f,
    board_size: Size,
) {
    let mut flag = VectorOfint::new();
    flag.push(IMWRITE_JPEG_QUALITY);
    flag.push(95);

    let mut image_mat: Mat = image.as_mat_view().clone().unwrap();
    let mut corners = VectorOfPoint2f::new();

    // imwrite("b.jpg", &image_mat, &flag).unwrap();

    let corners_found =
        find_chessboard_corners_sb(&image_mat, board_size, &mut corners, calib3d::CALIB_CB_ACCURACY + calib3d::CALIB_CB_NORMALIZE_IMAGE).unwrap();

    if !corners_found {
        opencv::highgui::imshow("chess", &image_mat).unwrap();
        return;
    }

    // println!("{:?}", corners.to_vec());

    draw_chessboard_corners(&mut image_mat, board_size, &corners, corners_found).unwrap();

    let mut rvec = Mat::default().unwrap();
    let mut tvec_mat = Mat::default().unwrap();
    // let dist_coeffs = Mat::default().unwrap();
    let dist_coeffs = Mat::from_slice(&[
        0.16882939064608002,
        -0.9243844884626233,
        0.006368686931626428,
        0.0055936556910287875,
        2.0772772661977683,
    ])
    .unwrap();

    let camera_mat = Mat::from_slice_2d(&[
        [1449.6970632013845, 0.0, 919.7416002537354],
        [0.0, 1456.6938719499683, 549.6275213145884],
        [0.0, 0.0, 1.0],
    ])
    .unwrap();

    solve_pnp(
        &obj_points,
        &corners,
        &camera_mat,
        &dist_coeffs,
        &mut rvec,
        &mut tvec_mat,
        false,
        opencv::calib3d::SOLVEPNP_IPPE,
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

    let x = camera_pose[[0, 0]];
    let y = camera_pose[[1, 0]];
    let z = camera_pose[[2, 0]];

    let theta = x.atan2(z);

    let r00 = rmat[[0, 0]];
    let r10 = rmat[[1, 0]];
    let r20 = rmat[[2, 0]];
    let r21 = rmat[[2, 1]];
    let r22 = rmat[[2, 2]];

    let yaw = r10.atan2(r00); // yaw
    let pitch = -r20.atan2((r21.powf(2.) + r22.powf(2.)).sqrt());
    let roll = r21.atan2(r22); // roll

    println!("---");
    println!("theta {}", theta.to_degrees());
    println!("yaw: {}", yaw.to_degrees());
    println!("pitch: {}", pitch.to_degrees());
    println!("roll: {}", roll.to_degrees());
    println!("dist away: {}", z);
    println!("dist up: {}", y);

    // imwrite("chess_corners.jpg", &image_mat, &flag).unwrap();
    opencv::highgui::imshow("ichess", &image_mat).unwrap();
}