use opencv::{
    calib3d::{
        draw_chessboard_corners, find_chessboard_corners, rodrigues, solve_pnp, CALIB_CB_FAST_CHECK,
    },
    core::{Mat, Size, Image2D},
    imgcodecs::imwrite,
    prelude::*,
    types::{VectorOfPoint2f, VectorOfPoint3f},
};

use standard_vision::{
    traits::ImageData,
    types::{CameraConfig, Image},
};

use opencv_camera::image::AsMat;

pub fn find_chessboard<I: ImageData>(
    image: &Image<I>,
    obj_points: &VectorOfPoint3f,
    config: &CameraConfig,
) {
    let image_mat: Mat = image.as_mat();
    let board_size = Size::new(7, 7);
    let mut corners = VectorOfPoint2f::new();

    find_chessboard_corners(&image_mat, board_size, &mut corners, CALIB_CB_FAST_CHECK);

    let mut rvec = Mat::default().unwrap();
    let mut tvec = Mat::default().unwrap();
    let dist_coeffs = Mat::default().unwrap();
    let camera_mat = Mat::from_slice_2d(&[
        [config.focal_length, 0.0, ((config.resolution.0) / 2) as f64],
        [0.0, config.focal_length, ((config.resolution.1) / 2) as f64],
        [0.0, 0.0, 1.0],
    ])
    .unwrap();

    solve_pnp(
        &obj_points,
        &corners,
        &camera_mat,
        &dist_coeffs,
        &mut rvec,
        &mut tvec,
        false,
        opencv::calib3d::SOLVEPNP_IPPE,
    );

    let mut rmat = Mat::default().unwrap();
    let mut jacobian = Mat::default().unwrap();

    rodrigues(&rvec, &mut rmat, &mut jacobian);
}
