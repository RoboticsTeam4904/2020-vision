use opencv::{
    calib3d::{
        draw_chessboard_corners, find_chessboard_corners, rodrigues, solve_pnp, CALIB_CB_FAST_CHECK,
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

use ndarray::prelude::*;

pub fn find_chessboard<I: ImageData>(image: &Image<I>, obj_points: &VectorOfPoint3f, config: &CameraConfig, board_size: Size) {
    let mut flag = VectorOfint::new();
    flag.push(IMWRITE_JPEG_QUALITY);
    flag.push(95);

    let mut image_mat: Mat = image.as_mat_view().clone().unwrap();
    let mut corners = VectorOfPoint2f::new();

    imwrite("b.jpg", &image_mat, &flag);

    let corners_found =
        find_chessboard_corners(&image_mat, board_size, &mut corners, CALIB_CB_FAST_CHECK).unwrap();

    if !corners_found {
        return;
    }
    
    draw_chessboard_corners(&mut image_mat, board_size, &corners, corners_found);
    imwrite("chess_corners.jpg", &image_mat, &flag);

    let mut rvec = Mat::default().unwrap();
    let mut tvec_mat = Mat::default().unwrap();
    let dist_coeffs = Mat::default().unwrap();

    let fx = config.resolution.0 as f64 / config.sensor_width;
    let fy = config.resolution.1 as f64 / config.sensor_height;
    let cx = config.resolution.0 as f64 / 2.;
    let cy = config.resolution.1 as f64 / 2.;
    let camera_mat =
        Mat::from_slice_2d(&[
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
            ]).unwrap();

    solve_pnp(
        &obj_points,
        &corners,
        &camera_mat,
        &dist_coeffs,
        &mut rvec,
        &mut tvec_mat,
        false,
        opencv::calib3d::SOLVEPNP_ITERATIVE,
    );

    let mut rmat_mat = Mat::default().unwrap();
    let mut jacobian_mat = Mat::default().unwrap();

    rodrigues(&rvec, &mut rmat_mat, &mut jacobian_mat);

    let tvec = tvec_mat.as_array_view::<f64>().into_shape((3, 1)).unwrap();
    let rmat = rmat_mat.as_array_view::<f64>().into_shape((3, 3,)).unwrap();
    let rmat_trans = rmat.t();
    
    let camera_pose = rmat_trans.dot(&tvec);

    let yaw = -rmat_trans[[1,0]].atan2(rmat_trans[[0,0]]);

    println!("---");
    println!("yaw: {}", yaw.to_degrees());
    println!("pitch: {}", rmat_trans[[2,0]].asin().to_degrees());
    println!("roll: {}", (-rmat_trans[[2,1]].atan2(rmat_trans[[2,2]])).to_degrees());

    // println!("distance: {}", camera_pose[[2, 0]]);
}
