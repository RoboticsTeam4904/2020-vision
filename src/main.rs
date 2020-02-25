// mod control_panel;
// mod extraction;
// mod analysis;
mod chessboard;

use chessboard::find_chessboard;
use opencv::{
    core::{Point3f, Size},
    prelude::*,
    types::VectorOfPoint3f,
};
use ndarray::prelude::*;
use stdvis_core::{
    traits::Camera,
    types::{Pose, CameraConfig},
};
use stdvis_opencv::camera::OpenCVCamera;

fn main() {
    let mut camera = OpenCVCamera::new(
        CameraConfig {
            id: 0,
            resolution: (1920, 1080),
            pose: Pose {
                angle: 0.0,
                dist: 0.0,
                height: 0.0,
                yaw: 0.0,
                pitch: 0.0,
                roll: 0.0,  
            },
            fov: (70.42, 43.3),
            distortion_coeffs: array![
                0.16882939064608002,
                -0.9243844884626233,
                0.006368686931626428,
                0.0055936556910287875,
                2.0772772661977683,
            ],
            intrinsic_matrix: array![
                [1449.6970632013845, 0.0, 919.7416002537354],
                [0.0, 1456.6938719499683, 549.6275213145884],
                [0.0, 0.0, 1.0]
            ]
        }
    )
    .unwrap();

    const SQUARE_SIZE: f32 = 18.9 / 1000.;

    let mut obj_points = VectorOfPoint3f::new();
    let board_size = Size::new(8, 5);

    let chess_width = board_size.width as f32 * SQUARE_SIZE;
    let chess_height = board_size.height as f32 * SQUARE_SIZE;

    for i in 0..board_size.height {
        for j in 0..board_size.width {
            obj_points.push(Point3f::new(
                chess_width / (board_size.width - 1) as f32 * j as f32 - chess_width * 0.5,
                chess_height / (board_size.height - 1) as f32 * i as f32 - chess_height * 0.5,
                0.,
            ));
        }
    }

    opencv::highgui::named_window("chess", opencv::highgui::WINDOW_NORMAL).unwrap();

    loop {
        let image = camera.grab_frame().unwrap();
        find_chessboard(&image, &obj_points, board_size);

        opencv::highgui::wait_key(30).unwrap();
    }
}
