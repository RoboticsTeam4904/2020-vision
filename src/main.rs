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
use stdvis_core::{
    traits::Camera,
    types::Pose,
};
use stdvis_opencv::camera::OpenCVCamera;

fn main() {
    let mut camera = OpenCVCamera::new(
        0,
        Pose {
            x: 0,
            y: 0,
            z: 0,
            angle: 0.0,
        },
        (70.42, 43.3),
        3.67,
        3.6,
        4.8,
    )
    .unwrap();

    const CHESS_WIDTH: f32 = 123. / 1000.;
    const CHESS_HEIGHT: f32 = 77. / 1000.;

    let mut obj_points = VectorOfPoint3f::new();
    let config = camera.config();
    let board_size = Size::new(9, 6);
    let distance = config.focal_length as f32 * CHESS_HEIGHT / config.sensor_height as f32;

    for i in 0..board_size.height {
        for j in 0..board_size.width {
            obj_points.push(Point3f::new(
                CHESS_WIDTH / (board_size.width - 1) as f32 * j as f32 - CHESS_WIDTH * 0.5,
                CHESS_HEIGHT / (board_size.height - 1) as f32 * i as f32 - CHESS_HEIGHT * 0.5,
                0.,
            ));
        }
    }

    loop {
        let image = camera.grab_frame().unwrap();
        find_chessboard(&image, &obj_points, &config, board_size);
    }
}
