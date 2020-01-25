// mod control_panel;
// mod extraction;
// mod analysis;
mod chessboard;

use control_panel::ControlPanelTracker;
use standard_vision::types::{CameraConfig, Pose};
use chessboard::find_chessboard;


fn main() {
    let b = ControlPanelTracker {};
    let config = CameraConfig { // config for c920
        id: 0,
        resolution: (1080, 720),
        pose: Pose {
            x: 0,
            y: 0,
            z: 0,
            angle: 0.0,
        },
        fov: (70.42, 43.3),
        focal_length: 3.67,
        sensor_height: 3.6,
    };

    let mut camera = OpenCVCamera::new_from_index(0, config.pose, config.fov, config.focal_length, config.sensor_height).unwrap();
    let image = camera.grab_frame().unwrap();
    let image_mat: Mat = image.as_mat();

}
