mod control_panel;
mod extraction;
mod analysis;

use control_panel::ControlPanelTracker;
use standard_vision::types::{CameraConfig, Pose};

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

    b.detect_color(config)
}
