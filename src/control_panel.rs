use opencv::core::{in_range, Mat};
// use opencv::prelude::*;
use opencv_camera::{OpenCVCamera, image::AsMat};
use standard_vision::traits::{Camera};
use standard_vision::types::{CameraConfig};
use std::ops::Range;

pub struct ControlPanelTracker {}

impl ControlPanelTracker {
    pub fn detect_color(&self, config: CameraConfig) {
        let mut camera = OpenCVCamera::new_from_index(0, config.pose, config.fov, config.focal_length, config.sensor_height).unwrap();
        let image = camera.grab_frame().unwrap();
        let image_mat: Mat = image.as_mat();

        // in BGR
        // blue = [255, 255, 0]
        // green = [0, 255, 0]
        // red = [0, 0, 255]
        // yellow = [0, 255, 255]

        const BLUE_HUE_RANGE: Range<[u8; 3]> = [155, 155, 0]..[255, 255, 100];
        const GREEN_HUE_RANGE: Range<[u8; 3]> = [0, 155, 0]..[100, 255, 100];
        const RED_HUE_RANGE: Range<[u8; 3]> = [0, 0, 155]..[100, 100, 255];
        const YELLOW_HUE_RANGE: Range<[u8; 3]> = [0, 155, 155]..[100, 255, 255];

        let mut blue_mat = Mat::default().unwrap();
        in_range(&image_mat, &Mat::from_slice(&BLUE_HUE_RANGE.start).unwrap(), &Mat::from_slice(&BLUE_HUE_RANGE.end).unwrap(), &mut blue_mat).unwrap();

        let mut green_mat = Mat::default().unwrap();
        in_range(&image_mat, &Mat::from_slice(&GREEN_HUE_RANGE.start).unwrap(), &Mat::from_slice(&GREEN_HUE_RANGE.end).unwrap(), &mut green_mat).unwrap();

        let mut red_mat = Mat::default().unwrap();
        in_range(&image_mat, &Mat::from_slice(&RED_HUE_RANGE.start).unwrap(), &Mat::from_slice(&RED_HUE_RANGE.end).unwrap(), &mut red_mat).unwrap();

        let mut yellow_mat = Mat::default().unwrap();
        in_range(&image_mat, &Mat::from_slice(&YELLOW_HUE_RANGE.start).unwrap(), &Mat::from_slice(&YELLOW_HUE_RANGE.end).unwrap(), &mut yellow_mat).unwrap();
        
        let mut out = "";
        let mut big = 0;
        for (mat, color) in vec![(blue_mat.rows().unwrap(), "blue"), (green_mat.rows().unwrap(), "green"), (red_mat.rows().unwrap(), "red"), (yellow_mat.rows().unwrap(), "yellow")] {
            if mat > big {
                big = mat;
                out = color;
            }

        println!("{}", out)
        
        }
    }
}