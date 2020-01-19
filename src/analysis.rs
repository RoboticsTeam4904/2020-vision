use standard_vision::traits::ContourAnalyzer;
use standard_vision::types::{ CameraConfig, Contour, Pose, Target };

pub struct WallTapeContourAnalyzer {}

impl WallTapeContourAnalyzer {
    fn make_bounding_rect() -> Contour {

    }

    fn get_distance(config: &CameraConfig, actual_height: &f64, contour: &Contour) -> f64 {
        return (config.focal_length * actual_height * image.pixels * config.dimensions[0]) / (OBJECT_HEIGHT * config.sensor_height) // figure out object height
    }

    fn get_theta(config: &CameraConfig, contour: &Contour, distance: &f64,) -> f64 {
        
    }

    fn get_beta() -> f64 {

    }
}

impl ContourAnalyzer for WallTapeContourAnalyzer {
    fn analyze(&self, contours: Vec<&Contour>) -> Vec<Target> {
        let distance = Self::get_distance;
        let theta = Self::get_theta;
        let beta = Self::get_beta;

        Target {
            theta: theta,
            beta: beta,
            dist: distance
        }
    }
}