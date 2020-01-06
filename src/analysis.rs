use standard_vision::traits::ContourAnalyzer;
use standard_vision::types::{ Pose, CameraConfig, Target };

pub struct WallTapeContourAnalyzer {}

impl WallTapeContourAnalyzer {
    fn get_distance(config: &CameraConfig, ) -> f32 {
        
    }

    fn get_theta() -> f32 {

    }

    fn get_beta() -> f32 {

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