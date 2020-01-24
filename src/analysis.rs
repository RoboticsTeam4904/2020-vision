use crate::extraction::HighPortContourExtractor;

use opencv::{
    calib3d::{solve_pnp},
    core::{Point, Mat},
    prelude::Vector,
    types::{VectorOfPoint, VectorOfVectorOfPoint, VectorOfPoint3f},
};
use standard_vision::{
    traits::{ContourAnalyzer, ContourExtractor, ImageData},
    types::{CameraConfig, Contour, Image, Target},
};

pub struct WallTapeContourAnalyzer {}

impl WallTapeContourAnalyzer {
    pub fn find_corners(&self) {
        
    }

    pub fn find_homography<I: ImageData>(&self, image: &Image<I>, obj_points: &VectorOfPoint, camera_mat: &Mat, dist_coef: &Mat) {
        let extractor = HighPortContourExtractor {};
        let contours = extractor.extract_from(image);
        let epic_contours = VectorOfVectorOfPoint::from_iter(contours.iter().map(|contour| {
            VectorOfPoint::from_iter(
                contour
                    .points
                    .iter()
                    .map(|point| Point::new(point.0 as i32, point.1 as i32)),
            )
        }));

        let corners = self.find_corners(); // Aaron moment

        let mut rotation_vec = VectorOfPoint3f::new();
        let mut translation_vec = VectorOfPoint3f::new();

        solve_pnp(
            &obj_points,
            &corners,
            &camera_mat,
            &dist_coef,
            &mut rotation_vec,
            &mut translation_vec,
            false,
            opencv::calib3d::SOLVEPNP_IPPE,
        );
    }

    // fn make_bounding_rect() -> Contour {

    // }

    // fn get_distance(config: &CameraConfig, actual_height: &f64, contour: &Contour) -> f64 {
    //     return (config.focal_length * actual_height * image.pixels * config.dimensions[0]) / (OBJECT_HEIGHT * config.sensor_height) // figure out object height
    // }

    // fn get_theta(config: &CameraConfig, contour: &Contour, distance: &f64,) -> f64 {

    // }

    // fn get_beta() -> f64 {

    // }
}

// impl ContourAnalyzer for WallTapeContourAnalyzer {
//     fn analyze(&self, config: &CameraConfig, contours: &Vec<Contour>) -> Vec<Target> {
//         let distance = Self::get_distance;
//         let theta = Self::get_theta;
//         let beta = Self::get_beta;
//     }
// }
