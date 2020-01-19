use std::ops::Range;

use opencv::{imgproc, prelude::*};

use standard_vision::{
    traits::{ContourExtractor, ImageData},
    types::{Contour, Image},
};

use opencv_camera::image::AsMat;

const RFTAPE_HSV_RANGE: Range<[u8; 3]> = [50, 103, 150]..[94, 255, 255];

pub(crate) struct HighPortContourExtractor {}

impl ContourExtractor for HighPortContourExtractor {
    fn extract_from<I: ImageData>(&self, image: &Image<I>) -> Vec<Contour> {
        let image_mat = image.as_mat();
        let mut hsv_image = Mat::default().unwrap();
        let mut thresholded_image = Mat::default().unwrap();
        let mut contours = Mat::default().unwrap();

        imgproc::cvt_color(&image_mat, &mut hsv_image, imgproc::COLOR_BGR2HSV, 0).unwrap();
        opencv::core::in_range(
            &hsv_image,
            &Mat::from_slice(&RFTAPE_HSV_RANGE.start).unwrap(),
            &Mat::from_slice(&RFTAPE_HSV_RANGE.end).unwrap(),
            &mut thresholded_image,
        )
        .unwrap();

        imgproc::find_contours(
            &mut thresholded_image,
            &mut contours,
            imgproc::RETR_EXTERNAL,
            imgproc::CHAIN_APPROX_SIMPLE,
            opencv::core::Point::new(0, 0),
        )
        .unwrap();

        todo!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contour_matching() {
        use opencv::imgcodecs;

        let image = imgcodecs::imread(
            "tests/images/BlueGoal-060in-Center.jpg",
            imgcodecs::IMREAD_COLOR,
        )
        .unwrap();

        let mut hsv_image = Mat::default().unwrap();
        let mut thresholded_image = Mat::default().unwrap();
        let mut contours = Mat::default().unwrap();

        imgproc::cvt_color(&image, &mut hsv_image, imgproc::COLOR_BGR2HSV, 0).unwrap();
        opencv::core::in_range(
            &hsv_image,
            &Mat::from_slice(&RFTAPE_HSV_RANGE.start).unwrap(),
            &Mat::from_slice(&RFTAPE_HSV_RANGE.end).unwrap(),
            &mut thresholded_image,
        )
        .unwrap();

        imgproc::find_contours(
            &mut thresholded_image,
            &mut contours,
            imgproc::RETR_EXTERNAL,
            imgproc::CHAIN_APPROX_SIMPLE,
            opencv::core::Point::new(0, 0),
        )
        .unwrap();
    }
}
