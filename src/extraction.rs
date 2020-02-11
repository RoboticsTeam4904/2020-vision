use std::ops::Range;

use opencv::{
    imgproc,
    prelude::*,
    types::{VectorOfPoint, VectorOfVectorOfPoint},
};

use stdvis_core::{
    traits::{ContourExtractor, ImageData},
    types::{Contour, Image},
};

use stdvis_opencv::convert::AsMatView;

const RFTAPE_HSV_RANGE: Range<[u8; 3]> = [50, 103, 150]..[94, 255, 255];

pub(crate) struct HighPortContourExtractor {}

impl HighPortContourExtractor {
    fn threshold_image(&self, image: &Mat) -> Mat {
        let mut hsv_image = Mat::default().unwrap();
        imgproc::cvt_color(&image, &mut hsv_image, imgproc::COLOR_BGR2HSV, 0).unwrap();

        let mut thresholded_image = Mat::default().unwrap();
        opencv::core::in_range(
            &hsv_image,
            &Mat::from_slice(&RFTAPE_HSV_RANGE.start).unwrap(),
            &Mat::from_slice(&RFTAPE_HSV_RANGE.end).unwrap(),
            &mut thresholded_image,
        )
        .unwrap();

        thresholded_image
    }

    fn find_contours(&self, image: &Mat) -> VectorOfVectorOfPoint {
        let mut contours = VectorOfVectorOfPoint::new();
        imgproc::find_contours(
            &image,
            &mut contours,
            imgproc::RETR_EXTERNAL,
            imgproc::CHAIN_APPROX_SIMPLE,
            opencv::core::Point::new(0, 0),
        )
        .unwrap();

        contours
    }

    fn filter_contour(&self, contours: &VectorOfPoint) -> bool {
        todo!()
    }
}

impl ContourExtractor for HighPortContourExtractor {
    fn extract_from<I: ImageData>(&self, image: &Image<I>) -> Vec<Contour> {
        let image_mat = image.as_mat_view();

        let thresholded_image = self.threshold_image(&image_mat);
        let contours = self.find_contours(&thresholded_image);

        let filtered_contours = contours
            .iter()
            .filter(|contour| self.filter_contour(contour));

        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::{rc::Rc, time};

    use stdvis_core::types::CameraConfig;
    use stdvis_opencv::camera::MatImageData;

    #[test]
    fn test_contour_matching() {
        use opencv::imgcodecs;
        use std::fs;

        let images = fs::read_dir("tests/images")
            .unwrap()
            .filter(|e| e.as_ref().unwrap().file_type().unwrap().is_file());

        for image in images {
            let image_path = image.unwrap().path();
            let image_name = image_path.file_name().unwrap().to_str().unwrap();

            let image_mat =
                imgcodecs::imread(image_path.to_str().unwrap(), imgcodecs::IMREAD_COLOR).unwrap();

            let image = Image::new(
                time::SystemTime::now(),
                Rc::new(CameraConfig::default()),
                MatImageData::new(image_mat),
            );

            let extractor = HighPortContourExtractor {};
            let thresholded_image = extractor.threshold_image(&image.as_mat_view());
            let contours = extractor.find_contours(&thresholded_image);

            let mut out_image = image.as_mat_view().clone().unwrap();
            imgproc::draw_contours(
                &mut out_image,
                &contours,
                -1,
                opencv::core::Scalar::new(255., 255., 255., 255.),
                3,
                imgproc::LINE_AA,
                &Mat::default().unwrap(),
                std::i32::MAX,
                opencv::core::Point::new(0, 0),
            )
            .unwrap();

            println!("tests/contours/{}", image_name);

            imgcodecs::imwrite(
                format!("tests/contours/{}", image_name).as_str(),
                &out_image,
                &opencv::types::VectorOfint::with_capacity(0),
            )
            .unwrap();
        }
    }
}
