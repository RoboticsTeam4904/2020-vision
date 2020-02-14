use std::ops::Range;

use opencv::{
    core::Point,
    imgproc,
    prelude::*,
    types::{VectorOfPoint, VectorOfVectorOfPoint, VectorOfVec4i},
};

use stdvis_core::{
    traits::{ContourExtractor, ImageData},
    types::{Contour, ContourGroup, Image},
};

use stdvis_opencv::convert::AsMatView;

pub(crate) struct RFTapeContourExtractor {}

impl RFTapeContourExtractor {
    fn threshold_image(&self, image: &Mat) -> Mat {
        const RFTAPE_HSV_RANGE: Range<[u8; 3]> = [50, 103, 125]..[94, 255, 255];

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

    fn find_contours(&self, image: &Mat) -> Vec<VectorOfVectorOfPoint> {
        let mut contours = VectorOfVectorOfPoint::new();
        let mut hierarchy = VectorOfVec4i::new();
        imgproc::find_contours_with_hierarchy(
            &image,
            &mut contours,
            &mut hierarchy,
            imgproc::RETR_TREE,
            imgproc::CHAIN_APPROX_SIMPLE,
            opencv::core::Point::new(0, 0),
        )
        .unwrap();

        let mut groups: Vec<Vec<i32>> = Vec::new();

        for (contour_index, indices) in hierarchy.iter().enumerate() {
            let parent_index = indices.get(3).unwrap();
            let group_indices = match groups
                .iter_mut()
                .find(|indices| indices.contains(&parent_index))
            {
                Some(indices) => indices,
                None => {
                    groups.push(Vec::new());
                    groups.last_mut().unwrap()
                }
            };

            group_indices.push(contour_index as i32);
        }

        groups
            .iter()
            .map(|indices| {
                VectorOfVectorOfPoint::from_iter(
                    indices
                        .iter()
                        .map(|index| contours.get(index.clone() as usize).unwrap()),
                )
            })
            .collect()
    }

    fn filter_high_port(&self, contours: &VectorOfVectorOfPoint) -> bool {
        const SOLIDITY_RANGE: Range<f64> = 0.15..0.25;

        if contours.len() != 1 {
            return false;
        }

        let external_contour = contours.get(0).unwrap();

        let mut hull_contour = VectorOfPoint::new();
        imgproc::convex_hull(&external_contour, &mut hull_contour, true, false).unwrap();

        let contour_area = imgproc::contour_area(&external_contour, false).unwrap();
        let hull_area = imgproc::contour_area(&hull_contour, false).unwrap();

        let solidity = contour_area / hull_area;

        if !SOLIDITY_RANGE.contains(&solidity) {
            return false;
        }

        true
    }

    fn filter_loading_port(&self, contours: &VectorOfVectorOfPoint) -> bool {
        const SOLIDITY_RANGE: Range<f64> = 0.25..0.50;
        const ASPECT_RATIO_RANGE: Range<f32> = 1.40..1.60;

        if contours.len() != 2 {
            return false;
        }

        let external_contour = contours.get(0).unwrap();
        let internal_contour = contours.get(1).unwrap();

        let external_area = imgproc::contour_area(&external_contour, false).unwrap();
        let internal_area = imgproc::contour_area(&internal_contour, false).unwrap();

        let solidity = internal_area / external_area;

        if !SOLIDITY_RANGE.contains(&solidity) {
            return false;
        }

        let rect = imgproc::min_area_rect(&external_contour).unwrap();
        let rect_size = rect.size().unwrap();
        let aspect_ratio = rect_size.width / rect_size.height;

        if !ASPECT_RATIO_RANGE.contains(&aspect_ratio) {
            return false;
        }

        true
    }

    fn find_vertices(&self, target_num_vertices: usize, contour: &VectorOfPoint) -> VectorOfPoint {
        let mut poly_contour = VectorOfPoint::new();
        let mut epsilon = 0.0;

        loop {
            epsilon += 0.01;
            imgproc::approx_poly_dp(&contour, &mut poly_contour, epsilon, true).unwrap();

            if poly_contour.len() <= target_num_vertices {
                break poly_contour;
            }
        }
    }

    fn order_vertices(&self, contour: &VectorOfPoint) -> Contour {
        let num_points = contour.len();
        let mut centroid = contour
            .iter()
            .fold((0, 0), |acc, p| (acc.0 + p.x, acc.1 + p.y));

        centroid = (
            centroid.0 / num_points as i32,
            centroid.1 / num_points as i32,
        );

        let vertex_theta = |point: &Point| -> f32 {
            let theta = (-(point.y - centroid.1) as f32).atan2((point.x - centroid.0) as f32);

            if theta.is_sign_positive() {
                theta
            } else {
                2. * std::f32::consts::PI + theta
            }
        };

        let (first_idx, _) = contour
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| vertex_theta(a).partial_cmp(&vertex_theta(b)).unwrap())
            .unwrap();

        let signed_area = (1..=num_points).fold(0, |acc, idx| {
            let p0 = contour.get(idx - 1).unwrap();
            let p1 = contour.get(idx % num_points).unwrap();

            acc + (p1.x - p0.x) * (p1.y + p0.y)
        });

        let is_clockwise = signed_area.is_negative();

        let mut points = contour
            .iter()
            .map(|point| (point.x as u32, point.y as u32))
            .collect::<Vec<_>>();
        
        if is_clockwise {
            points.rotate_right(num_points - first_idx - 1);
            points.reverse();
        } else {
            points.rotate_left(first_idx);
        }

        Contour { points }
    }

    fn extract_matching_contours<'a, F>(
        &'a self,
        contour_groups: &'a Vec<VectorOfVectorOfPoint>,
        id: u8,
        target_num_vertices: usize,
        filter_predicate: F,
    ) -> impl Iterator<Item = ContourGroup> + 'a
    where
        F: FnMut(&&VectorOfVectorOfPoint) -> bool + 'a,
    {
        contour_groups
            .iter()
            .filter(filter_predicate)
            .map(move |contours| ContourGroup {
                id,
                contours: contours
                    .iter()
                    .map(|contour| {
                        self.order_vertices(&self.find_vertices(target_num_vertices, &contour))
                    })
                    .collect(),
            })
    }
}

impl ContourExtractor for RFTapeContourExtractor {
    fn extract_from<I: ImageData>(&self, image: &Image<I>) -> Vec<ContourGroup> {
        let image_mat = image.as_mat_view();

        let thresholded_image = self.threshold_image(&image_mat);
        let contour_groups = self.find_contours(&thresholded_image);

        let high_port_contours = self.extract_matching_contours(&contour_groups, 0, 8, |contours| {
            self.filter_high_port(contours)
        });

        let loading_port_contours =
            self.extract_matching_contours(&contour_groups, 1, 4, |contours| {
                self.filter_loading_port(contours)
            });

        high_port_contours.chain(loading_port_contours).collect()
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
