use std::ops::Range;

use opencv::{
    core::{Point, Point2f, Size, TermCriteria},
    imgproc,
    prelude::*,
    types::{VectorOfPoint, VectorOfPoint2f, VectorOfVec4i, VectorOfVectorOfPoint},
};

use stdvis_core::{
    traits::{ContourExtractor, ImageData},
    types::{Contour, ContourGroup, Image},
};

use stdvis_opencv::convert::{AsMatView, MatView};

pub(crate) struct RFTapeContourExtractor {
    morph_elem: Mat,
    sub_pix_term_criteria: TermCriteria,
}

impl RFTapeContourExtractor {
    pub fn new() -> Self {
        const MORPH_KERNEL_SIZE: i32 = 1;
        const MORPH_SHAPE: i32 = imgproc::MORPH_RECT;

        let morph_elem = imgproc::get_structuring_element(
            MORPH_SHAPE,
            Size::new(MORPH_KERNEL_SIZE * 2 + 1, MORPH_KERNEL_SIZE * 2 + 1),
            Point::new(MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE),
        )
        .unwrap();

        let sub_pix_term_criteria = TermCriteria::new(
            opencv::core::TermCriteria_EPS + opencv::core::TermCriteria_COUNT,
            40,
            0.01,
        )
        .unwrap();

        Self {
            morph_elem,
            sub_pix_term_criteria,
        }
    }
}

impl RFTapeContourExtractor {
    fn threshold_image(&self, image: &Mat) -> Mat {
        const RFTAPE_HSV_RANGE: Range<[u8; 3]> = [50, 50, 125]..[94, 255, 255];

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

        let mut dilated_image = Mat::default().unwrap();
        imgproc::dilate(
            &thresholded_image,
            &mut dilated_image,
            &self.morph_elem,
            Point::new(-1, -1),
            1,
            opencv::core::BORDER_CONSTANT,
            imgproc::morphology_default_border_value().unwrap(),
        )
        .unwrap();

        let mut eroded_image = Mat::default().unwrap();
        imgproc::erode(
            &dilated_image,
            &mut eroded_image,
            &self.morph_elem,
            Point::new(-1, -1),
            1,
            opencv::core::BORDER_CONSTANT,
            imgproc::morphology_default_border_value().unwrap(),
        )
        .unwrap();

        eroded_image
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

    fn calc_solidity(&self, contour: &VectorOfPoint, hull: Option<&VectorOfPoint>) -> f64 {
        let contour_area = imgproc::contour_area(&contour, false).unwrap();

        let hull_area = match hull {
            None => {
                let mut hull_contour = VectorOfPoint::new();
                imgproc::convex_hull(&contour, &mut hull_contour, true, false).unwrap();
                imgproc::contour_area(&hull_contour, false).unwrap()
            }
            Some(hull_contour) => imgproc::contour_area(&hull_contour, false).unwrap(),
        };

        contour_area / hull_area
    }

    fn calc_aspect_ratio(&self, contour: &VectorOfPoint) -> f32 {
        let rect = imgproc::min_area_rect(&contour).unwrap();
        let rect_size = rect.size().unwrap();

        rect_size.width / rect_size.height
    }

    fn filter_high_port(&self, contours: &VectorOfVectorOfPoint) -> bool {
        const SOLIDITY_RANGE: Range<f64> = 0.10..0.25;
        const ASPECT_RATIO_RANGE: Range<f32> = 0.00..2.50;

        if contours.len() != 1 {
            return false;
        }

        let external_contour = contours.get(0).unwrap();

        if !SOLIDITY_RANGE.contains(&self.calc_solidity(&external_contour, None)) {
            return false;
        }

        if !ASPECT_RATIO_RANGE.contains(&self.calc_aspect_ratio(&external_contour)) {
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

        if !SOLIDITY_RANGE.contains(&self.calc_solidity(&internal_contour, Some(&external_contour)))
        {
            return false;
        }

        if !ASPECT_RATIO_RANGE.contains(&self.calc_aspect_ratio(&external_contour)) {
            return false;
        }

        true
    }

    fn find_vertices(
        &self,
        target_num_vertices: usize,
        contour: &VectorOfPoint,
    ) -> Option<VectorOfPoint> {
        let mut poly_contour = VectorOfPoint::new();
        let mut epsilon = 0.0;

        loop {
            epsilon += 0.01;
            imgproc::approx_poly_dp(&contour, &mut poly_contour, epsilon, true).unwrap();

            if poly_contour.len() == target_num_vertices {
                break Some(poly_contour);
            } else if poly_contour.len() < target_num_vertices {
                // Failed to coerce the contour to the target number of vertices.
                break None;
            }
        }
    }

    fn order_vertices(&self, contour: &VectorOfPoint) -> Vec<Point> {
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

        let mut points = contour.to_vec();

        if is_clockwise {
            points.rotate_right(num_points - first_idx - 1);
            points.reverse();
        } else {
            points.rotate_left(first_idx);
        }

        points
    }

    fn optimize_contour(&self, contour: &Vec<Point>, image_mat: &Mat) -> Contour {
        let mut corners = VectorOfPoint2f::from_iter(
            contour
                .iter()
                .map(|point| Point2f::new(point.x as f32, point.y as f32)),
        );

        let mut grayscale_image = Mat::default().unwrap();
        imgproc::cvt_color(&image_mat, &mut grayscale_image, imgproc::COLOR_BGR2GRAY, 0).unwrap();

        imgproc::corner_sub_pix(
            &grayscale_image,
            &mut corners,
            Size::new(5, 5),
            Size::new(-1, -1),
            &self.sub_pix_term_criteria,
        )
        .unwrap();

        let points = corners.iter().map(|corner| (corner.x, corner.y)).collect();

        Contour { points }
    }

    fn normalize_contours(
        &self,
        contours: &VectorOfVectorOfPoint,
        image_mat: MatView,
        target_num_vertices: usize,
    ) -> Option<Vec<Contour>> {
        let normalized_contours = contours
            .iter()
            .filter_map(|contour| {
                self.find_vertices(target_num_vertices, &contour)
                    .map(|verts| self.order_vertices(&verts))
                    .map(|contour| self.optimize_contour(&contour, &*image_mat))
            })
            .collect::<Vec<_>>();

        match normalized_contours.len() {
            0 => None,
            _ => Some(normalized_contours),
        }
    }

    fn extract_matching_contours<'src, F, I: ImageData>(
        &'src self,
        contour_groups: &Vec<VectorOfVectorOfPoint>,
        image: &Image<'src, I>,
        id: u8,
        target_num_vertices: usize,
        filter_predicate: F,
    ) -> Vec<ContourGroup<'src>>
    where
        F: FnMut(&&VectorOfVectorOfPoint) -> bool,
    {
        contour_groups
            .iter()
            .filter(filter_predicate)
            .filter_map(|contours| {
                self.normalize_contours(contours, image.as_mat_view(), target_num_vertices)
            })
            .map(move |contours| ContourGroup {
                id,
                camera: image.camera,
                contours,
            })
            .collect()
    }
}

impl ContourExtractor for RFTapeContourExtractor {
    fn extract_from<'src, I: ImageData>(
        &'src self,
        image: &Image<'src, I>,
    ) -> Vec<ContourGroup<'src>> {
        let image_mat = image.as_mat_view();

        let thresholded_image = self.threshold_image(&image_mat);
        let contour_groups = self.find_contours(&thresholded_image);

        let high_port_contours =
            self.extract_matching_contours(&contour_groups, image, 0, 8, |contours| {
                self.filter_high_port(contours)
            });

        let loading_port_contours =
            self.extract_matching_contours(&contour_groups, image, 1, 4, |contours| {
                self.filter_loading_port(contours)
            });

        let mut grouped_contours = Vec::new();
        grouped_contours.extend(high_port_contours);
        grouped_contours.extend(loading_port_contours);

        grouped_contours
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::time;

    use stdvis_core::types::CameraConfig;
    use stdvis_opencv::camera::MatImageData;

    #[test]
    fn test_contour_matching() {
        use ndarray::prelude::*;
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

            let config = CameraConfig::default();
            let image = Image::new(
                time::SystemTime::now(),
                &config,
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
