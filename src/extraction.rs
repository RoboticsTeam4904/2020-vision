use std::ops::Range;

use opencv::{
    core::{Point, Point2f, Size, TermCriteria},
    imgproc,
    prelude::*,
    types::{VectorOfPoint, VectorOfPoint2f, VectorOfVec4i, VectorOfVectorOfPoint},
};

#[cfg(feature = "opencl")]
use opencv::core::UMat;

use stdvis_core::{
    traits::{ContourExtractor, ImageData},
    types::{CameraConfig, Contour, ContourGroup, Image},
};

use stdvis_opencv::convert::AsMatView;

#[cfg(feature = "opencl")]
type SomeMat = UMat;

#[cfg(not(any(feature = "opencl")))]
type SomeMat = Mat;

#[cfg(feature = "opencl")]
fn default_mat() -> opencv::Result<UMat> {
    UMat::new(opencv::core::UMatUsageFlags::USAGE_DEFAULT)
}

#[cfg(not(any(feature = "opencl")))]
fn default_mat() -> opencv::Result<Mat> {
    Mat::default()
}

trait RFTapeTarget {
    const TYPE: u8;
    const NUM_VERTICES: usize;

    fn calc_solidity(contour: &VectorOfPoint, hull: Option<&VectorOfPoint>) -> f64 {
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

    fn calc_aspect_ratio(contour: &VectorOfPoint) -> f32 {
        let rect = imgproc::min_area_rect(&contour).unwrap();
        let rect_size = rect.size().unwrap();

        rect_size.width / rect_size.height
    }

    fn filter(contours: &VectorOfVectorOfPoint) -> bool;
}

struct HighPortTarget;

impl HighPortTarget {
    const SOLIDITY_RANGE: Range<f64> = 0.10..0.25;
    const ASPECT_RATIO_RANGE: Range<f32> = 0.00..2.50;
}

impl RFTapeTarget for HighPortTarget {
    const TYPE: u8 = 0;
    const NUM_VERTICES: usize = 8;

    fn filter(contours: &VectorOfVectorOfPoint) -> bool {
        if contours.len() != 1 {
            return false;
        }

        let external_contour = contours.get(0).unwrap();

        if !Self::SOLIDITY_RANGE.contains(&Self::calc_solidity(&external_contour, None)) {
            return false;
        }

        if !Self::ASPECT_RATIO_RANGE.contains(&Self::calc_aspect_ratio(&external_contour)) {
            return false;
        }

        true
    }
}

struct LoadingPortTarget;

impl LoadingPortTarget {
    const SOLIDITY_RANGE: Range<f64> = 0.25..0.50;
    const ASPECT_RATIO_RANGE: Range<f32> = 1.40..1.60;
}

impl RFTapeTarget for LoadingPortTarget {
    const TYPE: u8 = 1;
    const NUM_VERTICES: usize = 4;

    fn filter(contours: &VectorOfVectorOfPoint) -> bool {
        if contours.len() != 2 {
            return false;
        }

        let external_contour = contours.get(0).unwrap();
        let internal_contour = contours.get(1).unwrap();

        if !Self::SOLIDITY_RANGE.contains(&Self::calc_solidity(
            &internal_contour,
            Some(&external_contour),
        )) {
            return false;
        }

        if !Self::ASPECT_RATIO_RANGE.contains(&Self::calc_aspect_ratio(&external_contour)) {
            return false;
        }

        true
    }
}

pub struct RFTapeContourExtractor {
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
    pub fn threshold_image(&self, image: &SomeMat) -> SomeMat {
        const RFTAPE_HSV_RANGE: Range<[u8; 3]> = [50, 50, 50]..[94, 255, 255];

        let mut hsv_image = default_mat().unwrap();
        imgproc::cvt_color(&image, &mut hsv_image, imgproc::COLOR_BGR2HSV, 0).unwrap();

        let mut thresholded_image = default_mat().unwrap();
        opencv::core::in_range(
            &hsv_image,
            &Mat::from_slice(&RFTAPE_HSV_RANGE.start).unwrap(),
            &Mat::from_slice(&RFTAPE_HSV_RANGE.end).unwrap(),
            &mut thresholded_image,
        )
        .unwrap();

        let mut dilated_image = default_mat().unwrap();
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

        let mut eroded_image = default_mat().unwrap();
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

    fn grayscale_image(&self, image: &SomeMat) -> SomeMat {
        let mut grayscale_image = default_mat().unwrap();
        imgproc::cvt_color(&image, &mut grayscale_image, imgproc::COLOR_BGR2GRAY, 0).unwrap();

        grayscale_image
    }

    fn find_contours(&self, image: &SomeMat) -> Vec<VectorOfVectorOfPoint> {
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

    fn find_vertices(
        &self,
        target_num_vertices: usize,
        contour: &VectorOfPoint,
    ) -> Option<VectorOfPoint> {
        let mut poly_contour = VectorOfPoint::new();
        let mut epsilon = 0.0;

        loop {
            epsilon += 0.1;
            imgproc::approx_poly_dp(&contour, &mut poly_contour, epsilon, true).unwrap();

            if poly_contour.len() == target_num_vertices {
                break Some(poly_contour);
            } else if poly_contour.len() < target_num_vertices {
                // Failed to coerce the contour to the target number of vertices.
                break None;
            }
        }
    }

    fn order_vertices(&self, contour: &VectorOfPoint) -> VectorOfPoint {
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
            .map(|(idx, point)| (idx, vertex_theta(&point)))
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let signed_area = imgproc::contour_area(contour, true).unwrap();
        let is_clockwise = signed_area.is_sign_positive();

        let mut points = contour.to_vec();

        if is_clockwise {
            points.rotate_right(num_points - first_idx - 1);
            points.reverse();
        } else {
            points.rotate_left(first_idx);
        }

        VectorOfPoint::from_iter(points)
    }

    fn refine_vertices(
        &self,
        contour: &VectorOfPoint,
        grayscale_image_mat: &SomeMat,
    ) -> VectorOfPoint2f {
        let mut corners = VectorOfPoint2f::from_iter(
            contour
                .iter()
                .map(|point| Point2f::new(point.x as f32, point.y as f32)),
        );

        imgproc::corner_sub_pix(
            &grayscale_image_mat,
            &mut corners,
            Size::new(5, 5),
            Size::new(-1, -1),
            &self.sub_pix_term_criteria,
        )
        .unwrap();

        corners
    }

    fn convert_contour(&self, contour: &VectorOfPoint2f) -> Contour {
        let points = contour.iter().map(|point| (point.x, point.y)).collect();
        Contour { points }
    }

    fn normalize_contours(
        &self,
        contours: &VectorOfVectorOfPoint,
        grayscale_image_mat: &SomeMat,
        target_num_vertices: usize,
    ) -> Option<Vec<Contour>> {
        let normalized_contours = contours
            .iter()
            .filter_map(|contour| {
                self.find_vertices(target_num_vertices, &contour)
                    .map(|verts| self.order_vertices(&verts))
                    .map(|verts| self.refine_vertices(&verts, grayscale_image_mat))
                    .map(|verts| self.convert_contour(&verts))
            })
            .collect::<Vec<_>>();

        match normalized_contours.len() {
            0 => None,
            _ => Some(normalized_contours),
        }
    }

    fn extract_matching_contours<'src, T: RFTapeTarget>(
        &'src self,
        contour_groups: &Vec<VectorOfVectorOfPoint>,
        grayscale_image_mat: &SomeMat,
        camera: &'src CameraConfig,
    ) -> Vec<ContourGroup<'src>> {
        contour_groups
            .iter()
            .filter(|group| T::filter(group))
            .filter_map(|contours| {
                self.normalize_contours(contours, grayscale_image_mat, T::NUM_VERTICES)
            })
            .map(move |contours| ContourGroup {
                id: T::TYPE,
                camera,
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
        #[cfg(feature = "opencl")]
        let image_mat = image
            .as_mat_view()
            .get_umat(
                opencv::core::AccessFlag::ACCESS_READ,
                opencv::core::UMatUsageFlags::USAGE_DEFAULT,
            )
            .unwrap();

        #[cfg(not(any(feature = "opencl")))]
        let image_mat = image.as_mat_view();

        let thresholded_image = self.threshold_image(&image_mat);
        let contour_groups = self.find_contours(&thresholded_image);

        let grayscale_image_mat = self.grayscale_image(&image_mat);

        let high_port_contours = self.extract_matching_contours::<HighPortTarget>(
            &contour_groups,
            &grayscale_image_mat,
            image.camera,
        );

        let loading_port_contours = self.extract_matching_contours::<LoadingPortTarget>(
            &contour_groups,
            &grayscale_image_mat,
            image.camera,
        );

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
