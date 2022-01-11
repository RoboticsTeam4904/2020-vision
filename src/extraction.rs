use std::ops::{Range, RangeFrom};

use anyhow::Result;

use opencv::{
    core::{Point, Point2f, Size, TermCriteria, TermCriteria_Type},
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
type MatImpl = UMat;

#[cfg(not(any(feature = "opencl")))]
type MatImpl = Mat;

#[cfg(feature = "opencl")]
fn default_mat() -> MatImpl {
    UMat::new(opencv::core::UMatUsageFlags::USAGE_DEFAULT)
}

#[cfg(not(any(feature = "opencl")))]
fn default_mat() -> MatImpl {
    Mat::default()
}

trait RFTapeTarget {
    const TYPE: u8;
    const NUM_VERTICES: usize;

    /// Calculate the ratio between the area of `contour` and `hull`.
    /// If `hull` is not specified, the convex hull of `contour` will be used.
    fn calc_solidity(contour: &VectorOfPoint, hull: Option<&VectorOfPoint>) -> Result<f64> {
        let contour_area = imgproc::contour_area(&contour, false)?;

        let hull_area = match hull {
            None => {
                let mut hull_contour = VectorOfPoint::new();
                imgproc::convex_hull(&contour, &mut hull_contour, true, false)?;
                imgproc::contour_area(&hull_contour, false)?
            }
            Some(hull_contour) => imgproc::contour_area(&hull_contour, false)?,
        };

        Ok(contour_area / hull_area)
    }

    /// Calculate the aspect ratio (width:height) of `contour` by creating a rotated rectangular bounding box around it.
    fn calc_aspect_ratio(contour: &VectorOfPoint) -> Result<f32> {
        let rect = imgproc::min_area_rect(&contour)?;
        let rect_size = rect.size();

        Ok(rect_size.width / rect_size.height)
    }

    /// Filter a set of `contours`, returning `true` if they are of this type or `false` otherwise.
    fn filter(contours: &VectorOfVectorOfPoint) -> Result<bool>;
}

struct HighPortTarget;

impl HighPortTarget {
    const SOLIDITY_RANGE: Range<f64> = 0.10..0.25;
    const ASPECT_RATIO_RANGE: Range<f32> = 0.00..2.50;
    const AREA_RANGE: RangeFrom<f64> = 400.0..;
}

impl RFTapeTarget for HighPortTarget {
    const TYPE: u8 = 0;
    const NUM_VERTICES: usize = 8;

    fn filter(contours: &VectorOfVectorOfPoint) -> Result<bool> {
        if contours.len() != 1 {
            return Ok(false);
        }

        let external_contour = contours.get(0).unwrap();
        let external_area = imgproc::contour_area(&external_contour, false)?;

        if !Self::AREA_RANGE.contains(&external_area) {
            return Ok(false);
        }

        if !Self::SOLIDITY_RANGE.contains(&Self::calc_solidity(&external_contour, None)?) {
            return Ok(false);
        }

        if !Self::ASPECT_RATIO_RANGE.contains(&Self::calc_aspect_ratio(&external_contour)?) {
            return Ok(false);
        }

        Ok(true)
    }
}

struct LoadingPortTarget;

impl LoadingPortTarget {
    const SOLIDITY_RANGE: Range<f64> = 0.25..0.50;
    const ASPECT_RATIO_RANGE: Range<f32> = 1.40..1.60;
    const AREA_RANGE: RangeFrom<f64> = 1000.0..;
}

impl RFTapeTarget for LoadingPortTarget {
    const TYPE: u8 = 1;
    const NUM_VERTICES: usize = 4;

    fn filter(contours: &VectorOfVectorOfPoint) -> Result<bool> {
        if contours.len() != 2 {
            return Ok(false);
        }

        let external_contour = contours.get(0).unwrap();
        let internal_contour = contours.get(1).unwrap();

        let external_area = imgproc::contour_area(&external_contour, false)?;

        if !Self::AREA_RANGE.contains(&external_area) {
            return Ok(false);
        }

        if !Self::SOLIDITY_RANGE.contains(&Self::calc_solidity(
            &internal_contour,
            Some(&external_contour),
        )?) {
            return Ok(false);
        }

        if !Self::ASPECT_RATIO_RANGE.contains(&Self::calc_aspect_ratio(&external_contour)?) {
            return Ok(false);
        }

        Ok(true)
    }
}

pub struct RFTapeContourExtractor {
    /// A morphological element for dilating and eroding an image.
    morph_elem: Mat,
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

        Self { morph_elem }
    }

    /// Threshold (binarize) `image` such that only regions with the expected green tape color are white.
    pub fn threshold_image(&self, image: &MatImpl) -> Result<MatImpl> {
        const RFTAPE_HSV_RANGE: Range<[u8; 3]> = [50, 50, 50]..[94, 255, 255];

        let mut hsv_image = default_mat();
        imgproc::cvt_color(&image, &mut hsv_image, imgproc::COLOR_BGR2HSV, 0)?;

        let mut thresholded_image = default_mat();
        opencv::core::in_range(
            &hsv_image,
            &Mat::from_slice(&RFTAPE_HSV_RANGE.start).unwrap(),
            &Mat::from_slice(&RFTAPE_HSV_RANGE.end).unwrap(),
            &mut thresholded_image,
        )?;

        Ok(thresholded_image)
    }

    /// Denoise the image by dilating it and eroding it.
    /// This has the effect of removing noisy regions of the image,
    /// hopefully without compromising the integrity of the contours.
    pub fn denoise_image(&self, image: &MatImpl) -> Result<MatImpl> {
        let mut dilated_image = default_mat();
        imgproc::dilate(
            &image,
            &mut dilated_image,
            &self.morph_elem,
            Point::new(-1, -1),
            1,
            opencv::core::BORDER_CONSTANT,
            imgproc::morphology_default_border_value().unwrap(),
        )?;

        let mut eroded_image = default_mat();
        imgproc::erode(
            &dilated_image,
            &mut eroded_image,
            &self.morph_elem,
            Point::new(-1, -1),
            1,
            opencv::core::BORDER_CONSTANT,
            imgproc::morphology_default_border_value().unwrap(),
        )?;

        Ok(eroded_image)
    }

    fn grayscale_image(&self, image: &MatImpl) -> Result<MatImpl> {
        let mut grayscale_image = default_mat();
        imgproc::cvt_color(&image, &mut grayscale_image, imgproc::COLOR_BGR2GRAY, 0)?;

        Ok(grayscale_image)
    }

    fn find_contours(&self, image: &MatImpl) -> Result<Vec<VectorOfVectorOfPoint>> {
        let mut contours = VectorOfVectorOfPoint::new();
        let mut hierarchy = VectorOfVec4i::new();
        imgproc::find_contours_with_hierarchy(
            &image,
            &mut contours,
            &mut hierarchy,
            imgproc::RETR_TREE,
            imgproc::CHAIN_APPROX_SIMPLE,
            opencv::core::Point::new(0, 0),
        )?;

        // The contours returned from OpenCV are in a hierarchical structure.
        // In order to simplify this, the hierarchy is flattened such that each
        // contour is grouped with their highest-order ancestor.

        let mut groups: Vec<Vec<i32>> = Vec::new();

        for (contour_index, indices) in hierarchy.iter().enumerate() {
            // The hierarchy indices are (next, previous, child, parent).
            // This picks out the 3rd index: the parent index.
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

        Ok(groups
            .iter()
            .map(|indices| {
                VectorOfVectorOfPoint::from_iter(
                    indices
                        .iter()
                        .map(|index| contours.get(index.clone() as usize).unwrap()),
                )
            })
            .collect())
    }

    /// Coerce `contour` into being a polygon with `target_num_vertices` points.
    ///
    /// Returns `Ok(None)` if this coercion fails.
    fn find_vertices(
        &self,
        target_num_vertices: usize,
        contour: &VectorOfPoint,
    ) -> Result<Option<VectorOfPoint>> {
        let mut poly_contour = VectorOfPoint::new();
        let mut epsilon = 0.0;

        loop {
            epsilon += 0.1;
            imgproc::approx_poly_dp(&contour, &mut poly_contour, epsilon, true)?;

            if poly_contour.len() == target_num_vertices {
                break Ok(Some(poly_contour));
            } else if poly_contour.len() < target_num_vertices {
                // Failed to coerce the contour to the target number of vertices.
                break Ok(None);
            }
        }
    }

    /// Sort the vertices of `contour` so that the first vertex has the smallest
    /// theta on a unit circle and guarantee counter-clockwise winding.
    fn order_vertices(&self, contour: &VectorOfPoint) -> Result<VectorOfPoint> {
        let num_points = contour.len();
        let mut centroid = contour
            .iter()
            .fold((0, 0), |acc, p| (acc.0 + p.x, acc.1 + p.y));

        centroid = (
            centroid.0 / num_points as i32,
            centroid.1 / num_points as i32,
        );

        // Calculate the angle of `point` relative to `centroid` as if it was on
        // a unit circle.
        let vertex_theta = |point: &Point| -> f32 {
            let theta = (-(point.y - centroid.1) as f32).atan2((point.x - centroid.0) as f32);

            if theta.is_sign_positive() {
                theta
            } else {
                2. * std::f32::consts::PI + theta
            }
        };

        // Pick out the index of the point which should come first in the
        // contour.
        // This first point should be the one with the smallest angle, which is
        // calculated by `vertex_theta`.
        let (first_idx, _) = contour
            .iter()
            .enumerate()
            .map(|(idx, point)| (idx, vertex_theta(&point)))
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        // Contour winding can be determined by checking the sign of the contour's signed area.
        // See Green's Theorem for more information.
        let signed_area = imgproc::contour_area(contour, true)?;
        let is_clockwise = signed_area.is_sign_positive();

        let mut points = contour.to_vec();

        if is_clockwise {
            // Rotate the intended first point to the end and then reverse so
            // that the winding is counter-clockwise.
            points.rotate_right(num_points - first_idx - 1);
            points.reverse();
        } else {
            // Rotate the intended first point to the beginning.
            // The winding should already be counter-clockwise.
            points.rotate_left(first_idx);
        }

        Ok(VectorOfPoint::from_iter(points))
    }

    /// Refine the vertices of `contour` using a `grayscale_image_mat`.
    /// This is useful because thresholding the image erases any shading from
    /// the original image; by using the original image (in grayscale), the
    /// vertices can be refined to sub-pixel values based on the image's
    /// shading.
    fn refine_vertices(
        &self,
        contour: &VectorOfPoint,
        grayscale_image_mat: &MatImpl,
    ) -> Result<VectorOfPoint2f> {
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
            // This `TermCriteria` is the default value for the equivalent C++ function.
            // It is possible that these values could be optimized in the future.
            TermCriteria::new(
                TermCriteria_Type::EPS as i32 + TermCriteria_Type::COUNT as i32,
                40,
                0.01,
            )?,
        )?;

        Ok(corners)
    }

    fn convert_contour(&self, contour: &VectorOfPoint2f) -> Contour {
        let points = contour.iter().map(|point| (point.x, point.y)).collect();
        Contour { points }
    }

    /// Normalize a group of `contours` by coercing each contour into a polygon
    /// of `target_num_vertices` and then ordering and refining its vertices.
    ///
    /// Returns `Ok(None)` if coercion fails for any contour in `contours`.
    fn normalize_contours(
        &self,
        contours: &VectorOfVectorOfPoint,
        grayscale_image_mat: &MatImpl,
        target_num_vertices: usize,
    ) -> Result<Option<Vec<Contour>>> {
        let mut normalized_contours = Vec::with_capacity(contours.len());

        for contour in contours {
            if let Ok(Some(vertices)) = self.find_vertices(target_num_vertices, &contour) {
                let vertices = self.order_vertices(&vertices)?;
                let vertices = self.refine_vertices(&vertices, grayscale_image_mat)?;
                let contour = self.convert_contour(&vertices);

                normalized_contours.push(contour);
            } else {
                return Ok(None);
            }
        }

        Ok(Some(normalized_contours))
    }

    fn extract_matching_contours<'src, Target: RFTapeTarget>(
        &'src self,
        contour_groups: &Vec<VectorOfVectorOfPoint>,
        grayscale_image_mat: &MatImpl,
        camera: &'src CameraConfig,
    ) -> Result<Vec<ContourGroup<'src>>> {
        let mut result_groups = Vec::new();

        for group in contour_groups {
            if !Target::filter(group)? {
                continue;
            }

            if let Some(contours) =
                self.normalize_contours(group, grayscale_image_mat, Target::NUM_VERTICES)?
            {
                result_groups.push(ContourGroup {
                    id: Target::TYPE,
                    camera,
                    contours,
                })
            }
        }

        Ok(result_groups)
    }
}

impl ContourExtractor for RFTapeContourExtractor {
    fn extract_from<'src, I: ImageData>(
        &'src self,
        image: &Image<'src, I>,
    ) -> Result<Vec<ContourGroup<'src>>> {
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

        let thresholded_image = self.threshold_image(&image_mat)?;
        let denoised_image = self.denoise_image(&thresholded_image)?;
        let contour_groups = self.find_contours(&denoised_image)?;

        let grayscale_image_mat = self.grayscale_image(&image_mat)?;

        let high_port_contours = self
            .extract_matching_contours::<HighPortTarget>(
                &contour_groups,
                &grayscale_image_mat,
                image.camera,
            )
            .unwrap();

        let loading_port_contours = self
            .extract_matching_contours::<LoadingPortTarget>(
                &contour_groups,
                &grayscale_image_mat,
                image.camera,
            )
            .unwrap();

        let mut grouped_contours = Vec::new();
        grouped_contours.extend(high_port_contours);
        grouped_contours.extend(loading_port_contours);

        Ok(grouped_contours)
    }
}
