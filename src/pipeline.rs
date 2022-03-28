use anyhow::{Context, Result};
use std::time::{Duration, Instant};
use stdvis_core::{
    traits::{Camera as CameraTrait, ContourAnalyzer, ContourExtractor, ImageData},
    types::VisionTarget,
};
use stdvis_opencv::convert::AsMatView;

// use crate::filter::TargetFilterMap;

pub struct VisionPipeline<Camera: CameraTrait, Extractor, Analyzer> {
    camera: Camera,
    extractor: Extractor,
    analyzer: Analyzer,
    // target_filters: TargetFilterMap,
}

impl<I, Camera, Extractor, Analyzer> VisionPipeline<Camera, Extractor, Analyzer>
where
    I: ImageData,
    Camera: CameraTrait<ImageStorage = I>,
    Extractor: ContourExtractor,
    Analyzer: ContourAnalyzer,
{
    pub fn new(camera: Camera, extractor: Extractor, analyzer: Analyzer) -> Self {
        // let target_filters = TargetFilterMap::new();

        VisionPipeline {
            camera,
            extractor,
            analyzer,
            // target_filters,
        }
    }

    pub fn run(&mut self) -> Result<Vec<VisionTarget>> {
        let frame = self
            .camera
            .grab_frame()
            .context("Failed to read frame from camera")?;

        let frame_mat = frame.as_mat_view();

        let contour_groups = self
            .extractor
            .extract_from(&frame)
            .context("Contour extraction failed")?;

        let mut result_targets = Vec::new();

        for group in contour_groups {
            let target = self
                .analyzer
                .analyze(&group)
                .context("Contour analysis failed")?;

            // TODO: Evaluate later, but it is unclear if using Kalman filtering
            // at this stage is beneficial.

            // if let Some(filtered_target) =
            //     self.target_filters.filter_target(target, frame.timestamp)
            // {
            //     result_targets.push(filtered_target);
            // }

            result_targets.push(target);
        }

        Ok(result_targets)
    }
}
