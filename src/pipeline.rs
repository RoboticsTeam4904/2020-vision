use anyhow::{Context, Result};
use stdvis_core::{
    traits::{Camera, ContourAnalyzer, ContourExtractor},
    types::{CameraConfig, VisionTarget},
};
use stdvis_opencv::{camera::OpenCVCamera, convert::AsMatView};

use crate::{
    analysis::WallTapeContourAnalyzer,
    extraction::RFTapeContourExtractor,
    filter::TargetFilterMap,
};

pub struct VisionPipeline {
    camera: OpenCVCamera,
    extractor: RFTapeContourExtractor,
    analyzer: WallTapeContourAnalyzer,
    target_filters: TargetFilterMap,
}

impl VisionPipeline {
    pub fn new(config: CameraConfig) -> Result<Self> {
        let camera = OpenCVCamera::new(config)?;

        let extractor = RFTapeContourExtractor::new();
        let analyzer = WallTapeContourAnalyzer::new();

        let target_filters = TargetFilterMap::new();

        Ok(VisionPipeline {
            camera,
            extractor,
            analyzer,
            target_filters,
        })
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

            let filtered_target = self.target_filters.filter_target(target, frame.timestamp);

            if let Some(result_target) = filtered_target {
                result_targets.push(result_target);
            }
        }

        Ok(result_targets)
    }
}
