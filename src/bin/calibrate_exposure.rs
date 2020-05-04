use stdvis_core::{
    traits::{Camera, ContourExtractor},
    types::{CameraConfig, Image},
};

use stdvis_opencv::camera::OpenCVCamera;

use vision_2020::extraction::RFTapeContourExtractor;

fn main() {
    let mut camera = OpenCVCamera::new(CameraConfig::default()).unwrap();

    const NUM_PHOTOS: i32 = 5;
    const MAX_STEPS_SINCE_MIN: i32 = 10;

    let mut error_rate = 0.;
    let mut exposure = 3.;
    let mut epic_exposure = 3.;
    let mut min_error_rate = 1.;
    let mut steps_since_min = 0;

    let extractor = RFTapeContourExtractor::new();

    loop {
        camera.set_exposure(exposure).unwrap();
        for i in 0..NUM_PHOTOS {
            let frame = camera.grab_frame().unwrap();
            let contours = extractor.extract_from(&frame);
            if contours.len() == 0 {
                error_rate += 1. / NUM_PHOTOS as f64;
            }
        }
        if error_rate < min_error_rate {
            min_error_rate = error_rate;
            epic_exposure = exposure;
            steps_since_min = 0;
        }
        if steps_since_min == MAX_STEPS_SINCE_MIN {
            break
        }
        exposure += 1.;
        steps_since_min += 1;
    }
    println!("{}", epic_exposure);
    println!("{}", min_error_rate);
}
