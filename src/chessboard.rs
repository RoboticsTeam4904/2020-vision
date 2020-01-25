use crate::extraction::HighPortContourExtractor;

use opencv::{
    core::{Mat, Size},
    calib3d::{CALIB_CB_FAST_CHECK, find_chessboard_corners},
    types::VectorOfPoint2f,
};

use standard_vision::{
    traits::ImageData,
    types::Image,
};

pub fn b<I: ImageData(image: &Image<I>) {
    let board_size = Size::new(7, 7);
    let mut corners = VectorOfPoint2f::new();
    find_chessboard_corners(image, board_size, corners, CALIB_CB_FAST_CHECK); 
}
