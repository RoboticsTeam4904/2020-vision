use opencv::{
    calib3d::find_chessboard_corners;
}

// find outside corners  

pub fn find_chessboard (
    image: &dyn ToInputArray, 
    pattern_size: Size, 
    corners: &mut dyn ToOutputArray, 
    flags: i32
) -> Result<bool>




