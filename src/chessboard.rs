use opencv::{
    calib3d::find_chessboard_corners;
}

// find outside corners  

pub fn find_chessboard (
    image: &dyn ToInputArray, 
    // image goes to find outside corners
    // find oustide corners gives output of corners array
    // using image and output corners pass it into find_chesboard_corners
    // return the corners from the find_chessboard_corners
    pattern_size: Size, 
    let corners =
    corners: &mut dyn ToOutputArray, 
    flags: i32
) -> Result<bool>



