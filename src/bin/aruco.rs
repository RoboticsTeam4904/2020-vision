use anyhow::{Context, Result};
use opencv::{
    aruco::{
        self, detect_markers, draw_detected_markers, estimate_pose_single_markers,
        DetectorParameters,
    },
    calib3d::rodrigues,
    core::{no_array, Point2f, Scalar, Vec3d, Vector},
    prelude::Mat,
};
use serde_json;
use std::{
    fs::File,
    ops::{Deref, DerefMut},
};
use stdvis_core::{traits::Camera, types::VisionTarget};
use stdvis_opencv::{
    camera::OpenCVCamera,
    convert::{AsArrayView, AsMatView},
};

fn main() -> Result<()> {
    let config_file = File::open("config.json")?;
    let config = serde_json::from_reader(config_file)?;
    let mut camera = OpenCVCamera::new(config)?;

    loop {
        let image = camera
            .grab_frame()
            .context("Failed to read frame from camera")?;

        let mut mat_image = image.as_mat_view();
        let dictionary =
            aruco::get_predefined_dictionary(aruco::PREDEFINED_DICTIONARY_NAME::DICT_4X4_50)?;
        let mut corners = Vector::<Vector<Point2f>>::new();
        let mut ids = Vector::<i32>::new();
        let params = DetectorParameters::create()?;
        let mut rejected_img_points = no_array();

        let i = image.camera.intrinsic_matrix.view();
        let d = image.camera.distortion_coeffs.view();

        let intrinsic_matrix = Mat::from_slice_2d(&[
            &[i[[0, 0]], i[[0, 1]], i[[0, 2]]],
            &[i[[1, 0]], i[[1, 1]], i[[1, 2]]],
            &[i[[2, 0]], i[[2, 1]], i[[2, 2]]],
        ])?;

        let distortion_coeffs = Mat::from_slice(d.as_slice().unwrap())?;

        detect_markers(
            &*mat_image,
            &dictionary,
            &mut corners,
            &mut ids,
            &params,
            &mut rejected_img_points,
            &intrinsic_matrix,
            &distortion_coeffs,
        )?;

        let mut out_img = mat_image.deref_mut();

        let mut rvecs = Vector::<Vec3d>::new();
        let mut tvecs = Vector::<Vec3d>::new();

        draw_detected_markers(
            &mut out_img,
            &corners,
            &ids,
            Scalar::new(255.0, 0.0, 0.0, 0.0),
        )?;

        estimate_pose_single_markers(
            &corners,
            0.036,
            &intrinsic_matrix,
            &distortion_coeffs,
            &mut rvecs,
            &mut tvecs,
            &mut no_array(),
        )?;

        for idx in 0..ids.len() {
            let rvec_vec = rvecs.get(idx)?;
            let tvec_vec = tvecs.get(idx)?;

            opencv::calib3d::draw_frame_axes(
                out_img,
                &intrinsic_matrix,
                &distortion_coeffs,
                &rvecs.get(idx)?,
                &tvecs.get(idx)?,
                0.05,
                3,
            )?;

            let mut rmat_mat = Mat::default();
            let mut jacobian_mat = Mat::default();
            rodrigues(&rvec_vec, &mut rmat_mat, &mut jacobian_mat)?;

            let tvec_mat = Mat::from_exact_iter(rvec_vec.into_iter())?;

            // let tvec_mat = Mat::from_slice(rvec_vec.as_slice())?;

            let tvec = tvec_mat.as_array_view::<f64>().into_shape((3, 1))?;
            let rmat = rmat_mat.as_array_view::<f64>().into_shape((3, 3))?;
            let rmat_t = rmat.t();

            let camera_pose = rmat_t.dot(&tvec);

            let x = camera_pose[[0, 0]];
            let y = camera_pose[[1, 0]];
            let z = camera_pose[[2, 0]];

            let theta = x.atan2(z);

            let r00 = rmat[[0, 0]];
            let r10 = rmat[[1, 0]];
            let r20 = rmat[[2, 0]];
            let r21 = rmat[[2, 1]];
            let r22 = rmat[[2, 2]];

            let roll = r10.atan2(r00);
            let pitch = -r20.atan2((r21.powf(2.) + r22.powf(2.)).sqrt());
            let yaw = r21.atan2(r22);

            let target = VisionTarget {
                id: 0,
                theta: theta,
                beta: yaw,
                dist: z,
                height: y,
                confidence: 0.,
            };

            dbg!(target);
        }

        println!("-------------------------------");

        opencv::imgcodecs::imwrite(
            "aruco.png",
            out_img,
            &opencv::types::VectorOfi32::with_capacity(0),
        )?;

        std::thread::sleep(std::time::Duration::from_millis(200));
    }

    Ok(())
}
