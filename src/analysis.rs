use opencv::{
    calib3d::{rodrigues, solve_pnp},
    core::{Mat, Point2f, Point3f},
    prelude::*,
    types::{VectorOfPoint2f, VectorOfPoint3f},
};

use stdvis_core::{
    traits::{ContourAnalyzer},
    types::{CameraConfig, ContourGroup, Target},
};

use stdvis_opencv::convert::{AsArrayView, AsMatView};

const HIGH_PORT_OBJECT_POINTS: &[(f32, f32)] = &[
    (0.5, 0.2165),
    (0.4414, 0.2165),
    (0.2207, -0.1657),
    (-0.2207, -0.1657),
    (-0.4414, 0.2165),
    (-0.5, 0.2165),
    (-0.25, -0.2165),
    (0.25, -0.2165),
];
const LOADING_PORT_OBJECT_POINTS: &[(f32, f32)] = &[
    (0.0889, 0.1397),
    (-0.0889, 0.1397),
    (-0.0889, -0.1397),
    (0.0889, -0.1397),
    (0.0381, 0.0889),
    (-0.0381, 0.0889),
    (-0.0381, -0.0889),
    (0.0381, -0.0889),
];

const CV_HIGH_PORT_OBJECT_POINTS: VectorOfPoint3f = VectorOfPoint3f::from_iter(
    HIGH_PORT_OBJECT_POINTS
        .iter()
        .map(|point| Point3f::new(point.0, point.1, 0.)),
);

const CV_LOADING_PORT_OBJECT_POINTS: VectorOfPoint3f = VectorOfPoint3f::from_iter(
    LOADING_PORT_OBJECT_POINTS
        .iter()
        .map(|point| Point3f::new(point.0, point.1, 0.)),
);

pub struct WallTapeContourAnalyzer {}

impl WallTapeContourAnalyzer {
    pub fn find_target(
        &self,
        id: u8,
        img_points: &VectorOfPoint2f,
        obj_points: &VectorOfPoint3f,
        config: &CameraConfig,
    ) -> Target {
        let mut rvec = Mat::default().unwrap();
        let mut tvec_mat = Mat::default().unwrap();

        let intrinsic_matrix = config.intrinsic_matrix.as_mat_view();
        let dist_coeffs = config.distortion_coeffs.as_mat_view();

        solve_pnp(
            &obj_points,
            &img_points,
            &*intrinsic_matrix,
            &*dist_coeffs,
            &mut rvec,
            &mut tvec_mat,
            false,
            opencv::calib3d::SOLVEPNP_IPPE,
        )
        .unwrap();

        let mut rmat_mat = Mat::default().unwrap();
        let mut jacobian_mat = Mat::default().unwrap();

        rodrigues(&rvec, &mut rmat_mat, &mut jacobian_mat).unwrap();

        let tvec = tvec_mat.as_array_view::<f64>().into_shape((3, 1)).unwrap();
        let rmat = rmat_mat.as_array_view::<f64>().into_shape((3, 3)).unwrap();
        let rmat_trans = rmat.t();

        let camera_pose = rmat_trans.dot(&tvec);

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

        Target {
            id: id,
            theta: theta,
            beta: yaw,
            dist: z,
            height: y,
            confidence: 69.,
        }
    }
}

impl ContourAnalyzer for WallTapeContourAnalyzer {
    fn analyze(&self, contour_group: &ContourGroup) -> Target {
        let obj_points = match contour_group.id {
            0 => CV_HIGH_PORT_OBJECT_POINTS,
            1 => CV_LOADING_PORT_OBJECT_POINTS,
        };

        let mut targets = Vec::<Target>::with_capacity(contour_group.contours.len());
        let img_points = VectorOfPoint2f::from_iter(
            contour_group
                .contours
                .iter()
                .map(|contour| contour.points.iter())
                .flatten()
                .map(|point| Point2f::new(point.0 as f32, point.1 as f32)),
        );

        self.find_target(
            contour_group.id,
            &img_points,
            &obj_points,
            contour_group.camera,
        )
    }
}
