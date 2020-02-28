use opencv::{
    calib3d::{rodrigues, solve_pnp},
    core::{Mat, Point2f, Point3f},
    prelude::*,
    types::{VectorOfPoint2f, VectorOfPoint3f},
};

use stdvis_core::{
    traits::ContourAnalyzer,
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

pub struct PnPResult {
    pub intrinsic_matrix: Mat,
    pub distortion_coeffs: Mat,
    pub rvec_mat: Mat,
    pub tvec_mat: Mat,
}

pub struct WallTapeContourAnalyzer {
    high_port_object_points: VectorOfPoint3f,
    loading_port_object_points: VectorOfPoint3f,
}

impl WallTapeContourAnalyzer {
    pub fn new() -> Self {
        let high_port_object_points = VectorOfPoint3f::from_iter(
            HIGH_PORT_OBJECT_POINTS
                .iter()
                .map(|point| Point3f::new(point.0, point.1, 0.)),
        );

        let loading_port_object_points = VectorOfPoint3f::from_iter(
            LOADING_PORT_OBJECT_POINTS
                .iter()
                .map(|point| Point3f::new(point.0, point.1, 0.)),
        );

        WallTapeContourAnalyzer {
            high_port_object_points,
            loading_port_object_points,
        }
    }

    pub fn solve_pnp(
        &self,
        img_points: &VectorOfPoint2f,
        obj_points: &VectorOfPoint3f,
        config: &CameraConfig,
    ) -> PnPResult {
        let mut rvec_mat = Mat::default().unwrap();
        let mut tvec_mat = Mat::default().unwrap();

        // TODO: Would like to use as_mat_view(), but it currently does not work for
        // 2D matrices.
        let i = config.intrinsic_matrix.view();
        let d = config.distortion_coeffs.view();

        let intrinsic_matrix = Mat::from_slice_2d(&[
            &[i[[0, 0]], i[[0, 1]], i[[0, 2]]],
            &[i[[1, 0]], i[[1, 1]], i[[1, 2]]],
            &[i[[2, 0]], i[[2, 1]], i[[2, 2]]],
        ])
        .unwrap();

        let distortion_coeffs = Mat::from_slice(d.as_slice().unwrap()).unwrap();

        solve_pnp(
            &obj_points,
            &img_points,
            &intrinsic_matrix,
            &distortion_coeffs,
            &mut rvec_mat,
            &mut tvec_mat,
            false,
            opencv::calib3d::SOLVEPNP_IPPE,
        )
        .unwrap();

        PnPResult {
            intrinsic_matrix,
            distortion_coeffs,
            rvec_mat,
            tvec_mat,
        }
    }

    pub fn find_target(&self, id: u8, pnp_result: &PnPResult, config: &CameraConfig) -> Target {
        let PnPResult {
            rvec_mat, tvec_mat, ..
        } = pnp_result;

        let mut rmat_mat = Mat::default().unwrap();
        let mut jacobian_mat = Mat::default().unwrap();
        rodrigues(&rvec_mat, &mut rmat_mat, &mut jacobian_mat).unwrap();

        let tvec = tvec_mat.as_array_view::<f64>().into_shape((3, 1)).unwrap();
        let rmat = rmat_mat.as_array_view::<f64>().into_shape((3, 3)).unwrap();
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

        Target {
            id,
            theta: theta + config.pose.angle,
            beta: yaw + config.pose.angle,
            dist: config.pose.angle.cos() * config.pose.dist + config.pose.yaw.cos() * z
                - config.pose.yaw.sin() * x,
            height: y + config.pose.height,
            confidence: 0.,
        }
    }
}

impl ContourAnalyzer for WallTapeContourAnalyzer {
    fn analyze(&self, contour_group: &ContourGroup) -> Target {
        let obj_points = match contour_group.id {
            0 => &self.high_port_object_points,
            1 => &self.loading_port_object_points,
            _ => panic!("Unknown contour group type"),
        };

        let img_points = VectorOfPoint2f::from_iter(
            contour_group
                .contours
                .iter()
                .map(|contour| contour.points.iter())
                .flatten()
                .map(|point| Point2f::new(point.0 as f32, point.1 as f32)),
        );

        let pnp_result = self.solve_pnp(&img_points, &obj_points, contour_group.camera);

        self.find_target(contour_group.id, &pnp_result, contour_group.camera)
    }
}
