use opencv::{
    calib3d::{rodrigues, solve_pnp_generic, SolvePnPMethod},
    core::{Mat, Point2f, Point3f},
    prelude::*,
    types::{VectorOfPoint2f, VectorOfPoint3f, VectorOfMat},
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
    (0.3333, 0.0721), // symmetry-breaking point
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
    (-0.0254, -0.2286), // symmetry-breaking point
];

pub struct PnPParams<'src> {
    pub img_points: VectorOfPoint2f,
    pub obj_points: &'src VectorOfPoint3f,
    pub config: &'src CameraConfig,
}

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

    pub fn make_pnp_params<'src>(&'src self, contour_group: &'src ContourGroup) -> PnPParams<'src> {
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

        let config = contour_group.camera;

        PnPParams {
            img_points,
            obj_points,
            config,
        }
    }

    pub fn solve_pnp(
        &self,
        params: PnPParams,
    ) -> Vec<PnPResult> {
        let PnPParams { img_points, obj_points, config } = params;

        let mut rvec_mats = VectorOfMat::new();
        let mut tvec_mats = VectorOfMat::new();

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

        solve_pnp_generic(
            &obj_points,
            &img_points,
            &intrinsic_matrix,
            &distortion_coeffs,
            &mut rvec_mats,
            &mut tvec_mats,
            false,
            SolvePnPMethod::SOLVEPNP_IPPE,
            &opencv::core::no_array().unwrap(),
            &opencv::core::no_array().unwrap(),
            &mut opencv::core::no_array().unwrap(),
        )
        .unwrap();

        rvec_mats.iter().zip(tvec_mats).map(|(rvec_mat, tvec_mat)| {
            PnPResult {
                intrinsic_matrix: Mat::copy(&intrinsic_matrix).unwrap(),
                distortion_coeffs: Mat::copy(&distortion_coeffs).unwrap(),
                rvec_mat,
                tvec_mat,
            }
        }).collect()
    }

    pub fn find_target(&self, id: u8, pnp_results: &Vec<PnPResult>, config: &CameraConfig) -> Target {
        let (rmat, tvec, camera_pose) = pnp_results.iter().find_map(move |pnp_result| {
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

            if tvec[[2, 0]].is_sign_negative() {
                return None;
            }

            Some((rmat.into_owned(), tvec.into_owned(), camera_pose))
        }).expect("Failed to find valid PnP solution");

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
        let pnp_params = self.make_pnp_params(contour_group);
        let pnp_result = self.solve_pnp(pnp_params);

        self.find_target(contour_group.id, &pnp_result, contour_group.camera)
    }
}
