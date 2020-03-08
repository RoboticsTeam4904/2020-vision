use std::time::Instant;

use global_robot_localization::{ai::kalman_filter::KalmanFilterVision, utility::Pose};
use nalgebra::{Matrix3, RowVector3, Vector3};
use stdvis_core::types::VisionTarget;

pub struct TargetFilter {
    kalman_filter: KalmanFilterVision,
    pub last_timestamp: Instant,
}

impl TargetFilter {
    pub const TARGET_NOISE_BETA: f64 = 0.04363; // 2.5 deg
    pub const TARGET_NOISE_X: f64 = 0.005;
    pub const TARGET_NOISE_Y: f64 = 0.005;

    pub const CONTROL_NOISE_BETA: f64 = 0.2;
    pub const CONTROL_NOISE_X: f64 = 0.05;
    pub const CONTROL_NOISE_Y: f64 = 0.05;

    pub fn new(last_timestamp: Instant, initial_target: &VisionTarget) -> Self {
        let q = Matrix3::from_diagonal(&Vector3::new(
            Self::CONTROL_NOISE_BETA.powi(2),
            Self::CONTROL_NOISE_X.powi(2),
            Self::CONTROL_NOISE_Y.powi(2),
        ));

        let r = Matrix3::from_diagonal(&Vector3::new(
            Self::TARGET_NOISE_BETA.powi(2),
            Self::TARGET_NOISE_X.powi(2),
            Self::TARGET_NOISE_Y.powi(2),
        ));

        let beta = initial_target.beta;
        let x = initial_target.theta.cos() * initial_target.dist;
        let y = initial_target.theta.sin() * initial_target.dist;

        let init_state = RowVector3::new(beta, x, y);

        let alpha = 1e-5;
        let kappa = 0.0;
        let beta = 2.0;

        let covariance_matrix = r.clone();

        let kalman_filter =
            KalmanFilterVision::new(covariance_matrix, init_state, alpha, kappa, beta, q, r);

        Self {
            kalman_filter,
            last_timestamp,
        }
    }

    pub fn update(&mut self, timestamp: Instant, control_pose: Pose, target: &VisionTarget) {
        let delta_t = timestamp.duration_since(self.last_timestamp).as_secs_f64();

        let target_pose = Pose {
            angle: target.beta,
            position: (
                target.theta.cos() * target.dist,
                target.theta.sin() * target.dist,
            )
                .into(),
        };

        self.kalman_filter.prediction_update(delta_t, control_pose);
        self.kalman_filter.measurement_update(target_pose);

        self.last_timestamp = timestamp;
    }

    pub fn predict(&self, target: VisionTarget) -> VisionTarget {
        let predicted_pose = self.kalman_filter.known_state();

        VisionTarget {
            theta: predicted_pose.angle,
            dist: predicted_pose.position.x.hypot(predicted_pose.position.y),
            ..target
        }
    }
}
