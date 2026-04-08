import mujoco
import math
import numpy as np
import mink
from scipy.spatial.transform import Rotation as R


class IKSolver:
    def __init__(
        self,
        reset_qpos=None,
        xml_path='mj_assets/stanford_tidybot2/tidybot_yam_cube.xml',
        base_immobile=False,
        collision_avoidance=True,
        posture_cost=1e-3,
    ):
        """
        WBC IK Solver for the YAM 6-DOF arm on a mobile base.

        The YAM robot has:
          - 3 base DOF: joint_x, joint_y, joint_th
          - 6 arm DOF: joint1 through joint6
          - 2 gripper DOF: left_finger (controlled), right_finger (coupled)

        Args:
            reset_qpos: 6-element arm home configuration. Defaults to YAM home pose.
            xml_path: Path to MuJoCo XML model.
            base_immobile: If True, locks base in place.
            collision_avoidance: If True, enables arm-vs-base collision avoidance.
            posture_cost: Cost for posture regularisation task.
        """
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.model.body_gravcomp[:] = 1.0

        # DOF layout: base(3) + arm(6) + gripper(2)
        self.base_dofs = base_dofs = self.model.body('base_link').jntnum.item()
        self.arm_dofs = arm_dofs = 6
        self.qpos_base = self.data.qpos[:base_dofs]
        self.qpos_arm = self.data.qpos[base_dofs:(base_dofs + arm_dofs)]

        self.configuration = mink.Configuration(self.model)

        # Collision avoidance: keep arm away from the mobile base platform.
        arm_geoms = mink.get_subtree_geom_ids(self.model, self.model.body("arm").id)
        base_geoms = mink.get_body_geom_ids(self.model, self.model.body("base_link").id)
        collision_pairs = [(arm_geoms, base_geoms)]

        self.collision_avoidance_limit = mink.CollisionAvoidanceLimit(
            model=self.model,
            geom_pairs=collision_pairs,  # type: ignore
            minimum_distance_from_collisions=0.05,
            collision_detection_distance=0.1,
        )

        # End-effector task (pinch_site is the renamed grasp_site on link_6)
        self.end_effector_task = mink.FrameTask(
            frame_name="pinch_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )

        # Velocity limits
        if base_immobile:
            self.max_base_velocity = np.array([0, 0, 0])
        else:
            self.max_base_velocity = np.array([0.5, 0.5, np.pi / 2])
        # YAM: joints 1-3 are dm4340, joints 4-6 are dm4310 — use 80 deg/s for all
        self.max_arm_velocity = np.array([math.radians(80)] * 6)

        joint_names = [
            "joint_x", "joint_y", "joint_th",
            "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
        ]
        velocity_limits = {
            name: limit
            for name, limit in zip(
                joint_names,
                np.concatenate([self.max_base_velocity, self.max_arm_velocity]),
            )
        }
        self.velocity_limit = mink.VelocityLimit(self.model, velocity_limits)
        self.position_limit = mink.ConfigurationLimit(self.model)

        self.limits = [self.velocity_limit, self.position_limit]
        if collision_avoidance:
            self.limits.append(self.collision_avoidance_limit)

        # Posture task: regularise arm toward home pose
        self.posture_cost = np.zeros((self.model.nv,))
        self.posture_cost[3:] = posture_cost
        self.posture_task = mink.PostureTask(self.model, cost=self.posture_cost)

        # Soft damping to discourage unnecessary base motion
        immobile_base_cost = np.zeros((self.model.nv,))
        immobile_base_cost[:3] = 1.5
        self.damping_task = mink.DampingTask(self.model, immobile_base_cost)

        # Set retract/home configuration (6-DOF)
        if reset_qpos is None:
            reset_qpos = [0.0, 1.047, 1.047, 0.0, 0.0, 0.0]
        self.reset_qpos = reset_qpos

        self.retract_configuration = mink.Configuration(self.model)
        # qpos layout: zeros(3 base) + arm(6) + zeros(2 gripper)
        RETRACT_QPOS = np.hstack((np.zeros(3), np.array(self.reset_qpos), np.zeros(2)))
        self.retract_configuration.update(RETRACT_QPOS)
        self.posture_task.set_target_from_configuration(self.retract_configuration)

        self.tasks = [self.end_effector_task, self.posture_task]
        self.solver = "quadprog"
        self.pos_threshold = 1e-4
        self.ori_threshold = 1e-4
        self.max_iters = 20
        self.frequency = 100.0

    def solve(self, pos, quat, curr_qpos):
        """Solve WBC IK for the YAM arm.

        Args:
            pos: Target end-effector position [3].
            quat: Target end-effector quaternion [x, y, z, w].
            curr_qpos: Current full qpos [base(3) + arm(6) + gripper(2)].

        Returns:
            Full qpos after IK [11].
        """
        T_wt = np.eye(4)
        T_wt[:3, :3] = R.from_quat(quat).as_matrix()
        T_wt[:3, 3] = pos
        self.end_effector_task.set_target(mink.SE3.from_matrix(T_wt))

        self.data.qpos[:] = curr_qpos
        mujoco.mj_forward(self.model, self.data)

        for _ in range(self.max_iters):
            vel = mink.solve_ik(
                self.configuration,
                [*self.tasks, self.damping_task],
                1 / self.frequency,
                self.solver,
                1e-3,
                limits=self.limits,
            )
            self.configuration.integrate_inplace(vel, 1 / self.frequency)

            err = self.end_effector_task.compute_error(self.configuration)
            if (
                np.linalg.norm(err[:3]) <= self.pos_threshold
                and np.linalg.norm(err[3:]) <= self.ori_threshold
            ):
                break

        self.data.qpos[:] = self.configuration.q
        return self.data.qpos.copy()
