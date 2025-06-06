import mujoco
import math
import numpy as np
import mink
from scipy.spatial.transform import Rotation as R

class IKSolver:
    def __init__(self, reset_qpos, xml_path='mj_assets/stanford_tidybot2/tidybot.xml'):
        self.reset_qpos = reset_qpos
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        model_body_names = [self.model.body(i).name for i in range(self.model.nbody)]
        print(model_body_names)

        self.data = mujoco.MjData(self.model)

        self.model.body_gravcomp[:] = 1.0  # Enable gravity compensation for all bodies

        # Cache references to array slices
        self.base_dofs = base_dofs = self.model.body('base_link').jntnum.item()
        self.arm_dofs = arm_dofs = 7
        self.qpos_base = self.data.qpos[:base_dofs]
        self.qpos_arm = self.data.qpos[base_dofs:(base_dofs + arm_dofs)]
        self.qpos_gripper = self.data.qpos[(base_dofs + arm_dofs):(base_dofs + arm_dofs + 1)]

        # WBC Configuration and Tasks
        self.configuration = mink.Configuration(self.model)

        arm_geoms = mink.get_subtree_geom_ids(self.model, self.model.body("gen3/base_link").id)
        base_geoms = mink.get_body_geom_ids(self.model, self.model.body("base_link").id)
        gripper_geoms = mink.get_body_geom_ids(self.model, self.model.body("base").id)  # Gripper base
        gripper_finger_geoms = []
        for finger in ["right_driver", "right_follower", "left_driver", "left_follower"]:
            gripper_finger_geoms.extend(mink.get_body_geom_ids(self.model, self.model.body(finger).id))
        collision_pairs = [
            (gripper_geoms + gripper_finger_geoms, arm_geoms),
            (gripper_geoms + gripper_finger_geoms + arm_geoms, base_geoms),
        ]

        self.collision_avoidance_limit = mink.CollisionAvoidanceLimit(
            model=self.model,
            geom_pairs=collision_pairs,  # type: ignore
            minimum_distance_from_collisions=0.05,
            collision_detection_distance=0.1,
        )

        self.end_effector_task = mink.FrameTask(
            frame_name="pinch_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )

        # Base and arm velocity limits
        self.max_base_velocity = np.array([0.5, 0.5, np.pi/2])  # (x, y, yaw)
        self.max_arm_velocity = np.array([math.radians(80)] * 4 + [math.radians(140)] * 3)
        # Create a dictionary mapping joint names to velocity limits
        joint_names = [
            "joint_x",
            "joint_y",
            "joint_th",
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "joint_7",
        ]
        velocity_limits = {name: limit for name, limit in zip(joint_names, np.concatenate([self.max_base_velocity, self.max_arm_velocity]))}
        self.velocity_limit = mink.VelocityLimit(self.model, velocity_limits)
        self.position_limit = mink.ConfigurationLimit(self.model)

        #self.limits = [self.velocity_limit, self.position_limit, self.collision_avoidance_limit]
        self.limits = [self.velocity_limit, self.position_limit]

        # Posture Task (Encourages retraction-like configurations)
        self.posture_cost = np.zeros((self.model.nv,))
        self.posture_cost[3:] = 1e-3  # Encourage default posture
        self.posture_task = mink.PostureTask(self.model, cost=self.posture_cost)

        immobile_base_cost = np.zeros((self.model.nv,))
        immobile_base_cost[:3] = 1.5 
        self.damping_task = mink.DampingTask(self.model, immobile_base_cost)

        self.retract_configuration = mink.Configuration(self.model)
        RETRACT_QPOS = np.array(self.reset_qpos)
        RETRACT_QPOS = np.hstack((np.zeros(3), RETRACT_QPOS, np.zeros(8)))
        self.retract_configuration.update(RETRACT_QPOS)
        self.posture_task.set_target_from_configuration(self.retract_configuration)

        self.tasks = [self.end_effector_task, self.posture_task]  # Keep posture task
        self.solver = "quadprog"
        self.pos_threshold = 1e-4
        self.ori_threshold = 1e-4
        self.max_iters = 20
        self.frequency = 100.0  # Control frequency in Hz

    def solve(self, pos, quat, curr_qpos):
        """Solves inverse kinematics using Whole-Body Control (WBC)."""
        # Convert quaternion to rotation matrix and construct target transform
        T_wt = np.eye(4)
        T_wt[:3, :3] = R.from_quat(quat).as_matrix()
        T_wt[:3, 3] = pos
        T_wt = mink.SE3.from_matrix(T_wt)

        self.end_effector_task.set_target(T_wt)

        # Set initial joint configuration
        self.data.qpos[:] = curr_qpos
        #self.configuration.update(curr_qpos) 
   
        mujoco.mj_forward(self.model, self.data)  # Ensure kinematics update

        for _ in range(self.max_iters):
            # Solve IK to get joint velocity update
            vel = mink.solve_ik(
                self.configuration,
                [*self.tasks, self.damping_task],  # Includes posture task
                1 / self.frequency,
                self.solver,
                1e-3,
                limits=self.limits
            )

            # Integrate velocity update into joint positions
            self.configuration.integrate_inplace(vel, 1 / self.frequency)

            # Compute error and check convergence
            err = self.end_effector_task.compute_error(self.configuration)
            if (
                np.linalg.norm(err[:3]) <= self.pos_threshold
                and np.linalg.norm(err[3:]) <= self.ori_threshold
            ):
                break

        self.data.qpos[:] = self.configuration.q
        return self.data.qpos.copy()

# Test Script
if __name__ == '__main__':
    arm_reset_qpos = [
        0.0,
        -0.34906585,
        3.14159265,
        -2.54818071,
        0.0,
        -0.87266463,
        1.57079633,
    ]
    ik_solver = IKSolver(arm_reset_qpos)
    home_pos, home_quat = np.array([0.456, 0.0, 0.434]), np.array([0.5, 0.5, 0.5, 0.5])
    retract_qpos = np.deg2rad([0, -20, 180, -146, 0, -50, 90])  # Base and arm
    retract_qpos = np.hstack((np.zeros(3), retract_qpos, np.zeros(8)))

    import time
    start_time = time.time()
    for _ in range(1000):
        qpos = ik_solver.solve(home_pos, home_quat, retract_qpos)
    elapsed_time = time.time() - start_time
    print(f'Time per call: {elapsed_time:.3f} ms')

    # Home: 0, 15, 180, -130, 0, 55, 90
    print('start', retract_qpos.round(2))
    ik_solver.configuration.update(retract_qpos)
    print('end', ik_solver.solve(home_pos, home_quat, retract_qpos).round(2))
