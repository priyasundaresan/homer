import mujoco
import numpy as np

DAMPING_COEFF = 1e-12
MAX_ANGLE_CHANGE = np.deg2rad(45)


class IKSolver:
    """Jacobian-based IK solver for the YAM 6-DOF arm (base-arm / non-WBC mode).

    Solves in the base-local frame: the base is fixed at the origin and only
    arm joints are updated. Interface matches arm_ik_solver.IKSolver so that
    ArmController can use it unchanged.
    """

    def __init__(self):
        # Use the robot-only MJCF (no scene objects), base fixed at origin
        self.model = mujoco.MjModel.from_xml_path('mj_assets/stanford_tidybot2/tidybot_yam_cube.xml')
        self.data = mujoco.MjData(self.model)
        self.model.body_gravcomp[:] = 1.0

        # DOF layout: base(3) + arm(6) + gripper(2)
        self.base_dofs = 3
        self.arm_dofs = 6
        self.arm_slice = slice(self.base_dofs, self.base_dofs + self.arm_dofs)

        self.site_id = self.model.site('pinch_site').id
        self.site_pos = self.data.site(self.site_id).xpos
        self.site_mat = self.data.site(self.site_id).xmat

        # Preallocate
        self.err = np.empty(6)
        self.err_pos, self.err_rot = self.err[:3], self.err[3:]
        self.site_quat = np.empty(4)
        self.site_quat_inv = np.empty(4)
        self.err_quat = np.empty(4)
        self.jac = np.empty((6, self.model.nv))
        self.jac_pos, self.jac_rot = self.jac[:3], self.jac[3:]
        self.damping = DAMPING_COEFF * np.eye(6)

    def solve(self, pos, quat, curr_arm_qpos, max_iters=20, err_thresh=1e-4):
        """Solve IK for the YAM arm in base-local frame.

        Args:
            pos: target EE position in base-local frame [3]
            quat: target EE quaternion (x, y, z, w) [4]
            curr_arm_qpos: current arm joint positions [6]

        Returns:
            new arm joint positions [6]
        """
        quat_wxyz = quat[[3, 0, 1, 2]]  # (x,y,z,w) -> (w,x,y,z)

        # Fix base at origin, set arm joints, zero gripper
        self.data.qpos[:self.base_dofs] = 0.0
        self.data.qpos[self.arm_slice] = curr_arm_qpos
        self.data.qpos[self.base_dofs + self.arm_dofs:] = 0.0

        for _ in range(max_iters):
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)

            self.err_pos[:] = pos - self.site_pos

            mujoco.mju_mat2Quat(self.site_quat, self.site_mat)
            mujoco.mju_negQuat(self.site_quat_inv, self.site_quat)
            mujoco.mju_mulQuat(self.err_quat, quat_wxyz, self.site_quat_inv)
            mujoco.mju_quat2Vel(self.err_rot, self.err_quat, 1.0)

            if np.linalg.norm(self.err) < err_thresh:
                break

            mujoco.mj_jacSite(self.model, self.data, self.jac_pos, self.jac_rot, self.site_id)

            # Only update arm columns; keep base fixed
            jac_arm = self.jac[:, self.arm_slice]
            update_arm = jac_arm.T @ np.linalg.solve(jac_arm @ jac_arm.T + self.damping, self.err)

            update_max = np.abs(update_arm).max()
            if update_max > MAX_ANGLE_CHANGE:
                update_arm *= MAX_ANGLE_CHANGE / update_max

            self.data.qpos[self.arm_slice] += update_arm

        return self.data.qpos[self.arm_slice].copy()
