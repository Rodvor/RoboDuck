import gymnasium as gym
from gymnasium import spaces

import mujoco
import mujoco.viewer

import numpy as np


class MicroDuckEnv(gym.Env):

    # Physically authored standing pose in actuator order. The policy controls
    # small offsets around this pose instead of the raw +/-10 radian actuator
    # range, which made early exploration violently unstable.
    STAND_POSE = np.array([
        0.0, -0.08726646, -0.457924, -0.004940, 0.452984,
        0.34906585, 0.34906585, 0.0, 0.0,
        0.0, 0.08726646, 0.457924, 0.004940, -0.452984,
    ], dtype=np.float64)
    ACTION_SCALE = 0.35

    metadata = {
        "render_modes": ["human"]
    }

    def __init__(self, render_mode=None):

        super().__init__()

        self.render_mode = render_mode

        # ==========================================
        # LOAD OFFICIAL MICRODUCK
        # ==========================================

        self.model = mujoco.MjModel.from_xml_path(
            "duck/microduck/robot_allcollisions.xml"
        )

        self.data = mujoco.MjData(self.model)

        # ==========================================
        # JOINT NAMES
        # ==========================================

        self.joint_names = [
            "left_hip_yaw",
            "left_hip_roll",
            "left_hip_pitch",
            "left_knee",
            "left_ankle",
            "neck_pitch",
            "head_pitch",
            "head_yaw",
            "head_roll",
            "right_hip_yaw",
            "right_hip_roll",
            "right_hip_pitch",
            "right_knee",
            "right_ankle",
        ]

        # Resolve all joint addresses once. Looking names up inside every step
        # is unnecessary Python overhead because model addresses are constant.
        joint_ids = np.asarray([
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )
            for name in self.joint_names
        ])
        self.joint_qpos_addresses = self.model.jnt_qposadr[joint_ids]
        self.joint_dof_addresses = self.model.jnt_dofadr[joint_ids]
        self.head_qpos_addresses = self.joint_qpos_addresses[5:9]
        self.head_dof_addresses = self.joint_dof_addresses[5:9]

        # The jaw_soft body is near the center of the complete head assembly.
        self.head_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "jaw_soft",
        )
        self.trunk_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "trunk_base",
        )
        self.foot_site_ids = np.asarray([
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_SITE,
                name,
            )
            for name in ("left_foot", "right_foot")
        ])
        self.left_foot_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "left_foot_collision",
        )
        self.right_foot_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "right_foot_collision",
        )
        self.floor_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "floor",
        )
        self.joint_lower = self.model.jnt_range[joint_ids, 0].copy()
        self.joint_upper = self.model.jnt_range[joint_ids, 1].copy()
        self.last_action = np.zeros(14, dtype=np.float64)

        # ==========================================
        # ACTION SPACE
        # ==========================================

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(14,),
            dtype=np.float32
        )

        # ==========================================
        # OBSERVATION
        #
        # quaternion        4
        # angular velocity  3
        # linear velocity   3
        # joint offsets    14
        # joint velocities 14
        #
        # TOTAL = 38
        # ==========================================

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(38,),
            dtype=np.float32
        )

        self.frame_skip = 5
        self.max_episode_steps = 1000

        self.step_count = 0

        self.viewer = None

    # ==================================================
    # RESET
    # ==================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        mujoco.mj_resetData(
            self.model,
            self.data
        )

        self.step_count = 0

        # ==========================================
        # INITIAL BASE POSE
        # ==========================================

        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = 0.120

        # Horizontal reference used by the standing-position reward.
        self.initial_xy = self.data.qpos[:2].copy()

        roll = self.np_random.uniform(-0.03, 0.03)
        pitch = self.np_random.uniform(-0.03, 0.03)
        yaw = self.np_random.uniform(-0.02, 0.02)

        quaternion = np.array([
            1.0,
            roll / 2.0,
            pitch / 2.0,
            yaw / 2.0,
        ])
        self.data.qpos[3:7] = quaternion / np.linalg.norm(quaternion)

        # ==========================================
        # INITIAL JOINT POSE
        #
        # Replace these values later with the
        # exact good standing pose.
        # ==========================================

        standing_pose = self.STAND_POSE + self.np_random.uniform(
            low=-0.025,
            high=0.025,
            size=14,
        )
        self.data.qpos[self.joint_qpos_addresses] = standing_pose
        self.data.ctrl[:] = self.STAND_POSE
        self.last_action.fill(0.0)

        # ==========================================
        # ZERO VELOCITY
        # ==========================================

        self.data.qvel[:] = 0.0

        # ==========================================
        # FORWARD DYNAMICS
        # ==========================================

        mujoco.mj_forward(
            self.model,
            self.data
        )

        observation = self._get_observation()

        return observation, {}

    # ==================================================
    # STEP
    # ==================================================

    def step(self, action):

        self.step_count += 1

        action = np.asarray(
            action,
            dtype=np.float32
        )

        # ==========================================
        # NORMALIZED ACTION -> STANDING-POSE OFFSET
        # ==========================================

        np.clip(action, -1.0, 1.0, out=self.last_action)
        np.multiply(
            self.last_action,
            self.ACTION_SCALE,
            out=self.data.ctrl,
        )
        self.data.ctrl += self.STAND_POSE
        np.clip(
            self.data.ctrl,
            self.joint_lower,
            self.joint_upper,
            out=self.data.ctrl,
        )

        # ==========================================
        # PHYSICS
        # ==========================================

        mujoco.mj_step(
            self.model,
            self.data,
            nstep=self.frame_skip,
        )

        # ==========================================
        # OBSERVATION
        # ==========================================

        observation = self._get_observation()

        # ==========================================
        # REWARD
        # ==========================================

        reward = self._get_reward()

        # ==========================================
        # TERMINATION
        # ==========================================

        terminated = self._is_fallen()

        truncated = (
            self.step_count
            >= self.max_episode_steps
        )

        return (
            observation,
            reward,
            terminated,
            truncated,
            {}
        )

    # ==================================================
    # OBSERVATION
    # ==================================================

    def _get_observation(self):

        linear_velocity = self.data.qvel[:3]

        # ==========================================
        # BODY ORIENTATION
        # ==========================================

        quat = self.data.qpos[3:7]

        # ==========================================
        # BODY ANGULAR VELOCITY
        # ==========================================

        angular_velocity = self.data.qvel[3:6]

        # ==========================================
        # JOINT POSITIONS
        # ==========================================

        joint_positions = (
            self.data.qpos[self.joint_qpos_addresses]
            - self.STAND_POSE
        )

        # ==========================================
        # JOINT VELOCITIES
        # ==========================================

        joint_velocities = self.data.qvel[
            self.joint_dof_addresses
        ]

        # ==========================================
        # COMBINE
        # ==========================================

        observation = np.concatenate([
            quat,
            angular_velocity,
            linear_velocity,
            joint_positions,
            joint_velocities,
        ], dtype=np.float32)

        return observation

    def _get_com_support_error(self):

        center_of_mass_xy = self.data.subtree_com[
            self.trunk_body_id, :2
        ]
        feet_midpoint_xy = np.mean(
            self.data.site_xpos[self.foot_site_ids, :2],
            axis=0,
        )

        return center_of_mass_xy - feet_midpoint_xy

    # ==================================================
    # REWARD
    # ==================================================

    def _get_reward(self):

        # ==========================================
        # TORSO UPRIGHTNESS
        # ==========================================

        quat = self.data.qpos[3:7]
        up_z = 1.0 - 2.0 * (
            quat[1] * quat[1]
            + quat[2] * quat[2]
        )
        upright_reward = max(
            0.0,
            float(np.clip(up_z, -1.0, 1.0)),
        )

        # ==========================================
        # BODY HEIGHT
        # ==========================================

        height_error = self.data.qpos[2] - 0.115
        height_reward = np.exp(
            -150.0 * height_error ** 2
        )

        # ==========================================
        # BOTH FEET ON THE FLOOR
        # ==========================================

        left_contact = False
        right_contact = False

        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1 = contact.geom1
            geom2 = contact.geom2

            if (
                (geom1 == self.left_foot_geom_id and geom2 == self.floor_geom_id)
                or
                (geom2 == self.left_foot_geom_id and geom1 == self.floor_geom_id)
            ):
                left_contact = True

            if (
                (geom1 == self.right_foot_geom_id and geom2 == self.floor_geom_id)
                or
                (geom2 == self.right_foot_geom_id and geom1 == self.floor_geom_id)
            ):
                right_contact = True

        foot_contact_reward = float(left_contact) + float(right_contact)

        # ==========================================
        # MOTION PENALTIES
        # ==========================================

        angular_velocity_penalty = (
            0.10 * np.sum(self.data.qvel[3:6] ** 2)
        )
        joint_velocity_penalty = (
            0.0005 * np.sum(self.data.qvel[6:] ** 2)
        )

        # ==========================================
        # HEAD AND WHOLE-BODY STANDING POSE
        # ==========================================

        head_joint_positions = (
            self.data.qpos[self.head_qpos_addresses]
            - self.STAND_POSE[5:9]
        )
        head_joint_velocities = self.data.qvel[
            self.head_dof_addresses
        ]
        head_pose_reward = np.exp(
            -8.0 * np.sum(head_joint_positions ** 2)
        )
        head_velocity_penalty = (
            0.01 * np.sum(head_joint_velocities ** 2)
        )

        joint_pose_error = (
            self.data.qpos[self.joint_qpos_addresses]
            - self.STAND_POSE
        )
        stand_pose_reward = np.exp(
            -2.0 * np.sum(joint_pose_error ** 2)
        )

        # Penalize policy offsets rather than the non-zero standing targets.
        control_penalty = 0.02 * np.sum(self.last_action ** 2)

        # ==========================================
        # ALIVE BONUS
        # ==========================================

        alive_bonus = 1.0

        # ==========================================
        # TOTAL
        # ==========================================

        reward = (
            8.0 * upright_reward
            + 2.0 * height_reward
            + 2.0 * foot_contact_reward
            + 3.0 * head_pose_reward
            + 2.0 * stand_pose_reward
            + alive_bonus
            - angular_velocity_penalty
            - joint_velocity_penalty
            - head_velocity_penalty
            - control_penalty
        )

        # Keep critic targets numerically comfortable without changing the
        # relative reward trade-offs.
        return float(0.05 * reward)

    # ==================================================
    # FALL DETECTION
    # ==================================================

    def _is_fallen(self):

        height = self.data.qpos[2]

        if height < 0.055:
            return True

        quat = self.data.qpos[3:7]
        up_z = 1.0 - 2.0 * (
            quat[1] * quat[1]
            + quat[2] * quat[2]
        )

        if up_z < 0.35:
            return True

        return False

    # ==================================================
    # RENDER
    # ==================================================

    def render(self):

        if self.render_mode != "human":
            return

        if self.viewer is None:

            self.viewer = (
                mujoco.viewer.launch_passive(
                    self.model,
                    self.data
                )
            )

        self.viewer.sync()

    # ==================================================
    # CLOSE
    # ==================================================

    def close(self):

        if self.viewer is not None:

            self.viewer.close()

            self.viewer = None
