import gymnasium as gym
from gymnasium import spaces

import mujoco
import mujoco.viewer

import numpy as np


class MicroDuckEnv(gym.Env):

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
        # quaternion       4
        # angular velocity 3
        # joint positions 14
        # joint velocities 14
        #
        # TOTAL = 35
        # ==========================================

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(35,),
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
        self.data.qpos[2] = 0.101

        self.data.qpos[3:7] = np.array([
            1.0,
            0.0,
            0.0,
            0.0
        ])

        # ==========================================
        # INITIAL JOINT POSE
        #
        # Replace these values later with the
        # exact good standing pose.
        # ==========================================

        standing_pose = np.zeros(14)

        for i, name in enumerate(self.joint_names):

            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name
            )

            qpos_addr = self.model.jnt_qposadr[
                joint_id
            ]

            self.data.qpos[qpos_addr] = (
                standing_pose[i]
            )

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

        action = np.clip(
            action,
            -1.0,
            1.0
        )

        # ==========================================
        # NORMALIZED ACTION -> ACTUATOR CONTROL
        # ==========================================

        for i in range(self.model.nu):

            minimum = self.model.actuator_ctrlrange[
                i, 0
            ]

            maximum = self.model.actuator_ctrlrange[
                i, 1
            ]

            self.data.ctrl[i] = (
                minimum
                + (action[i] + 1.0)
                * 0.5
                * (maximum - minimum)
            )

        # ==========================================
        # PHYSICS
        # ==========================================

        for _ in range(self.frame_skip):

            mujoco.mj_step(
                self.model,
                self.data
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

        joint_positions = []

        for name in self.joint_names:

            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name
            )

            qpos_addr = self.model.jnt_qposadr[
                joint_id
            ]

            joint_positions.append(
                self.data.qpos[qpos_addr]
            )

        # ==========================================
        # JOINT VELOCITIES
        # ==========================================

        joint_velocities = []

        for name in self.joint_names:

            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name
            )

            dof_addr = self.model.jnt_dofadr[
                joint_id
            ]

            joint_velocities.append(
                self.data.qvel[dof_addr]
            )

        # ==========================================
        # COMBINE
        # ==========================================

        observation = np.concatenate([
            quat,
            angular_velocity,
            np.asarray(joint_positions),
            np.asarray(joint_velocities)
        ])

        return observation.astype(
            np.float32
        )

    # ==================================================
    # REWARD
    # ==================================================

    def _get_reward(self):

        # ==========================================
        # BODY HEIGHT
        # ==========================================

        height = self.data.qpos[2]

        target_height = 0.45

        height_error = abs(
            height - target_height
        )

        height_reward = np.exp(
            -20.0 * height_error ** 2
        )

        # ==========================================
        # UPRIGHTNESS
        # ==========================================

        quat = self.data.qpos[3:7]

        upright_reward = quat[0] ** 2

        # ==========================================
        # JOINT VELOCITY PENALTY
        # ==========================================

        joint_velocity_penalty = (
            0.0005
            * np.sum(
                self.data.qvel[6:] ** 2
            )
        )

        # ==========================================
        # ACTION PENALTY
        # ==========================================

        action_penalty = (
            0.001
            * np.sum(
                self.data.ctrl ** 2
            )
        )

        # ==========================================
        # ALIVE BONUS
        # ==========================================

        alive_bonus = 1.0

        # ==========================================
        # TOTAL
        # ==========================================

        reward = (
            3.0 * height_reward
            + 5.0 * upright_reward
            + alive_bonus
            - joint_velocity_penalty
            - action_penalty
        )

        return float(reward)

    # ==================================================
    # FALL DETECTION
    # ==================================================

    def _is_fallen(self):

        height = self.data.qpos[2]

        if height < 0.20:

            return True

        quat = self.data.qpos[3:7]

        if quat[0] ** 2 < 0.30:

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