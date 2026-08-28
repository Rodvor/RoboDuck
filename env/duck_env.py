import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np


class DuckEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):

        super().__init__()

        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(
            "duck/microduck/robot_walk.xml"
        )

        self.data = mujoco.MjData(self.model)

        # ==============================================
        # 14 SERVO ACTIONS
        # ==============================================

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(14,),
            dtype=np.float32
        )

        # ==============================================
        # OBSERVATION
        # ==============================================

        observation_size = self.model.nq + self.model.nv

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float32
        )

        self.frame_skip = 5

        self.max_episode_steps = 1000
        self.step_count = 0

        # ==============================================
        # VIEWER
        # ==============================================

        self.viewer = None

    # ==============================================
    # RESET
    # ==============================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        mujoco.mj_resetData(
            self.model,
            self.data
        )

        self.step_count = 0

        # Starting position
        self.data.qpos[0] = 0
        self.data.qpos[1] = 0
        self.data.qpos[2] = 0.45

        # Upright quaternion
        self.data.qpos[3:7] = np.array([
            1.0,
            0.0,
            0.0,
            0.0
        ])

        # Start joints close to neutral
        self.data.qpos[7:] = 0.0

        # Zero velocities
        self.data.qvel[:] = 0.0

        mujoco.mj_forward(
            self.model,
            self.data
        )

        return self._get_observation(), {}

    # ==============================================
    # STEP
    # ==============================================

    def step(self, action):

        self.step_count += 1

        action = np.clip(
            action,
            -1.0,
            1.0
        )

        # Scale action
        self.data.ctrl[:] = action * 0.5

        for _ in range(self.frame_skip):

            mujoco.mj_step(
                self.model,
                self.data
            )

        observation = self._get_observation()

        reward = self._get_reward()

        terminated = self._is_fallen()

        truncated = (
            self.step_count >=
            self.max_episode_steps
        )

        return (
            observation,
            reward,
            terminated,
            truncated,
            {}
        )

    # ==============================================
    # OBSERVATION
    # ==============================================

    def _get_observation(self):

        qpos = self.data.qpos.copy()

        qvel = self.data.qvel.copy()

        return np.concatenate([
            qpos,
            qvel
        ]).astype(np.float32)

    # ==============================================
    # REWARD
    # ==============================================

    def _get_reward(self):

        height = self.data.qpos[2]

        # ------------------------------------------
        # Height
        # ------------------------------------------

        height_reward = np.clip(
            height / 0.45,
            0,
            1
        )

        # ------------------------------------------
        # Upright orientation
        # ------------------------------------------

        quat = self.data.qpos[3:7]

        # For an upright quaternion, w should be near 1
        upright_reward = quat[0] ** 2

        # ------------------------------------------
        # Penalize movement
        # ------------------------------------------

        velocity_penalty = (
            0.0005 *
            np.sum(
                self.data.qvel ** 2
            )
        )

        # ------------------------------------------
        # Penalize large actions
        # ------------------------------------------

        action_penalty = (
            0.001 *
            np.sum(
                self.data.ctrl ** 2
            )
        )

        reward = (
            2.0 * height_reward
            + 2.0 * upright_reward
            - velocity_penalty
            - action_penalty
        )

        return float(reward)

    # ==============================================
    # FALL DETECTION
    # ==============================================

    def _is_fallen(self):

        height = self.data.qpos[2]

        if height < 0.20:
            return True

        # Quaternion
        quat = self.data.qpos[3:7]

        # If the body is no longer upright
        if quat[0] ** 2 < 0.30:

            return True

        return False

    # ==============================================
    # RENDER
    # ==============================================

    def render(self):

        if self.render_mode != "human":
            return

        if self.viewer is None:

            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data
            )

        self.viewer.sync()

    # ==============================================
    # CLOSE
    # ==============================================

    def close(self):

        if self.viewer is not None:

            self.viewer.close()

            self.viewer = None