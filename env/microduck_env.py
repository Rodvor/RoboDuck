import gymnasium as gym
from gymnasium import spaces

import mujoco
import mujoco.viewer

import numpy as np


class MicroDuckEnv(gym.Env):

    # Stable pose authored with the robot model (actuator order). The policy
    # commands small offsets around this pose instead of the actuator's very
    # broad +/-10 rad electrical control range.
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

        # Resolve model metadata once. Name lookup is surprisingly expensive
        # when repeated for every observation and reward calculation.
        joint_ids = np.asarray([
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            for name in self.joint_names
        ])
        self._joint_qpos_adr = self.model.jnt_qposadr[joint_ids].copy()
        self._joint_dof_adr = self.model.jnt_dofadr[joint_ids].copy()
        self._head_qpos_adr = self._joint_qpos_adr[5:9]
        self._head_dof_adr = self._joint_dof_adr[5:9]

        self._left_foot_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision"
        )
        self._right_foot_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision"
        )
        self._floor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )

        self._joint_lower = self.model.jnt_range[joint_ids, 0].copy()
        self._joint_upper = self.model.jnt_range[joint_ids, 1].copy()
        self._last_action = np.zeros(14, dtype=np.float64)

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
        # linear velocity  3
        # joint positions 14
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

        # ==========================================
        # SMALL RANDOM BODY TILT
        # ==========================================

        roll = self.np_random.uniform(-0.03, 0.03)
        pitch = self.np_random.uniform(-0.03, 0.03)
        yaw = self.np_random.uniform(-0.02, 0.02)

        quat = np.array([
            1.0,
            roll / 2.0,
            pitch / 2.0,
            yaw / 2.0
        ])

        quat = quat / np.linalg.norm(quat)

        self.data.qpos[3:7] = quat

        # ==========================================
        # INITIAL JOINT POSE
        #
        # Replace these values later with the
        # exact good standing pose.
        # ==========================================

        # ==========================================
        # RANDOMIZED INITIAL POSE
        # ==========================================

        # Start near a physically meaningful stance. A small perturbation still
        # teaches recovery without making every early rollout an instant fall.
        standing_pose = self.STAND_POSE + self.np_random.uniform(
            low=-0.025,
            high=0.025,
            size=14
        )
        self.data.qpos[self._joint_qpos_adr] = standing_pose
        self.data.ctrl[:] = self.STAND_POSE
        self._last_action.fill(0.0)

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

        action = np.asarray(action, dtype=np.float32)

        # ==========================================
        # NORMALIZED ACTION -> ACTUATOR CONTROL
        # ==========================================

        # A normalized action is a local adjustment around the standing pose.
        # Mapping it to the raw +/-10 rad actuator range made random exploration
        # violently saturate every servo and prevented PPO from seeing useful
        # standing experience.
        np.clip(action, -1.0, 1.0, out=self._last_action)
        np.multiply(self._last_action, self.ACTION_SCALE, out=self.data.ctrl)
        self.data.ctrl += self.STAND_POSE
        np.clip(
            self.data.ctrl,
            self._joint_lower,
            self._joint_upper,
            out=self.data.ctrl,
        )

        # ==========================================
        # PHYSICS
        # ==========================================

        # MuJoCo performs the loop in native code, avoiding five Python/C
        # boundary crossings per environment step.
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

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
        # BODY LINEAR VELOCITY
        # ==========================================

        linear_velocity = self.data.qvel[0:3]

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
            self.data.qpos[self._joint_qpos_adr] - self.STAND_POSE
        )

        # ==========================================
        # JOINT VELOCITIES
        # ==========================================

        joint_velocities = self.data.qvel[self._joint_dof_adr]

        # ==========================================
        # COMBINE
        # ==========================================

        observation = np.concatenate([
            quat,
            angular_velocity,
            linear_velocity,
            joint_positions,
            joint_velocities
        ], dtype=np.float32)

        return observation

    # ==================================================
    # REWARD
    # ==================================================

    def _get_reward(self):

        # ==========================================
        # 1. TORSO UPRIGHTNESS
        # ==========================================

        quat = self.data.qpos[3:7]

        # Z component of the local up vector for a MuJoCo wxyz quaternion.
        up_z = 1.0 - 2.0 * (quat[1] * quat[1] + quat[2] * quat[2])

        # 1.0 = perfectly upright
        # 0.0 = sideways
        # -1.0 = upside down
        upright = np.clip(up_z, -1.0, 1.0)

        upright_reward = max(0.0, upright)

        # ==========================================
        # 2. BODY HEIGHT
        # ==========================================

        height = self.data.qpos[2]

        target_height = 0.115

        height_error = height - target_height

        height_reward = np.exp(
            -150.0 * height_error ** 2
        )

        # ==========================================
        # 3. FOOT CONTACT
        # ==========================================

        left_contact = False
        right_contact = False

        for i in range(self.data.ncon):

            contact = self.data.contact[i]

            g1 = contact.geom1
            g2 = contact.geom2

            # Left foot touching floor
            if (
                    (g1 == self._left_foot_id and g2 == self._floor_id)
                    or
                    (g2 == self._left_foot_id and g1 == self._floor_id)
            ):
                left_contact = True

            # Right foot touching floor
            if (
                    (g1 == self._right_foot_id and g2 == self._floor_id)
                    or
                    (g2 == self._right_foot_id and g1 == self._floor_id)
            ):
                right_contact = True

        foot_contact_reward = (
                float(left_contact)
                + float(right_contact)
        )

        # ==========================================
        # 4. BODY ANGULAR VELOCITY
        # ==========================================

        angular_velocity = self.data.qvel[3:6]

        angular_velocity_penalty = (
                0.10 *
                np.sum(angular_velocity ** 2)
        )

        # ==========================================
        # 5. JOINT VELOCITY
        # ==========================================

        joint_velocity_penalty = (
                0.0005 *
                np.sum(self.data.qvel[6:] ** 2)
        )

        # ==========================================
        # 6. HEAD/NECK STABILITY
        # ==========================================

        head_joint_positions = (
            self.data.qpos[self._head_qpos_adr] - self.STAND_POSE[5:9]
        )
        head_joint_velocities = self.data.qvel[self._head_dof_adr]

        # A Gaussian gives the policy a strong, smooth incentive to keep all
        # four head joints near the neutral standing pose.
        head_pose_reward = np.exp(
            -8.0 * np.sum(head_joint_positions ** 2)
        )

        head_velocity_penalty = (
            0.01 * np.sum(head_joint_velocities ** 2)
        )

        # Upright + height alone permits awkward, splayed-leg solutions. This
        # broad Gaussian favors the authored whole-body stance while still
        # leaving enough freedom for active balance corrections.
        joint_pose_error = (
            self.data.qpos[self._joint_qpos_adr] - self.STAND_POSE
        )
        stand_pose_reward = np.exp(-2.0 * np.sum(joint_pose_error ** 2))

        # ==========================================
        # 7. CONTROL EFFORT
        # ==========================================

        # Penalize policy offsets, not absolute servo targets: the authored
        # standing pose itself contains non-zero commands.
        control_penalty = 0.02 * np.sum(self._last_action ** 2)

        # ==========================================
        # 8. ALIVE BONUS
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

        # Keep critic targets in a numerically comfortable range. PPO
        # normalizes advantages, so this preserves the reward trade-offs while
        # avoiding value losses in the tens of thousands.
        return float(0.05 * reward)


    # ==================================================
    # FALL DETECTION
    # ==================================================

    def _is_fallen(self):

        height = self.data.qpos[2]

        if height < 0.055:
            return True

        quat = self.data.qpos[3:7]

        up_z = 1.0 - 2.0 * (quat[1] * quat[1] + quat[2] * quat[2])

        # Terminate if torso tilts more than about 60 degrees.
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
