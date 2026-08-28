import math

import mujoco
import numpy as np
from gymnasium import spaces

from env.microduck_walk_env import MicroDuckWalkEnv


class MicroDuckRobustEnv(MicroDuckWalkEnv):
    """Commanded walking with a staged robustness curriculum."""

    MODEL_XML = "scene_robust.xml"
    RECOVERY_GRACE_STEPS = 250
    SEVERE_FALL_GRACE_STEPS = 100

    def __init__(
        self,
        render_mode=None,
        curriculum_horizon=312_500,
        curriculum_start_fraction=0.0,
        evaluation=False,
    ):
        super().__init__(render_mode=render_mode)
        self.curriculum_horizon = max(1, int(curriculum_horizon))
        self.evaluation = evaluation
        start_fraction = float(np.clip(curriculum_start_fraction, 0.0, 1.0))
        self.curriculum_step = int(start_fraction * self.curriculum_horizon)
        self.target_speed = 0.0
        self._push_steps_remaining = 0
        self._push_force = np.zeros(3, dtype=np.float64)
        self._next_push_step = 10**9
        self._fall_steps = 0
        self._previous_recovery_potential = 0.0

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(39,),
            dtype=np.float32,
        )

        self._original_floor_geom_id = self.floor_geom_id
        self.terrain_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "training_terrain_collision",
        )
        self.terrain_hfield_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_HFIELD,
            "training_terrain",
        )
        self.floor_geom_id = self.terrain_geom_id

        # The original plane remains as a safety floor below the heightfield.
        self.model.geom_pos[self._original_floor_geom_id, 2] = -0.05
        self._default_force_range = self.model.actuator_forcerange.copy()
        self._default_body_mass = self.model.body_mass.copy()
        self._terrain_base_quat = self.model.geom_quat[
            self.terrain_geom_id
        ].copy()

    @property
    def curriculum_progress(self):
        return min(1.0, self.curriculum_step / self.curriculum_horizon)

    def _get_observation(self):
        base = super()._get_observation()
        return np.concatenate(
            [base, np.asarray([self.target_speed], dtype=np.float32)],
            dtype=np.float32,
        )

    def _set_flat_terrain(self):
        address = self.model.hfield_adr[self.terrain_hfield_id]
        count = (
            self.model.hfield_nrow[self.terrain_hfield_id]
            * self.model.hfield_ncol[self.terrain_hfield_id]
        )
        self.model.hfield_data[address:address + count] = 0.5
        self.model.geom_quat[self.terrain_geom_id] = self._terrain_base_quat

    def _randomize_terrain(self, maximum_bump, maximum_slope_degrees):
        self._set_flat_terrain()
        if maximum_bump <= 0.0 and maximum_slope_degrees <= 0.0:
            return

        rows = self.model.hfield_nrow[self.terrain_hfield_id]
        columns = self.model.hfield_ncol[self.terrain_hfield_id]
        address = self.model.hfield_adr[self.terrain_hfield_id]
        x = np.linspace(-math.pi, math.pi, columns)
        y = np.linspace(-math.pi, math.pi, rows)[:, None]
        phase_x = self.np_random.uniform(-math.pi, math.pi)
        phase_y = self.np_random.uniform(-math.pi, math.pi)
        bumps = (
            np.sin(2.0 * x[None, :] + phase_x)
            + 0.5 * np.sin(5.0 * x[None, :] - phase_x)
            + 0.5 * np.sin(3.0 * y + phase_y)
        ) / 2.0
        height_scale = self.model.hfield_size[self.terrain_hfield_id, 2]
        normalized_amplitude = maximum_bump / height_scale
        values = np.clip(0.5 + normalized_amplitude * bumps, 0.0, 1.0)
        self.model.hfield_data[address:address + rows * columns] = values.ravel()

        slope = math.radians(
            self.np_random.uniform(
                -maximum_slope_degrees,
                maximum_slope_degrees,
            )
        )
        self.model.geom_quat[self.terrain_geom_id] = np.array([
            math.cos(slope / 2.0),
            0.0,
            math.sin(slope / 2.0),
            0.0,
        ])

    def _configure_episode(self):
        progress = 1.0 if self.evaluation else self.curriculum_progress

        # Exact zero commands are deliberately common; uniform sampling would
        # almost never teach deployment's most important stop command.
        if self.np_random.random() < 0.30:
            self.target_speed = 0.0
        else:
            maximum_speed = 0.10 + 0.10 * progress
            self.target_speed = self.np_random.uniform(0.04, maximum_speed)

        if progress < 0.40:
            bump, slope = 0.0, 0.0
        elif progress < 0.70:
            blend = (progress - 0.40) / 0.30
            bump, slope = 0.003 * blend, 2.0 * blend
        else:
            blend = (progress - 0.70) / 0.30
            bump, slope = 0.003 + 0.005 * blend, 2.0 + 4.0 * blend

        # Preserve clean flat experience throughout the curriculum.
        if self.np_random.random() < 0.35:
            bump, slope = 0.0, 0.0
        self._randomize_terrain(bump, slope)

        friction_spread = 0.0 if progress < 0.25 else 0.30 * progress
        friction = self.np_random.uniform(
            1.0 - friction_spread,
            1.0 + friction_spread,
        )
        self.model.geom_friction[self.terrain_geom_id, 0] = friction

        self.model.actuator_forcerange[:] = self._default_force_range
        self.model.body_mass[:] = self._default_body_mass
        if progress >= 0.50:
            motor_scale = self.np_random.uniform(0.85, 1.15)
            self.model.actuator_forcerange[:] *= motor_scale
            trunk_scale = self.np_random.uniform(0.95, 1.05)
            self.model.body_mass[self.trunk_body_id] *= trunk_scale

        self._push_steps_remaining = 0
        self._push_force.fill(0.0)
        if progress >= 0.20:
            self._next_push_step = int(self.np_random.integers(100, 300))
        else:
            self._next_push_step = 10**9

    def reset(self, seed=None, options=None):
        if not hasattr(self, "terrain_hfield_id"):
            return super().reset(seed=seed, options=options)
        # Let Gym establish the requested seed before sampling any curriculum
        # randomization, then update the model and run forward dynamics again.
        observation, info = super().reset(seed=seed, options=options)
        self._configure_episode()
        progress = 1.0 if self.evaluation else self.curriculum_progress

        # Terrain height at the spawn is centered around zero; slightly more
        # clearance avoids initial penetration on the largest bumps.
        self.data.qpos[2] += 0.008 * progress

        # Occasionally begin moderately off balance. These are catch/recovery
        # states, not fully prone get-up states, and become harder gradually.
        if progress >= 0.25 and self.np_random.random() < 0.30 * progress:
            maximum_tilt = 0.08 + 0.40 * progress
            roll = self.np_random.uniform(-maximum_tilt, maximum_tilt)
            pitch = self.np_random.uniform(-maximum_tilt, maximum_tilt)
            yaw = self.np_random.uniform(-0.04, 0.04)
            self.data.qpos[3:7] = self._euler_to_quaternion(roll, pitch, yaw)
            maximum_angular_speed = 0.25 + 1.25 * progress
            self.data.qvel[3:5] = self.np_random.uniform(
                -maximum_angular_speed,
                maximum_angular_speed,
                size=2,
            )

        self._fall_steps = 0
        self.data.xfrc_applied[self.trunk_body_id] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._previous_recovery_potential = self._recovery_potential()
        observation = self._get_observation()
        info.update({
            "target_speed": self.target_speed,
            "curriculum_progress": self.curriculum_progress,
        })
        return observation, info

    @staticmethod
    def _euler_to_quaternion(roll, pitch, yaw):
        cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
        cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
        cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
        return np.array([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ])

    def _apply_push_if_due(self):
        self.data.xfrc_applied[self.trunk_body_id] = 0.0
        progress = 1.0 if self.evaluation else self.curriculum_progress
        if self._push_steps_remaining <= 0 and self.step_count >= self._next_push_step:
            angle = self.np_random.uniform(-math.pi, math.pi)
            maximum_force = 0.40 + 1.60 * progress
            magnitude = self.np_random.uniform(0.40, 1.0) * maximum_force
            self._push_force[:] = (
                magnitude * math.cos(angle),
                magnitude * math.sin(angle),
                0.0,
            )
            self._push_steps_remaining = int(self.np_random.integers(5, 16))
            self._next_push_step += int(self.np_random.integers(100, 300))

        if self._push_steps_remaining > 0:
            self.data.xfrc_applied[self.trunk_body_id, :3] = self._push_force
            self._push_steps_remaining -= 1

    def step(self, action):
        self._apply_push_if_due()
        result = super().step(action)
        self.curriculum_step += 1
        observation, reward, terminated, truncated, info = result
        if terminated:
            reward -= 2.0
        info.update({
            "target_speed": self.target_speed,
            "curriculum_progress": self.curriculum_progress,
            "recovery_steps": self._fall_steps,
            "push_force": float(np.linalg.norm(self._push_force))
            if self._push_steps_remaining > 0 else 0.0,
        })
        return observation, reward, terminated, truncated, info

    def _recovery_potential(self):
        quat = self.data.qpos[3:7]
        up_z = 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)
        height = self.data.qpos[2]
        return 2.0 * np.clip(up_z, -1.0, 1.0) + 8.0 * np.clip(
            height,
            0.0,
            0.12,
        )

    def _get_reward(self):
        quat = self.data.qpos[3:7]
        up_z = 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)
        upright = np.clip(up_z, 0.0, 1.0)
        vx, vy = self.data.qvel[0:2]
        yaw_rate = self.data.qvel[5]

        velocity_error = vx - self.target_speed
        velocity_tracking = np.exp(-(velocity_error / 0.07) ** 2)
        stopped = self.target_speed < 0.01
        command_reward = velocity_tracking * upright
        progress_reward = 0.0 if stopped else np.clip(vx, -0.30, 0.30)

        contacts = self._foot_contacts()
        step_dt = self.frame_skip * self.model.opt.timestep
        self._air_time[~contacts] += step_dt
        touchdown = contacts & ~self._was_contact
        touchdown_reward = 0.0 if stopped else np.sum(
            touchdown * np.exp(-((self._air_time - 0.18) / 0.09) ** 2)
        )
        self._air_time[contacts] = 0.0
        self._was_contact[:] = contacts

        action_rate = np.sum((self.last_action - self._previous_action) ** 2)
        self._previous_action[:] = self.last_action
        height_reward = np.exp(-150.0 * (self.data.qpos[2] - 0.115) ** 2)

        recovery_potential = self._recovery_potential()
        recovery_progress = np.clip(
            recovery_potential - self._previous_recovery_potential,
            -0.20,
            0.20,
        )
        self._previous_recovery_potential = recovery_potential
        posture_deficit = (
            max(0.0, 0.60 - up_z)
            + 8.0 * max(0.0, 0.075 - self.data.qpos[2])
        )

        reward = (
            6.0 * command_reward
            + 12.0 * progress_reward
            + 2.0 * touchdown_reward
            + 1.0 * upright
            + 0.25 * height_reward
            + 8.0 * recovery_progress
            - 2.0 * posture_deficit
            - 1.5 * vy ** 2
            - 0.20 * yaw_rate ** 2
            - 0.04 * action_rate
            - 0.01 * np.sum(self.last_action ** 2)
        )
        return float(0.10 * reward)

    def _is_fallen(self):
        """Give the policy time to catch itself before ending the episode."""
        progress = 1.0 if self.evaluation else self.curriculum_progress
        if progress < 0.20:
            # Preserve the easier walking objective before recovery states are
            # introduced to a model transferred from the walking checkpoint.
            return super()._is_fallen()

        height = self.data.qpos[2]
        quat = self.data.qpos[3:7]
        up_z = 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)

        needs_recovery = height < 0.075 or up_z < 0.45
        if needs_recovery:
            self._fall_steps += 1
        else:
            # Require a sustained recovery, while forgiving brief threshold
            # crossings rather than resetting all progress immediately.
            self._fall_steps = max(0, self._fall_steps - 5)

        severe_fall = height < 0.025 or up_z < -0.50
        grace_steps = (
            self.SEVERE_FALL_GRACE_STEPS
            if severe_fall
            else self.RECOVERY_GRACE_STEPS
        )
        return self._fall_steps >= grace_steps
