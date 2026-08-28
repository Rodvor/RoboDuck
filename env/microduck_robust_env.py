import math

import mujoco
import numpy as np
from gymnasium import spaces

from env.microduck_walk_env import MicroDuckWalkEnv


class MicroDuckRobustEnv(MicroDuckWalkEnv):
    """Commanded walking with a staged robustness curriculum."""

    MODEL_XML = "scene_robust.xml"

    def __init__(self, render_mode=None, curriculum_horizon=312_500, evaluation=False):
        super().__init__(render_mode=render_mode)
        self.curriculum_horizon = max(1, int(curriculum_horizon))
        self.evaluation = evaluation
        self.curriculum_step = 0
        self.target_speed = 0.0
        self._push_steps_remaining = 0
        self._next_push_step = 10**9

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
        if progress >= 0.20:
            self._next_push_step = self.np_random.integers(150, 350)
        else:
            self._next_push_step = 10**9

    def reset(self, seed=None, options=None):
        # Configure model-level randomization before reset/forward dynamics.
        if not hasattr(self, "terrain_hfield_id"):
            return super().reset(seed=seed, options=options)
        self._configure_episode()
        observation, info = super().reset(seed=seed, options=options)
        # Terrain height at the spawn is centered around zero; slightly more
        # clearance avoids initial penetration on the largest bumps.
        self.data.qpos[2] += 0.008 * self.curriculum_progress
        mujoco.mj_forward(self.model, self.data)
        observation = self._get_observation()
        info.update({
            "target_speed": self.target_speed,
            "curriculum_progress": self.curriculum_progress,
        })
        return observation, info

    def _apply_push_if_due(self):
        progress = 1.0 if self.evaluation else self.curriculum_progress
        if self.step_count != self._next_push_step:
            return
        maximum_push = 0.03 + 0.12 * progress
        self.data.qvel[0] += self.np_random.uniform(-maximum_push, maximum_push)
        self.data.qvel[1] += self.np_random.uniform(-maximum_push, maximum_push)
        self._next_push_step += int(self.np_random.integers(150, 350))

    def step(self, action):
        self._apply_push_if_due()
        result = super().step(action)
        self.curriculum_step += 1
        observation, reward, terminated, truncated, info = result
        info.update({
            "target_speed": self.target_speed,
            "curriculum_progress": self.curriculum_progress,
        })
        return observation, reward, terminated, truncated, info

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

        reward = (
            6.0 * command_reward
            + 12.0 * progress_reward
            + 2.0 * touchdown_reward
            + 1.0 * upright
            + 0.25 * height_reward
            - 1.5 * vy ** 2
            - 0.20 * yaw_rate ** 2
            - 0.04 * action_rate
            - 0.01 * np.sum(self.last_action ** 2)
        )
        return float(0.10 * reward)

