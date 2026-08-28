import math

import mujoco
import numpy as np
from gymnasium import spaces

from env.microduck_robust_env import MicroDuckRobustEnv


class MicroDuckGoalRobustEnv(MicroDuckRobustEnv):
    """Robust locomotion toward automatically sampled local waypoints."""

    GOAL_REACHED_DISTANCE = 0.08
    GOAL_NORMALIZATION_DISTANCE = 1.0
    FULL_RECOVERY_GRACE_STEPS = 400

    def __init__(
        self,
        render_mode=None,
        curriculum_horizon=12_500_000,
        curriculum_start_fraction=0.0,
        evaluation=False,
    ):
        # These fields are needed by the dynamically dispatched observation
        # method during the first reset in the parent environment.
        self.goal_world = np.zeros(2, dtype=np.float64)
        self._previous_goal_distance = 0.0
        self._goal_reached_pending = False
        self._goals_reached = 0
        self._push_enabled = False
        self._started_tipped = False
        self._was_recovering = False
        self._upright_recovery_streak = 0

        super().__init__(
            render_mode=render_mode,
            curriculum_horizon=curriculum_horizon,
            curriculum_start_fraction=curriculum_start_fraction,
            evaluation=evaluation,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(41,),
            dtype=np.float32,
        )

    def _curriculum_goal_limits(self):
        progress = 1.0 if self.evaluation else self.curriculum_progress
        if progress < 0.15:
            return math.radians(10.0), 0.70
        if progress < 0.50:
            blend = (progress - 0.15) / 0.35
            return math.radians(10.0 + 20.0 * blend), 0.70 + 0.20 * blend
        if progress < 0.80:
            blend = (progress - 0.50) / 0.30
            return math.radians(30.0 + 20.0 * blend), 0.90 + 0.10 * blend
        return math.radians(50.0), 1.00

    def _configure_episode(self):
        super()._configure_episode()
        progress = 1.0 if self.evaluation else self.curriculum_progress

        # Preserve stop-command training without allowing it to dominate the
        # navigation experience.
        if self.np_random.random() < 0.20:
            self.target_speed = 0.0
        else:
            maximum_speed = 0.10 + 0.10 * progress
            self.target_speed = self.np_random.uniform(0.06, maximum_speed)

        # Push-free episodes remain common even at maximum difficulty. This
        # prevents clean steering from being forgotten while still exposing
        # most episodes to randomly timed, directed and sized disturbances.
        push_probability = 0.70 * np.clip((progress - 0.15) / 0.35, 0.0, 1.0)
        self._push_enabled = self.np_random.random() < push_probability
        if not self._push_enabled:
            self._next_push_step = 10**9

    @staticmethod
    def _yaw_from_quaternion(quaternion):
        w, x, y, z = quaternion
        sin_yaw = 2.0 * (w * z + x * y)
        cos_yaw = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(sin_yaw, cos_yaw)

    def _goal_delta_world(self):
        return self.goal_world - self.data.qpos[:2]

    def _goal_distance(self):
        return float(np.linalg.norm(self._goal_delta_world()))

    def _goal_local_vector(self):
        delta = self._goal_delta_world()
        quaternion = self.data.qpos[3:7]
        if np.linalg.norm(quaternion) < 0.5:
            return delta
        yaw = self._yaw_from_quaternion(quaternion)
        cosine, sine = math.cos(yaw), math.sin(yaw)
        return np.array([
            cosine * delta[0] + sine * delta[1],
            -sine * delta[0] + cosine * delta[1],
        ])

    def _get_observation(self):
        base = super()._get_observation()
        local_goal = np.clip(
            self._goal_local_vector() / self.GOAL_NORMALIZATION_DISTANCE,
            -2.0,
            2.0,
        ).astype(np.float32)
        return np.concatenate([base, local_goal], dtype=np.float32)

    def _spawn_goal(self):
        maximum_bearing, maximum_distance = self._curriculum_goal_limits()
        bearing = self.np_random.uniform(-maximum_bearing, maximum_bearing)
        distance = self.np_random.uniform(0.35, maximum_distance)
        yaw = self._yaw_from_quaternion(self.data.qpos[3:7])
        world_angle = yaw + bearing
        self.goal_world[:] = self.data.qpos[:2] + distance * np.array([
            math.cos(world_angle),
            math.sin(world_angle),
        ])
        self._previous_goal_distance = distance

    def _apply_tipped_start_if_due(self):
        progress = 1.0 if self.evaluation else self.curriculum_progress
        tipped_probability = 0.10 * np.clip((progress - 0.80) / 0.20, 0.0, 1.0)
        self._started_tipped = self.np_random.random() < tipped_probability
        if not self._started_tipped:
            return

        direction = int(self.np_random.integers(4))
        angle = math.radians(self.np_random.uniform(75.0, 82.0))
        orientations = (
            (0.0, angle, 0.0),
            (0.0, -angle, 0.0),
            (angle, 0.0, 0.0),
            (-angle, 0.0, 0.0),
        )
        self.data.qpos[2] = 0.08
        self.data.qpos[3:7] = self._euler_to_quaternion(*orientations[direction])
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self._apply_tipped_start_if_due()
        self._spawn_goal()
        self._goal_reached_pending = False
        self._goals_reached = 0
        self._fall_steps = 0
        self._upright_recovery_streak = 0
        self._was_recovering = self._needs_recovery()
        self._previous_recovery_potential = self._recovery_potential()
        observation = self._get_observation()
        info.update({
            "goal_distance": self._goal_distance(),
            "goals_reached": self._goals_reached,
            "push_enabled": self._push_enabled,
            "started_tipped": self._started_tipped,
        })
        return observation, info

    def _needs_recovery(self):
        quaternion = self.data.qpos[3:7]
        up_z = 1.0 - 2.0 * (quaternion[1] ** 2 + quaternion[2] ** 2)
        return self.data.qpos[2] < 0.075 or up_z < 0.45

    def _get_reward(self):
        quaternion = self.data.qpos[3:7]
        up_z = 1.0 - 2.0 * (quaternion[1] ** 2 + quaternion[2] ** 2)
        upright = float(np.clip(up_z, 0.0, 1.0))
        height = float(self.data.qpos[2])
        yaw_rate = float(self.data.qvel[5])

        goal_delta = self._goal_delta_world()
        goal_distance = float(np.linalg.norm(goal_delta))
        if goal_distance > 1e-6:
            goal_direction = goal_delta / goal_distance
        else:
            goal_direction = np.array([1.0, 0.0])
        lateral_direction = np.array([-goal_direction[1], goal_direction[0]])
        horizontal_velocity = self.data.qvel[:2]
        toward_speed = float(np.dot(horizontal_velocity, goal_direction))
        lateral_speed = float(np.dot(horizontal_velocity, lateral_direction))

        stopped = self.target_speed < 0.01
        if stopped:
            velocity_error = float(np.linalg.norm(horizontal_velocity))
        else:
            velocity_error = toward_speed - self.target_speed
        velocity_tracking = math.exp(-(velocity_error / 0.07) ** 2)
        command_reward = velocity_tracking * upright
        step_dt = self.frame_skip * self.model.opt.timestep
        distance_progress = np.clip(
            (self._previous_goal_distance - goal_distance) / step_dt,
            -0.30,
            0.30,
        )
        self._previous_goal_distance = goal_distance
        progress_reward = 0.0 if stopped else distance_progress * upright

        local_goal = self._goal_local_vector()
        heading_alignment = (
            float(np.clip(local_goal[0] / goal_distance, -1.0, 1.0))
            if goal_distance > 1e-6
            else 1.0
        )

        contacts = self._foot_contacts()
        self._air_time[~contacts] += step_dt
        touchdown = contacts & ~self._was_contact
        touchdown_reward = 0.0 if stopped else np.sum(
            touchdown * np.exp(-((self._air_time - 0.18) / 0.09) ** 2)
        )
        self._air_time[contacts] = 0.0
        self._was_contact[:] = contacts

        action_rate = np.sum((self.last_action - self._previous_action) ** 2)
        self._previous_action[:] = self.last_action
        height_reward = math.exp(-150.0 * (height - 0.115) ** 2)

        recovery_potential = self._recovery_potential()
        recovery_progress = np.clip(
            recovery_potential - self._previous_recovery_potential,
            -0.20,
            0.20,
        )
        self._previous_recovery_potential = recovery_potential
        posture_deficit = max(0.0, 0.60 - up_z) + 8.0 * max(0.0, 0.075 - height)

        recovering = self._needs_recovery()
        recovery_bonus = 0.0
        if recovering:
            self._was_recovering = True
            self._upright_recovery_streak = 0
        elif self._was_recovering:
            self._upright_recovery_streak += 1
            if self._upright_recovery_streak >= 25:
                recovery_bonus = 5.0
                self._was_recovering = False
                self._upright_recovery_streak = 0

        self._goal_reached_pending = (
            not stopped
            and goal_distance <= self.GOAL_REACHED_DISTANCE
            and up_z >= 0.75
            and height >= 0.09
        )
        arrival_bonus = 8.0 if self._goal_reached_pending else 0.0

        reward = (
            6.0 * command_reward
            + 12.0 * progress_reward
            + (0.0 if stopped else 1.0 * upright * heading_alignment)
            + 2.0 * touchdown_reward
            + 0.25 * height_reward
            + 8.0 * recovery_progress
            + recovery_bonus
            + arrival_bonus
            - 2.0 * posture_deficit
            - 1.5 * lateral_speed ** 2
            - 0.20 * yaw_rate ** 2
            - 0.04 * action_rate
            - 0.01 * np.sum(self.last_action ** 2)
        )
        return float(0.10 * reward)

    def _is_fallen(self):
        progress = 1.0 if self.evaluation else self.curriculum_progress
        if progress < 0.15:
            return super()._is_fallen()

        if self._needs_recovery():
            self._fall_steps += 1
        else:
            self._fall_steps = max(0, self._fall_steps - 5)
        return self._fall_steps >= self.FULL_RECOVERY_GRACE_STEPS

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        reached_goal = self._goal_reached_pending and not terminated
        if reached_goal:
            self._goals_reached += 1
            self._spawn_goal()
            self._goal_reached_pending = False
            observation = self._get_observation()

        info.update({
            "goal_distance": self._goal_distance(),
            "goals_reached": self._goals_reached,
            "reached_goal": reached_goal,
            "push_enabled": self._push_enabled,
            "started_tipped": self._started_tipped,
        })
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        if self.viewer is None:
            super().render()
        if self.viewer is not None and self.viewer.user_scn is not None:
            scene = self.viewer.user_scn
            scene.ngeom = 1
            mujoco.mjv_initGeom(
                scene.geoms[0],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.03, 0.03, 0.03]),
                np.array([self.goal_world[0], self.goal_world[1], 0.03]),
                np.eye(3).ravel(),
                np.array([0.15, 0.95, 0.25, 0.85], dtype=np.float32),
            )
        super().render()
