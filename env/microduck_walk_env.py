import numpy as np

from env.microduck_env import MicroDuckEnv


class MicroDuckWalkEnv(MicroDuckEnv):
    """Forward walking task built on the proven standing controller contract."""

    TARGET_SPEED = 0.15  # m/s; brisk but realistic for a 25 cm robot
    ACTION_SCALE = 1.0

    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self._previous_action = np.zeros(14, dtype=np.float64)
        self._air_time = np.zeros(2, dtype=np.float64)
        self._was_contact = np.ones(2, dtype=bool)

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self._previous_action.fill(0.0)
        self._air_time.fill(0.0)
        self._was_contact.fill(True)
        return observation, info

    def _foot_contacts(self):
        contacts = np.zeros(2, dtype=bool)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            if g1 == self.floor_geom_id:
                other = g2
            elif g2 == self.floor_geom_id:
                other = g1
            else:
                continue
            contacts[0] |= other == self.left_foot_geom_id
            contacts[1] |= other == self.right_foot_geom_id
        return contacts

    def _get_reward(self):
        quat = self.data.qpos[3:7]
        up_z = 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)
        upright = np.clip(up_z, 0.0, 1.0)

        vx, vy = self.data.qvel[0:2]
        yaw_rate = self.data.qvel[5]
        height_error = self.data.qpos[2] - 0.115

        # A narrow velocity target prevents both standing still and learning to
        # launch itself forward. Multiplication by upright makes falling an
        # unusable way to collect forward-speed reward.
        speed_tracking = np.exp(-((vx - self.TARGET_SPEED) / 0.06) ** 2)
        walking_reward = upright * speed_tracking

        # Potential-like progress keeps a useful gradient when velocity is far
        # from the target and explicitly makes backward travel undesirable.
        forward_progress = np.clip(vx, -0.30, 0.30)

        height_reward = np.exp(-150.0 * height_error ** 2)
        joint_pose_error = (
            self.data.qpos[self.joint_qpos_addresses] - self.STAND_POSE
        )
        loose_pose_reward = np.exp(-0.75 * np.sum(joint_pose_error ** 2))

        action_rate = np.sum((self.last_action - self._previous_action) ** 2)
        effort = np.sum(self.last_action ** 2)
        joint_speed = np.sum(self.data.qvel[self.joint_dof_addresses] ** 2)

        contacts = self._foot_contacts()
        step_dt = self.frame_skip * self.model.opt.timestep
        self._air_time[~contacts] += step_dt
        touchdown = contacts & ~self._was_contact
        # Reward useful swing durations only; holding a foot in the air forever
        # pays nothing, and two-foot jumping gets no clearance reward.
        touchdown_reward = np.sum(
            touchdown * np.exp(-((self._air_time - 0.18) / 0.09) ** 2)
        )
        foot_z = np.array([
            self.data.geom_xpos[self.left_foot_geom_id, 2],
            self.data.geom_xpos[self.right_foot_geom_id, 2],
        ])
        single_support = contacts[0] ^ contacts[1]
        swing_index = 1 if contacts[0] else 0
        clearance_reward = (
            np.exp(-((foot_z[swing_index] - 0.025) / 0.015) ** 2)
            if single_support else 0.0
        )
        self._air_time[contacts] = 0.0
        self._was_contact[:] = contacts

        reward = (
            # Standing still earns only a small balance baseline. Reaching the
            # requested speed is worth several times more, preventing the
            # standing policy from remaining a locally optimal solution.
            5.0 * walking_reward
            + 15.0 * forward_progress
            + 2.0 * touchdown_reward
            + 0.5 * clearance_reward
            + 1.0 * upright
            + 0.25 * height_reward
            + 0.10 * loose_pose_reward
            - 1.0 * vy ** 2
            - 0.15 * yaw_rate ** 2
            - 0.04 * action_rate
            - 0.01 * effort
            - 0.0002 * joint_speed
        )

        self._previous_action[:] = self.last_action
        return float(0.10 * reward)
