import mujoco
import mujoco.viewer
import time
import numpy as np


MODEL_PATH = "duck/microduck/robot_allcollisions.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)


# ==========================================
# MICRODUCK INITIAL POSE
# ==========================================

# Use the XML's original body position.
data.qpos[0:3] = [0, 0, 0.3]

# Upright quaternion
data.qpos[3:7] = [1, 0, 0, 0]

# ==========================================
# JOINT POSITIONS
# ==========================================
#
# Order:
#
# 0 left_hip_yaw
# 1 left_hip_roll
# 2 left_hip_pitch
# 3 left_knee
# 4 left_ankle
#
# 5 neck_pitch
# 6 head_pitch
# 7 head_yaw
# 8 head_roll
#
# 9 right_hip_yaw
# 10 right_hip_roll
# 11 right_hip_pitch
# 12 right_knee
# 13 right_ankle
#
# ==========================================


joint_pose = np.zeros(14)


# Legs
joint_pose[0] = 0.0
joint_pose[1] = 0.0
joint_pose[2] = 0.0
joint_pose[3] = 0.0
joint_pose[4] = 0.0

joint_pose[9] = 0.0
joint_pose[10] = 0.0
joint_pose[11] = 0.0
joint_pose[12] = 0.0
joint_pose[13] = 0.0


# ==========================================
# APPLY POSE
# ==========================================

# qpos[7:] contains the joint positions
#
# However, MuJoCo joint addresses are safer
# to use explicitly.

actuator_joint_names = [
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


for i, name in enumerate(actuator_joint_names):

    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name
    )

    qpos_addr = model.jnt_qposadr[joint_id]

    data.qpos[qpos_addr] = joint_pose[i]


data.qvel[:] = 0

mujoco.mj_forward(model, data)


# ==========================================
# VIEWER
# ==========================================

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    print()
    print("MicroDuck pose test")
    print("===================")
    print()
    print("Current joint pose:")
    print(joint_pose)
    print()
    print("Close the viewer when finished.")

    while viewer.is_running():
        viewer.sync()

        time.sleep(0.01)