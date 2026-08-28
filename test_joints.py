import mujoco
import mujoco.viewer
import time
import numpy as np

# Load MicroDuck model
model = mujoco.MjModel.from_xml_path("duck/duck.xml")
data = mujoco.MjData(model)

# Order must match the XML actuator order
joints = [
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

print("===================================")
print("        MICRODUCK JOINT TEST")
print("===================================")
print(f"Actuators: {model.nu}")
print()

for i, name in enumerate(joints):
    print(f"{i:2d}  {name}")

print()
print("Starting joint test...")
print("Each joint will move back and forth.")
print("Close the MuJoCo window to stop.")
print()

# Make sure the robot starts roughly upright
mujoco.mj_resetData(model, data)

# Give the duck a small initial height
data.qpos[2] = 0.45

mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        for actuator_id, joint_name in enumerate(joints):

            print(f"Testing {actuator_id}: {joint_name}")

            # Move positive
            for target in np.linspace(0.0, 0.3, 40):

                if not viewer.is_running():
                    break

                data.ctrl[:] = 0
                data.ctrl[actuator_id] = target

                mujoco.mj_step(model, data)
                viewer.sync()

                time.sleep(0.02)

            # Move negative
            for target in np.linspace(0.3, -0.3, 80):

                if not viewer.is_running():
                    break

                data.ctrl[:] = 0
                data.ctrl[actuator_id] = target

                mujoco.mj_step(model, data)
                viewer.sync()

                time.sleep(0.02)

            # Return to zero
            for target in np.linspace(-0.3, 0.0, 40):

                if not viewer.is_running():
                    break

                data.ctrl[:] = 0
                data.ctrl[actuator_id] = target

                mujoco.mj_step(model, data)
                viewer.sync()

                time.sleep(0.02)

            # Small pause
            data.ctrl[:] = 0

            for _ in range(50):

                if not viewer.is_running():
                    break

                mujoco.mj_step(model, data)
                viewer.sync()

                time.sleep(0.002)

        print("Test completed. Restarting...\n")