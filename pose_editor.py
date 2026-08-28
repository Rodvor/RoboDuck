import mujoco
import mujoco.viewer
import time

MODEL_PATH = "duck/microduck/robot_allcollisions.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)

# Put robot above the ground
data.qpos[2] = 0.50

mujoco.mj_forward(model, data)

print("MicroDuck pose editor")
print()
print("Use the MuJoCo GUI to inspect the robot.")
print("Close the window when finished.")

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    while viewer.is_running():

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(0.002)
