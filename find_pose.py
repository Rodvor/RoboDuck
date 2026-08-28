import mujoco
import numpy as np

MODEL_PATH = "duck/microduck/robot_allcollisions.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)

print("========================================")
print("       MICRODUCK JOINT INFORMATION")
print("========================================")
print()

for i in range(model.njnt):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        i
    )

    qpos_addr = model.jnt_qposadr[i]

    print(
        f"{i:2d}  "
        f"{name:25s} "
        f"qpos address = {qpos_addr}"
    )

print()
print("Joint ranges:")
print("----------------------------------------")

for i in range(model.njnt):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        i
    )

    if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:

        minimum = model.jnt_range[i, 0]
        maximum = model.jnt_range[i, 1]

        print(
            f"{name:25s} "
            f"{minimum:8.3f} "
            f"to "
            f"{maximum:8.3f}"
        )
