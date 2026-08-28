import mujoco
import mujoco.viewer
import time


MODEL_PATH = "duck/microduck/robot_allcollisions.xml"


print("Loading official MicroDuck model...")
print(MODEL_PATH)

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print()
print("====================================")
print("       OFFICIAL MICRODUCK")
print("====================================")
print()

print("Bodies:     ", model.nbody)
print("Joints:     ", model.njnt)
print("qpos:       ", model.nq)
print("qvel:       ", model.nv)
print("Actuators:  ", model.nu)

print()
print("ACTUATORS")
print("------------------------------------")

for i in range(model.nu):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        i
    )

    print(f"{i:2d}: {name}")


print()
print("Starting viewer...")


with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    while viewer.is_running():

        mujoco.mj_step(
            model,
            data
        )

        viewer.sync()

        time.sleep(0.002)
