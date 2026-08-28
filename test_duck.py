import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path("duck/duck.xml")
data = mujoco.MjData(model)

print("Model loaded successfully")
print("Joints:", model.njnt)
print("Actuators:", model.nu)

print("\nActuators:")
for i in range(model.nu):
    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        i
    )
    print(f"{i}: {name}")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.002)