import mujoco

MODEL_PATH = "duck/microduck/robot_allcollisions.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)

print("========================================")
print("        MICRODUCK MODEL INFO")
print("========================================")

print(f"qpos:      {model.nq}")
print(f"qvel:      {model.nv}")
print(f"joints:    {model.njnt}")
print(f"actuators: {model.nu}")

print()
print("ACTUATORS")
print("----------------------------------------")

for i in range(model.nu):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        i
    )

    joint_id = model.actuator_trnid[i, 0]

    joint_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_id
    )

    ctrl_min = model.actuator_ctrlrange[i, 0]
    ctrl_max = model.actuator_ctrlrange[i, 1]

    print(
        f"{i:2d}  "
        f"{name:25s} "
        f"joint={joint_name:20s} "
        f"ctrl=[{ctrl_min:8.3f}, {ctrl_max:8.3f}]"
    )

print()
print("JOINT LIMITS")
print("----------------------------------------")

for i in range(model.njnt):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        i
    )

    joint_type = model.jnt_type[i]

    if joint_type == mujoco.mjtJoint.mjJNT_HINGE:

        limited = model.jnt_limited[i]

        if limited:

            minimum = model.jnt_range[i, 0]
            maximum = model.jnt_range[i, 1]

            print(
                f"{i:2d}  "
                f"{name:25s} "
                f"range=[{minimum:8.3f}, {maximum:8.3f}]"
            )
