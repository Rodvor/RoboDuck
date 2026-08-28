import mujoco
import mujoco.viewer
import time

xml = """
<mujoco>
    <worldbody>
        <body name="box" pos="0 0 1">
            <freejoint/>
            <geom type="box" size=".2 .2 .2" mass="1"/>
        </body>

        <geom type="plane"
              size="5 5 .1"
              rgba=".8 .8 .8 1"/>
    </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.002)