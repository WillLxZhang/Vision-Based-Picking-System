"""MuJoCo scene construction and simulation access."""

from pathlib import Path
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PANDA_DIR = ROOT / "third_party" / "mujoco_menagerie" / "franka_emika_panda"


def panda_assets() -> dict[str, bytes]:
    """Return model meshes for MuJoCo's in-memory loader.

    The Windows XML path loader cannot reliably open this project's Chinese path.
    """
    return {
        f"assets/{mesh.name}": mesh.read_bytes()
        for mesh in (PANDA_DIR / "assets").iterdir()
        if mesh.is_file()
    }


def load_panda_model() -> mujoco.MjModel:
    xml = (PANDA_DIR / "panda.xml").read_text(encoding="utf-8")
    return mujoco.MjModel.from_xml_string(xml, assets=panda_assets())


PARTS = ("cube", "cylinder", "prism")
COLORS = {
    "cube": "0.93 0.15 0.10 1",
    "cylinder": "0.12 0.77 0.18 1",
    "prism": "0.12 0.30 0.94 1",
}
BIN_CENTERS = {
    "cube": (0.30, -0.31),
    "cylinder": (0.30, 0.0),
    "prism": (0.30, 0.31),
}


def _box(parent, name, pos, size, rgba, **extra):
    return ET.SubElement(parent, "geom", name=name, type="box", pos=pos,
                         size=size, rgba=rgba, **extra)


def make_scene_xml() -> str:
    """Extend the credited Panda MJCF without altering the upstream file."""
    root = ET.fromstring((PANDA_DIR / "panda.xml").read_text(encoding="utf-8"))
    root.attrib["model"] = "conveyor vision sorting"
    root.find("option").set("timestep", "0.004")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="720")
    assets = root.find("asset")
    ET.SubElement(assets, "mesh", name="triangular_prism",
                  vertex="-0.024 -0.020 -0.018 0.024 -0.020 -0.018 0 0.027 -0.018 "
                         "-0.024 -0.020 0.018 0.024 -0.020 0.018 0 0.027 0.018",
                  face="0 2 1 3 4 5 0 1 4 0 4 3 1 2 5 1 5 4 2 0 3 2 3 5")
    world = root.find("worldbody")
    ET.SubElement(world, "light", name="work_light", pos="0.5 -0.2 1.6",
                  dir="0 0 -1", directional="true", diffuse="0.95 0.95 0.95")
    ET.SubElement(world, "camera", name="top", pos="0.50 -0.04 1.26",
                  fovy="55", resolution="640 480")
    ET.SubElement(world, "camera", name="overview", pos="1.25 -1.0 1.25",
                  xyaxes="0.72 0.69 0 -0.42 0.44 0.79", fovy="52",
                  resolution="1280 720")
    ET.SubElement(world, "geom", name="floor", type="plane", pos="0 0 -0.02",
                  size="2 2 0.01", rgba="0.78 0.82 0.84 1")
    for x in (0.37, 0.73):
        for y in (-0.40, 0.40):
            ET.SubElement(world, "geom", name=f"fiducial_{x}_{y}", type="sphere",
                          pos=f"{x} {y} 0.20", size="0.014",
                          rgba="1 0.85 0.05 1", contype="0", conaffinity="0")
    belt = ET.SubElement(world, "body", name="conveyor", pos="0.55 0 0.14")
    _box(belt, "belt_surface", "0 0 0", "0.14 0.56 0.02", "0.20 0.25 0.29 1",
         friction="1.4 0.005 0.0001", surfacevel="0 0.12 0 0 0 0")
    for x in (-0.155, 0.155):
        _box(belt, f"belt_rail_{'left' if x < 0 else 'right'}", f"{x} 0 0.028",
             "0.009 0.57 0.013", "0.68 0.70 0.72 1", contype="0", conaffinity="0")
    for name, (x, y) in BIN_CENTERS.items():
        bin_body = ET.SubElement(world, "body", name=f"bin_{name}", pos=f"{x} {y} 0.025")
        _box(bin_body, f"bin_{name}_base", "0 0 0", "0.08 0.08 0.012", "0.35 0.38 0.42 1")
        for side in (-1, 1):
            _box(bin_body, f"bin_{name}_x{side}", f"{side * 0.08} 0 0.05",
                 "0.006 0.08 0.05", "0.43 0.45 0.48 1")
            _box(bin_body, f"bin_{name}_y{side}", f"0 {side * 0.08} 0.05",
                 "0.08 0.006 0.05", "0.43 0.45 0.48 1")
    for kind in PARTS:
        body = ET.SubElement(world, "body", name=f"part_{kind}", pos="0.55 -0.35 0.185")
        ET.SubElement(body, "freejoint", name=f"part_{kind}_free")
        kwargs = {"name": f"part_{kind}_geom", "rgba": COLORS[kind],
                  "mass": "0.035", "friction": "1.4 0.005 0.0001"}
        if kind == "cube":
            ET.SubElement(body, "geom", type="box", size="0.023 0.023 0.023", **kwargs)
        elif kind == "cylinder":
            ET.SubElement(body, "geom", type="cylinder", size="0.022 0.023", **kwargs)
        else:
            ET.SubElement(body, "geom", type="mesh", mesh="triangular_prism", **kwargs)
    hand = world.find(".//body[@name='hand']")
    ET.SubElement(hand, "site", name="grasp_center", pos="0 0 0.103", size="0.005")
    # Model rubberized fingertip contact without editing Menagerie's source XML.
    for name in ("left_finger", "right_finger"):
        finger = hand.find(f"body[@name='{name}']")
        for geom in finger.findall("geom"):
            if geom.get("class") != "visual":
                geom.set("friction", "2.5 0.02 0.001")
    root.find("keyframe").clear()
    return ET.tostring(root, encoding="unicode")


class Scene:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(make_scene_xml(), assets=panda_assets())
        self.data = mujoco.MjData(self.model)
        self._belt_id = self.model.geom("belt_surface").id
        self.grasp_site_id = self.model.site("grasp_center").id
        self.jaw_actuator_id = self.model.actuator("actuator8").id
        self._renderers = {}
        self.kind = "cube"
        self.reset("cube", seed=0)
        self.home_rotation = self.data.site_xmat[self.grasp_site_id].reshape(3, 3).copy()

    def set_jaw(self, opened: bool) -> None:
        self.data.ctrl[self.jaw_actuator_id] = 255.0 if opened else 0.0

    def reset(self, kind: str, seed: int = 0, lateral: float | None = None,
              yaw: float | None = None):
        if kind not in PARTS:
            raise ValueError(f"Unknown part kind: {kind}")
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        home = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
        self.data.qpos[:7] = home
        self.data.qpos[7:9] = 0.04
        self.data.ctrl[:7] = home
        self.set_jaw(opened=True)
        self.kind = kind
        for other in PARTS:
            joint = self.model.joint(f"part_{other}_free")
            start = int(joint.qposadr[0])
            if other == kind:
                x = 0.55 + (float(rng.uniform(-0.045, 0.045)) if lateral is None else lateral)
                y = -0.34 + float(rng.uniform(-0.015, 0.015))
                angle = float(rng.uniform(-math.pi, math.pi)) if yaw is None else yaw
                self.data.qpos[start:start + 7] = [x, y, 0.19,
                                                     math.cos(angle / 2), 0, 0,
                                                     math.sin(angle / 2)]
            else:
                self.data.qpos[start:start + 7] = [3.0, 2.0 + PARTS.index(other), 0.2, 1, 0, 0, 0]
        self.set_belt_speed(0.0)
        mujoco.mj_forward(self.model, self.data)
        self.step(0.1)

    def set_belt_speed(self, speed: float):
        self.model.geom_surfacevel[self._belt_id, :] = [0, speed, 0, 0, 0, 0]

    def step(self, seconds: float):
        n = max(1, math.ceil(seconds / self.model.opt.timestep))
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def part_position(self) -> np.ndarray:
        return self.data.xpos[self.model.body(f"part_{self.kind}").id].copy()

    def rgb(self, camera: str = "top", width: int = 640, height: int = 480) -> np.ndarray:
        key = (width, height)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height=height, width=width)
        renderer = self._renderers[key]
        renderer.update_scene(self.data, camera=camera)
        return renderer.render().copy()

    def calibration_homography(self) -> np.ndarray:
        from conveyor.vision import calibrate, fiducial_centers

        pixels = fiducial_centers(self.rgb("top", 640, 480))
        world = np.array([[0.37, 0.40], [0.73, 0.40],
                          [0.37, -0.40], [0.73, -0.40]], dtype=np.float32)
        return calibrate(pixels, world)

    def close(self):
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
