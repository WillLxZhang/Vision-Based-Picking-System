"""Damped least-squares Panda control and pick/place state machine."""

from dataclasses import dataclass
from typing import Callable

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from conveyor.scene import Scene, BIN_CENTERS
from conveyor.vision import detect


@dataclass
class EpisodeResult:
    kind: str
    predicted_kind: str | None = None
    detected: bool = False
    grasped: bool = False
    sorted: bool = False
    failure: str = "not_run"
    elapsed: float = 0.0
    error_m: float | None = None
    states: tuple[tuple[str, float], ...] = ()
    final_xy: tuple[float, float] | None = None
    max_object_angular_speed_rad_s: float | None = None
    max_relative_orientation_deg: float | None = None


def _relative_rotation_change_deg(first: Rotation, current: Rotation) -> float:
    return float(np.degrees((first.inv() * current).magnitude()))


@dataclass
class RotationTelemetry:
    """Ground-truth evaluation only; never used to choose control actions."""

    max_object_angular_speed_rad_s: float | None = None
    max_relative_orientation_deg: float | None = None
    _first_relative: Rotation | None = None

    def sample(self, scene: Scene, state: str) -> None:
        if state not in ("LIFT", "TRANSFER"):
            return
        if state == "TRANSFER" and self._first_relative is None:
            return
        joint = scene.model.joint(f"part_{scene.kind}_free")
        dof = int(joint.dofadr[0])
        omega = float(np.linalg.norm(scene.data.qvel[dof + 3:dof + 6]))
        part_rot = scene.data.xmat[scene.model.body(f"part_{scene.kind}").id].reshape(3, 3)
        hand_rot = scene.data.xmat[scene.model.body("hand").id].reshape(3, 3)
        relative = Rotation.from_matrix(hand_rot.T @ part_rot)
        if self._first_relative is None:
            self._first_relative = relative
            self.max_relative_orientation_deg = 0.0
        angle = _relative_rotation_change_deg(self._first_relative, relative)
        self.max_relative_orientation_deg = max(self.max_relative_orientation_deg, angle)
        self.max_object_angular_speed_rad_s = max(
            self.max_object_angular_speed_rad_s or 0.0, omega)


def solve_ik(scene: Scene, target_xyz: np.ndarray,
             seed_q: np.ndarray | None = None) -> np.ndarray | None:
    """Solve 6D pose with the tool orientation held at the home pose."""
    target = np.asarray(target_xyz, dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("target_xyz must be a finite 3-vector")
    if np.linalg.norm(target) > 1.2:
        return None
    model = scene.model
    data = mujoco.MjData(model)
    data.qpos[:] = scene.data.qpos
    if seed_q is not None:
        data.qpos[:7] = seed_q
    tip = scene.grasp_site_id
    desired_rot = scene.home_rotation
    jac_p = np.zeros((3, model.nv))
    jac_r = np.zeros((3, model.nv))
    for _ in range(350):
        mujoco.mj_forward(model, data)
        position_error = target - data.site_xpos[tip]
        rotation_error = Rotation.from_matrix(
            desired_rot @ data.site_xmat[tip].reshape(3, 3).T).as_rotvec()
        if np.linalg.norm(position_error) < 0.003 and np.linalg.norm(rotation_error) < 0.045:
            return data.qpos[:7].copy()
        mujoco.mj_jacSite(model, data, jac_p, jac_r, tip)
        jacobian = np.vstack((jac_p[:, :7], jac_r[:, :7]))
        error = np.concatenate((position_error, rotation_error * 0.45))
        damping = 0.035
        dq = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping**2 * np.eye(6), error)
        dq = np.clip(dq, -0.075, 0.075)
        data.qpos[:7] += dq
        for index in range(7):
            joint = model.joint(f"joint{index + 1}")
            data.qpos[index] = np.clip(data.qpos[index],
                                       float(joint.range[0]) + 0.01,
                                       float(joint.range[1]) - 0.01)
    return None


def _advance(scene: Scene, duration: float, state: str,
             observer: Callable[[Scene, str], None] | None,
             monitor: Callable[[Scene, str], None] | None = None):
    if observer is None and monitor is None:
        scene.step(duration)
        return
    steps = max(1, int(np.ceil(duration / scene.model.opt.timestep)))
    for _ in range(steps):
        mujoco.mj_step(scene.model, scene.data)
        if monitor is not None:
            monitor(scene, state)
        if observer is None:
            continue
        observer(scene, state)


def _move(scene: Scene, xyz: np.ndarray, state: str,
          observer: Callable[[Scene, str], None] | None,
          monitor: Callable[[Scene, str], None] | None = None,
          timeout: float = 3.0, max_joint_speed: float = 0.8) -> bool:
    q = solve_ik(scene, xyz)
    if q is None:
        return False
    start = scene.data.time
    tip = scene.grasp_site_id
    dt = float(scene.model.opt.timestep)
    jac_p = np.zeros((3, scene.model.nv))
    jac_r = np.zeros((3, scene.model.nv))
    iterations = 0
    while scene.data.time - start < timeout:
        actual = scene.data.site_xpos[tip]
        position_error = xyz - actual
        joint_lag = np.max(np.abs(scene.data.ctrl[:7] - scene.data.qpos[:7]))
        command_lag = np.max(np.abs(q - scene.data.ctrl[:7]))
        if (iterations % 10 == 0 and joint_lag < 0.04 and command_lag < 0.02
                and np.linalg.norm(position_error) < 0.08):
            mujoco.mj_jacSite(scene.model, scene.data, jac_p, jac_r, tip)
            rotation_error = Rotation.from_matrix(
                scene.home_rotation @ scene.data.site_xmat[tip].reshape(3, 3).T
            ).as_rotvec()
            jacobian = np.vstack((jac_p[:, :7], jac_r[:, :7]))
            error = np.concatenate((position_error, 0.45 * rotation_error))
            correction = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + 0.035**2 * np.eye(6), error)
            q += np.clip(0.6 * correction, -0.015, 0.015)
            for index in range(7):
                limits = scene.model.joint(f"joint{index + 1}").range
                q[index] = np.clip(q[index], limits[0] + 0.01, limits[1] - 0.01)
        delta = np.clip(q - scene.data.ctrl[:7],
                        -max_joint_speed * dt, max_joint_speed * dt)
        scene.data.ctrl[:7] += delta
        _advance(scene, dt, state, observer, monitor)
        iterations += 1
        position_error = xyz - scene.data.site_xpos[tip]
        position_ready = (np.linalg.norm(position_error[:2]) < 0.003
                          and abs(position_error[2]) < 0.008)
        joints_settled = np.max(np.abs(scene.data.ctrl[:7] - scene.data.qpos[:7])) < 0.02
        if position_ready and joints_settled:
            return True
    return False


def _finger_contacts(scene: Scene) -> tuple[bool, bool]:
    left_body = scene.model.body("left_finger").id
    right_body = scene.model.body("right_finger").id
    part_geom = scene.model.geom(f"part_{scene.kind}_geom").id
    touched = {left_body: False, right_body: False}
    for contact in scene.data.contact:
        if contact.geom1 == part_geom:
            other = contact.geom2
        elif contact.geom2 == part_geom:
            other = contact.geom1
        else:
            continue
        body = int(scene.model.geom_bodyid[other])
        if body in touched and contact.dist < 0.006:
            touched[body] = True
    return touched[left_body], touched[right_body]


def is_correct_bin(kind: str, predicted_kind: str | None,
                   final_xyz: np.ndarray) -> bool:
    """Evaluate the final state against ground truth, outside the controller."""
    if kind not in BIN_CENTERS or predicted_kind != kind:
        return False
    bin_x, bin_y = BIN_CENTERS[kind]
    return bool(abs(final_xyz[0] - bin_x) < 0.065
                and abs(final_xyz[1] - bin_y) < 0.065
                and final_xyz[2] < 0.13)


def run_episode(scene: Scene, kind: str, seed: int, use_vision: bool = True,
                observer: Callable[[Scene, str], None] | None = None,
                belt_speed: float = 0.12) -> EpisodeResult:
    result = EpisodeResult(kind=kind)
    telemetry = RotationTelemetry()
    transitions = []
    safe_retract_ready = False

    def mark(state: str):
        transitions.append((state, round(float(scene.data.time), 3)))
        if observer is not None:
            observer(scene, state)

    def finish(reason: str, recover: bool = True) -> EpisodeResult:
        scene.set_belt_speed(0)
        scene.set_jaw(opened=True)
        if reason and recover:
            mark("SAFE_OPEN")
            _advance(scene, 0.12, "SAFE_OPEN", observer)
            if safe_retract_ready:
                mark("SAFE_RETRACT")
                current = scene.data.site_xpos[scene.grasp_site_id].copy()
                height = float(current[2])
                steps = int(np.ceil(max(0.0, 0.34 - height) / 0.04))
                for target_height in np.linspace(height, 0.34, steps + 1)[1:]:
                    target = np.array([current[0], current[1], target_height])
                    if not _move(scene, target, "SAFE_RETRACT", observer, timeout=3.0):
                        reason += "+safe_retract_failed"
                        break
        result.failure = reason
        result.elapsed = float(scene.data.time)
        result.states = tuple(transitions)
        result.max_object_angular_speed_rad_s = telemetry.max_object_angular_speed_rad_s
        result.max_relative_orientation_deg = telemetry.max_relative_orientation_deg
        return result

    scene.reset(kind, seed=seed)
    mark("RESET")
    mark("CONVEY")
    scene.set_belt_speed(belt_speed)
    _advance(scene, 1.35 * (0.12 / belt_speed), "CONVEY", observer)
    scene.set_belt_speed(0)
    _advance(scene, 0.25, "STOP", observer)
    mark("DETECT")
    if use_vision:
        image = scene.rgb("top", 640, 480)
        detected = detect(image, scene.calibration_homography())
        if detected is None:
            return finish("no_detection")
        result.predicted_kind = detected.kind
        target_xy = np.asarray(detected.world_xy)
        result.error_m = float(np.linalg.norm(target_xy - scene.part_position()[:2]))
        if observer is not None and hasattr(observer, "set_detection"):
            observer.set_detection(detected.kind, result.error_m, image, detected.contour)
    else:
        result.predicted_kind = kind
        target_xy = scene.part_position()[:2].copy()
        result.error_m = 0.0
    result.detected = True
    if result.predicted_kind not in BIN_CENTERS:
        return finish("unknown_class")
    scene.set_jaw(opened=True)
    mark("PREGRASP")
    above = np.array([target_xy[0], target_xy[1], 0.34])
    if not _move(scene, above, "PREGRASP", observer):
        return finish("pregrasp_ik_or_timeout")
    safe_retract_ready = True
    mark("DESCEND")
    grasp_height = 0.18 if result.predicted_kind == "prism" else 0.185
    contact_pose = np.array([target_xy[0], target_xy[1], grasp_height])
    if not _move(scene, contact_pose, "DESCEND", observer):
        return finish("descend_ik_or_timeout")
    mark("CLOSE")
    scene.set_jaw(opened=False)
    _advance(scene, 0.35, "CLOSE", observer)
    if observer is not None:
        observer(scene, "CLOSE_COMPLETE")
    if not all(_finger_contacts(scene)):
        return finish("no_bilateral_contact")
    mark("LIFT")
    if not _move(scene, above, "LIFT", observer, monitor=telemetry.sample):
        return finish("lift_ik_or_timeout")
    # Ground truth is recorded for evaluation only; it does not gate control.
    result.grasped = bool(scene.part_position()[2] >= 0.24)
    bin_x, bin_y = BIN_CENTERS[result.predicted_kind]
    mark("TRANSFER")
    if not _move(scene, np.array([bin_x, bin_y, 0.34]), "TRANSFER", observer,
                 monitor=telemetry.sample, timeout=6.0, max_joint_speed=0.35):
        return finish("transfer_ik_or_timeout")
    mark("LOWER")
    if not _move(scene, np.array([bin_x, bin_y, 0.17]), "LOWER", observer):
        return finish("lower_ik_or_timeout")
    _advance(scene, 0.12, "LOWER", observer)
    mark("OPEN")
    scene.set_jaw(opened=True)
    _advance(scene, 0.25, "OPEN", observer)
    if observer is not None:
        observer(scene, "OPEN_COMPLETE")
    mark("RETRACT")
    retract_ok = _move(scene, np.array([bin_x, bin_y, 0.34]), "RETRACT", observer)
    _advance(scene, 0.5, "RETRACT", observer)
    mark("VERIFY")
    final = scene.part_position()
    result.final_xy = (float(final[0]), float(final[1]))
    result.sorted = is_correct_bin(kind, result.predicted_kind, final)
    if not retract_ok:
        return finish("retract_ik_or_timeout")
    mark("RETURN")
    if not _move(scene, np.array([0.55, 0.0, 0.50]), "RETURN", observer):
        return finish("return_ik_or_timeout", recover=False)
    return finish("" if result.sorted else "outside_target_bin", recover=False)
