# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Concrete OMY env config for the elevator call-button task."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from robotis_lab.assets.robots.OMY import OMY_CFG

from robotis_lab.real_world_tasks.manager_based.OMY.pick_place.mdp import (
    omy_pick_place_events,
)

from .elevator_call_env_cfg import ElevatorCallEnvCfg, EventCfg


@configclass
class OMYElevatorCallEventCfg(EventCfg):
    """Extend shared events with OMY-specific joint init and light randomization."""

    init_omy_arm_pose = EventTerm(
        func=omy_pick_place_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0, -1.55, 2.66, -1.1, 1.6, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
    )

    randomize_omy_joint_state = EventTerm(
        func=omy_pick_place_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    randomize_scene_light = EventTerm(
        func=omy_pick_place_events.randomize_scene_lighting_domelight,
        mode="reset",
        params={
            "intensity_range": (1000.0, 3000.0),
            "color_range": ((0.7, 1.0), (0.7, 1.0), (0.7, 1.0)),
            "asset_cfg": SceneEntityCfg("light"),
        },
    )


@configclass
class OMYElevatorCallEnvCfg(ElevatorCallEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Swap events with the OMY-specific variant.
        self.events = OMYElevatorCallEventCfg()

        # --- Robot ----------------------------------------------------------
        # activate_contact_sensors=True is required for ContactSensor reads on
        # the gripper.
        self.scene.robot = OMY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn = self.scene.robot.spawn.replace(
            activate_contact_sensors=True)
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Add semantics to ground.
        self.scene.plane.semantic_tags = [("class", "ground")]

        # --- Sensors --------------------------------------------------------
        # Wrist camera (reused from pick-place pattern).
        self.scene.cam_wrist = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/OMY/link6/cam_wrist",
            update_period=0.0,
            height=480,
            width=848,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=11.8, focus_distance=200.0,
                horizontal_aperture=20.955, clipping_range=(0.01, 100.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, -0.08, 0.07),
                rot=(0.5, -0.5, -0.5, -0.5),
                convention="isaac",
            ),
        )

        # Top camera: placeholder under env root (always exists). Final
        # position TBD — reattach to a meaningful prim later.
        self.scene.cam_top = CameraCfg(
            prim_path="{ENV_REGEX_NS}/cam_top",
            update_period=0.0,
            height=480,
            width=848,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=10.0, focus_distance=200.0,
                horizontal_aperture=20.955, clipping_range=(0.01, 100.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 1.6),                    # env origin 기준 높이
                rot=(0.0, 0.7071068, 0.7071068, 0.0),   # facing +x, pitched down
                convention="isaac",
            ),
        )

        # Contact sensor on arm end + gripper links.
        # The elevator USD doubles-wraps its root (Elevator/Elevator/...), so
        # the real button path is <env>/Elevator/Elevator/HallExterior/CallBtn_<idx>.
        # Isaac Lab requires exactly ONE prim match per env for filter_prim_paths_expr,
        # so we pin to CallBtn_0 (press_call_button() in our env class also targets
        # call_index=0, keeping observation and trigger aligned).
        self.scene.contact_gripper = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/OMY/link6",
            update_period=0.0,
            history_length=1,
            filter_prim_paths_expr=[
                "{ENV_REGEX_NS}/Elevator/Elevator/HallExterior/CallBtn_0",
            ],
        )

        # End-effector frame transformer.
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/OMY/world",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/OMY/link6",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, -0.248, 0.0]),
                ),
            ],
        )
