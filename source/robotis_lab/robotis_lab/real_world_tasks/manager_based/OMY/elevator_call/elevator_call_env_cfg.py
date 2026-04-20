# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Base env config for OMY elevator call-button task.

Elevator USD is randomized once at scene construction (option A, resolved in
plan). A separate concrete cfg (`joint_pos_env_cfg.OMYElevatorCallEnvCfg`)
injects the OMY robot, cameras, and contact sensor.
"""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg as RecordTerm
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

from isaac_sim.assets.elevator import ElevatorBehaviorCfg, make_elevator_cfg

from robotis_lab.assets.object.pedestal import PEDESTAL_CFG

from . import mdp


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
@configclass
class ElevatorCallSceneCfg(InteractiveSceneCfg):
    """Scene: elevator + pedestal + OMY robot (+ sensors injected by agent cfg)."""

    # Robot and its sensors are injected by the concrete OMY cfg.
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    cam_wrist: CameraCfg = MISSING
    cam_top: CameraCfg = MISSING
    contact_gripper: ContactSensorCfg = MISSING

    # Pedestal (1 m gray box). xy is overwritten every reset via event.
    pedestal = PEDESTAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Pedestal")

    # Elevator: random USD at construction time (seed=None).
    # Auto cycles off so the robot drives the triggers.
    elevator = make_elevator_cfg(
        seed=None,
        prim_path="{ENV_REGEX_NS}/Elevator",
        behavior=ElevatorBehaviorCfg(
            auto_door_cycle=False,
            auto_button_cycle=False,
        ),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


# ---------------------------------------------------------------------------
# MDP blocks
# ---------------------------------------------------------------------------
@configclass
class ActionsCfg:
    arm_action: mdp.ActionTermCfg = MISSING
    gripper_action: mdp.ActionTermCfg = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)

        joint_pos = ObsTerm(
            func=mdp.joint_pos_name,
            params={
                "joint_names": ["joint1", "joint2", "joint3", "joint4",
                                "joint5", "joint6", "rh_r1_joint"],
                "asset_name": "robot",
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_name,
            params={
                "joint_names": ["joint1", "joint2", "joint3", "joint4",
                                "joint5", "joint6", "rh_r1_joint"],
                "asset_name": "robot",
            },
        )
        cam_wrist = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("cam_wrist"),
                    "data_type": "rgb", "normalize": False},
        )
        cam_top = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("cam_top"),
                    "data_type": "rgb", "normalize": False},
        )
        eef_pose = ObsTerm(
            func=mdp.eef_pose,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame"),
                    "robot_cfg": SceneEntityCfg("robot")},
        )
        joint_pos_target = ObsTerm(
            func=mdp.joint_pos_target_name,
            params={
                "joint_names": ["joint1", "joint2", "joint3", "joint4",
                                "joint5", "joint6", "rh_r1_joint"],
                "asset_name": "robot",
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        call_button_contact = ObsTerm(func=mdp.call_button_contact)
        call_button_lit = ObsTerm(func=mdp.call_button_lit)
        door_open_progress = ObsTerm(func=mdp.door_open_progress)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class EventCfg:
    """Shared reset events. Light + joint randomization override / extend from OMY cfg."""

    randomize_robot_spawn = EventTerm(
        func=mdp.randomize_robot_spawn_near_call_button,
        mode="reset",
        params={
            "distance_range": (0.4, 0.5),
            "pedestal_height": 1.0,
            "robot_cfg": SceneEntityCfg("robot"),
            "pedestal_cfg": SceneEntityCfg("pedestal"),
        },
    )

    reset_door_latch = EventTerm(
        func=mdp.reset_door_open_latch,
        mode="reset",
    )

    # NOTE: per-step contact->press trigger is implemented in the custom env
    # class ``OMYElevatorCallEnv.step`` override (elevator_call_env.py), which
    # is more reliable than interval-mode events across Isaac Lab versions.


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=mdp.door_open_sustained,
        params={"hold_seconds": 1.0, "progress_threshold": 0.9},
    )


# ---------------------------------------------------------------------------
# Env cfg
# ---------------------------------------------------------------------------
@configclass
class ElevatorCallEnvCfg(ManagerBasedRLEnvCfg):
    """Base env cfg. Concrete agent cfg injects robot, cameras, contact sensor."""

    scene: ElevatorCallSceneCfg = ElevatorCallSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    recorders: RecordTerm = RecordTerm()

    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 5
        self.episode_length_s = 30.0
        self.sim.dt = 0.01
        self.sim.render_interval = 2

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

    def init_action_cfg(self, mode: str):
        print(f"Initializing action configuration for device: {mode}")
        if mode in ["record", "inference"]:
            self.actions.arm_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["joint.*"],
                scale=1.0,
                use_default_offset=False,
            )
            self.actions.gripper_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["rh_r1_joint"],
                scale=1.0,
                use_default_offset=False,
            )
        elif mode in ["mimic_ik"]:
            self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=["joint[1-6]"],
                body_name="link6",
                controller=DifferentialIKControllerCfg(
                    command_type="pose", ik_params={"lambda_val": 0.05},
                    ik_method="dls",
                    use_relative_mode=False,
                ),
                body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                    pos=[0.0, -0.248, 0.0]),
            )
            self.actions.gripper_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["rh_r1_joint"],
                scale=1.0,
                use_default_offset=False,
            )
        else:
            self.actions.arm_action = None
            self.actions.gripper_action = None
