# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Reset events for OMY elevator call-button task."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.assets import Articulation, AssetBase
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _yaw_to_quat(yaw: float, device) -> torch.Tensor:
    return math_utils.quat_from_euler_xyz(
        torch.zeros(1, device=device),
        torch.zeros(1, device=device),
        torch.tensor([yaw], device=device),
    )


def randomize_robot_spawn_near_call_button(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    distance_range: tuple[float, float] = (0.4, 0.5),
    pedestal_height: float = 1.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pedestal_cfg: SceneEntityCfg = SceneEntityCfg("pedestal"),
):
    """Place the pedestal + OMY root directly in front of the call button.

    - Robot base (+x) faces the button (yaw locked).
    - Horizontal distance sampled from `distance_range`.
    - Robot z = env_origin.z + pedestal_height (pedestal top).
    """
    if env_ids is None:
        return

    elevator = env.scene["elevator"]
    robot: Articulation = env.scene[robot_cfg.name]
    pedestal: AssetBase = env.scene[pedestal_cfg.name]
    device = env.device

    for eid in env_ids.tolist():
        btn_pos, _ = elevator.get_call_button_world_pose(eid, call_index=0)
        facing = elevator.get_call_button_facing(eid, call_index=0)
        if btn_pos is None or facing is None:
            continue

        # Project facing onto xy plane and normalize.
        facing_xy = np.array(facing[:2], dtype=np.float64)
        n = np.linalg.norm(facing_xy)
        if n < 1e-6:
            continue
        facing_xy = facing_xy / n

        d = random.uniform(*distance_range)
        env_origin = env.scene.env_origins[eid].cpu().numpy()

        # World-frame target position of the robot root.
        robot_xy_world = np.array(btn_pos[:2]) + d * facing_xy
        robot_pos_world = np.array([
            robot_xy_world[0],
            robot_xy_world[1],
            env_origin[2] + pedestal_height,
        ])
        # Pedestal xy under the robot; z = env_origin + height/2 (cuboid center).
        pedestal_pos_world = np.array([
            robot_xy_world[0],
            robot_xy_world[1],
            env_origin[2] + pedestal_height * 0.5,
        ])

        # Robot yaw: face the button (local +x pointing toward the button).
        to_button = -facing_xy
        yaw = float(math.atan2(to_button[1], to_button[0]))

        env_idx = torch.tensor([eid], device=device)

        robot_pose = torch.cat([
            torch.tensor(robot_pos_world, dtype=torch.float32, device=device).unsqueeze(0),
            _yaw_to_quat(yaw, device),
        ], dim=-1)
        robot.write_root_pose_to_sim(robot_pose, env_ids=env_idx)
        robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device), env_ids=env_idx)

        pedestal_pose = torch.cat([
            torch.tensor(pedestal_pos_world, dtype=torch.float32, device=device).unsqueeze(0),
            _yaw_to_quat(0.0, device),
        ], dim=-1)
        pedestal.write_root_pose_to_sim(pedestal_pose, env_ids=env_idx)


def reset_door_open_latch(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
):
    """Clear the per-env ``open_since`` timestamps used by DoorOpenSustained."""
    if env_ids is None:
        return
    latch = getattr(env, "_elev_door_open_since", None)
    if latch is None:
        return
    for eid in env_ids.tolist():
        latch.pop(eid, None)


def randomize_scene_lighting_domelight(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float] = (1000.0, 3000.0),
    color_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    = ((0.7, 1.0), (0.7, 1.0), (0.7, 1.0)),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("light"),
):
    """Lightweight light randomizer copy (avoid pick_place dep)."""
    from pxr import Gf
    asset: AssetBase = env.scene[asset_cfg.name]
    light_prim = asset.prims[0]
    light_prim.GetAttribute("inputs:intensity").Set(
        random.uniform(*intensity_range))
    light_prim.GetAttribute("inputs:color").Set(Gf.Vec3f(
        random.uniform(*color_range[0]),
        random.uniform(*color_range[1]),
        random.uniform(*color_range[2]),
    ))
