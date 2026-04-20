# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Observation terms for OMY elevator call-button task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def call_button_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_gripper"),
    threshold: float = 0.3,
) -> torch.Tensor:
    """Bool (as float) whether any filtered CallBtn contact exceeds ``threshold``."""
    sensor = env.scene[sensor_cfg.name]
    forces = sensor.data.force_matrix_w
    if forces is None:
        return torch.zeros(env.num_envs, device=env.device)
    mags = torch.linalg.norm(forces, dim=-1)
    return (mags.amax(dim=(-1, -2)) > threshold).float()


def call_button_lit(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Whether the elevator's first call button is currently lit."""
    elevator = env.elevator
    result = torch.zeros(env.num_envs, device=env.device)
    for eid in range(env.num_envs):
        lit = False
        for info_idx, info in enumerate(elevator.button.btn_infos[eid]):
            if info["kind"] == "call":
                lit = bool(elevator.button.lit[eid][info_idx])
                break
        result[eid] = 1.0 if lit else 0.0
    return result


def door_open_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Door open progress 0.0 (closed) – 1.0 (open) per env."""
    elevator = env.elevator
    return torch.as_tensor(
        elevator.door.progress, device=env.device, dtype=torch.float32)
