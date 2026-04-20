# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Termination terms for OMY elevator call-button task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def door_open_sustained(
    env: ManagerBasedRLEnv,
    hold_seconds: float = 1.0,
    progress_threshold: float = 0.9,
) -> torch.Tensor:
    """Success when ``door.progress > threshold`` holds for ``hold_seconds``.

    State is kept on ``env._elev_door_open_since: dict[int, float]`` (env_id →
    simulated time when the door first crossed the threshold). ``events.
    reset_door_open_latch`` clears this dict on reset.
    """
    elevator = env.elevator
    if not hasattr(env, "_elev_door_open_since"):
        env._elev_door_open_since = {}
    latch = env._elev_door_open_since

    now = (env.episode_length_buf.float() * env.step_dt).detach().cpu().tolist()
    progress = torch.as_tensor(elevator.door.progress, dtype=torch.float32)

    result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for eid in range(env.num_envs):
        if progress[eid].item() > progress_threshold:
            start = latch.setdefault(eid, now[eid])
            if (now[eid] - start) >= hold_seconds:
                result[eid] = True
        else:
            latch.pop(eid, None)
    return result
