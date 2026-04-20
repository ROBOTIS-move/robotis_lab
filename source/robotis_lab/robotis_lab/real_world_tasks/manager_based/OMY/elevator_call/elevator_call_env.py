# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Custom env class for the OMY elevator call-button task.

Hooks into ``step()`` to translate a gripper-CallBtn contact into an
``elevator.press_call_button`` trigger on the rising edge. This is more
reliable than relying on EventTerm(mode="interval", (0,0)) which has
version-dependent semantics.
"""

from __future__ import annotations

from typing import Any

import torch
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg


_DEFAULT_FORCE_THRESHOLD = 0.3


class OMYElevatorCallEnv(ManagerBasedRLEnv):
    def __init__(
        self,
        cfg: ManagerBasedRLEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        self._contact_prev = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------
    def step(self, action: torch.Tensor) -> Any:
        result = super().step(action)
        self._update_call_button_trigger()
        return result

    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if env_ids is not None and len(env_ids) > 0:
            self._contact_prev[env_ids] = False

    # ------------------------------------------------------------------
    def _update_call_button_trigger(self) -> None:
        try:
            elevator = self.scene["elevator"]
            sensor = self.scene["contact_gripper"]
        except KeyError:
            return

        forces = sensor.data.force_matrix_w
        if forces is None or forces.numel() == 0:
            return
        mags = torch.linalg.norm(forces, dim=-1)
        # mags: [num_envs, num_bodies, num_filter_prims]
        contact_now = mags.amax(dim=(-1, -2)) > _DEFAULT_FORCE_THRESHOLD

        rising = contact_now & (~self._contact_prev)
        self._contact_prev = contact_now.clone()

        for eid in rising.nonzero(as_tuple=False).flatten().tolist():
            elevator.press_call_button(eid, call_index=0)
