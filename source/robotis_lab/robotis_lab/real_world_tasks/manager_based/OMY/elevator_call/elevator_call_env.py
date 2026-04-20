# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Custom env class for the OMY elevator call-button task.

Responsibilities this custom class takes on beyond ManagerBasedRLEnv:
1. Instantiate the ``Elevator`` runtime (the InteractiveScene only spawns the
   USD; it does not create the behaviour-carrying ``Elevator`` instance).
2. Drive the elevator FSM lifecycle (``update`` / ``write_data_to_sim`` /
   ``reset``) on every physics step / reset.
3. Translate a gripper-CallBtn contact into an
   ``elevator.press_call_button`` trigger on the rising edge.

Observations / events / terminations read the elevator via ``env.elevator``
instead of ``env.scene["elevator"]`` — the scene dict only holds the static
USD wrapper, not this runtime object.
"""

from __future__ import annotations

from typing import Any

import torch
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

from isaac_sim.assets.elevator import Elevator


_DEFAULT_FORCE_THRESHOLD = 0.3


class OMYElevatorCallEnv(ManagerBasedRLEnv):
    def __init__(
        self,
        cfg: ManagerBasedRLEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # --- Elevator runtime ------------------------------------------------
        # Each env spawns its elevator USD under "/World/envs/env_{i}/Elevator".
        elevator_cfg = cfg.scene.elevator
        env_paths = [
            f"/World/envs/env_{i}/Elevator" for i in range(self.num_envs)
        ]
        self.elevator = Elevator(elevator_cfg)
        self.elevator._initialize(env_paths)

        # Per-env contact rising-edge tracker for call-button press trigger.
        self._contact_prev = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------
    def step(self, action: torch.Tensor) -> Any:
        # Advance FSMs (door/button/LED/floor) with the env time step.
        self.elevator.update(self.step_dt)
        result = super().step(action)
        # Flush FSM state back to USD (panel translate, button/LED material).
        self.elevator.write_data_to_sim()
        self._update_call_button_trigger()
        return result

    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if env_ids is None:
            return
        ids = env_ids.tolist() if hasattr(env_ids, "tolist") else list(env_ids)
        if not ids:
            return
        self.elevator.reset(ids)
        # Reset contact rising-edge memory for the reset envs.
        self._contact_prev[env_ids] = False

    # ------------------------------------------------------------------
    def _update_call_button_trigger(self) -> None:
        try:
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
            self.elevator.press_call_button(eid, call_index=0)
