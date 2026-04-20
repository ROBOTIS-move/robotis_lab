# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Custom env class for the OMY elevator call-button task.

The InteractiveScene only spawns the elevator USD; it does not create a
behaviour-carrying ``Elevator`` instance. This wrapper provides:

1. A lazy ``env.elevator`` property that instantiates ``Elevator`` and calls
   ``_initialize`` the first time any observation / event / termination reads
   it. Lazy because ObservationManager runs obs functions inside
   ``super().__init__()`` before attribute assignments in our __init__ would
   take effect.
2. FSM lifecycle hooks (``update`` / ``write_data_to_sim`` / ``reset``) driven
   from ``step()`` and ``_reset_idx()``.
3. Rising-edge contact → ``press_call_button`` trigger using the filtered
   ContactSensor data on the gripper.
"""

from __future__ import annotations

from typing import Any, Optional

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
        # These must exist before super().__init__ because the ObservationManager
        # runs obs funcs (which read self.elevator) during its setup below.
        self._elevator_inst: Optional[Elevator] = None
        self._contact_prev: Optional[torch.Tensor] = None

        super().__init__(cfg, render_mode, **kwargs)

        # Once num_envs/device are known, allocate the contact rising-edge tracker.
        self._contact_prev = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------
    @property
    def elevator(self) -> Elevator:
        """Return the lazily-constructed Elevator runtime.

        Instantiation is deferred so that ``self.scene`` has fully spawned the
        elevator USD prims by the time ``_initialize(env_paths)`` needs to walk
        the stage.
        """
        if self._elevator_inst is None:
            env_paths = [
                f"/World/envs/env_{i}/Elevator" for i in range(self.num_envs)
            ]
            inst = Elevator(self.cfg.scene.elevator)
            inst._initialize(env_paths)
            self._elevator_inst = inst
        return self._elevator_inst

    # ------------------------------------------------------------------
    def step(self, action: torch.Tensor) -> Any:
        self.elevator.update(self.step_dt)
        result = super().step(action)
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
        # Trigger lazy init if needed (first reset may fire before step).
        self.elevator.reset(ids)
        if self._contact_prev is not None:
            self._contact_prev[env_ids] = False

    # ------------------------------------------------------------------
    def _update_call_button_trigger(self) -> None:
        if self._contact_prev is None:
            return
        try:
            sensor = self.scene["contact_gripper"]
        except KeyError:
            return

        forces = sensor.data.force_matrix_w
        if forces is None or forces.numel() == 0:
            return
        mags = torch.linalg.norm(forces, dim=-1)
        contact_now = mags.amax(dim=(-1, -2)) > _DEFAULT_FORCE_THRESHOLD

        rising = contact_now & (~self._contact_prev)
        self._contact_prev = contact_now.clone()

        for eid in rising.nonzero(as_tuple=False).flatten().tolist():
            self.elevator.press_call_button(eid, call_index=0)
