# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import gymnasium as gym


gym.register(
    id="RobotisLab-Real-Elevator-Call-OMY-v0",
    entry_point=(
        "robotis_lab.real_world_tasks.manager_based.OMY.elevator_call."
        "elevator_call_env:OMYElevatorCallEnv"
    ),
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:OMYElevatorCallEnvCfg",
    },
    disable_env_checker=True,
)
