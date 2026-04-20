# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Static pedestal (gray box) used as the base for the OMY arm in elevator tasks.

Declared as ``RigidObjectCfg`` (not AssetBaseCfg) so the reset event can call
``write_root_pose_to_sim`` to re-place the pedestal directly beneath the robot.
``kinematic_enabled=True`` keeps it immovable under gravity / robot load.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg


PEDESTAL_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Pedestal",
    spawn=sim_utils.CuboidCfg(
        size=(0.5, 0.5, 1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
)
