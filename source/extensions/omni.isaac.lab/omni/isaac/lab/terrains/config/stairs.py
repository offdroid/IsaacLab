# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import omni.isaac.lab.terrains as terrain_gen

from ..terrain_generator_cfg import TerrainGeneratorCfg

STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(
        15.0,
        30.5,
    ),  # make it long enough to make sure robot doesn't fall down at end of platform (1ms * 20s = 20m)
    border_width=0.2,
    num_rows=13,
    num_cols=20,
    horizontal_scale=0.1,  # not relevant, I think
    vertical_scale=0.005,  # not relevant, I think
    slope_threshold=0.75,  # not relevant, I think
    use_cache=False,
    sub_terrains={
        "stairs": terrain_gen.MeshStairsTerrainCfg(
            step_height_range=(0.08, 0.16),  # demo was done for step height 0.14
            width_to_height_ratio=34 / 14,  # demo recorded for 34 / 14,
            y_coordinate_origin_relative_to_first_stair_step=-1.5,
            mode="scaled_norm",
        ),
    },
)
"""Stair terrains configuration."""
