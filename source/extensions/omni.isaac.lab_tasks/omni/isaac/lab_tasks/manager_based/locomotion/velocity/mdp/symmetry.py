import torch
from typing import Optional, Tuple


# https://github.com/jloganolson/g1_23dof_locomotion_isaac/blob/d6c1c13c03cbe22b1e08bb1fc324e641023c930e/source/g1_23dof_locomotion_isaac/g1_23dof_locomotion_isaac/tasks/manager_based/g1_23dof_locomotion_isaac/agents/rsl_rl_ppo_cfg.py#L120
def mirror_joint_tensor(
    original: torch.Tensor, mirrored: torch.Tensor, offset: int = 0
) -> torch.Tensor:
    """Mirror a tensor of joint values by swapping left/right pairs and inverting yaw/roll joints.

    Args:
        original: Input tensor of shape [..., num_joints] where num_joints is 23
        mirrored: Output tensor of same shape to store mirrored values
        offset: Optional offset to add to indices if tensor has additional dimensions

    Returns:
        Mirrored tensor with same shape as input
    """
    lut = torch.tensor([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10])
    mirrored = original[..., lut + offset]
    return mirrored


def obs_prime(obs):
    """
    assuming a uniform history of 5 observations for every obs group

    base_lin_vel
    size 3
    offset 0
    ---
    base_ang_vel
    size 3
    offset 15
    ---
    velocity_commands
    size 4
    offset 30
    ---
    projected_gravity
    size 3
    offset 50
    ---
    joint_pos
    size 12
    offset 65
    ---
    joint_vel
    size 12
    offset 125
    ---
    actions
    size 12
    offset 185
    ---
    relative_position_on_stairs
    size 6
    offset 245
    ---
    stair_parameters
    size 2
    offset 275
    ---
    """
    if obs is None:
        return obs

    _obs = torch.clone(obs)
    flipped_obs = torch.clone(obs)

    for i in range(5):
        # Mirror base linear velocity (flip y)
        idx = 0 + 1 + 3 * i
        flipped_obs[..., idx] = -_obs[..., idx]  # y component of base_lin_vel

        # Mirror base angular velocity (flip z)
        idx = 15 + 2 + 3 * i
        flipped_obs[..., idx] = -_obs[..., idx]  # z component of base_ang_vel

        # Mirror projected gravity (flip y)
        idx = 50 + 1 + 3 * i
        flipped_obs[..., idx] = -_obs[..., idx]  # y component of projected_gravity

        # Mirror velocity commands (flip y, z, and yaw)
        idx = 30 + 1 + 4 * i
        flipped_obs[..., idx] = -_obs[..., idx]  # y component of velocity_commands
        idx = 30 + 2 + 4 * i
        flipped_obs[..., idx] = -_obs[..., idx]  # z component of velocity_commands
        idx = 30 + 3 + 4 * i
        flipped_obs[..., idx] = -_obs[..., idx]  # yaw component of velocity_commands

        mirror_joint_tensor(_obs, flipped_obs, 65 + 12 * i)
        mirror_joint_tensor(_obs, flipped_obs, 125 + 12 * i)
        mirror_joint_tensor(_obs, flipped_obs, 185 + 12 * i)

        idx = 245 + 2 + 6 * i
        _obs[..., idx : idx + 4] = torch.cat(
            (
                -_obs[..., 0 + idx : 1 + idx],
                _obs[..., 1 + idx : 2 + idx],
                -_obs[..., 2 + idx : 3 + idx],
                -_obs[..., 3 + idx : 4 + idx],
            ),
            dim=1,
        )
        flipped_obs[..., idx] = -_obs[
            ..., idx
        ]  # cosine encoding (yaw) of relative_position_on_stairs

    return torch.vstack((_obs, flipped_obs))


def actions_prime(actions):
    if actions is None:
        return None

    _actions = torch.clone(actions)
    flip_actions = torch.zeros_like(_actions)
    mirror_joint_tensor(_actions, flip_actions)
    return torch.vstack((_actions, flip_actions))


def data_augmentation_func(env, obs, actions, obs_type="policy"):
    assert (
        obs is None or obs.shape[-1] == 285
    ), "Obs shape varies from expected. Manual reconfiguration of the data data_augmentation_func (for symmetry) might be necessary"
    obs_mirrored= obs_prime(obs)
    actions_mirrored = actions_prime(actions)
    if obs is not None:
        obs =  torch.vstack((obs, obs_mirrored))
    if actions is not None:
        actions = torch.vstack((actions, actions_mirrored))
    return obs, actions


@torch.no_grad()
def get_symmetric_states(
    obs: torch.Tensor | None,
    actions: torch.Tensor | None,
    env: "LocomotionVelocityRoughEnvCfg" = None,
    obs_type: str = "policy",
):
    am = env.unwrapped.action_manager
    action = am.get_term("joint_pos")

    # Get joint names and indices
    joint_indices, joint_names = action._asset.find_joints(
        action.cfg.joint_names, preserve_order=True
    )

    # Build a mapping from name to index for quick lookup
    name_to_index = {name: idx for name, idx in zip(joint_names, joint_indices)}

    # Function to swap 'L' and 'R' in joint names
    def swap_lr(name: str):
        if "L" in name[1:]:
            return name[0] + name[1:].replace("L", "R", 1)
        elif "R" in name[1:]:
            return name[0] + name[1:].replace("R", "L", 1)
        else:
            return name

    # Build the lookup table: index -> symmetric index
    symmetric_indices = []
    for name, idx in zip(joint_names, joint_indices):
        sym_name = swap_lr(name)
        sym_idx = name_to_index.get(sym_name, idx)  # fallback to self if not found
        symmetric_indices.append(sym_idx)

    print("Symmetric indices lookup table:", symmetric_indices)

    # Process observations
    assert not isinstance(obs, torch.Tensor)
    offset = 0
    for v, k in obs.items():
        print(v)
        print("size", k.shape[-1] // 5)
        print("offset", offset)
        offset += k.shape[-1]
        print("---")

    # TODO: Finish the obs part
    def fn(
        obs: torch.Tensor | None,
        actions: torch.Tensor | None,
        env: "LocomotionVelocityRoughEnvCfg" = None,
        obs_type: str = "policy",
    ):
        return obs, actions[symmetric_indices]

    return fn
