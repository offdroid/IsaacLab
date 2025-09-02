def get_vel_dependent_actor_latent_dim_for_action_manager_class(
    action_manager_class: str,
) -> int:
    if action_manager_class == "LegwiseLatentActionManager":
        return 9  # main freq, 4 legs with freq and amp each 1 + 4 * 2
    elif action_manager_class == "InterpolatedStyleActionManager":
        return 0  # no latent actions
    elif action_manager_class in [
        "FrequencyInterpolatedStyleActionManager",
        "ResidualRLActionManager",
    ]:
        return 1  # main frequency
    elif action_manager_class in ["ActionManager", "MotionBlendingActionManager", "OscillatorActionManager"]:
        return 0  # Default
    else:
        raise ValueError(f"Unknown action_manager_class: {action_manager_class}")
