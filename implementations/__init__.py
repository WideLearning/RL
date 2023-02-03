from gymnasium.envs.registration import register

register(
    id="bandit",
    entry_point="implementations.envs.bandit:Bandit",
    max_episode_steps=1000,
)
