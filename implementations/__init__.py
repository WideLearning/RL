from gymnasium.envs.registration import register

register(
    id="MultiArmedBandit",
    entry_point="implementations.envs.MultiArmedBandit:MultiArmedBanditEnv",
    max_episode_steps=1000,
)
