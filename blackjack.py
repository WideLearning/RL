import gymnasium as gym

env = gym.make("Blackjack-v1", render_mode="human")
_observation, _info = env.reset()
for i in range(1000):
    action = env.action_space.sample()
    _observation, _reward, terminated, _truncated, _info = env.step(action)
    env.render()
    if terminated:
        _observation, _info = env.reset()
