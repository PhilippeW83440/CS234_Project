from gym.envs.registration import register

register(
    id='Act-v0',
    entry_point='gym_act.envs:ActEnv',
)
