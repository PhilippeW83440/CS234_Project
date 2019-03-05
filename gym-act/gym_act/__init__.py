import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Act-v0',
    entry_point='gym_act.envs:ActEnv',
)
