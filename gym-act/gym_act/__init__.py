import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# Default settings just 2 objs with a cv (Constant Velocity) driver model
register(
    id='Act-v0',
    entry_point='gym_act.envs:ActEnv',
)

# Act with a discrete action space: -2,-1,0,1,2
register(
    id='ActDiscrete-v0',
    entry_point='gym_act.envs:ActEnv',
    kwargs={'discrete' : True},
)

register(
    id='Act10cv-v0',
    entry_point='gym_act.envs:ActEnv',
    kwargs={'nobjs' : 10, 'driver_model' : 'cv'},
)
