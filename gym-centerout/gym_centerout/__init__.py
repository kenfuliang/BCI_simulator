import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Centerout-v1',
    entry_point='gym_centerout.envs:CenteroutEnv',
    #timestep_limit=4000,
    #reward_threshold=1.0,
    #nondeterministic = True,
)

#register(
#    id='Centerout2-v0',
#    entry_point='gym_centerout.envs:CenteroutEnv2',
#    #timestep_limit=4000,
#    #reward_threshold=1.0,
#    #nondeterministic = True,
#)

register(
    id='DelayedCenterOut-v1',
    entry_point='gym_centerout.envs:DelayedCenterOutEnv'
)