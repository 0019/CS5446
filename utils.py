## file that contains utility functions 
from supersuit import  frame_stack_v1, color_reduction_v0, \
    max_observation_v0, frame_skip_v0, resize_v0, clip_reward_v0, black_death_v3

def preset_settings_env(env):
    '''
    Presets certain conditions on the environment for 
    better scaling of the observations etc. 
    '''
    ## doing the max and skip as defined by deepmind. 
    env = max_observation_v0(env, memory=2)
    env = frame_skip_v0(env, 4)
    env = black_death_v3(env)
    
    env = frame_stack_v1(color_reduction_v0(env, 'R'), 4)
    
    env = resize_v0(env, 84, 84, linear_interp=True)

    env = clip_reward_v0(env,-1, 1)

    return env






